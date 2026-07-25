"""
One-off migration: Chinook SQLite → Hopsworks feature groups.

Run this once, after `feature_pipeline.py`, to move the agent off SQLite:

    python feature_pipeline.py            # catalogue name embeddings
    python migrate_to_feature_store.py    # everything else

## What the shapes are, and why

The online feature store is a keyed lookup, not a query engine: you fetch a
feature vector by primary key. SQLite let the agent write arbitrary joins and
WHERE clauses; the feature store will not, so the migration denormalises around
the two questions the agent actually asks and makes each one a single keyed
read.

| Feature group | Key | Answers |
|---|---|---|
| `chinook_artist_catalog` | `artist_name` | "what albums/tracks does this artist have?" |
| `chinook_customers` | `customer_key` | "who is this, and which lines are theirs?" |
| `chinook_purchases` | `invoice_line_id` | one row per purchased line |
| `chinook_refunds` | `invoice_line_id` | "has this line already been refunded?" |

`customer_key` is a deterministic hash of the first name, last name and phone
the customer gives — the same three fields the refund flow already asks for, so
the agent can compute the key from the conversation without a lookup first.

Fuzzy name matching is not modelled here: `chinook_catalog_embeddings` (from
`feature_pipeline.py`) already resolves what a customer typed to the canonical
name, which is then the key into `chinook_artist_catalog`.

## Refunds become events, not deletions

The SQLite version implemented a refund by DELETEing the Invoice and InvoiceLine
rows. That cannot be ported, and should not be: **a feature group has no
row-level delete reachable from a Python client.** `FeatureGroup.delete()` drops
the entire feature group, `commit_delete_record()` needs Spark and a
Hudi/Delta/Iceberg table, and the Python engine has no per-row delete at all.

So `chinook_refunds` is an append-only ledger keyed by `invoice_line_id`, and a
line counts as refunded when a row exists for it. That is better domain
modelling regardless of the platform — destroying the record of a sale to
represent a refund loses the audit trail that any real store needs — but it does
change agent behaviour: refunded purchases still appear in a customer's history,
marked as refunded, rather than vanishing.
"""

import hashlib
import json
import logging
import os
import sqlite3
import urllib.request
from collections import defaultdict

import hopsworks
import pandas as pd
from hsfs.feature import Feature

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_DB = os.environ.get("CHINOOK_DB_PATH", "chinook.db")

ARTIST_FG = "chinook_artist_catalog"
CUSTOMERS_FG = "chinook_customers"
PURCHASES_FG = "chinook_purchases"
LEGACY_PURCHASES_FG = "chinook_customer_purchases"
REFUNDS_FG = "chinook_refunds"
FG_VERSION = 1


def customer_key(first_name: str, last_name: str, phone: str) -> str:
    """Stable key from the three fields the refund flow asks the customer for.

    Normalised so "Aaron", " aaron " and "AARON" agree, and so phone formatting
    differences do not produce a different customer.
    """
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    raw = f"{(first_name or '').strip().lower()}|{(last_name or '').strip().lower()}|{digits}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def ensure_db(path: str = CHINOOK_DB) -> str:
    if not os.path.exists(path):
        log.info("Downloading Chinook DB → %s", path)
        urllib.request.urlretrieve(CHINOOK_URL, path)
    return path


def build_artist_catalog(conn) -> pd.DataFrame:
    """One row per artist, carrying their albums and tracks as JSON."""
    rows = conn.execute(
        """
        SELECT ar.Name, al.Title, t.Name, t.UnitPrice, g.Name
        FROM Artist ar
        LEFT JOIN Album al ON al.ArtistId = ar.ArtistId
        LEFT JOIN Track t  ON t.AlbumId  = al.AlbumId
        LEFT JOIN Genre g  ON g.GenreId  = t.GenreId
        """
    ).fetchall()

    albums: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for artist, album, track, price, genre in rows:
        if not artist:
            continue
        if album:
            albums[artist][album].append(
                {"track_name": track, "unit_price": price, "genre": genre}
                if track
                else None
            )

    records = []
    for artist, by_album in albums.items():
        payload = [
            {"album_title": album, "tracks": [t for t in tracks if t]}
            for album, tracks in by_album.items()
        ]
        records.append(
            {
                "artist_name": artist,
                "album_count": len(payload),
                "track_count": sum(len(a["tracks"]) for a in payload),
                # JSON in a TEXT column: nothing queries inside it, and the
                # online store's JSON support is weaker than plain text
                "albums": json.dumps(payload),
            }
        )
    # artists with no albums still deserve a row, so a lookup returns "no albums"
    # rather than nothing at all
    for (artist,) in conn.execute("SELECT Name FROM Artist").fetchall():
        if artist and artist not in albums:
            records.append(
                {
                    "artist_name": artist,
                    "album_count": 0,
                    "track_count": 0,
                    "albums": json.dumps([]),
                }
            )
    return pd.DataFrame(records)


def build_customers_and_purchases(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two frames: one row per customer, and one row per purchased line.

    The online store does point lookups by primary key — it cannot scan for
    "every line belonging to this customer". So the customer row carries the
    list of their line ids, and that list is the index: read the customer, then
    batch-read the lines. Small (a few hundred bytes at Chinook's ~38 lines per
    customer) and, unlike the JSON blob this replaces, adding one purchase
    rewrites one short list rather than the customer's entire history.
    """
    rows = conn.execute(
        """
        SELECT c.FirstName, c.LastName, c.Phone, c.Email,
               il.InvoiceLineId, il.InvoiceId, t.Name, art.Name, alb.Title,
               i.InvoiceDate, il.Quantity, il.UnitPrice
        FROM InvoiceLine il
        JOIN Invoice  i   ON il.InvoiceId = i.InvoiceId
        JOIN Customer c   ON i.CustomerId = c.CustomerId
        JOIN Track    t   ON il.TrackId  = t.TrackId
        JOIN Album    alb ON t.AlbumId   = alb.AlbumId
        JOIN Artist   art ON alb.ArtistId = art.ArtistId
        """
    ).fetchall()

    customers: dict[str, dict] = {}
    purchases: list[dict] = []
    for (first, last, phone, email, line_id, invoice_id, track, artist, album,
         date, qty, price) in rows:
        key = customer_key(first, last, phone)
        entry = customers.setdefault(
            key,
            {"customer_key": key, "first_name": first, "last_name": last,
             "phone": phone, "email": email, "_ids": []},
        )
        entry["_ids"].append(int(line_id))
        purchases.append(
            {
                "invoice_line_id": int(line_id),
                "customer_key": key,
                "invoice_id": int(invoice_id),
                "track_name": track,
                "artist_name": artist,
                "album_title": album,
                "purchase_date": date,
                "quantity_purchased": int(qty),
                "price_per_unit": float(price),
            }
        )

    records = []
    for entry in customers.values():
        ids = sorted(entry.pop("_ids"))
        entry["line_count"] = len(ids)
        entry["line_ids"] = json.dumps(ids)
        records.append(entry)
    return pd.DataFrame(records), pd.DataFrame(purchases)



def _json_feature(name: str) -> Feature:
    """A JSON payload column, declared TEXT rather than left to inference.

    hsfs sizes string columns from the widest value it sees and emits
    `varchar(N)`; an online feature group's row must fit in 30000 bytes, and
    utf8mb4 charges 4 bytes per character. The widest artist catalogue here is
    ~17k characters, so inference produced varchar(17100) — an estimated 68832
    byte row, which the backend rejects outright:

        Cannot create an online feature group because row size > 30000 bytes

    TEXT is stored out of row in RonDB, so it does not count against that limit,
    and hsfs skips the widening entirely for a non-varchar online type. Verified
    against a live feature store: the same frame is rejected with inferred types
    and accepted with this one.
    """
    return Feature(name, type="string", online_type="text")


def main() -> None:
    db = ensure_db()
    conn = sqlite3.connect(db)
    try:
        artists = build_artist_catalog(conn)
        customers, purchases = build_customers_and_purchases(conn)
    finally:
        conn.close()
    log.info("artists: %d | customers: %d | purchase lines: %d",
             len(artists), len(customers), len(purchases))

    now = pd.Timestamp.utcnow().tz_localize(None)
    artists["migrated_at"] = now
    customers["migrated_at"] = now
    purchases["migrated_at"] = now

    project = hopsworks.login()
    fs = project.get_feature_store()

    artist_fg = fs.get_or_create_feature_group(
        name=ARTIST_FG,
        version=FG_VERSION,
        description="Chinook catalogue denormalised per artist (albums + tracks as JSON)",
        primary_key=["artist_name"],
        event_time="migrated_at",
        online_enabled=True,
        features=[
            Feature("artist_name", type="string", online_type="varchar(200)"),
            # bigint, not int: pandas defaults integer columns to int64, which
            # hsfs derives as 'bigint'. Declaring 'int' (its name for int32)
            # fails schema verification before anything is written.
            Feature("album_count", type="bigint"),
            Feature("track_count", type="bigint"),
            _json_feature("albums"),
            Feature("migrated_at", type="timestamp"),
        ],
    )
    artist_fg.insert(artists, write_options={"wait_for_job": True})

    customers_fg = fs.get_or_create_feature_group(
        name=CUSTOMERS_FG,
        version=FG_VERSION,
        description="Chinook customers; line_ids indexes their purchase lines",
        primary_key=["customer_key"],
        event_time="migrated_at",
        online_enabled=True,
        features=[
            Feature("customer_key", type="string", online_type="varchar(64)"),
            Feature("first_name", type="string", online_type="varchar(100)"),
            Feature("last_name", type="string", online_type="varchar(100)"),
            Feature("phone", type="string", online_type="varchar(50)"),
            Feature("email", type="string", online_type="varchar(200)"),
            Feature("line_count", type="bigint"),
            # a few hundred bytes even for the busiest customer, so a plain
            # varchar is fine here — unlike the blob this replaces
            Feature("line_ids", type="string", online_type="varchar(2000)"),
            Feature("migrated_at", type="timestamp"),
        ],
    )
    customers_fg.insert(customers, write_options={"wait_for_job": True})

    purchases_fg = fs.get_or_create_feature_group(
        name=PURCHASES_FG,
        version=FG_VERSION,
        description="One row per purchased invoice line",
        primary_key=["invoice_line_id"],
        event_time="migrated_at",
        online_enabled=True,
        features=[
            Feature("invoice_line_id", type="bigint"),
            Feature("customer_key", type="string", online_type="varchar(64)"),
            Feature("invoice_id", type="bigint"),
            Feature("track_name", type="string", online_type="varchar(400)"),
            Feature("artist_name", type="string", online_type="varchar(200)"),
            Feature("album_title", type="string", online_type="varchar(400)"),
            Feature("purchase_date", type="string", online_type="varchar(40)"),
            Feature("quantity_purchased", type="bigint"),
            Feature("price_per_unit", type="double"),
            Feature("migrated_at", type="timestamp"),
        ],
    )
    purchases_fg.insert(purchases, write_options={"wait_for_job": True})

    # Created empty and written to by the agent, one row per refunded line.
    refunds_fg = fs.get_or_create_feature_group(
        name=REFUNDS_FG,
        version=FG_VERSION,
        description="Append-only refund ledger; a line is refunded when a row exists",
        primary_key=["invoice_line_id"],
        event_time="refunded_at",
        online_enabled=True,
    )
    seed = pd.DataFrame(
        [
            {
                "invoice_line_id": -1,
                "invoice_id": -1,
                "customer_key": "__seed__",
                "amount": 0.0,
                "refunded_at": now,
            }
        ]
    )
    # A feature group has no schema until something is written, and the agent
    # can only append online — so seed one sentinel row here, where an offline
    # write is available, rather than making the agent's first refund special.
    refunds_fg.insert(seed, write_options={"wait_for_job": True})

    # Online point lookups go through a feature view, not the feature group.
    for name, fg in (
        (ARTIST_FG, artist_fg),
        (CUSTOMERS_FG, customers_fg),
        (PURCHASES_FG, purchases_fg),
        (REFUNDS_FG, refunds_fg),
    ):
        fs.get_or_create_feature_view(
            name=name, version=FG_VERSION, query=fg.select_all()
        )
        log.info("feature view ready: %s v%d", name, FG_VERSION)

    try:
        legacy = fs.get_feature_group(LEGACY_PURCHASES_FG, version=FG_VERSION)
    except Exception:  # noqa: BLE001
        legacy = None
    if legacy is not None:
        log.warning(
            "%s is superseded by %s + %s and is no longer read. Delete it when "
            "you are happy with the new shape.",
            LEGACY_PURCHASES_FG, CUSTOMERS_FG, PURCHASES_FG,
        )

    log.info("Migration complete. The agent no longer needs chinook.db.")


if __name__ == "__main__":
    main()
