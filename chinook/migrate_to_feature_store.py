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
| `chinook_customer_purchases` | `customer_key` | "what did this customer buy?" |
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_DB = os.environ.get("CHINOOK_DB_PATH", "chinook.db")

ARTIST_FG = "chinook_artist_catalog"
PURCHASES_FG = "chinook_customer_purchases"
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


def build_customer_purchases(conn) -> pd.DataFrame:
    """One row per customer, carrying every invoice line they bought as JSON."""
    rows = conn.execute(
        """
        SELECT c.FirstName, c.LastName, c.Phone,
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

    by_customer: dict[str, dict] = {}
    for (
        first,
        last,
        phone,
        line_id,
        invoice_id,
        track,
        artist,
        album,
        date,
        qty,
        price,
    ) in rows:
        key = customer_key(first, last, phone)
        entry = by_customer.setdefault(
            key,
            {
                "customer_key": key,
                "first_name": first,
                "last_name": last,
                "phone": phone,
                "_lines": [],
            },
        )
        entry["_lines"].append(
            {
                "invoice_line_id": line_id,
                "invoice_id": invoice_id,
                "track_name": track,
                "artist_name": artist,
                "album_title": album,
                "purchase_date": date,
                "quantity_purchased": qty,
                "price_per_unit": price,
            }
        )

    records = []
    for entry in by_customer.values():
        lines = entry.pop("_lines")
        entry["line_count"] = len(lines)
        entry["purchases"] = json.dumps(lines)
        records.append(entry)
    return pd.DataFrame(records)


def main() -> None:
    db = ensure_db()
    conn = sqlite3.connect(db)
    try:
        artists = build_artist_catalog(conn)
        purchases = build_customer_purchases(conn)
    finally:
        conn.close()
    log.info("artists: %d rows | customers: %d rows", len(artists), len(purchases))

    now = pd.Timestamp.utcnow().tz_localize(None)
    artists["migrated_at"] = now
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
    )
    artist_fg.insert(artists, write_options={"wait_for_job": True})

    purchases_fg = fs.get_or_create_feature_group(
        name=PURCHASES_FG,
        version=FG_VERSION,
        description="Chinook invoice lines denormalised per customer (purchases as JSON)",
        primary_key=["customer_key"],
        event_time="migrated_at",
        online_enabled=True,
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
        (PURCHASES_FG, purchases_fg),
        (REFUNDS_FG, refunds_fg),
    ):
        fs.get_or_create_feature_view(
            name=name, version=FG_VERSION, query=fg.select_all()
        )
        log.info("feature view ready: %s v%d", name, FG_VERSION)

    log.info("Migration complete. The agent no longer needs chinook.db.")


if __name__ == "__main__":
    main()
