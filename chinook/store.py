"""The store, and everything an agent can do to it.

Framework-agnostic on purpose. `support_agent.py` wraps these as LangChain tools
for a LangGraph supervisor and `support_agent_llamaindex.py` wraps the same
functions as LlamaIndex FunctionTools — two agents over one implementation,
because the interesting rules live here and a second copy of them is a second
set of rules.

The docstrings are load-bearing: both frameworks read them as the tool
description the model chooses on, so they are written for the model rather than
for a reader of this file.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from typing import Literal

import hopsworks
import pandas as pd
from sentence_transformers import SentenceTransformer

from hopsworks_agent_protocol import AgentError

log = logging.getLogger(__name__)

from typing_extensions import Annotated, TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Set by the platform on a deployment meant for evaluation. AgentApp reports it
# in the manifest as `eval_mode`, and the eval runner refuses to run a sandboxed
# suite against a deployment that does not — a suite that makes the agent place
# orders must not place them against real customers.
#
# What it changes here is exactly one thing: the three writes below do not
# happen. Everything the run can observe is identical — the same tools are
# called with the same arguments, and every tool returns the same text — because
# a deployment that behaved differently under evaluation would be measuring
# something other than the agent that serves customers.
# Named EVAL_MODE rather than HOPSWORKS_EVAL_MODE: the platform reserves the
# HOPS_, HOPSWORKS_, HOPSFS_ and AGENT_ prefixes, so a deployment cannot set
# either of those and the flag would be unusable. Matches
# hopsworks_agent_protocol.conventions.EVAL_MODE_ENV, which is what AgentApp
# reports in the manifest and what the eval runner checks.
EVAL_MODE = os.environ.get("EVAL_MODE", "").strip().lower() in (
    "1", "true", "yes",
)

CATALOG_FG = "chinook_catalog_embeddings"
ARTIST_FG = "chinook_artist_catalog"
CUSTOMERS_FG = "chinook_customers"
PURCHASES_FG = "chinook_purchases"
REFUNDS_FG = "chinook_refunds"
FG_VERSION = 1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Same model as the RAGBench agents in ../ragbench, at temperature 0 for the
# same reason: routing and purchase-info extraction are schema-constrained
# steps that run on every turn, and a refund flow should not take a different
# branch on a re-ask. Override the answering model if the open-ended path
# needs more headroom.
ROUTER_MODEL = os.environ.get("CHINOOK_ROUTER_MODEL", "claude-haiku-4-5")
ANSWER_MODEL = os.environ.get("CHINOOK_ANSWER_MODEL", "claude-haiku-4-5")

IDENTITY_KEYS = ("customer_first_name", "customer_last_name", "customer_phone")

# Identity is stored per conversation, not per person, because nothing here can
# tell us who the person is: the chat transport authenticates a project-wide
# serving key, so `subject` is whatever the client claims. From the Hopsworks
# panel that is the logged-in Hopsworks user (`meb10000`) — one stable value
# shared by every conversation and every customer the agent talks to. So
# `user` scope here means "everyone who reaches this deployment", not "this
# person", and anything written there is read back by the next conversation.
# Writing identity to `session` scope says that plainly.
#
# When real end-user identity arrives — the deployment-scoped chat token the
# panel design calls for — flipping this to "user" is the whole change needed
# to make the ask happen once per customer rather than once per conversation.
IDENTITY_SCOPE = "session"

# The identity of the customer in the turn being handled. A LangChain tool is
# invoked by the model and gets no graph state, so the handler puts it here and
# the purchase-history tool reads it back. A ContextVar rather than a plain
# global because two turns can be in flight at once and must not see each
# other's customer.
_current_identity: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "chinook_identity", default={}
)

# Durable, cross-conversation memory is keyed on the customer, not the
# conversation — using the same customer_key the purchases feature group is
# keyed on. The agent asks for name and phone anyway, so the application can
# decide what identity means rather than waiting for the platform to provide
# one. It is a client-asserted identity (a customer could claim to be someone
# else), but that is already the trust model for looking up their orders, so
# this adds no authority they did not have.
INTEREST_SCOPE = "user"
WANTS_PREFIX = "wants:"
LIKES_PREFIX = "likes:"


# ── data access ──────────────────────────────────────────────────────────────


_embed = SentenceTransformer(EMBEDDING_MODEL)
_project = hopsworks.login()
_fs = _project.get_feature_store()
_catalog_fg = None
_catalog_features: list[str] = []
_views: dict[str, object] = {}


def customer_key(first_name: str, last_name: str, phone: str) -> str:
    """Must match migrate_to_feature_store.customer_key exactly — it is the
    primary key the purchases feature group is written under."""
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    raw = f"{(first_name or '').strip().lower()}|{(last_name or '').strip().lower()}|{digits}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _view(name: str):
    """A serving-initialised feature view, cached for the process.

    Online point lookups go through a feature view rather than the feature
    group; ``init_serving`` opens the online client and is the expensive part,
    so it happens once per view rather than per turn.
    """
    if name not in _views:
        view = _fs.get_feature_view(name=name, version=FG_VERSION)
        if view is None:
            # hsfs returns None rather than raising for a view that was never
            # created, so without this the caller sees an AttributeError on
            # None and has to guess why. The cause is almost always that the
            # migration job has not run against this project yet.
            raise LookupError(
                f"Feature view {name!r} v{FG_VERSION} does not exist — "
                "run chinook/migrate_to_feature_store.py first"
            )
        view.init_serving()
        _views[name] = view
        log.info("Feature view ready: %s", name)
    return _views[name]


def _lookup_one(view_name: str, entry: dict) -> dict | None:
    """One keyed online read, or None when the key is absent."""
    try:
        view = _view(view_name)
        row = view.get_feature_vector(entry, return_type="pandas")
    except Exception:  # noqa: BLE001 — a missing key raises on some versions
        log.exception("Lookup in %s failed for %s", view_name, entry)
        return None
    if row is None or getattr(row, "empty", False):
        return None
    record = row.iloc[0].to_dict()
    # A key that does not exist neither raises nor returns an empty frame: it
    # returns one row of nulls. Without this check the caller gets a truthy
    # dict of NaNs — which would make _refunded_line_ids treat every line as
    # already refunded, so every refund would quietly come to $0.00.
    if all(pd.isna(value) for value in record.values()):
        return None
    return record


def _lookup_many(view_name: str, entries: list[dict]) -> list[dict]:
    """One batched online read for many keys.

    Purchase lines are one row each now, so a customer's history is N point
    lookups. Batching them keeps that a single round trip.
    """
    if not entries:
        return []
    try:
        view = _view(view_name)
        frame = view.get_feature_vectors(entries, return_type="pandas")
    except Exception:  # noqa: BLE001
        log.exception("Batch lookup in %s failed for %d keys", view_name, len(entries))
        return []
    if frame is None or getattr(frame, "empty", True):
        return []
    # Unlike the single-key read, which returns one row of nulls for a missing
    # key, the batch read simply omits it: N keys can come back as fewer than N
    # rows, in unspecified order. So nothing here may assume the result lines up
    # positionally with `entries` — the rows carry their own keys. The null-row
    # filter below is belt-and-braces for the single-key shape leaking in.
    return [
        row.to_dict()
        for _, row in frame.iterrows()
        if not row.isnull().all()
    ]


def _catalog():
    """The name-embedding feature group, opened lazily and cached."""
    global _catalog_fg, _catalog_features
    if _catalog_fg is None:
        _catalog_fg = _fs.get_feature_group(CATALOG_FG, version=FG_VERSION)
        if _catalog_fg is not None:
            _catalog_features = [f.name for f in _catalog_fg.features]
            log.info("Catalogue index ready (%s)", CATALOG_FG)
    return _catalog_fg


def resolve_name(text: str, kind: Literal["track", "artist", "album"]) -> str:
    """Snap what the customer typed to the closest name the catalogue stores.

    "prince" → "Prince", which is then a valid key into the artist catalogue.
    Falls back to the raw input if the index is unavailable.
    """
    fg = _catalog()
    if fg is None or not text:
        return text
    try:
        vector = _embed.encode(text, normalize_embeddings=True).tolist()
        # Build the filter with get_feature(), never getattr(fg, "entity_kind").
        # Attribute access returns a FeatureGroup attribute when the column name
        # collides with one, and the resulting `False` is translated to *no
        # filter at all* — unfiltered neighbours, with no error.
        condition = fg.get_feature("entity_kind") == kind
        hits = fg.find_neighbors(vector, col="embedding", k=1, filter=condition)
    except Exception:  # noqa: BLE001 — disambiguation is best-effort
        log.exception("Catalogue lookup failed for %r; using the raw input", text)
        return text
    if not hits:
        return text
    return dict(zip(_catalog_features, hits[0][1])).get("name") or text


def artist_catalog(artist_name: str) -> list[dict]:
    """Every album (with its tracks) for one canonical artist name."""
    row = _lookup_one(ARTIST_FG, {"artist_name": artist_name})
    if not row:
        return []
    return json.loads(row.get("albums") or "[]")


def _refunded_line_ids(line_ids: list[int]) -> set[int]:
    """Which of these invoice lines already have a row in the refund ledger."""
    refunded = set()
    for line_id in line_ids:
        if _lookup_one(REFUNDS_FG, {"invoice_line_id": int(line_id)}):
            refunded.add(int(line_id))
    return refunded


def _refund(
    invoice_id: int | None, invoice_line_ids: list[int] | None, mock: bool = False
) -> float:
    """Record a refund for the given invoice lines, returning the total refunded.

    This *appends to a ledger*; it does not delete the sale. A feature group has
    no row-level delete reachable from a Python client (``FeatureGroup.delete()``
    drops the whole group, ``commit_delete_record`` needs Spark), and destroying
    the record of a sale to represent a refund would lose the audit trail
    anyway. A line counts as refunded once a row exists for it.
    """
    lines = list(invoice_line_ids or [])
    if invoice_id is not None:
        # every line on the invoice, minus anything already refunded
        lines += [
            line["invoice_line_id"]
            for line in _purchases_for_invoice(invoice_id)
            if line["invoice_line_id"] not in lines
        ]
    if not lines:
        return 0.0

    known = {line["invoice_line_id"]: line for line in _all_known_lines(lines)}
    already = _refunded_line_ids(lines)
    to_refund = [line_id for line_id in lines if line_id not in already]
    total = sum(
        (known[i]["price_per_unit"] or 0) * (known[i]["quantity_purchased"] or 1)
        for i in to_refund
        if i in known
    )
    if mock or not to_refund:
        return float(total)

    now = pd.Timestamp.utcnow().tz_localize(None)
    frame = pd.DataFrame(
        [
            {
                "invoice_line_id": int(i),
                "invoice_id": int(known.get(i, {}).get("invoice_id") or -1),
                "customer_key": known.get(i, {}).get("customer_key") or "",
                "amount": float(
                    (known.get(i, {}).get("price_per_unit") or 0)
                    * (known.get(i, {}).get("quantity_purchased") or 1)
                ),
                "refunded_at": now,
            }
            for i in to_refund
        ]
    )
    if EVAL_MODE:
        log.info("eval mode: not recording a refund of %.2f for lines %s",
                 total, to_refund)
        return float(total)
    try:
        # storage="online" is required: the offline write goes through HopsFS,
        # which an agent pod cannot reach.
        _fs.get_feature_group(REFUNDS_FG, version=FG_VERSION).insert(
            frame, storage="online", write_options={"wait_for_job": False}
        )
    except Exception:  # noqa: BLE001 — never fail the turn on a ledger write
        log.exception("Could not record the refund for lines %s", to_refund)
        return 0.0
    return float(total)


# Purchases are fetched per customer, so the refund path keeps the lines it
# looked up for this turn rather than re-reading them by invoice.
_turn_lines: dict[int, dict] = {}


def _all_known_lines(line_ids: list[int]) -> list[dict]:
    return [_turn_lines[i] for i in line_ids if i in _turn_lines]


def _purchases_for_invoice(invoice_id: int) -> list[dict]:
    return [
        line for line in _turn_lines.values() if line.get("invoice_id") == invoice_id
    ]


def _lookup(
    customer_first_name: str,
    customer_last_name: str,
    customer_phone: str,
    track_name: str | None,
    album_title: str | None,
    artist_name: str | None,
    purchase_date_iso_8601: str | None,
) -> list[dict]:
    """A customer's purchases, filtered by the optional criteria.

    Two reads: the customer row for the ids of their lines, then one batched
    read for the lines themselves. The online store cannot scan for "every line
    belonging to this customer", so that id list is the index. Filtering that
    used to be SQL happens here, over tens of rows. Already-refunded lines are
    annotated rather than hidden — the ledger records refunds, it does not erase
    the sale.
    """
    key = customer_key(customer_first_name, customer_last_name, customer_phone)
    customer = _lookup_one(CUSTOMERS_FG, {"customer_key": key})
    if not customer:
        return []
    try:
        line_ids = json.loads(customer.get("line_ids") or "[]")
    except (TypeError, ValueError):
        log.exception("Malformed line_ids for customer %s", key)
        return []
    lines = _lookup_many(
        PURCHASES_FG, [{"invoice_line_id": int(i)} for i in line_ids]
    )

    if track_name:
        wanted = resolve_name(track_name, "track")
        lines = [line for line in lines if line["track_name"] == wanted]
    if album_title:
        wanted = resolve_name(album_title, "album")
        lines = [line for line in lines if line["album_title"] == wanted]
    if artist_name:
        wanted = resolve_name(artist_name, "artist")
        lines = [line for line in lines if line["artist_name"] == wanted]
    if purchase_date_iso_8601:
        day = purchase_date_iso_8601[:10]
        lines = [line for line in lines if (line["purchase_date"] or "")[:10] == day]

    refunded = _refunded_line_ids([line["invoice_line_id"] for line in lines])
    for line in lines:
        line["refunded"] = line["invoice_line_id"] in refunded
    _turn_lines.update({line["invoice_line_id"]: line for line in lines})
    return lines


# ── refund sub-agent ─────────────────────────────────────────────────────────


def lookup_track(
    track_name: str | None = None,
    album_title: str | None = None,
    artist_name: str | None = None,
) -> list[dict]:
    """Look up tracks in the store's catalogue.

    Returns a list of dicts with keys {'track_name', 'artist_name', 'album_name'}.
    """
    if artist_name:
        canonical = resolve_name(artist_name, "artist")
        results = [
            {
                "track_name": track["track_name"],
                "artist_name": canonical,
                "album_name": album["album_title"],
            }
            for album in artist_catalog(canonical)
            for track in album["tracks"]
        ]
    elif album_title or track_name:
        # no artist given: resolve the album or track name, then find the artist
        # it belongs to by resolving that name back through the catalogue index
        wanted_album = resolve_name(album_title, "album") if album_title else None
        wanted_track = resolve_name(track_name, "track") if track_name else None
        canonical = resolve_name(wanted_album or wanted_track, "artist")
        results = [
            {
                "track_name": track["track_name"],
                "artist_name": canonical,
                "album_name": album["album_title"],
            }
            for album in artist_catalog(canonical)
            for track in album["tracks"]
        ]
    else:
        return []

    if album_title:
        wanted = resolve_name(album_title, "album")
        results = [r for r in results if r["album_name"] == wanted]
    if track_name:
        wanted = resolve_name(track_name, "track")
        results = [r for r in results if r["track_name"] == wanted]
    return results


def lookup_album(
    track_name: str | None = None,
    album_title: str | None = None,
    artist_name: str | None = None,
) -> list[dict]:
    """Look up albums in the store's catalogue.

    Returns a list of dicts with keys {'album_name', 'artist_name'}.
    """
    canonical = resolve_name(
        artist_name or album_title or track_name or "", "artist"
    )
    albums = artist_catalog(canonical)
    if album_title:
        wanted = resolve_name(album_title, "album")
        albums = [a for a in albums if a["album_title"] == wanted]
    if track_name:
        wanted = resolve_name(track_name, "track")
        albums = [
            a for a in albums if any(t["track_name"] == wanted for t in a["tracks"])
        ]
    return [{"album_name": a["album_title"], "artist_name": canonical} for a in albums]


def lookup_artist(
    track_name: str | None = None,
    album_title: str | None = None,
    artist_name: str | None = None,
) -> list[str]:
    """Look up artists in the store's catalogue. Returns matching artist names."""
    hint = artist_name or album_title or track_name
    if not hint:
        return []
    canonical = resolve_name(hint, "artist")
    # a row exists for every artist, so an empty catalogue still confirms the name
    return [canonical] if artist_catalog(canonical) is not None else []


def purchase_history(
    artist_name: str | None = None, album_title: str | None = None
) -> list[dict] | str:
    """What this customer has already bought from the store, grouped by album.

    Call this whenever the customer asks about their own orders, their history,
    what they own, or whether they already bought something — "what albums have
    I purchased", "have I bought this before", "what did I order in March".
    Optionally narrow to one artist or album. Do not use it for questions about
    the catalogue in general; that is what the lookup tools are for.

    Returns one entry per album with the tracks bought from it, the dates, and
    how many of those lines have since been refunded.
    """
    identity = _current_identity.get()
    missing = [key for key in IDENTITY_KEYS if not identity.get(key)]
    if missing:
        return (
            "I don't know who I'm speaking to yet, so I can't look up an order "
            "history. Ask the customer for their first name, last name and the "
            "phone number on their account."
        )

    lines = _lookup(
        identity["customer_first_name"],
        identity["customer_last_name"],
        identity["customer_phone"],
        None,
        album_title,
        artist_name,
        None,
    )
    if not lines:
        return []

    albums: dict[tuple, dict] = {}
    for line in lines:
        key = (line["album_title"], line["artist_name"])
        entry = albums.setdefault(
            key,
            {
                "album_title": line["album_title"],
                "artist_name": line["artist_name"],
                "tracks_purchased": [],
                "purchase_dates": [],
                "refunded_tracks": 0,
            },
        )
        entry["tracks_purchased"].append(line["track_name"])
        day = (line["purchase_date"] or "")[:10]
        if day and day not in entry["purchase_dates"]:
            entry["purchase_dates"].append(day)
        if line.get("refunded"):
            entry["refunded_tracks"] += 1
    return list(albums.values())


# Ids for orders the agent places. Well above the migrated range (Chinook's
# invoice lines end at 2240) so anything the agent created is obvious at a
# glance, and wide enough that a collision is not a practical concern.
_NEW_ID_FLOOR = 10_000_000
_NEW_ID_CEIL = 10**12


def _new_id() -> int:
    return random.randrange(_NEW_ID_FLOOR, _NEW_ID_CEIL)


def place_order(
    artist_name: str,
    album_title: str | None = None,
    track_name: str | None = None,
    customer_asked_to_buy: bool = False,
) -> str:
    """Record an order for an album or a single track.

    Set `customer_asked_to_buy` only when the customer has *told you to buy
    it* — "buy it", "order it", "I'll take it", or a plain yes to a direct
    question you asked about placing the order. Saying they are **interested**,
    that they **like** it, that they **want** it, or that they are thinking
    about it is NOT that: it is an interest, and it is recorded as one.

    Leave the flag false when you are unsure. Doing so records the interest and
    tells you to ask, which costs one question. Getting it wrong the other way
    puts an order on someone's account that they did not ask for.

    Give `artist_name` always, then either `album_title` for the whole album or
    `track_name` for one track.

    This RECORDS the order against their account. It does not take payment:
    there is no card, no charge and no delivery in this chat. Say that when you
    confirm, so nobody believes they have paid.
    """
    if not customer_asked_to_buy:
        # The safe default is deliberate. An order the customer did not ask for
        # is worse than an extra question, so an unconfirmed call degrades into
        # the thing they probably did mean.
        item = album_title or track_name or artist_name
        noted = _record_interest(item, "wants_to_buy")
        return (
            f"No order placed — that sounded like interest rather than an "
            f"instruction to buy. {noted} Ask them directly whether they want "
            f"you to place the order for {item!r}, and call this again with "
            f"customer_asked_to_buy=True only if they say yes."
        )
    memory_identity = _current_identity.get()
    missing = [key for key in IDENTITY_KEYS if not memory_identity.get(key)]
    if missing:
        return (
            "I can't place an order without knowing who the customer is. Ask "
            "for their first name, last name and account phone number first."
        )
    if not album_title and not track_name:
        return "Ask which album or which track they want before ordering."

    canonical_artist = resolve_name(artist_name, "artist")
    albums = artist_catalog(canonical_artist)
    if not albums:
        return f"I couldn't find {artist_name!r} in the catalogue."

    if album_title:
        wanted = resolve_name(album_title, "album")
        matched = [a for a in albums if a["album_title"] == wanted]
        if not matched:
            return (
                f"{canonical_artist} doesn't have an album called "
                f"{album_title!r} in our catalogue."
            )
        album = matched[0]
        chosen = [(album["album_title"], t) for t in album["tracks"]]
    else:
        wanted = resolve_name(track_name, "track")
        chosen = [
            (a["album_title"], t)
            for a in albums
            for t in a["tracks"]
            if t["track_name"] == wanted
        ][:1]
        if not chosen:
            return (
                f"I couldn't find a track called {track_name!r} by "
                f"{canonical_artist}."
            )

    key = customer_key(
        memory_identity["customer_first_name"],
        memory_identity["customer_last_name"],
        memory_identity["customer_phone"],
    )
    invoice_id = _new_id()
    now = pd.Timestamp.utcnow().tz_localize(None)
    rows = [
        {
            "invoice_line_id": _new_id(),
            "customer_key": key,
            "invoice_id": invoice_id,
            "track_name": track["track_name"],
            "artist_name": canonical_artist,
            "album_title": album_name,
            "purchase_date": now.isoformat(),
            "quantity_purchased": 1,
            "price_per_unit": float(track.get("unit_price") or 0.0),
            "migrated_at": now,
        }
        for album_name, track in chosen
    ]
    if not _record_order(key, rows):
        return (
            "Something went wrong writing the order — tell the customer it did "
            "not go through and that nothing has been charged."
        )

    total = sum(r["price_per_unit"] for r in rows)
    what = (
        f"{len(rows)} tracks from {chosen[0][0]!r}"
        if album_title
        else f"{chosen[0][1]['track_name']!r}"
    )
    return (
        f"Order recorded: {what} by {canonical_artist}, ${total:.2f}. "
        "No payment was taken — say the order is on their account and that "
        "nothing has been charged."
    )


def _record_order(key: str, rows: list[dict]) -> bool:
    """Append the lines, then add their ids to the customer's index.

    Order matters: the lines go in first, so a failure between the two writes
    leaves rows nothing points at — invisible, and harmless. The reverse would
    leave the index advertising lines that do not exist, and every later read
    of that customer would come back short.
    """
    if EVAL_MODE:
        # Both writes skipped together. Doing the lines and not the index would
        # leave the customer advertising ids that do not exist, which is the
        # failure the ordering below exists to avoid.
        log.info("eval mode: not recording %d order line(s) for %s", len(rows), key)
        return True
    if not _ensure_ready():
        return False
    try:
        _fs.get_feature_group(PURCHASES_FG, version=FG_VERSION).insert(
            pd.DataFrame(rows),
            storage="online",
            write_options={"wait_for_job": False},
        )
    except Exception:  # noqa: BLE001
        log.exception("Could not write order lines for %s", key)
        return False

    # Re-read immediately before writing rather than reusing anything fetched
    # earlier in the turn, to keep the read-modify-write window as short as
    # possible. It is still a window: two orders placed for the same customer
    # at the same instant can lose one set of ids from the index. Acceptable
    # here, and the fix would be a per-customer lock or an append-only index.
    customer = _lookup_one(CUSTOMERS_FG, {"customer_key": key})
    if not customer:
        log.error("No customer row for %s; order lines are orphaned", key)
        return False
    try:
        existing = json.loads(customer.get("line_ids") or "[]")
    except (TypeError, ValueError):
        existing = []
    merged = sorted({*existing, *(int(r["invoice_line_id"]) for r in rows)})
    customer["line_ids"] = json.dumps(merged)
    customer["line_count"] = len(merged)
    try:
        _fs.get_feature_group(CUSTOMERS_FG, version=FG_VERSION).insert(
            pd.DataFrame([customer]),
            storage="online",
            write_options={"wait_for_job": False},
        )
    except Exception:  # noqa: BLE001
        log.exception("Wrote order lines for %s but could not update the index", key)
        return False
    return True


def _ensure_ready() -> bool:
    return _fs is not None


def _interest_owner(identity: dict | None = None) -> str | None:
    """The durable-memory owner for the customer in this turn, or None.

    The single definition of "who this is" in the agent. `ctx.rebind_subject`
    is fed from here too, so `ctx.subject` and this value are the same string
    by construction — durable state written through the SDK's own `remember`
    and state written here land on the same customer.
    """
    identity = identity if identity is not None else _current_identity.get()
    if not all(identity.get(key) for key in IDENTITY_KEYS):
        return None
    return customer_key(
        identity["customer_first_name"],
        identity["customer_last_name"],
        identity["customer_phone"],
    )


def _rebind_to_customer(identity: dict | None = None) -> str | None:
    """Point the SDK's durable memory at the customer, not the serving key.

    Called from both places identity becomes known: `identify` when the model
    has just extracted it, and the handler when it was already in session state
    from an earlier turn. Idempotent — the second call in a conversation is a
    no-op because the subject already matches.
    """
    owner = _interest_owner(identity)
    if owner is None:
        return None
    _, ctx = _memory_and_ctx()
    if ctx is not None:
        ctx.rebind_subject(owner)
    return owner


def _record_interest(item: str, kind: str) -> str:
    """Store an interest. Shared by the tool and by an unconfirmed order.

    A plain function rather than a call into the decorated tool: `remember_interest`
    is a StructuredTool once decorated, and reaching through it for the callable
    couples this to the decorator's internals.
    """
    memory, ctx = _memory_and_ctx()
    owner = _interest_owner()
    if memory is None or owner is None:
        return "I can't store that yet — I don't know who I'm speaking to."
    prefix = WANTS_PREFIX if kind == "wants_to_buy" else LIKES_PREFIX
    memory.set_state(
        INTEREST_SCOPE,
        owner,
        f"{prefix}{item.strip()}",
        json.dumps({"item": item.strip(), "noted_at": _today()}),
        source_ref=json.dumps(
            {"conversation_id": ctx.conversation_id, "turn_id": ctx.turn_id}
        ),
    )
    if kind == "wants_to_buy":
        # Worded to be unmistakable. The model previously read "recorded" as
        # "ordered" and told a customer their order was confirmed, when the
        # only thing that happened was this row being written.
        return (
            f"Saved a note that they are interested in buying {item.strip()!r}. "
            "This is only a note of interest: NO ORDER HAS BEEN PLACED and "
            "nothing has been charged. If they want to buy it now, use "
            "place_order; otherwise tell them you have made a note and will "
            "follow it up next time."
        )
    return f"Saved a note that they like {item.strip()!r}."


def remember_interest(item: str, kind: Literal["wants_to_buy", "likes"]) -> str:
    """Record something the customer feels about an album, artist or track.

    Call this the moment they express either:
      - `wants_to_buy` — they are interested in it, like the sound of it, want
        it, or are thinking about buying it, but have not told you to buy it.
        "I'm interested in this album", "I'd love to have this one" and "I
        might get this" all belong here. This records an *interest only*: it
        places no order and charges nothing. Use `place_order` only once they
        tell you to buy it. You will be reminded next time so you can follow up
        on whether they went ahead.
      - `likes` — they simply enjoy it, with no intent to buy. Use this for
        taste, so recommendations can be tailored later.

    Use their words for `item` (an album, artist or track name). Do not use
    this for anything they have already bought — that is in their order
    history.
    """
    return _record_interest(item, kind)


def _memory_and_ctx():
    """The active store and request context, resolved the way the SDK's own
    memory tools do — a tool is called by the model and is handed neither."""
    from hopsworks_agent_protocol.autoevents import current_context

    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        return None, None
    return ctx.memory, ctx


def _today() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def interests_block(memory, identity: dict) -> str:
    """What we remember about this customer, checked against what they bought.

    This is the part worth watching: an intention lives in agent memory, the
    purchase lives in the feature store, and the two are reconciled here. A
    want that has since been fulfilled is reported as fulfilled rather than
    nagged about.
    """
    owner = _interest_owner(identity)
    if memory is None or owner is None:
        return ""
    rows = memory.list_state(INTEREST_SCOPE, owner)
    wants = [r for r in rows if r["key"].startswith(WANTS_PREFIX)]
    likes = [r for r in rows if r["key"].startswith(LIKES_PREFIX)]
    if not wants and not likes:
        return ""

    owned = {
        (album["album_title"] or "").lower()
        for album in _purchased_albums(identity)
    }

    lines = []
    for row in wants:
        item = row["key"][len(WANTS_PREFIX):]
        canonical = resolve_name(item, "album")
        bought = canonical.lower() in owned or item.lower() in owned
        lines.append(
            f"- wanted to buy {item!r} — {'HAS SINCE BOUGHT IT' if bought else 'still not purchased'}"
        )
    for row in likes:
        lines.append(f"- likes {row['key'][len(LIKES_PREFIX):]!r}")
    return (
        "\n\nFrom earlier conversations with this customer:\n"
        + "\n".join(lines)
        + "\nIf it fits naturally, follow up on anything they wanted to buy — "
        "congratulate them if they bought it, or offer to help if they did not. "
        "Mention it once; do not nag.\n"
    )


def _purchased_albums(identity: dict) -> list[dict]:
    previous = _current_identity.get()
    _current_identity.set(identity)
    try:
        result = purchase_history.invoke({})
    except Exception:  # noqa: BLE001 — a check-in must never fail the turn
        log.exception("Could not read purchases while reconciling interests")
        result = []
    finally:
        _current_identity.set(previous)
    return result if isinstance(result, list) else []


qa_llm = ChatAnthropic(model=ANSWER_MODEL, max_tokens=1024, temperature=0.0)
# The memory tools go in the same list: a support agent that remembers a
# returning customer's name and phone number is the whole point of durable
# memory, and this is the only place they can be registered — the SDK cannot
# reach into a framework's tool list on your behalf.
def recall_interests() -> str:
    """What this customer has told you they like or want to buy, from any
    earlier conversation.

    Use this whenever they ask what they were interested in, what you remember
    about them, or what they were looking at last time. Takes no arguments:
    it returns everything stored for them, already checked against what they
    have since bought.

    This is the only way to read their interests. Do not reach for `search`,
    which looks through the words of past conversations rather than what was
    remembered from them.
    """
    memory, _ = _memory_and_ctx()
    identity = _current_identity.get()
    if memory is None or _interest_owner(identity) is None:
        return "I don't know who I'm speaking to yet, so I can't look that up."
    block = interests_block(memory, identity)
    if not block:
        return (
            "Nothing is stored for this customer yet — they have not told you "
            "about anything they like or want. Say that plainly rather than "
            "guessing at what they might have meant."
        )
    return block
