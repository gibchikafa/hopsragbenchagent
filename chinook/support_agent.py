"""
Chinook customer-support agent — native Hopsworks Agent Protocol implementation.

A supervisor routes each turn to one of two sub-agents:

    intent_classifier ─┬─▶ refund_agent          (gather_info → lookup | refund)
                       └─▶ question_answering_agent (ReAct over catalogue lookups)
                                    │
                              compile_followup

Ported from a LangGraph/LangSmith notebook, with two substantive changes:

**Catalogue lookup comes from the feature store.** The notebook embedded every
artist, album and track name into three in-process vector stores at import time.
Here that is a pipeline (`feature_pipeline.py`) writing one embedding
feature group, and the agent queries it online. The agent no longer re-embeds a
catalogue on every pod start, replicas share one index, and the index can be
rebuilt without redeploying.

**The SDK owns the serving surface.** Manifest, `/v1/chat`, `/v1/chat/stream`,
health/readiness, CORS, tracing and memory are all `AgentApp`; the code here is
just the domain.

There is no SQLite anywhere: `migrate_to_feature_store.py` denormalises Chinook
into keyed feature groups, and refunds became an append-only ledger because a
feature group has no row-level delete reachable from Python. See that module's
docstring for the shapes and the reasoning.

Deploy:
    python feature_pipeline.py            # catalogue name embeddings
    python migrate_to_feature_store.py    # invoices, catalogue, refund ledger
    hops agent create support_agent.py --name chinooksupport \
        --requirements requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start chinooksupport --wait 600
"""

import contextvars
import hashlib
import json
import logging
import os
from typing import Literal

import pandas as pd

import hopsworks
from hopsworks_agent_protocol import (  # noqa: E501
    AgentApp,
    AgentError,
    PersistentAgentMemory,
    anthropic_summarizer,
    memory_tools,
    remember,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from typing_extensions import Annotated, TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CATALOG_FG = "chinook_catalog_embeddings"
ARTIST_FG = "chinook_artist_catalog"
PURCHASES_FG = "chinook_customer_purchases"
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
# serving key, so `subject` is whatever the client claims and defaults to the
# conversation id. Writing to `session` scope says that plainly instead of
# leaning on that fallback and pretending it is per-user.
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

    One keyed read returns everything the customer bought; the filtering that
    used to be SQL happens here, over tens of rows. Already-refunded lines are
    annotated rather than hidden — the ledger records refunds, it does not erase
    the sale.
    """
    key = customer_key(customer_first_name, customer_last_name, customer_phone)
    row = _lookup_one(PURCHASES_FG, {"customer_key": key})
    if not row:
        return []
    lines = json.loads(row.get("purchases") or "[]")
    for line in lines:
        line["customer_key"] = key

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


class State(TypedDict):
    """Shared by every sub-agent, so they compose as plain graph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    followup: str | None
    invoice_id: int | None
    invoice_line_ids: list[int] | None
    customer_first_name: str | None
    customer_last_name: str | None
    customer_phone: str | None
    track_name: str | None
    album_title: str | None
    artist_name: str | None
    purchase_date_iso_8601: str | None


GATHER_INFO_INSTRUCTIONS = """You are managing an online music store that sells song tracks. \
Customers can buy multiple tracks at a time and these purchases are recorded in a database as \
an Invoice per purchase and an associated set of Invoice Lines for each purchased track.

Your task is to help customers who would like a refund for one or more of the tracks they've \
purchased. In order for you to be able refund them, the customer must specify the Invoice ID \
to get a refund on all the tracks they bought in a single transaction, or one or more Invoice \
Line IDs if they would like refunds on individual tracks.

Often a user will not know the specific Invoice ID(s) or Invoice Line ID(s) for which they \
would like a refund. In this case you can help them look up their invoices by asking them to \
specify:
- Required: Their first name, last name, and phone number.
- Optionally: The track name, artist name, album name, or purchase date.

If the customer has not specified the required information (either Invoice/Invoice Line IDs \
or first name, last name, phone) then please ask them to specify it."""


class PurchaseInformation(TypedDict):
    """Everything known about the invoice lines to refund. Do not make up values;
    leave fields null when unknown."""

    invoice_id: int | None
    invoice_line_ids: list[int] | None
    customer_first_name: str | None
    customer_last_name: str | None
    customer_phone: str | None
    track_name: str | None
    album_title: str | None
    artist_name: str | None
    purchase_date_iso_8601: str | None
    followup: Annotated[
        str | None,
        ...,
        "If the user hasn't given enough identifying information, tell them what is "
        "required and ask them to specify it.",
    ]


info_llm = ChatAnthropic(
    model=ROUTER_MODEL, max_tokens=1024, temperature=0.0
).with_structured_output(
    PurchaseInformation
)


async def gather_info(state: State) -> Command[Literal["lookup", "refund", "__end__"]]:
    parsed = await info_llm.ainvoke(
        [{"role": "system", "content": GATHER_INFO_INSTRUCTIONS}, *state["messages"]]
    ) or {}
    if any(parsed.get(k) for k in ("invoice_id", "invoice_line_ids")):
        goto = "refund"
    elif all(
        parsed.get(k)
        for k in ("customer_first_name", "customer_last_name", "customer_phone")
    ):
        goto = "lookup"
    else:
        goto = END
    # NB: do not put info["raw"] into `messages`. ChatAnthropic implements
    # with_structured_output via tool calling, so the raw reply is an assistant
    # message carrying a `tool_use` block. Appending it leaves a tool_use with
    # no matching tool_result, and the next model call is rejected:
    #   messages.N: `tool_use` ids were found without `tool_result` blocks
    # The extracted fields go into state, which is the part that matters; the
    # tool call itself is not conversation content. (The notebook this came
    # from used OpenAI with method="json_schema", which returns plain content
    # and so had nothing to trip over.)
    return Command(update=dict(parsed), goto=goto)


def refund(state: State) -> dict:
    refunded = _refund(
        invoice_id=state.get("invoice_id"),
        invoice_line_ids=state.get("invoice_line_ids"),
    )
    response = (
        f"You have been refunded a total of: ${refunded:.2f}. "
        "Is there anything else I can help with?"
    )
    return {
        "messages": [{"role": "assistant", "content": response}],
        "followup": response,
    }


def lookup(state: State) -> dict:
    results = _lookup(
        *(
            state.get(k)
            for k in (
                "customer_first_name",
                "customer_last_name",
                "customer_phone",
                "track_name",
                "album_title",
                "artist_name",
                "purchase_date_iso_8601",
            )
        )
    )
    if not results:
        response = followup = (
            "We did not find any purchases associated with the information you've "
            "provided. Are you sure you've entered all of your information correctly?"
        )
    else:
        response = (
            "Which of the following purchases would you like to be refunded for?\n\n"
            f"```json\n{json.dumps(results, indent=2)}\n```"
        )
        followup = (
            "Which of the following purchases would you like to be refunded for?\n\n"
            f"{tabulate(results, headers='keys')}"
        )
    return {
        "messages": [{"role": "assistant", "content": response}],
        "followup": followup,
        "invoice_line_ids": [r["invoice_line_id"] for r in results],
    }


_refund_builder = StateGraph(State)
_refund_builder.add_node(gather_info)
_refund_builder.add_node(refund)
_refund_builder.add_node(lookup)
_refund_builder.set_entry_point("gather_info")
_refund_builder.add_edge("lookup", END)
_refund_builder.add_edge("refund", END)
refund_graph = _refund_builder.compile()


# ── question-answering sub-agent ─────────────────────────────────────────────


@tool
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


@tool
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


@tool
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


@tool
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


def _interest_owner(identity: dict | None = None) -> str | None:
    """The durable-memory owner for the customer in this turn, or None."""
    identity = identity if identity is not None else _current_identity.get()
    if not all(identity.get(key) for key in IDENTITY_KEYS):
        return None
    return customer_key(
        identity["customer_first_name"],
        identity["customer_last_name"],
        identity["customer_phone"],
    )


@tool
def remember_interest(item: str, kind: Literal["wants_to_buy", "likes"]) -> str:
    """Record something the customer feels about an album, artist or track.

    Call this the moment they express either:
      - `wants_to_buy` — they intend to buy it, are thinking about it, or ask
        how to get it. This records an *interest only*: it places no order,
        charges nothing, and reserves nothing. You will be reminded next time
        so you can follow up on whether they went ahead.
      - `likes` — they simply enjoy it, with no intent to buy. Use this for
        taste, so recommendations can be tailored later.

    Use their words for `item` (an album, artist or track name). Do not use
    this for anything they have already bought — that is in their order
    history.
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
            "NO ORDER HAS BEEN PLACED and nothing has been charged — you cannot "
            "sell anything in this chat. Tell them you have made a note and "
            "that they need to complete the purchase in the store itself."
        )
    return f"Saved a note that they like {item.strip()!r}."


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
qa_graph = create_react_agent(
    qa_llm,
    [
        lookup_track,
        lookup_artist,
        lookup_album,
        purchase_history,
        remember_interest,
        *memory_tools("langgraph"),
    ],
)


# ── supervisor ───────────────────────────────────────────────────────────────


IDENTIFY_INSTRUCTIONS = """You are the front desk of an online music store. Your only \
job right now is to work out who you are speaking to. From the conversation so far, \
extract the customer's first name, last name, and the phone number on their account. \
Do not guess or invent any of them — leave a field null if the customer has not said \
it. Do not answer any other question yet."""


class CustomerIdentity(TypedDict):
    """Who the customer is. Leave a field null when they have not said it."""

    customer_first_name: str | None
    customer_last_name: str | None
    customer_phone: str | None


identity_llm = ChatAnthropic(
    model=ROUTER_MODEL, max_tokens=512, temperature=0.0
).with_structured_output(CustomerIdentity)

ASK_FOR_IDENTITY = (
    "Before we start — could you give me your first name, last name, and the "
    "phone number on your account? I'll keep them for the rest of this chat."
)


_MISSING_MARKERS = {"", "none", "null", "n/a", "na", "unknown", "not provided",
                    "not specified", "string"}


def _clean_identity(parsed: dict) -> dict:
    """Drop values the model filled in rather than read.

    Structured output makes every key of the schema required, so the model
    cannot simply omit a field it was not told — it produces *something*,
    usually a placeholder and occasionally an invention. Telling it not to
    guess helps but does not bind it. So placeholders are treated as missing,
    and a phone number has to contain enough digits to be one; otherwise a
    turn like "hello" sails past the identity gate on fabricated details and
    the refund flow goes looking up a customer who does not exist.
    """
    cleaned = {}
    for key in IDENTITY_KEYS:
        value = parsed.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text.lower() in _MISSING_MARKERS:
            continue
        if key == "customer_phone" and sum(c.isdigit() for c in text) < 7:
            continue
        cleaned[key] = text
    return cleaned


async def identify(state: State) -> Command[Literal["intent_classifier", "__end__"]]:
    """Establish who we are talking to before doing any work.

    Asked once per conversation, not once per turn: the handler seeds the graph
    state from `session`-scoped memory, so every turn after the first falls
    straight through without consulting the model. The refund flow downstream
    never has to ask for name and phone either — they are already in state by
    the time it runs.

    It is once per *conversation* rather than once per *customer* only because
    there is no login to key on; see IDENTITY_SCOPE.
    """
    if all(state.get(key) for key in IDENTITY_KEYS):
        return Command(goto="intent_classifier")

    parsed = _clean_identity(
        await identity_llm.ainvoke(
            [{"role": "system", "content": IDENTIFY_INSTRUCTIONS}, *state["messages"]]
        )
        or {}
    )
    if all(parsed.get(key) for key in IDENTITY_KEYS):
        for key in IDENTITY_KEYS:
            # `user` scope: durable across every future conversation for this
            # subject, and auto-injected via ctx.system_context()
            remember(key, str(parsed[key]), scope=IDENTITY_SCOPE)
        # NB: do not put info["raw"] into `messages`. ChatAnthropic implements
        # with_structured_output via tool calling, so the raw reply is an assistant
        # message carrying a `tool_use` block. Appending it leaves a tool_use with
        # no matching tool_result, and the next model call is rejected:
        #   messages.N: `tool_use` ids were found without `tool_result` blocks
        # The extracted fields go into state, which is the part that matters; the
        # tool call itself is not conversation content. (The notebook this came
        # from used OpenAI with method="json_schema", which returns plain content
        # and so had nothing to trip over.)
        return Command(update=dict(parsed), goto="intent_classifier")

    return Command(
        update={
            "messages": [{"role": "assistant", "content": ASK_FOR_IDENTITY}],
            "followup": ASK_FOR_IDENTITY,
            **{k: v for k, v in parsed.items() if v},
        },
        goto=END,
    )


class UserIntent(TypedDict):
    """The user's current intent in the conversation."""

    intent: Literal["refund", "question_answering"]


ROUTE_INSTRUCTIONS = """You are managing an online music store that sells song tracks. \
You can help customers in two types of ways: (1) answering general questions about \
tracks sold at your store, (2) helping them get a refund on a purchase they made at your store.

Based on the following conversation, determine if the user is currently seeking general \
information about song tracks or if they are trying to refund a specific purchase.

Return 'refund' if they are trying to get a refund and 'question_answering' if they are \
asking a general music question. Do NOT return anything else. Do NOT try to respond to \
the user.
"""

router_llm = ChatAnthropic(
    model=ROUTER_MODEL, max_tokens=256, temperature=0.0
).with_structured_output(UserIntent)


async def intent_classifier(
    state: State,
) -> Command[Literal["refund_agent", "question_answering_agent"]]:
    response = await router_llm.ainvoke(
        [{"role": "system", "content": ROUTE_INSTRUCTIONS}, *state["messages"]]
    )
    return Command(goto=response["intent"] + "_agent")


def compile_followup(state: State) -> dict:
    """Ensure `followup` is set, defaulting to the last message."""
    if not state.get("followup"):
        return {"followup": state["messages"][-1].content}
    return {}


_builder = StateGraph(State)
_builder.add_node(identify)
_builder.add_node(intent_classifier)
_builder.add_node("refund_agent", refund_graph)
_builder.add_node("question_answering_agent", qa_graph)
_builder.add_node(compile_followup)
_builder.set_entry_point("identify")
_builder.add_edge("refund_agent", "compile_followup")
_builder.add_edge("question_answering_agent", "compile_followup")
_builder.add_edge("compile_followup", END)
graph = _builder.compile()


# ── the protocol app ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a customer support agent for an online music store. Answer questions about \
the catalogue, and help customers get refunds on tracks they bought.

You CANNOT sell anything. There is no checkout in this chat: you cannot place an \
order, take payment, reserve stock, or confirm a purchase, and no tool you have \
does any of those things. Never tell a customer an order is placed, confirmed or \
paid for. When they say they want to buy something, save it with \
`remember_interest` and say plainly that you have noted it and they can complete \
the purchase in the store — then you will be able to follow up next time.

You are told below what you already know about this customer. Never ask again for \
anything that appears there - use it and carry on. If a detail is missing, ask for \
it once, then remember it for the rest of this conversation.

When a customer says they want to buy something, or that they simply like an \
album or artist, record it with `remember_interest`. That outlives the \
conversation, so next time you can pick the thread back up. Anything else that \
stays true - how they prefer to be addressed, that they want refunds as store \
credit - goes in `remember`. Do not store catalogue facts in either; look \
those up.\
"""

agent_app = AgentApp(
    name="Chinook support agent",
    description="Customer support for an online music store: catalogue questions "
    "and refunds (LangGraph supervisor + Hopsworks feature store lookup).",
    framework="langgraph",
    welcome_message="Hi! I can answer questions about our catalogue or help you "
    "with a refund. I'll ask who you are first — just once per chat.",
    suggested_prompts=[
        "What albums do you have by Led Zeppelin?",
        "Who are the artists similar to Prince?",
        "My name is Aaron Mitchell, phone +1 (204) 452-6452 — I'd like a refund.",
    ],
    placeholder="Ask about the catalogue, or request a refund...",
    memory=PersistentAgentMemory(
        summarize=anthropic_summarizer(),
        long_term=True,
    ),
    tool_events=True,
    graph=graph,
)


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints; ctx.stream_langchain yields token
    deltas and turns each sub-agent's tool calls into progress chips."""
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.",
            code="invalid_request",
            status_code=400,
        )

    # ctx.system_context() carries the rolling summary of compacted turns plus
    # what the agent has stored about this customer; "" until there is any.
    known = ctx.state(IDENTITY_SCOPE)
    identity = {key: known[key] for key in IDENTITY_KEYS if known.get(key)}
    # published for the tools, which the model calls without graph state
    _current_identity.set(identity)

    system = SYSTEM_PROMPT + ctx.system_context()
    # What we remember about this customer from previous conversations,
    # reconciled against what they actually bought. Keyed on the customer, so
    # unlike ctx.system_context() this survives a new conversation.
    system += interests_block(ctx.memory, identity)

    messages = [{"role": "system", "content": system}]
    messages += ctx.history  # turns since the last fold
    messages.append({"role": "user", "content": request.text})

    # Seed the graph with whatever we already know about this customer. This is
    # what makes the identity gate a one-time ask rather than a per-conversation
    # one: `user`-scoped state outlives the conversation, so a returning
    # customer arrives already identified and `identify` falls straight through.
    # Not every reply comes from a model. `identify` asks for details, `lookup`
    # renders the purchase table and `refund` confirms the amount — all written
    # straight into graph state by a node. ctx.stream_langchain only yields
    # model token deltas, so those turns stream nothing at all and the client
    # shows an empty response. Tap the event stream on its way through to keep
    # the graph's final state, and fall back to its `followup` when the turn
    # produced no tokens.
    final_state: dict = {}

    async def _capture(events):
        async for event in events:
            if event.get("event") == "on_chain_end":
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict) and "followup" in output:
                    final_state.update(output)
            yield event

    streamed = False
    async for delta in ctx.stream_langchain(
        _capture(graph.astream_events({"messages": messages, **identity}, version="v2"))
    ):
        streamed = True
        yield delta

    if not streamed:
        followup = final_state.get("followup")
        if followup:
            yield str(followup)
        else:
            # a turn that neither streamed nor set a followup would otherwise
            # look like the agent ignored the customer
            yield (
                "Sorry — I wasn't able to put together a reply just then. "
                "Could you try rephrasing?"
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
