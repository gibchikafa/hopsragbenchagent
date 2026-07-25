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
    return row.iloc[0].to_dict()


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
    PurchaseInformation, include_raw=True
)


async def gather_info(state: State) -> Command[Literal["lookup", "refund", "__end__"]]:
    info = await info_llm.ainvoke(
        [{"role": "system", "content": GATHER_INFO_INSTRUCTIONS}, *state["messages"]]
    )
    parsed = info["parsed"] or {}
    if any(parsed.get(k) for k in ("invoice_id", "invoice_line_ids")):
        goto = "refund"
    elif all(
        parsed.get(k)
        for k in ("customer_first_name", "customer_last_name", "customer_phone")
    ):
        goto = "lookup"
    else:
        goto = END
    return Command(update={"messages": [info["raw"]], **parsed}, goto=goto)


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


qa_llm = ChatAnthropic(model=ANSWER_MODEL, max_tokens=1024, temperature=0.0)
# The memory tools go in the same list: a support agent that remembers a
# returning customer's name and phone number is the whole point of durable
# memory, and this is the only place they can be registered — the SDK cannot
# reach into a framework's tool list on your behalf.
qa_graph = create_react_agent(
    qa_llm,
    [lookup_track, lookup_artist, lookup_album, *memory_tools("langgraph")],
)


# ── supervisor ───────────────────────────────────────────────────────────────


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
_builder.add_node(intent_classifier)
_builder.add_node("refund_agent", refund_graph)
_builder.add_node("question_answering_agent", qa_graph)
_builder.add_node(compile_followup)
_builder.set_entry_point("intent_classifier")
_builder.add_edge("refund_agent", "compile_followup")
_builder.add_edge("question_answering_agent", "compile_followup")
_builder.add_edge("compile_followup", END)
graph = _builder.compile()


# ── the protocol app ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a customer support agent for an online music store. Answer questions about \
the catalogue, and help customers get refunds on tracks they bought.

When a customer tells you something that will still be true next time — their name, \
their phone number, how they prefer to be addressed — store it with `remember` so \
they do not have to repeat it. Do not store catalogue facts there; look those up.\
"""

agent_app = AgentApp(
    name="Chinook support agent",
    description="Customer support for an online music store: catalogue questions "
    "and refunds (LangGraph supervisor + Hopsworks feature store lookup).",
    framework="langgraph",
    welcome_message="Hi! I can answer questions about our catalogue or help you "
    "with a refund.",
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
    messages = [{"role": "system", "content": SYSTEM_PROMPT + ctx.system_context()}]
    messages += ctx.history  # turns since the last fold
    messages.append({"role": "user", "content": request.text})

    async for delta in ctx.stream_langchain(
        graph.astream_events({"messages": messages}, version="v2")
    ):
        yield delta


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
