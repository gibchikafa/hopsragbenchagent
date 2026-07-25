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
Here that is a pipeline (`chinook_feature_pipeline.py`) writing one embedding
feature group, and the agent queries it online. The agent no longer re-embeds a
catalogue on every pod start, replicas share one index, and the index can be
rebuilt without redeploying.

**The SDK owns the serving surface.** Manifest, `/v1/chat`, `/v1/chat/stream`,
health/readiness, CORS, tracing and memory are all `AgentApp`; the code here is
just the domain.

Chinook itself stays in SQLite: refunds delete rows transactionally, which is
not what a feature store is for. See the note on durability by `ensure_db`.

Deploy:
    python chinook_feature_pipeline.py        # once, to build the index
    hops agent create chinook_support_agent.py --name chinooksupport \
        --requirements chinook_support_requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start chinooksupport --wait 600
"""

import json
import logging
import os
import sqlite3
import urllib.request
from typing import Literal

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

FG_NAME = "chinook_catalog_embeddings"
FG_VERSION = 1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_DB = os.environ.get("CHINOOK_DB_PATH", "chinook.db")

# Haiku for the two structured-output steps (routing and extraction) — they are
# short, schema-constrained, and run on every turn. Override for the answering
# model if you want more headroom on the open-ended path.
ROUTER_MODEL = os.environ.get("CHINOOK_ROUTER_MODEL", "claude-haiku-4-5")
ANSWER_MODEL = os.environ.get("CHINOOK_ANSWER_MODEL", "claude-haiku-4-5")


# ── data access ──────────────────────────────────────────────────────────────


def ensure_db(path: str = CHINOOK_DB) -> str:
    """Make sure the Chinook SQLite file exists locally.

    Chinook stays in SQLite rather than moving to the feature store because the
    refund path deletes invoice rows — transactional writes the feature store is
    not designed for. Note the consequence: the file is pod-local, so refunds do
    not survive a restart. Point ``CHINOOK_DB_PATH`` at a mounted dataset if you
    need them to.
    """
    if not os.path.exists(path):
        log.info("Downloading Chinook DB → %s", path)
        urllib.request.urlretrieve(CHINOOK_URL, path)
    return path


ensure_db()
_embed = SentenceTransformer(EMBEDDING_MODEL)
_fs = hopsworks.login().get_feature_store()
_catalog_fg = None
_catalog_features: list[str] = []


def _catalog():
    """The catalogue feature group, opened lazily and cached."""
    global _catalog_fg, _catalog_features
    if _catalog_fg is None:
        _catalog_fg = _fs.get_feature_group(FG_NAME, version=FG_VERSION)
        if _catalog_fg is not None:
            _catalog_features = [f.name for f in _catalog_fg.features]
            log.info("Catalogue index ready (%s)", FG_NAME)
    return _catalog_fg


def resolve_name(text: str, kind: Literal["track", "artist", "album"]) -> str:
    """Snap what the customer typed to the closest name the database stores.

    "prince" → "Prince", so the SQL below matches. Falls back to the raw input
    when the index is unavailable, which degrades to a plain LIKE rather than
    failing the turn.
    """
    fg = _catalog()
    if fg is None or not text:
        return text
    try:
        vector = _embed.encode(text, normalize_embeddings=True).tolist()
        # Build the filter with get_feature(), never getattr(fg, "entity_kind").
        # Attribute access silently returns a FeatureGroup attribute when the
        # column name collides with one (`subject` and `description` are real
        # examples), and the resulting `False` is translated to *no filter at
        # all* — you get unfiltered neighbours with no error.
        condition = fg.get_feature("entity_kind") == kind
        hits = fg.find_neighbors(vector, col="embedding", k=1, filter=condition)
    except Exception:  # noqa: BLE001 — disambiguation is best-effort
        log.exception("Catalogue lookup failed for %r; using the raw input", text)
        return text
    if not hits:
        return text
    row = dict(zip(_catalog_features, hits[0][1]))
    return row.get("name") or text


def _refund(
    invoice_id: int | None, invoice_line_ids: list[int] | None, mock: bool = False
) -> float:
    """Delete the given Invoice / InvoiceLine records, returning the amount refunded."""
    if invoice_id is None and invoice_line_ids is None:
        return 0.0

    conn = sqlite3.connect(ensure_db())
    cursor = conn.cursor()
    total_refund = 0.0
    try:
        if invoice_id is not None:
            row = cursor.execute(
                "SELECT Total FROM Invoice WHERE InvoiceId = ?", (invoice_id,)
            ).fetchone()
            if row:
                total_refund += row[0]
            if not mock:
                # invoice lines first: foreign key constraints
                cursor.execute(
                    "DELETE FROM InvoiceLine WHERE InvoiceId = ?", (invoice_id,)
                )
                cursor.execute("DELETE FROM Invoice WHERE InvoiceId = ?", (invoice_id,))

        if invoice_line_ids:
            placeholders = ",".join("?" for _ in invoice_line_ids)
            row = cursor.execute(
                f"SELECT SUM(UnitPrice * Quantity) FROM InvoiceLine "
                f"WHERE InvoiceLineId IN ({placeholders})",
                invoice_line_ids,
            ).fetchone()
            if row and row[0]:
                total_refund += row[0]
            if not mock:
                cursor.execute(
                    f"DELETE FROM InvoiceLine WHERE InvoiceLineId IN ({placeholders})",
                    invoice_line_ids,
                )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()
    return float(total_refund)


def _lookup(
    customer_first_name: str,
    customer_last_name: str,
    customer_phone: str,
    track_name: str | None,
    album_title: str | None,
    artist_name: str | None,
    purchase_date_iso_8601: str | None,
) -> list[dict]:
    """Invoice lines matching a customer and optional purchase filters."""
    conn = sqlite3.connect(ensure_db())
    cursor = conn.cursor()
    query = """
    SELECT il.InvoiceLineId, t.Name, art.Name, i.InvoiceDate, il.Quantity, il.UnitPrice
    FROM InvoiceLine il
    JOIN Invoice i ON il.InvoiceId = i.InvoiceId
    JOIN Customer c ON i.CustomerId = c.CustomerId
    JOIN Track t ON il.TrackId = t.TrackId
    JOIN Album alb ON t.AlbumId = alb.AlbumId
    JOIN Artist art ON alb.ArtistId = art.ArtistId
    WHERE c.FirstName = ? AND c.LastName = ? AND c.Phone = ?
    """
    params: list = [customer_first_name, customer_last_name, customer_phone]
    if track_name:
        query += " AND t.Name = ?"
        params.append(resolve_name(track_name, "track"))
    if album_title:
        query += " AND alb.Title = ?"
        params.append(resolve_name(album_title, "album"))
    if artist_name:
        query += " AND art.Name = ?"
        params.append(resolve_name(artist_name, "artist"))
    if purchase_date_iso_8601:
        query += " AND date(i.InvoiceDate) = date(?)"
        params.append(purchase_date_iso_8601)

    rows = cursor.execute(query, params).fetchall()
    conn.close()
    return [
        {
            "invoice_line_id": r[0],
            "track_name": r[1],
            "artist_name": r[2],
            "purchase_date": r[3],
            "quantity_purchased": r[4],
            "price_per_unit": r[5],
        }
        for r in rows
    ]


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


info_llm = ChatAnthropic(model=ROUTER_MODEL, max_tokens=1024).with_structured_output(
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
    conn = sqlite3.connect(ensure_db())
    query = """
    SELECT DISTINCT t.Name, ar.Name, al.Title
    FROM Track t
    JOIN Album al ON t.AlbumId = al.AlbumId
    JOIN Artist ar ON al.ArtistId = ar.ArtistId
    WHERE 1=1
    """
    params: list = []
    if track_name:
        query += " AND t.Name LIKE ?"
        params.append(f"%{resolve_name(track_name, 'track')}%")
    if album_title:
        query += " AND al.Title LIKE ?"
        params.append(f"%{resolve_name(album_title, 'album')}%")
    if artist_name:
        query += " AND ar.Name LIKE ?"
        params.append(f"%{resolve_name(artist_name, 'artist')}%")
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [
        {"track_name": r[0], "artist_name": r[1], "album_name": r[2]} for r in rows
    ]


@tool
def lookup_album(
    track_name: str | None = None,
    album_title: str | None = None,
    artist_name: str | None = None,
) -> list[dict]:
    """Look up albums in the store's catalogue.

    Returns a list of dicts with keys {'album_name', 'artist_name'}.
    """
    conn = sqlite3.connect(ensure_db())
    query = """
    SELECT DISTINCT al.Title, ar.Name
    FROM Album al
    JOIN Artist ar ON al.ArtistId = ar.ArtistId
    LEFT JOIN Track t ON t.AlbumId = al.AlbumId
    WHERE 1=1
    """
    params: list = []
    if track_name:
        query += " AND t.Name LIKE ?"
        params.append(f"%{resolve_name(track_name, 'track')}%")
    if album_title:
        query += " AND al.Title LIKE ?"
        params.append(f"%{resolve_name(album_title, 'album')}%")
    if artist_name:
        query += " AND ar.Name LIKE ?"
        params.append(f"%{resolve_name(artist_name, 'artist')}%")
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [{"album_name": r[0], "artist_name": r[1]} for r in rows]


@tool
def lookup_artist(
    track_name: str | None = None,
    album_title: str | None = None,
    artist_name: str | None = None,
) -> list[str]:
    """Look up artists in the store's catalogue. Returns matching artist names."""
    conn = sqlite3.connect(ensure_db())
    query = """
    SELECT DISTINCT ar.Name
    FROM Artist ar
    LEFT JOIN Album al ON al.ArtistId = ar.ArtistId
    LEFT JOIN Track t ON t.AlbumId = al.AlbumId
    WHERE 1=1
    """
    params: list = []
    if track_name:
        query += " AND t.Name LIKE ?"
        params.append(f"%{resolve_name(track_name, 'track')}%")
    if album_title:
        query += " AND al.Title LIKE ?"
        params.append(f"%{resolve_name(album_title, 'album')}%")
    if artist_name:
        query += " AND ar.Name LIKE ?"
        params.append(f"%{resolve_name(artist_name, 'artist')}%")
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [r[0] for r in rows]


qa_llm = ChatAnthropic(model=ANSWER_MODEL, max_tokens=1024)
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

router_llm = ChatAnthropic(model=ROUTER_MODEL, max_tokens=256).with_structured_output(
    UserIntent
)


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
