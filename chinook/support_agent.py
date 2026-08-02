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

import json
import re
from typing import Literal


from hopsworks_agent_protocol import (  # noqa: E501
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
    memory_tools,
    remember,
)
from typing_extensions import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from tabulate import tabulate
# The store and every tool, shared with the other framework entrypoints. Wrapped
# below as LangChain tools; the other agents wrap the same functions in their
# own tool type, so the rules live in one place rather than several.
from store import (  # noqa: F401 — re-exported for the graph and the app below
    ANSWER_MODEL,
    CUSTOMERS_FG,
    IDENTITY_KEYS,
    IDENTITY_SCOPE,
    ROUTER_MODEL,
    _current_identity,
    _interest_owner,
    _lookup,
    _lookup_one,
    _rebind_to_customer,
    _record_interest,
    _refund,
    interests_block,
    log,
    lookup_album,
    lookup_artist,
    lookup_track,
    place_order,
    purchase_history,
    recall_interests,
    remember_interest,
)

# Wrapped here rather than in store.py: each entrypoint needs a framework-native
# tool object, and a decorator in the shared module would make it one framework's
# object for all.
lookup_track = tool(lookup_track)
lookup_album = tool(lookup_album)
lookup_artist = tool(lookup_artist)
purchase_history = tool(purchase_history)
place_order = tool(place_order)
remember_interest = tool(remember_interest)
recall_interests = tool(recall_interests)



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



# The answering model. Here rather than in store.py: it is a LangChain client,
# and the store is what every entrypoint shares.
qa_llm = ChatAnthropic(model=ANSWER_MODEL, max_tokens=1024, temperature=0.0)

qa_graph = create_react_agent(
    qa_llm,
    [
        lookup_track,
        lookup_artist,
        lookup_album,
        purchase_history,
        place_order,
        remember_interest,
        recall_interests,
        # `search` only, of the SDK's memory tools. `remember` is left out
        # because `remember_interest` is the one writer for this kind of fact,
        # and `recall` because it looks a value up by exact key — interests are
        # stored as "likes:<name>", which the model cannot enumerate, so it
        # guesses a key, misses, and tells the customer there is no record.
        # `recall_interests` answers that question properly.
        *memory_tools("langgraph", include=("search",)),
    ],
)


# ── interest extraction (post-answer) ────────────────────────────────────────

# Recording an interest is not left to the model choosing a tool. Measured over
# a live deployment: `remember_interest` was never called once, across every
# conversation, while the agent repeatedly told customers it had noted their
# interest. Three rounds of sharper tool docstrings and prompt rules did not
# change that — a tool the model declines to call is indistinguishable, from
# the customer's side, from one that does not exist.
#
# So this runs deterministically after the answer, in the same awaited slot the
# SDK uses for summarization: cost lands on request duration, not on
# time-to-answer, and nothing depends on the model volunteering.


class StatedInterest(TypedDict):
    """What the customer said they feel about an item, if anything."""

    item: str | None
    kind: Literal["wants_to_buy", "likes", "none"]


INTEREST_INSTRUCTIONS = """Decide whether the customer expressed interest in a \
specific album, artist or track. You are given what you said to them and their \
reply.

  - "wants_to_buy" — they are interested in it, want it, are thinking about
    buying it, or asked you to remember it for later. This includes "I'm
    interested in this album" and "I'd like this one but not now".
  - "likes" — they simply enjoy it, with no suggestion of buying.
  - "none" — anything else: browsing, asking what tracks are on an album,
    asking about their orders, requesting a refund, or actually buying.

`item` must be the **full name of the thing**, as it appears in the \
conversation — "Led Zeppelin I", not "the album". Customers refer to things \
indirectly ("this one", "it", "the album"); resolve that against what you were \
just discussing and write the real name. If you cannot tell which item they \
mean, use kind "none" rather than guessing: a note that says "the album" is \
worse than no note, because it is read back to them later as if it meant \
something."""

interest_llm = ChatAnthropic(
    model=ROUTER_MODEL, max_tokens=256, temperature=0.0
).with_structured_output(StatedInterest)


#: Words a customer uses to point at something rather than name it. An item
#: that is only one of these carries no information: it cannot be looked up,
#: cannot be matched against the catalogue, and reads as nonsense when the
#: agent recalls it in a later conversation.
_VAGUE_ITEMS = frozenset(
    {
        "it", "this", "that", "one", "them", "these", "those",
        "album", "the album", "this album", "that album", "the record",
        "song", "the song", "this song", "track", "the track", "this track",
        "artist", "the artist", "this artist", "this one", "that one",
    }
)


#: How much of the conversation the extractor sees. Enough to resolve "this
#: one" against the album that was named a turn or two ago, not so much that a
#: cheap model loses the thread.
_INTEREST_CONTEXT_TURNS = 6


def _is_vague(item: str) -> bool:
    cleaned = item.strip().strip('"\'').lower()
    # <UNKNOWN>, N/A and friends: structured output makes `item` required, so a
    # model that cannot name the thing returns a placeholder rather than
    # omitting it. Storing one would be worse than storing nothing.
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return True
    return cleaned in _VAGUE_ITEMS or cleaned in _MISSING_MARKERS


async def _record_stated_interest(
    user_text: str, reply: str, history: list | None = None
) -> None:
    """Note an interest the customer expressed, whatever the model chose to do.

    Grounded on purpose: structured output makes every field required, so the
    model returns *something* even when there is nothing to record. An item
    that does not appear verbatim in the conversation is discarded rather than
    written, because a memory the customer never expressed is worse than a
    missing one — it is read back to them later as fact.
    """
    if _interest_owner() is None:
        return  # nobody to attribute it to yet
    try:
        # The recent conversation, not just this turn. "I also like this one"
        # names nothing; the album it points at was named a turn or two back,
        # so an extractor given only the latest exchange answers <UNKNOWN> —
        # which it did, on exactly this conversation.
        context = [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:1500]}
            for m in (history or [])[-_INTEREST_CONTEXT_TURNS:]
        ]
        parsed = await interest_llm.ainvoke(
            [
                {"role": "system", "content": INTEREST_INSTRUCTIONS},
                *context,
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply[-1500:]},
            ]
        ) or {}
    except Exception:  # noqa: BLE001 — never fail a turn over a note
        log.exception("Interest extraction failed")
        return

    kind = parsed.get("kind")
    item = (parsed.get("item") or "").strip()
    if kind not in ("wants_to_buy", "likes") or not item:
        return
    if _is_vague(item):
        # "the album" is what the customer said, but it is not what they meant,
        # and a note keyed on it is unreadable next week
        log.info("Discarding interest %r: refers to something without naming it", item)
        return
    haystack = f"{user_text}\n{reply}".lower()
    if item.lower() not in haystack:
        log.info("Discarding interest %r: not said in this turn", item)
        return
    log.info("Recording interest: %s (%s)", item, kind)
    _record_interest(item, kind)


# ── supervisor ───────────────────────────────────────────────────────────────


IDENTIFY_INSTRUCTIONS = """You are the front desk of an online music store. Your only \
job right now is to work out who you are speaking to. From the conversation so far, \
extract the customer's first name, last name, and the phone number on their account. \
If instead they gave a customer key — a long string of letters and digits — put that \
in customer_key and leave the other fields null; it identifies them on its own. \
Do not guess or invent any of them — leave a field null if the customer has not said \
it. Do not answer any other question yet."""


class CustomerIdentity(TypedDict):
    """Who the customer is. Leave a field null when they have not said it."""

    customer_first_name: str | None
    customer_last_name: str | None
    customer_phone: str | None
    # An alternative to the three fields above, not an addition: the primary key
    # of the customers feature group, which resolves to all three.
    customer_key: str | None


identity_llm = ChatAnthropic(
    model=ROUTER_MODEL, max_tokens=512, temperature=0.0
).with_structured_output(CustomerIdentity)

ASK_FOR_IDENTITY = (
    "Before we start — could you give me your first name, last name, and the "
    "phone number on your account? If you have your customer key to hand, that "
    "works on its own. I'll keep them for the rest of this chat."
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


# customer_key is a truncated sha256 (see customer_key()), so anything that is
# not exactly that shape was not one — checked before it reaches the store so a
# hallucinated "key" is a re-ask rather than a lookup for nobody.
_KEY_RE = re.compile(r"^[0-9a-f]{24}$")


def _identity_from_key(parsed: dict) -> dict:
    """Resolve a customer key the customer quoted back into name and phone.

    The key is the primary key of `chinook_customers`, so this is one point
    lookup. Returning the row's own name and phone rather than the key alone
    keeps every downstream caller working on the three fields it already
    expects — including `_interest_owner`, which recomputes the key from them
    and, because the row is what the key was derived from, arrives back at the
    same string.

    This is identification, not authentication, exactly as name-and-phone is:
    the key is derived deterministically from those same three fields and is
    visible in the feature-group browser, so it proves possession of a
    reference, not ownership of the account. It grants nothing that answering
    "I'm Aaron Mitchell on +1 (204) 452-6452" would not.
    """
    raw = parsed.get("customer_key")
    if raw is None:
        return {}
    key = str(raw).strip().lower()
    if key in _MISSING_MARKERS or not _KEY_RE.match(key):
        return {}
    row = _lookup_one(CUSTOMERS_FG, {"customer_key": key})
    if not row:
        log.info("Customer key %r matched no customer", key)
        return {}
    resolved = {
        "customer_first_name": row.get("first_name"),
        "customer_last_name": row.get("last_name"),
        "customer_phone": row.get("phone"),
    }
    if not all(resolved.values()):
        return {}
    return {k: str(v) for k, v in resolved.items()}


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

    extracted = (
        await identity_llm.ainvoke(
            [{"role": "system", "content": IDENTIFY_INSTRUCTIONS}, *state["messages"]]
        )
        or {}
    )
    parsed = _clean_identity(extracted)
    if not all(parsed.get(key) for key in IDENTITY_KEYS):
        # A quoted customer key stands in for all three fields. Only consulted
        # when they are not already complete, so a customer who gave their name
        # and phone is never overridden by a key they mentioned in passing.
        parsed = _identity_from_key(extracted) or parsed
    if all(parsed.get(key) for key in IDENTITY_KEYS):
        # The chatbot just asked who it is talking to and got an answer, so
        # bind durable memory to that customer for the rest of the turn. Until
        # now `subject` was whatever the client asserted — from the Hopsworks
        # panel, the logged-in operator, which is the same value for every
        # customer. Everything subject-keyed (`user` scope, system_context,
        # search) follows this.
        _rebind_to_customer(parsed)
        for key in IDENTITY_KEYS:
            # `session` scope (see IDENTITY_SCOPE): owned by the conversation
            # id, so it dies with the conversation and the next one asks again.
            # Deliberately not `user` — that is owned by ctx.subject, which is
            # the serving-key holder, not the customer.
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

You can record an order with `place_order`, but you CANNOT take payment. Nothing \
in this chat charges a card, and no money moves. So when an order goes through, \
say it is recorded on their account and that nothing has been charged — never \
say it is paid for, or that a card has been billed.

Being interested in something is not asking to buy it. "I'm interested in this \
album", "I like this one" and "I want this" are interests: record them with \
`remember_interest` and then ask whether they would like you to place the \
order. Only order when they have told you to — "buy it", "order it", or a yes \
to that question. If you asked "would you like to buy this?" and they replied \
with something other than a clear yes, ask again rather than assuming.

Only call `place_order` once the customer has clearly said they want a specific \
album or track. If they are still browsing, or only said they like something, \
use `remember_interest` instead so you can follow it up next time.

You are told below what you already know about this customer. Never ask again for \
anything that appears there - use it and carry on. If a detail is missing, ask for \
it once, then remember it for the rest of this conversation.

When a customer says they want to buy something, or that they simply like an \
album or artist, record it with `remember_interest`. That outlives the \
conversation, so next time you can pick the thread back up. Never tell a \
customer you have noted, recorded or saved an interest unless you called \
`remember_interest` in this turn and it succeeded - if you did not call it, \
you did not record anything. `search` only looks at what is already stored; it \
never stores. Do not store catalogue facts; look those up.\
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
    memory=ManagedMemoryService(
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
    # Before ctx.system_context(), which reads `user` scope keyed on the
    # subject: rebinding after it would build the prompt for the wrong person.
    # On the very first turn of a conversation identity is not known yet, so
    # `identify` rebinds mid-turn instead and this turn's prompt carries no
    # durable block — the turn after it does.
    _rebind_to_customer(identity)

    system = SYSTEM_PROMPT + ctx.system_context()
    # What we remember about this customer from previous conversations,
    # reconciled against what they actually bought. Keyed on the customer, so
    # unlike ctx.system_context() this survives a new conversation.
    system += interests_block(ctx.memory, identity)

    messages = [{"role": "system", "content": system}]
    messages += ctx.history  # turns since the last fold
    messages.append({"role": "user", "content": request.text})

    # Seed the graph with whatever we already know about this customer. This is
    # what makes the identity gate a once-per-conversation ask rather than a
    # once-per-turn one: `session`-scoped state outlives the turn, so every turn
    # after the first arrives already identified and `identify` falls straight
    # through. A new conversation asks again, by design — nothing here can
    # authenticate anyone, so identity must not outlive the conversation it was
    # asserted in.
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
    spoken: list[str] = []
    async for delta in ctx.stream_langchain(
        _capture(graph.astream_events({"messages": messages, **identity}, version="v2"))
    ):
        streamed = True
        spoken.append(str(delta))
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

    # After the last token, so it costs request duration rather than
    # time-to-answer — the same slot the SDK folds the summary in.
    await _record_stated_interest(
        request.text,
        "".join(spoken) or str(final_state.get("followup") or ""),
        ctx.history,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
