"""The Chinook support agent, as a LlamaIndex workflow agent.

The same store, the same seven tools and the same rules as `support_agent.py`.
Both import them from `store.py`, so the two agents differ only in how they are
wired — which is the point of having both: an evaluation suite written against
one should hold against the other, and any difference in its results is a
difference in the framework rather than in what the agent was told to do.

What is genuinely different, and deliberately so:

`support_agent.py` is a LangGraph supervisor with sub-agents — one to identify
the customer, one to route, one to answer, one to refund. This is a single
`FunctionAgent` with all seven tools. Splitting a graph into sub-agents buys
control over what the model may do at each step; a function agent buys
simplicity, and the rules that matter are enforced in the tools either way:

  - `purchase_history` and `place_order` refuse without all three identity
    fields, whoever asks them
  - `place_order` records an interest instead of an order unless it is told the
    customer said to buy, so an agent that guesses wrong costs a question rather
    than an unwanted order

That is why they are in the tools rather than in a graph. A rule that only holds
because a supervisor routed correctly is a rule that holds until the routing
changes.

The identity gate works the same way: `session`-scoped state outlives the turn,
so the ask happens once per conversation rather than once per turn, and a new
conversation asks again — nothing here can authenticate anyone, so identity must
not outlive the conversation it was asserted in.
"""

from __future__ import annotations

import logging
import os
import re

from langchain_anthropic import ChatAnthropic
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from typing_extensions import TypedDict

from hopsworks_agent_protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
    memory_tools,
    remember,
)

from store import (
    ANSWER_MODEL,
    CUSTOMERS_FG,
    IDENTITY_KEYS,
    IDENTITY_SCOPE,
    ROUTER_MODEL,
    _current_identity,
    _lookup_one,
    _rebind_to_customer,
    interests_block,
    lookup_album,
    lookup_artist,
    lookup_track,
    place_order,
    purchase_history,
    recall_interests,
    remember_interest,
)

log = logging.getLogger(__name__)

# Same rules, same words as the LangGraph agent. Kept identical on purpose: a
# suite that passes against one and fails against the other should be telling
# you about the framework, and a prompt that had drifted would make that
# comparison meaningless.
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

# The store's functions, as this framework's tool type. `from_defaults` reads the
# name, signature and docstring, which is why the docstrings in store.py are
# written for the model.
TOOLS = [
    FunctionTool.from_defaults(fn=fn)
    for fn in (
        lookup_track,
        lookup_artist,
        lookup_album,
        purchase_history,
        place_order,
        remember_interest,
        recall_interests,
    )
] + [
    # `search` only, for the same reason as the LangGraph agent: `remember` is
    # left out because `remember_interest` is the one writer for this kind of
    # fact, and `recall` because it looks a value up by exact key — interests are
    # stored as "likes:<name>", which the model cannot enumerate, so it guesses a
    # key, misses, and tells the customer there is no record.
    *memory_tools("llamaindex", include=("search",)),
]

_llm = Anthropic(model=os.environ.get("CHINOOK_ANSWER_MODEL", ANSWER_MODEL))


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


_MISSING_MARKERS = {
    "", "none", "null", "n/a", "na", "unknown", "not provided",
    "not specified", "string",
}


def _clean_identity(parsed: dict) -> dict:
    """Drop values the model filled in rather than read."""
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


_KEY_RE = re.compile(r"^[0-9a-f]{24}$")


def _identity_from_key(parsed: dict) -> dict:
    """Resolve a customer key the customer quoted back into name and phone."""
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


async def _identify_from_turn(messages: list[dict]) -> dict:
    extracted = (
        await identity_llm.ainvoke(
            [{"role": "system", "content": IDENTIFY_INSTRUCTIONS}, *messages]
        )
        or {}
    )
    parsed = _clean_identity(extracted)
    if not all(parsed.get(key) for key in IDENTITY_KEYS):
        # A quoted customer key stands in for all three fields.
        parsed = _identity_from_key(extracted) or parsed
    return parsed


def _agent(system_prompt: str) -> FunctionAgent:
    """A fresh agent per turn, carrying this turn's system prompt.

    Rebuilt rather than kept module-level because the prompt is not constant: it
    carries the rolling summary and what is known about this particular customer,
    and a shared agent would answer one customer with another's context.
    """
    return FunctionAgent(tools=TOOLS, llm=_llm, system_prompt=system_prompt)


agent_app = AgentApp(
    name="Chinook support agent (LlamaIndex)",
    description="Customer support for an online music store: catalogue questions "
    "and refunds (LlamaIndex FunctionAgent + Hopsworks feature store lookup).",
    framework="llamaindex",
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
)


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints; ctx.stream_llamaindex yields token
    deltas and turns each tool call into a progress chip."""
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.",
            code="invalid_request",
            status_code=400,
        )

    # What this conversation already established about who is asking. Session
    # scope, so the gate is a once-per-conversation ask rather than a per-turn
    # one; a new conversation asks again.
    known = ctx.state(IDENTITY_SCOPE)
    identity = {key: known[key] for key in IDENTITY_KEYS if known.get(key)}
    if not all(identity.get(key) for key in IDENTITY_KEYS):
        parsed = {
            **identity,
            **await _identify_from_turn(
                [
                    *ctx.history,
                    {"role": "user", "content": request.text},
                ]
            ),
        }
        if not all(parsed.get(key) for key in IDENTITY_KEYS):
            yield ASK_FOR_IDENTITY
            return
        identity = {key: str(parsed[key]) for key in IDENTITY_KEYS}
        _rebind_to_customer(identity)
        for key in IDENTITY_KEYS:
            remember(key, str(identity[key]), scope=IDENTITY_SCOPE)

    # Published for the tools, which the model calls without any of this in scope.
    _current_identity.set(identity)
    # Before ctx.system_context(), which reads `user` scope keyed on the subject:
    # rebinding after it would build the prompt for the wrong person.
    _rebind_to_customer(identity)

    system = SYSTEM_PROMPT + ctx.system_context()
    # Keyed on the customer rather than the conversation, so unlike
    # ctx.system_context() this survives a new conversation.
    system += interests_block(ctx.memory, identity)

    # History as chat messages: the agent is rebuilt each turn, so nothing but
    # this carries the conversation forward.
    from llama_index.core.llms import ChatMessage

    history = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in ctx.history
        if message.get("content")
    ]

    handler = _agent(system).run(request.text, chat_history=history)

    spoken = False
    async for delta in ctx.stream_llamaindex(handler):
        spoken = True
        yield delta

    if not spoken:
        # A turn that called only tools and produced no tokens would otherwise
        # show the customer an empty reply. The final response is what the agent
        # meant to say; the LangGraph agent has the same fallback for the same
        # reason.
        response = await handler
        text = str(getattr(response, "response", response) or "").strip()
        if text:
            yield text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
