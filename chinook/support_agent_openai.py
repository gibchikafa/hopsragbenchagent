"""The Chinook support agent, as an OpenAI Agents SDK agent.

Same store, same tools and same identity gate as the LangGraph and LlamaIndex
entrypoints. This file only changes the framework wiring: the model-facing
tools are OpenAI Agents `function_tool`s, and the Hopsworks protocol app
advertises `framework="openai_agents"` so tracing can pick the right
OpenInference instrumentor.
"""

from __future__ import annotations

import json
import os
import re

from agents import Agent, RunConfig, Runner, function_tool
from pydantic import BaseModel

from hopsworks_agent_protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
    memory_tools,
    remember,
)

from store import (
    CUSTOMERS_FG,
    IDENTITY_KEYS,
    IDENTITY_SCOPE,
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


TOOLS = [
    function_tool(fn)
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
    # `search` only, for the same reason as the other single-agent entrypoints:
    # `remember_interest` is the one writer for this kind of durable fact, and
    # `recall_interests` is the only reader that can enumerate them properly.
    *memory_tools("openai_agents", include=("search",)),
]


IDENTIFY_INSTRUCTIONS = """You are the front desk of an online music store. Your only \
job right now is to work out who you are speaking to. From the conversation so far, \
extract the customer's first name, last name, and the phone number on their account. \
If instead they gave a customer key — a long string of letters and digits — put that \
in customer_key and leave the other fields null; it identifies them on its own. \
Do not guess or invent any of them — leave a field null if the customer has not said \
it. Do not answer any other question yet."""


class CustomerIdentity(BaseModel):
    """Who the customer is. Leave a field null when they have not said it."""

    customer_first_name: str | None = None
    customer_last_name: str | None = None
    customer_phone: str | None = None
    customer_key: str | None = None


ASK_FOR_IDENTITY = (
    "Before we start — could you give me your first name, last name, and the "
    "phone number on your account? If you have your customer key to hand, that "
    "works on its own. I'll keep them for the rest of this chat."
)


_MISSING_MARKERS = {
    "", "none", "null", "n/a", "na", "unknown", "not provided",
    "not specified", "string",
}
_KEY_RE = re.compile(r"^[0-9a-f]{24}$")


def _model_kwargs(env_name: str) -> dict:
    model = os.environ.get(env_name)
    return {"model": model} if model else {}


_identity_agent = Agent(
    name="Chinook identity extractor",
    instructions=IDENTIFY_INSTRUCTIONS,
    output_type=CustomerIdentity,
    **_model_kwargs("CHINOOK_OPENAI_ROUTER_MODEL"),
)
_run_config = RunConfig(tool_not_found_behavior="return_error_to_model")


def _agent(system_prompt: str) -> Agent:
    return Agent(
        name="Chinook support agent",
        instructions=system_prompt,
        tools=TOOLS,
        **_model_kwargs("CHINOOK_OPENAI_ANSWER_MODEL"),
    )


def _conversation_text(messages: list[dict], user_text: str | None = None) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    if user_text:
        lines.append(f"user: {user_text.strip()}")
    return "\n".join(lines) or (user_text or "")


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
        return {}
    resolved = {
        "customer_first_name": row.get("first_name"),
        "customer_last_name": row.get("last_name"),
        "customer_phone": row.get("phone"),
    }
    if not all(resolved.values()):
        return {}
    return {k: str(v) for k, v in resolved.items()}


def _identity_dict(value) -> dict:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except Exception:  # noqa: BLE001
        return {}


async def _identify_from_turn(messages: list[dict]) -> dict:
    result = await Runner.run(
        _identity_agent,
        _conversation_text(messages),
        max_turns=1,
    )
    extracted = _identity_dict(result.final_output)
    parsed = _clean_identity(extracted)
    if not all(parsed.get(key) for key in IDENTITY_KEYS):
        # A quoted customer key stands in for all three fields.
        parsed = _identity_from_key(extracted) or parsed
    return parsed


agent_app = AgentApp(
    name="Chinook support agent (OpenAI Agents)",
    description="Customer support for an online music store: catalogue questions "
    "and refunds (OpenAI Agents SDK + Hopsworks feature store lookup).",
    framework="openai_agents",
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
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.",
            code="invalid_request",
            status_code=400,
        )

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

    _current_identity.set(identity)
    _rebind_to_customer(identity)

    system = SYSTEM_PROMPT + ctx.system_context()
    system += interests_block(ctx.memory, identity)

    result = await Runner.run(
        _agent(system),
        _conversation_text(ctx.history, request.text),
        max_turns=8,
        run_config=_run_config,
    )
    text = str(result.final_output or "").strip()
    yield text or (
        "Sorry — I wasn't able to put together a reply just then. "
        "Could you try rephrasing?"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
