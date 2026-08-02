"""The Chinook support agent, as a Claude Agent SDK agent.

Same store, same tools and same identity gate as the LangGraph and LlamaIndex
entrypoints. The store functions are exposed through the Claude Agent SDK's
in-process MCP server, and built-in Claude Code file/shell tools are disabled:
this is a customer-support agent, not a repo-editing agent.
"""

from __future__ import annotations

import json
import os
import re
import types as types_module
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

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


IDENTIFY_INSTRUCTIONS = """You are the front desk of an online music store. Your only \
job right now is to work out who you are speaking to. From the conversation so far, \
extract the customer's first name, last name, and the phone number on their account. \
If instead they gave a customer key — a long string of letters and digits — put that \
in customer_key and leave the other fields null; it identifies them on its own. \
Do not guess or invent any of them — leave a field null if the customer has not said \
it. Do not answer any other question yet."""


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

_IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_first_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "customer_last_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "customer_phone": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "customer_key": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": [
        "customer_first_name",
        "customer_last_name",
        "customer_phone",
        "customer_key",
    ],
    "additionalProperties": False,
}


def _json_schema(annotation: object) -> dict:
    origin = get_origin(annotation)
    if origin is Literal:
        values = [value for value in get_args(annotation) if value is not None]
        return {"type": "string", "enum": [str(value) for value in values]}
    if origin in (Union, types_module.UnionType):
        args = get_args(annotation)
        return {"anyOf": [_json_schema(arg) for arg in args]}
    if annotation is None or annotation is type(None):
        return {"type": "null"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is list or origin is list:
        item_args = get_args(annotation)
        return {
            "type": "array",
            "items": _json_schema(item_args[0]) if item_args else {},
        }
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _tool_schema(fn) -> dict:
    try:
        type_hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001
        type_hints = {}
    import inspect

    properties = {}
    required = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = _json_schema(type_hints.get(name, param.annotation))
        if param.default is inspect.Signature.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool_text(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _claude_tool(fn):
    import inspect

    params = inspect.signature(fn).parameters

    async def invoke(args: dict[str, Any]) -> dict[str, Any]:
        values = {}
        for name, param in params.items():
            if name in args:
                values[name] = args[name]
            elif param.default is inspect.Signature.empty:
                return {
                    "content": [
                        {"type": "text", "text": f"Missing required argument {name!r}."}
                    ],
                    "is_error": True,
                }
        try:
            result = fn(**values)
        except Exception as err:  # noqa: BLE001
            return {"content": [{"type": "text", "text": str(err)}], "is_error": True}
        return {"content": [{"type": "text", "text": _tool_text(result)}]}

    invoke.__name__ = fn.__name__
    invoke.__doc__ = fn.__doc__
    return tool(fn.__name__, fn.__doc__ or fn.__name__, _tool_schema(fn))(invoke)


TOOLS = [
    _claude_tool(fn)
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
    *memory_tools("claude_agents", include=("search",)),
]

_MCP_SERVER = create_sdk_mcp_server(
    name="chinook",
    version="1.0.0",
    tools=TOOLS,
)
_ALLOWED_TOOLS = [f"mcp__chinook__{item.name}" for item in TOOLS]
_IDENTITY_MAX_TURNS = 3


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


def _model(env_name: str, default: str | None = None) -> str | None:
    return os.environ.get(env_name) or default


def _base_options(system_prompt: str, *, with_tools: bool) -> dict:
    options = {
        "system_prompt": system_prompt,
        "tools": [],
        "setting_sources": [],
    }
    model = _model(
        "CHINOOK_CLAUDE_ANSWER_MODEL" if with_tools else "CHINOOK_CLAUDE_ROUTER_MODEL",
        ANSWER_MODEL,
    )
    if model:
        options["model"] = model
    if with_tools:
        options.update(
            {
                "mcp_servers": {"chinook": _MCP_SERVER},
                "allowed_tools": _ALLOWED_TOOLS,
                "strict_mcp_config": True,
                "max_turns": 8,
            }
        )
    else:
        options.update(
            {
                "output_format": {
                    "type": "json_schema",
                    "schema": _IDENTITY_SCHEMA,
                },
                "max_turns": _IDENTITY_MAX_TURNS,
            }
        )
    return options


async def _run_claude(prompt: str, options: ClaudeAgentOptions) -> tuple[str, dict | None]:
    text_chunks: list[str] = []
    final_text = ""
    structured = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_chunks.append(block.text)
        if isinstance(message, ResultMessage):
            value = getattr(message, "result", None)
            if value:
                final_text = str(value)
            structured = getattr(message, "structured_output", None) or structured
    return (final_text or "".join(text_chunks)).strip(), structured


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


async def _identify_from_turn(messages: list[dict]) -> dict:
    text, structured = await _run_claude(
        _conversation_text(messages),
        ClaudeAgentOptions(**_base_options(IDENTIFY_INSTRUCTIONS, with_tools=False)),
    )
    extracted = structured or {}
    if not extracted:
        try:
            extracted = json.loads(text)
        except Exception:  # noqa: BLE001
            extracted = {}
    parsed = _clean_identity(extracted)
    if not all(parsed.get(key) for key in IDENTITY_KEYS):
        parsed = _identity_from_key(extracted) or parsed
    return parsed


agent_app = AgentApp(
    name="Chinook support agent (Claude Agent SDK)",
    description="Customer support for an online music store: catalogue questions "
    "and refunds (Claude Agent SDK + Hopsworks feature store lookup).",
    framework="claude_agents",
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

    text, _ = await _run_claude(
        _conversation_text(ctx.history, request.text),
        ClaudeAgentOptions(**_base_options(system, with_tools=True)),
    )
    yield text or (
        "Sorry — I wasn't able to put together a reply just then. "
        "Could you try rephrasing?"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
