"""
Personalized shopping on the OpenAI Agents SDK: the same store, the same prompt, a
different agent runtime.

The tools are the store's plain functions wrapped as `function_tool`s; the
recipe's interaction flow is the prompt; the Hopsworks protocol app is the
serving surface, as in `shopping_agent.py`. A shopper's photo reaches the
model as an image input.

Deploy (git-backed, so `store.py` and `prompts.py` come along):
    agents.deploy_agent("personalized_shopping/shopping_agent_openai.py", name="webshop-openai", ...)
"""

from __future__ import annotations

import os
from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    openai_summarizer,
)
from openai.types.responses import ResponseTextDeltaEvent

from prompts import SYSTEM_PROMPT
from store import TOOLS as STORE_TOOLS

#: OpenAI by default (the SDK's own default model); "anthropic/claude-sonnet-4-5" and the like go
#: through LiteLLM, which the requirements bring in
MODEL_NAME = os.environ.get("WEBSHOP_OPENAI_MODEL", "")


def _model_kwargs() -> dict[str, Any]:
    if "/" in MODEL_NAME:
        from agents.extensions.models.litellm_model import LitellmModel  # noqa: PLC0415

        return {"model": LitellmModel(model=MODEL_NAME)}
    return {"model": MODEL_NAME} if MODEL_NAME else {}


def _plain(tool: Any):
    """The function under a LangChain tool, or the function itself."""
    return getattr(tool, "func", None) or getattr(tool, "coroutine", None) or tool


TOOLS = [function_tool(_plain(t)) for t in STORE_TOOLS]

agent = Agent(
    name="Personalized shopping agent",
    instructions=SYSTEM_PROMPT,
    tools=TOOLS,
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
)
_run_config = RunConfig(tool_not_found_behavior="return_error_to_model")

agent_app = AgentApp(
    eval_per_request=True,
    name="Personalized shopping (OpenAI Agents)",
    description="Finds products in the shop's catalogue, explores them with you and places "
    "the order when you say so (OpenAI Agents SDK over Hopsworks feature-store search).",
    framework="openai_agents",
    welcome_message="Hi! I can help you find something in our shop, tell you all about "
    "it, and order it for you when you're ready. What are you looking for?",
    suggested_prompts=[
        "Hello, who are you?",
        "Can you help me find a summer dress? Something flowy and floral.",
        "I'm looking for a rustic console table for my entryway.",
    ],
    placeholder="What are you looking for? You can also send a photo...",
    input_modalities=["text", "image"],
    memory=ManagedMemoryService(summarize=openai_summarizer()),
    tool_events=True,
)


def _user_item(request) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if request.text:
        parts.append({"type": "input_text", "text": request.text})
    for image in request.images:
        parts.append({"type": "input_image", "image_url": f"data:{image.media_type};base64,{image.data}", "detail": "auto"})
    if not parts:
        raise AgentError("Send some text, or an image of what you are looking for.", code="invalid_request", status_code=400)
    if len(parts) == 1 and parts[0]["type"] == "input_text":
        return {"role": "user", "content": request.text}
    if not request.text:
        parts.insert(0, {"type": "input_text", "text": "Find me this product."})
    return {"role": "user", "content": parts}


def _input_items(ctx, request) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    summary = ctx.system_context()
    if summary:
        items.append({"role": "user", "content": "(" + summary + ")"})
    for turn in ctx.history:
        content = str(turn.get("content") or "").strip()
        if content:
            items.append({"role": "user" if turn.get("role") == "user" else "assistant", "content": content})
    items.append(_user_item(request))
    return items


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints: the reply streams, and each tool call is a progress chip."""
    result = Runner.run_streamed(agent, _input_items(ctx, request), max_turns=10, run_config=_run_config)
    streamed = False
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                streamed = True
                yield event.data.delta
        elif isinstance(event, RunItemStreamEvent):
            raw = getattr(event.item, "raw_item", None)
            if event.name == "tool_called":
                await ctx.emit_event(getattr(raw, "name", "tool"), status="running",
                                     message=str(getattr(raw, "arguments", "") or "")[:200],
                                     event_id=getattr(raw, "call_id", None))
            elif event.name == "tool_output":
                await ctx.emit_event("tool", status="done",
                                     event_id=raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None))
    if not streamed:
        final = str(result.final_output or "").strip()
        yield final or "Sorry — I wasn't able to put together a reply just then. Could you try rephrasing?"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
