"""
A small agent whose subject is the memory service itself.

Every other example in this repo uses memory in passing. This one exists to
make it visible: three tiers, four tools, and a `memory_report` that prints
what is in each tier right now, so a person chatting with it can watch the
history fold into a summary and watch a fact outlive the conversation it was
mentioned in.

What it demonstrates, in the order you meet it:

1. **The conversation buffer.** Every turn arrives with the ones before it.
   Nothing to configure; `ctx.history` is it.
2. **The rolling summary.** Older turns are folded into a running summary
   instead of dropped. Folding is set aggressively here (every 6 messages,
   keeping 4) so it happens inside a demo chat rather than after an hour.
3. **Durable memory.** `remember` / `recall` / `forget` / `search` write and
   read facts that outlive the conversation, keyed by the person
   (`ctx.subject`) once `identify` has been called.

Deploy it as a single file; it needs no feature store and no data:

    agents = hopsworks.login().get_agent_serving()
    agent = agents.deploy_agent("agent_memory/memory_agent.py", name="memorydemo")
    agent.start()
"""

from __future__ import annotations

import json
import os
from typing import Any

from agents import Agent, ModelSettings, Runner, function_tool
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    identity_tools,
    memory_tools,
    openai_summarizer,
)
from hopsworks_agents.protocol.autoevents import current_context
from openai.types.responses import ResponseTextDeltaEvent

from prompts import SYSTEM_PROMPT, WELCOME

MODEL_NAME = os.environ.get("MEMORY_DEMO_MODEL", "")

#: Fold early and keep little, so the summary tier is reachable in a demo. A
#: real agent leaves these alone: the defaults (20 and 10) fold about once an
#: hour of conversation instead of twice a coffee break.
FOLD_AFTER_MESSAGES = int(os.environ.get("MEMORY_DEMO_FOLD_AFTER", "6"))
KEEP_RECENT_MESSAGES = int(os.environ.get("MEMORY_DEMO_KEEP_RECENT", "4"))

#: Vector search over earlier conversations. Off by default: it loads an
#: embedding model and writes an embedding feature group, which is more than a
#: demo of the first three tiers needs. With it off, `search` still works -- on
#: keywords rather than meaning, which is the difference worth showing.
VECTOR_SEARCH = os.environ.get("MEMORY_DEMO_VECTOR_SEARCH", "").lower() in (
    "1",
    "true",
    "yes",
)


def _model_kwargs() -> dict[str, Any]:
    if "/" in MODEL_NAME:
        from agents.extensions.models.litellm_model import LitellmModel  # noqa: PLC0415

        return {"model": LitellmModel(model=MODEL_NAME)}
    return {"model": MODEL_NAME} if MODEL_NAME else {}


def _vector_memory() -> dict[str, Any]:
    """The embedder and its feature group, when the demo asks for semantic search.

    The embedding model comes from the project's model registry when it was
    registered there, so this does not pull it from the internet on every pod
    start; see `register_sentence_transformer`.
    """
    if not VECTOR_SEARCH:
        return {}
    from hopsworks_agents.protocol import (  # noqa: PLC0415
        sentence_transformer_embedder,
        vector_store_for,
    )

    embedder = sentence_transformer_embedder()
    return {"embedder": embedder, "vector_store": vector_store_for(embedder)}


# ── the tool that makes the memory visible ───────────────────────────────────


@function_tool
def memory_report() -> str:
    """Show what is currently held in each memory tier.

    Use this when the user asks what you remember, or wants to see how the
    memory works. It reads the tiers; it never changes them.
    """
    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        return "No memory is configured for this deployment."
    history = ctx.history
    summary = ctx.summary
    report: dict[str, Any] = {
        "who_you_are_talking_to": {
            "subject": ctx.subject,
            # "conversation" means nobody has identified themselves yet, so
            # durable facts are keyed by this chat and will not follow them
            "known_from": ctx.subject_source,
        },
        "tier_1_conversation_buffer": {
            "turns_held": len(history),
            "folds_after_messages": FOLD_AFTER_MESSAGES,
            "oldest_turn_held": (history[0]["content"][:120] if history else None),
        },
        "tier_2_rolling_summary": summary or "nothing folded yet",
        "tier_3_durable_memory": {
            "about_this_person": ctx.state("user"),
            "this_conversation_only": ctx.state("session"),
        },
        "search_mode": "vector" if VECTOR_SEARCH else "keyword",
    }
    return json.dumps(report, indent=2, default=str)


TOOLS = [
    *memory_tools("openai_agents"),
    *identity_tools("openai_agents"),
    memory_report,
]

agent = Agent(
    name="Memory demo",
    instructions=SYSTEM_PROMPT,
    tools=TOOLS,
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
)

agent_app = AgentApp(
    eval_per_request=True,
    name="Agent memory demo",
    description="A small assistant that shows the agent memory service working: "
    "the conversation buffer, the rolling summary, and durable per-person facts "
    "it can store, recall, search and forget.",
    framework="openai_agents",
    welcome_message=WELCOME,
    suggested_prompts=[
        "I'm Dana, I live in Oslo and I'm vegetarian.",
        "What do you remember about me?",
        "Show me what's in each memory tier.",
        "Forget where I live.",
    ],
    placeholder="Tell me something about yourself, or ask what I remember...",
    memory=ManagedMemoryService(
        summarize=openai_summarizer(),
        summarize_after_messages=FOLD_AFTER_MESSAGES,
        keep_recent_messages=KEEP_RECENT_MESSAGES,
        long_term=True,
        **_vector_memory(),
    ),
    tool_events=True,
)


def _input_items(ctx, request) -> list[dict[str, Any]]:
    """The turn, as the model sees it.

    `ctx.system_context()` is the whole point of the summary tier: it carries
    the folded summary and the durable facts, built with no model round-trip.
    `ctx.history` is only what has not been folded yet, so the two together are
    the conversation without the token cost of all of it.
    """
    items: list[dict[str, Any]] = []
    context = ctx.system_context()
    if context:
        items.append({"role": "user", "content": f"({context})"})
    for turn in ctx.history:
        content = str(turn.get("content") or "").strip()
        if content:
            items.append(
                {
                    "role": "user" if turn.get("role") == "user" else "assistant",
                    "content": content,
                }
            )
    items.append({"role": "user", "content": request.text})
    return items


@agent_app.stream
async def stream(request, ctx):
    """One handler for both endpoints: the answer streams, and every memory tool
    the model reaches for shows as a progress chip -- which is how a person
    watching the demo sees that `remember` really was called."""
    if not request.text:
        raise AgentError(
            "Say something and I'll remember it.",
            code="invalid_request",
            status_code=400,
        )
    result = Runner.run_streamed(agent, _input_items(ctx, request), max_turns=8)
    streamed = False
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                streamed = True
                yield event.data.delta
        elif isinstance(event, RunItemStreamEvent):
            raw = getattr(event.item, "raw_item", None)
            if event.name == "tool_called":
                await ctx.emit_event(
                    getattr(raw, "name", "tool"),
                    status="running",
                    message=str(getattr(raw, "arguments", "") or "")[:200],
                    event_id=getattr(raw, "call_id", None),
                )
            elif event.name == "tool_output":
                await ctx.emit_event(
                    "tool",
                    status="done",
                    event_id=(
                        raw.get("call_id")
                        if isinstance(raw, dict)
                        else getattr(raw, "call_id", None)
                    ),
                )
    if not streamed:
        yield str(result.final_output or "").strip() or "(no answer)"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
