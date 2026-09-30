"""
Travel concierge — Google's ADK travel-concierge recipe on the OpenAI Agents
SDK and the Hopsworks Agent Protocol.

    root_agent ──▶ inspiration_agent ──▶ planning_agent ──▶ booking_agent      (before booking)
        └────────▶ pre_trip_agent ──▶ in_trip_agent ──▶ post_trip_agent          (after: the trip has an itinerary)

The recipe is a root agent with six sub-agents that transfer the conversation
to one another, and whichever agent last had it keeps it across turns. Each
sub-agent carries agent-tools of its own (a place finder, a flight search, an
itinerary writer) that are LLMs asked for JSON, plus mocks for what a demo
cannot do (flight status, payment). Everything reads and writes ADK's session
state: the profile, the itinerary, the selections, the "current time on the
trip".

That is the OpenAI Agents SDK's own shape, which is why it was chosen over
LangGraph for this one: `handoffs` are the transfers, under the same tool
names (`transfer_to_planning_agent`); `as_tool` is the AgentTool; a run's
`last_agent` is the agent that keeps the conversation, remembered in working
memory so the next turn starts there. Working memory is the SDK's session
scope; the preferences the post-trip agent learns go to user scope, so they
are there next time, which is what the recipe wanted from them.

Deploy (git-backed, so the modules and profiles come along):
    agents = hopsworks.login().get_agent_serving()
    agents.deploy_agent("travel_concierge/concierge_agent.py", name="concierge",
                        git_url="https://github.com/<you>/hopsragbenchagent.git",
                        git_provider="GitHub", git_branch="main",
                        requirements="travel_concierge/requirements.txt").start()
"""

from __future__ import annotations

from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
)
from openai.types.responses import ResponseTextDeltaEvent

import memory
from prompts import (
    AGENTS,
    BOOKING_AGENT_INSTR,
    CONTEXT_BLOCK,
    INSPIRATION_AGENT_INSTR,
    INTRIP_INSTR,
    PERSONA,
    PLANNING_AGENT_INSTR,
    POSTTRIP_INSTR,
    PRETRIP_AGENT_INSTR,
    ROOT_AGENT_INSTR,
)
from tools import TOOLS_BY_AGENT, _model_kwargs

INSTRUCTIONS = {
    "inspiration_agent": INSPIRATION_AGENT_INSTR,
    "planning_agent": PLANNING_AGENT_INSTR,
    "booking_agent": BOOKING_AGENT_INSTR,
    "pre_trip_agent": PRETRIP_AGENT_INSTR,
    "in_trip_agent": INTRIP_INSTR,
    "post_trip_agent": POSTTRIP_INSTR,
}
TEMPERATURE = {"planning_agent": 0.1, "booking_agent": 0.0}
#: a stored itinerary is a few thousand characters; the default 4k would cut it
MAX_STATE_CHARS = 32_000
MAX_TURNS = 12


def _instructions(agent_name: str):
    """The agent's instructions for this turn: the recipe's text, filled from working memory."""

    def build(_ctx, _agent) -> str:
        values = memory.prompt_context()
        body = INSTRUCTIONS[agent_name].format(**values)
        return PERSONA + "\n\nYou are the " + agent_name + ".\n\n" + body + CONTEXT_BLOCK.format(**values)

    return build


def _root_instructions(_ctx, _agent) -> str:
    values = memory.prompt_context()
    return (PERSONA + "\n\n" + ROOT_AGENT_INSTR.format(itinerary_summary=memory.itinerary_summary())
            + CONTEXT_BLOCK.format(**values))


SUB_AGENTS: dict[str, Agent] = {
    name: Agent(
        name=name,
        handoff_description=what,
        instructions=_instructions(name),
        tools=TOOLS_BY_AGENT[name],
        model_settings=ModelSettings(temperature=TEMPERATURE.get(name, 0.3)),
        **_model_kwargs(),
    )
    for name, what in AGENTS.items()
}
root_agent = Agent(
    name="root_agent",
    handoff_description="The concierge itself: routes the traveler to the right agent.",
    instructions=_root_instructions,
    handoffs=list(SUB_AGENTS.values()),
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
)
# The recipe lets every sub-agent transfer to its peers and back to the root.
for name, agent in SUB_AGENTS.items():
    agent.handoffs = [root_agent, *[peer for peer_name, peer in SUB_AGENTS.items() if peer_name != name]]

ALL_AGENTS = {"root_agent": root_agent, **SUB_AGENTS}
_run_config = RunConfig(tool_not_found_behavior="return_error_to_model")


def starting_agent() -> Agent:
    """Whichever agent last had the conversation, as the recipe's transfers leave it; the root at first."""
    return ALL_AGENTS.get(memory.get(memory.ACTIVE_AGENT, ""), root_agent)


# What the chat panel's Graph tab shows.
GRAPH = {
    "nodes": [
        {"id": "__start__", "label": "__start__"},
        {"id": "root_agent", "label": "root_agent"},
        *[{"id": name, "label": name} for name in AGENTS],
        {"id": "__end__", "label": "__end__"},
    ],
    "edges": [
        {"source": "__start__", "target": "root_agent"},
        *[{"source": "root_agent", "target": name, "conditional": True} for name in AGENTS],
        {"source": "inspiration_agent", "target": "planning_agent", "label": "start planning"},
        {"source": "planning_agent", "target": "booking_agent", "label": "go ahead and book"},
        {"source": "pre_trip_agent", "target": "in_trip_agent", "label": "trip starts"},
        {"source": "in_trip_agent", "target": "post_trip_agent", "label": "trip ends"},
        *[{"source": name, "target": "__end__"} for name in AGENTS],
    ],
}

agent_app = AgentApp(
    name="Travel concierge",
    description="A cohort of agents for a traveler: inspiration, planning, booking, then pre-trip, "
    "in-trip and post-trip support, handing the conversation to one another (OpenAI Agents SDK; "
    "working memory on Hopsworks).",
    framework="openai_agents",
    welcome_message="Hello! I'm your travel concierge. Looking for inspiration, ready to plan a trip, "
    "or already booked and want help before, during or after it?",
    suggested_prompts=[
        "Inspire me about the Americas",
        "Find flights to London from JFK on April 20th for 4 days. Pick any flights and seats, any hotel and room, without my input; confirm with me before generating an itinerary.",
        "transfer to pre_trip",
    ],
    placeholder="Where would you like to go?",
    input_modalities=["text", "image"],
    memory=ManagedMemoryService(
        summarize=anthropic_summarizer(),
        long_term=True,
        max_state_value_chars=MAX_STATE_CHARS,
    ),
    tool_events=True,
    graph=GRAPH,
)


def _user_item(request) -> dict[str, Any]:
    """The turn's message as a Responses input item: its text, and any image the traveler sent."""
    parts: list[dict[str, Any]] = []
    if request.text:
        parts.append({"type": "input_text", "text": request.text})
    for image in request.images:
        parts.append({"type": "input_image", "image_url": f"data:{image.media_type};base64,{image.data}", "detail": "auto"})
    if not parts:
        raise AgentError("Say what you need, or send a photo of where you are.", code="invalid_request", status_code=400)
    if len(parts) == 1 and parts[0]["type"] == "input_text":
        return {"role": "user", "content": request.text}
    if not request.text:
        parts.insert(0, {"type": "input_text", "text": "What is this place? Tell me about it."})
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
    """One handler serves both endpoints: the active agent's text streams to the traveler, and
    each tool call and hand-off shows as a progress chip. Agent-tools' own model calls are inner
    runs and never stream."""
    memory.load_scenario()
    result = Runner.run_streamed(
        starting_agent(), _input_items(ctx, request), max_turns=MAX_TURNS, run_config=_run_config
    )
    streamed = False
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                streamed = True
                yield event.data.delta
        elif isinstance(event, RunItemStreamEvent):
            item = event.item
            raw = getattr(item, "raw_item", None)
            if event.name == "tool_called":
                await ctx.emit_event(getattr(raw, "name", "tool"), status="running",
                                     message=str(getattr(raw, "arguments", "") or "")[:200],
                                     event_id=getattr(raw, "call_id", None))
            elif event.name == "tool_output":
                await ctx.emit_event(getattr(getattr(item, "raw_item", None), "name", None) or "tool", status="done",
                                     event_id=(raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)))
            elif event.name == "handoff_occured":
                target = getattr(item, "target_agent", None)
                await ctx.emit_event("transfer", status="done",
                                     message=f"→ {getattr(target, 'name', 'agent')}")
    # the agent that ended the turn keeps the conversation, as the recipe's transfers leave it
    memory.put(memory.ACTIVE_AGENT, result.last_agent.name)
    if not streamed:
        final = str(result.final_output or "").strip()
        yield final or "Sorry — I wasn't able to put together a reply just then. Could you try rephrasing?"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
