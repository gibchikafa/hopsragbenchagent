"""
Cyber Guardian on the OpenAI Agents SDK: the recipe's own shape, an orchestrator whose
execution plan is its prompt, with the four analysts as tools.

`guardian_agent.py` turned the recipe's plan into a graph. This entry point
keeps it as the recipe had it: an orchestrator LLM told, at length, to
classify the alert, triage first, stop on a duplicate, run threat intel and
the investigation in the order the alert type calls for, get the playbook,
open the incident and report. The analysts are agents attached as tools, each
wrapping the store's lookups; the writes stay guarded under evaluation.
Actions that need approval are held in the conversation's working memory and
executed on the analyst's word, a turn later.

Deploy (git-backed, so `store.py` and `prompts.py` come along):
    agents.deploy_agent("cyber_guardian/guardian_agent_openai.py", name="cyberguardian-openai", ...)
"""

from __future__ import annotations

import json
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
from hopsworks_agents.protocol.autoevents import current_context
from openai.types.responses import ResponseTextDeltaEvent

from prompts import INVESTIGATION_PROMPT, PERSONA
from store import (
    create_incident,
    execute_response,
    get_playbook,
    investigation_query,
    threat_intel_query,
    triage_query,
)

MODEL_NAME = os.environ.get("GUARDIAN_OPENAI_MODEL", "")
PENDING_KEY = "pending_actions"


def _model_kwargs() -> dict[str, Any]:
    if "/" in MODEL_NAME:
        from agents.extensions.models.litellm_model import LitellmModel  # noqa: PLC0415

        return {"model": LitellmModel(model=MODEL_NAME)}
    return {"model": MODEL_NAME} if MODEL_NAME else {}


def _plain(tool: Any):
    return getattr(tool, "func", None) or getattr(tool, "coroutine", None) or tool


# ── the analysts, as the recipe had them: an agent around one lookup ───────

ANALYST_INSTR = {
    "triage_agent": """You are the Triage Agent: initial alert assessment. Call `triage_query` with the hostname and \
the alert type. Return JSON with is_duplicate (bool), existing_incident (if any), asset_context (owner, \
criticality, os, asset_type, ip_address, or "Unknown") and a one-sentence summary. Nothing else.""",
    "threat_intel_agent": """You are the Threat Intel Agent. Call `threat_intel_query` once with the whole list of \
indicators (IPs, domains, URLs, hashes; a list even for one). Return its JSON as it came: one entry per \
indicator with is_malicious, threat_name and confidence, Unknown where not found.""",
    "investigation_agent": PERSONA + "\n\n" + INVESTIGATION_PROMPT + """

You have one tool, `investigation_query`, which pulls the host's process events and network connections \
(for EDR_DETECTION pass the parent process named in the alert; for IOC_MATCH pass the indicator IP). Call \
it once, then return JSON with attack_timeline, confirmed_connections, responsible_processes, \
derived_iocs and summary.""",
    "response_agent": """You are the Response Agent. Given the triggering condition ("ThreatName = 'LockbitC2'" \
for a confirmed threat, else "AlertType = 'PHISHING_EMAIL'"), call `get_playbook` with it; if that \
returns no steps, call it again with the alert-type condition. Return JSON: recommended_actions, a list of \
{action, target, requires_approval} in step order, and the playbook_id.""",
}

TOOL_FOR = {
    "triage_agent": triage_query,
    "threat_intel_agent": threat_intel_query,
    "investigation_agent": investigation_query,
    "response_agent": get_playbook,
}
DESCRIPTIONS = {
    "triage_agent": "Check whether this host already has an open incident of this type (24h) and enrich it from the asset inventory. Input: 'hostname: X, alert_type: Y'.",
    "threat_intel_agent": "Look up indicators of compromise in the threat intelligence knowledge base. Input: the indicators, comma-separated.",
    "investigation_agent": "Investigate the host's logs: timeline, connections, responsible processes, derived indicators. Input: the alert type, the entities, and any threat intel so far.",
    "response_agent": "The response playbook for a confirmed threat or the alert type: the recommended actions and which need approval. Input: the triggering condition and the findings.",
}

ANALYSTS = {
    name: Agent(
        name=name,
        instructions=instr,
        tools=[function_tool(_plain(TOOL_FOR[name]))],
        model_settings=ModelSettings(temperature=0.0),
        **_model_kwargs(),
    )
    for name, instr in ANALYST_INSTR.items()
}
ANALYST_TOOLS = [agent.as_tool(tool_name=name, tool_description=DESCRIPTIONS[name]) for name, agent in ANALYSTS.items()]


# ── the orchestrator's own tools ────────────────────────────────────────────


def _session_get(key: str) -> Any:
    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        return None
    raw = ctx.memory.get_state("session", ctx.conversation_id, key)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _session_set(key: str, value: Any) -> None:
    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        return
    ctx.memory.set_state("session", ctx.conversation_id, key, json.dumps(value))


@function_tool
def hold_for_approval(actions_json: str) -> str:
    """Hold the recommended actions that require approval until the analyst approves them.

    Args:
        actions_json: A JSON list of {action, target} for the actions that need approval.
    """
    try:
        actions = json.loads(actions_json)
    except ValueError:
        return "Nothing held: actions_json must be a JSON list."
    actions = [a for a in actions if isinstance(a, dict) and a.get("action")] if isinstance(actions, list) else []
    _session_set(PENDING_KEY, actions)
    return json.dumps({"held": actions, "hitl_approval_required": bool(actions)})


@function_tool
def pending_approvals() -> str:
    """The actions held for the analyst's approval in this conversation, if any."""
    return json.dumps(_session_get(PENDING_KEY) or [])


@function_tool
def approve_pending(approved: bool) -> str:
    """Execute the held actions (simulated) when the analyst approves, or clear them when they decline.

    Args:
        approved: True to execute the held actions, False to drop them.
    """
    pending = _session_get(PENDING_KEY) or []
    if not pending:
        return json.dumps({"executed": [], "note": "nothing was waiting for approval"})
    _session_set(PENDING_KEY, [])
    if not approved:
        return json.dumps({"executed": [], "cleared": pending})
    run = _plain(execute_response)
    return json.dumps({"executed": [json.loads(run(a["action"], a.get("target", ""))) for a in pending]})


ORCHESTRATOR_INSTR = PERSONA + """

You are the central Orchestrator: you receive raw alert text, classify it and extract its entities, \
delegate all analysis to the analyst tools, and synthesize their findings into a report. You are a \
coordinator, not an analyst: never deduplicate, correlate logs or look up threats yourself; call the \
analysts.

Step 1, classify (your task). IOC_MATCH when the text says an indicator matched (entities: hostname, \
user, ip_address, the IOCs). EDR_DETECTION for an endpoint detection or a process tree (hostname, user, \
processes, parent process, file paths, command lines, any IOCs). PHISHING_EMAIL for an email report \
(sender, recipient, smtp_ip, url, hostname if named, the IOCs: domain, IP, URL). UNCATEGORIZED otherwise. \
If the message is not an alert but an approval or refusal of held actions, call `approve_pending` and \
report what ran; if it is a question or a greeting, say what you do and ask for raw alert text.

Step 2, triage, always first: `triage_agent`. If it reports a duplicate, report that with the existing \
incident id and stop.

Step 3, by classification, without pausing for the user between calls:
- IOC_MATCH or PHISHING_EMAIL: `threat_intel_agent` on the extracted IOCs first, then \
`investigation_agent` with the alert type, the entities and the threat intel.
- EDR_DETECTION or UNCATEGORIZED: `investigation_agent` first; then, if it derived indicators not yet \
checked, `threat_intel_agent` on them.
Then `response_agent` with the triggering condition: "ThreatName = '<name>'" for a threat the intel \
confirmed, else "AlertType = '<classification>'".

Step 4, act: call `execute_response` for each recommended action that does not require approval; \
call `hold_for_approval` with the ones that do, and say plainly that approval is required and ask for it.

Step 5, record: call `create_incident` with the alert type, the host, the user, a severity (Critical \
when a confirmed threat hits a Critical asset, High for a confirmed threat, else the alert's own) and a \
one-line summary.

Step 6, report, step by step: the classification and entities; what triage found; what threat \
intelligence said about each indicator, naming the threat; what the investigation established with the \
timeline; the response plan, saying per step whether it ran or waits for approval; the incident opened. \
Finish with a fenced JSON block, the incident log: alert_type, entities, triage, threat_intel, \
investigation, recommended_actions, incident_id, hitl_approval_required. Use the findings exactly as \
returned; add nothing.
"""

orchestrator = Agent(
    name="cyber_guardian_orchestrator",
    instructions=ORCHESTRATOR_INSTR,
    tools=[*ANALYST_TOOLS, function_tool(_plain(execute_response)), function_tool(_plain(create_incident)),
           hold_for_approval, pending_approvals, approve_pending],
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
)
_run_config = RunConfig(tool_not_found_behavior="return_error_to_model")

agent_app = AgentApp(
    eval_per_request=True,
    name="Cyber Guardian (OpenAI Agents)",
    description="Incident response for raw alerts: triage, threat intelligence, log investigation, the "
    "response playbook, and approval before anything disruptive (OpenAI Agents SDK orchestrator with "
    "analysts as tools, over Hopsworks feature-store lookups).",
    framework="openai_agents",
    welcome_message="Paste a raw alert — an EDR detection, an IOC match or a phishing report — and "
    "I'll triage it, check the indicators, investigate the logs and propose the playbook.",
    suggested_prompts=["What can you do?"],
    placeholder="Paste the raw alert text...",
    memory=ManagedMemoryService(summarize=openai_summarizer()),
    tool_events=True,
)


def _input_items(ctx, request) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    summary = ctx.system_context()
    if summary:
        items.append({"role": "user", "content": "(" + summary + ")"})
    for turn in ctx.history:
        content = str(turn.get("content") or "").strip()
        if content:
            items.append({"role": "user" if turn.get("role") == "user" else "assistant", "content": content})
    items.append({"role": "user", "content": request.text})
    return items


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints: the report streams to the analyst and every tool the
    orchestrator calls shows as a progress chip; the analysts' own runs are inner and never stream."""
    if not request.text:
        raise AgentError("Paste the raw alert text.", code="invalid_request", status_code=400)
    result = Runner.run_streamed(orchestrator, _input_items(ctx, request), max_turns=16, run_config=_run_config)
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
        yield final or "The workflow finished without a report; see the log."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
