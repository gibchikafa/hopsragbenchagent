"""
Cyber Guardian — Google's ADK cyber-guardian recipe as a LangGraph workflow on
the Hopsworks Agent Protocol.

    classify ─▶ triage ─┬─ duplicate ──────────────────────────────▶ report
                        ├─ IOC_MATCH / PHISHING ─▶ threat_intel ─▶ investigate ─▶ respond ─▶ record ─▶ report
                        └─ EDR / UNCATEGORIZED ──▶ investigate ─▶ threat_intel ─▶ respond ─▶ record ─▶ report
    approval ─▶ approve ─▶ report          (the analyst's word on the actions held back)
    other    ─▶ converse

The recipe is an orchestrator LLM with four sub-agent LLMs, each wrapping one
BigQuery tool and told, at length, to hand control back. Its orchestrator
prompt is an execution plan with conditional routing: triage first, stop on a
duplicate, threat intel before investigation for IOC-heavy alerts and after it
for EDR alerts, then the playbook, then a flag for human approval. Here that
plan is the graph. The routing is edges, not prompt text; the tools are called
by the nodes that need them; the model is asked three things: what kind of
alert this is and what is in it, what the logs say happened, and how to tell
the analyst. The chat panel's Graph tab shows the workflow as it runs.

Deploy (git-backed, so `store.py` and `prompts.py` come along):
    agents = hopsworks.login().get_agent_serving()
    agents.deploy_agent("cyber_guardian/guardian_agent.py", name="cyberguardian",
                        git_url="https://github.com/<you>/hopsragbenchagent.git",
                        git_provider="GitHub", git_branch="main",
                        requirements="cyber_guardian/requirements.txt").start()
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
)
from hopsworks_agents.protocol.autoevents import current_context
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from prompts import CLASSIFY_PROMPT, CONVERSE_PROMPT, INVESTIGATION_PROMPT, PERSONA, REPORT_PROMPT
from store import (
    create_incident,
    execute_response,
    get_playbook,
    investigation_query,
    threat_intel_query,
    triage_query,
)

MODEL = os.environ.get("GUARDIAN_MODEL", "claude-sonnet-4-5")
#: model calls whose tokens are findings for the graph, not words for the analyst
ANALYST_TAG = "analyst"
#: actions held back for approval, in session memory under this key
PENDING_KEY = "pending_actions"
IOC_FIRST = ("IOC_MATCH", "PHISHING_EMAIL")

_llm = ChatAnthropic(model=MODEL, max_tokens=4096, temperature=0.0)
_writer = ChatAnthropic(model=MODEL, max_tokens=4096, temperature=0.0)


# ── what the model is asked for ──────────────────────────────────────────────


class Classification(BaseModel):
    kind: Literal["IOC_MATCH", "EDR_DETECTION", "PHISHING_EMAIL", "UNCATEGORIZED", "APPROVAL", "OTHER"]
    hostname: str | None = None
    user: str | None = None
    ip_address: str | None = None
    iocs: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)
    parent_process: str | None = None
    command_lines: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    sender: str | None = None
    recipient: str | None = None
    smtp_ip: str | None = None
    url: str | None = None
    severity_hint: str | None = None
    approved: bool | None = Field(default=None, description="For APPROVAL: True if approving, False if declining")


class InvestigationReport(BaseModel):
    attack_timeline: list[dict[str, Any]] = Field(default_factory=list)
    confirmed_connections: list[dict[str, Any]] = Field(default_factory=list)
    responsible_processes: list[dict[str, Any]] = Field(default_factory=list)
    derived_iocs: list[str] = Field(default_factory=list)
    summary: str = ""


class State(TypedDict, total=False):
    alert_text: str
    context: str
    kind: str
    entities: dict[str, Any]
    approved: bool | None
    triage: dict[str, Any]
    threat_intel: list[dict[str, Any]]
    investigation: dict[str, Any]
    response: dict[str, Any]
    incident: dict[str, Any]
    hitl_approval_required: bool
    executed: list[dict[str, Any]]
    reply: str


# ── session memory: what is waiting for the analyst ─────────────────────────


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


# ── the nodes ────────────────────────────────────────────────────────────────


async def classify(state: State, config: RunnableConfig) -> State:
    pending = _session_get(PENDING_KEY) or []
    prompt = CLASSIFY_PROMPT
    if pending:
        prompt += (
            "\n\nActions proposed earlier in this conversation and waiting for approval: "
            + json.dumps(pending)
        )
    parsed: Classification = await _llm.with_structured_output(Classification).ainvoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["alert_text"])],
        config={**config, "tags": [ANALYST_TAG]},
    )
    entities = parsed.model_dump(exclude={"kind", "approved"}, exclude_none=True)
    entities = {k: v for k, v in entities.items() if v not in ([], "")}
    return {"kind": parsed.kind, "entities": entities, "approved": parsed.approved}


def route_after_classify(state: State) -> str:
    kind = state.get("kind")
    if kind == "APPROVAL":
        return "approve"
    if kind == "OTHER":
        return "converse"
    return "triage"


async def triage(state: State, config: RunnableConfig) -> State:
    entities = state["entities"]
    result = json.loads(
        await triage_query.ainvoke(
            {"hostname": entities.get("hostname") or "", "alert_type": state["kind"]}, config=config
        )
    )
    return {"triage": result}


def route_after_triage(state: State) -> str:
    if state["triage"].get("is_duplicate"):
        return "report"
    return "threat_intel" if state["kind"] in IOC_FIRST else "investigate"


def _indicators(state: State) -> list[str]:
    """What to look up: the alert's own IOCs, plus what the investigation derived, once each."""
    seen: list[str] = []
    for value in [*state["entities"].get("iocs", []), *state.get("investigation", {}).get("derived_iocs", [])]:
        value = str(value).strip()
        if value and value not in seen:
            seen.append(value)
    already = {r["ioc"] for r in state.get("threat_intel", [])}
    return [v for v in seen if v not in already]


async def threat_intel(state: State, config: RunnableConfig) -> State:
    indicators = _indicators(state)
    found = json.loads(await threat_intel_query.ainvoke({"indicators": indicators}, config=config)) if indicators else []
    return {"threat_intel": [*state.get("threat_intel", []), *found]}


def route_after_threat_intel(state: State) -> str:
    return "investigate" if state["kind"] in IOC_FIRST else "respond"


async def investigate(state: State, config: RunnableConfig) -> State:
    entities = state["entities"]
    ip_iocs = [i for i in entities.get("iocs", []) if i.replace(".", "").isdigit()]
    logs = await investigation_query.ainvoke(
        {
            "alert_type": state["kind"],
            "hostname": entities.get("hostname") or "",
            "parent_process": entities.get("parent_process") or "",
            "destination_ip": ip_iocs[0] if ip_iocs else "",
        },
        config=config,
    )
    brief = (
        f"alert_type: {state['kind']}\n"
        f"entities: {json.dumps(entities)}\n"
        f"threat_intel_report: {json.dumps(state.get('threat_intel', []))}\n\n"
        f"logs: {logs}"
    )
    report: InvestigationReport = await _llm.with_structured_output(InvestigationReport).ainvoke(
        [SystemMessage(content=INVESTIGATION_PROMPT), HumanMessage(content=brief)],
        config={**config, "tags": [ANALYST_TAG]},
    )
    return {"investigation": report.model_dump()}


def route_after_investigate(state: State) -> str:
    if state["kind"] in IOC_FIRST:
        return "respond"
    return "threat_intel" if _indicators(state) else "respond"


async def respond(state: State, config: RunnableConfig) -> State:
    """The playbook for the confirmed threat, else for the alert type. Steps that need no approval
    run now (simulated); the rest wait for the analyst."""
    conditions = [
        f"ThreatName = '{r['threat_name']}'"
        for r in state.get("threat_intel", [])
        if r.get("is_malicious") and r.get("threat_name") not in (None, "Unknown")
    ]
    conditions.append(f"AlertType = '{state['kind']}'")
    playbook: dict[str, Any] = {"playbook_id": None, "steps": []}
    condition = conditions[-1]
    for candidate in dict.fromkeys(conditions):
        found = json.loads(await get_playbook.ainvoke({"triggering_condition": candidate}, config=config))
        if found.get("steps"):
            playbook, condition = found, candidate
            break
    actions = [
        {"action": s["action"], "target": s["target"], "requires_approval": bool(s.get("requires_approval"))}
        for s in playbook.get("steps", [])
    ]
    executed = []
    for action in actions:
        if not action["requires_approval"]:
            executed.append(
                json.loads(await execute_response.ainvoke({"action": action["action"], "target": action["target"]}, config=config))
            )
    pending = [a for a in actions if a["requires_approval"]]
    _session_set(PENDING_KEY, pending)
    return {
        "response": {"triggering_condition": condition, "playbook_id": playbook.get("playbook_id"), "recommended_actions": actions},
        "executed": executed,
        "hitl_approval_required": bool(pending),
    }


def _severity(state: State) -> str:
    malicious = [r for r in state.get("threat_intel", []) if r.get("is_malicious")]
    criticality = str((state.get("triage", {}).get("asset_context") or {}).get("criticality") or "") if isinstance(
        state.get("triage", {}).get("asset_context"), dict
    ) else ""
    if malicious and criticality.lower() == "critical":
        return "Critical"
    if malicious:
        return "High"
    return str(state["entities"].get("severity_hint") or "Medium").title()


async def record(state: State, config: RunnableConfig) -> State:
    entities = state["entities"]
    result = json.loads(
        await create_incident.ainvoke(
            {
                "alert_type": state["kind"],
                "hostname": entities.get("hostname") or "",
                "user": entities.get("user") or entities.get("recipient") or "",
                "severity": _severity(state),
                "summary": state.get("investigation", {}).get("summary", "")[:500],
            },
            config=config,
        )
    )
    return {"incident": result}


async def approve(state: State, config: RunnableConfig) -> State:
    pending = _session_get(PENDING_KEY) or []
    if not pending:
        return {"executed": [], "reply": "There is nothing waiting for approval in this conversation. Paste an alert to start."}
    if state.get("approved") is False:
        _session_set(PENDING_KEY, [])
        return {"executed": [], "reply": "Understood: the held-back actions were not executed and are cleared. " + json.dumps(pending)}
    executed = [
        json.loads(await execute_response.ainvoke({"action": a["action"], "target": a["target"]}, config=config))
        for a in pending
    ]
    _session_set(PENDING_KEY, [])
    return {"executed": executed, "hitl_approval_required": False}


def _findings(state: State) -> str:
    return json.dumps(
        {
            "alert_type": state.get("kind"),
            "entities": state.get("entities"),
            "triage": state.get("triage"),
            "threat_intel": state.get("threat_intel", []),
            "investigation": state.get("investigation"),
            "response": state.get("response"),
            "executed_now": state.get("executed", []),
            "incident": state.get("incident"),
            "hitl_approval_required": state.get("hitl_approval_required", False),
        },
        default=str,
    )


async def report(state: State, config: RunnableConfig) -> State:
    if state.get("reply"):
        return {}
    if state.get("kind") == "APPROVAL":
        brief = "The analyst approved the held-back actions; these were executed (simulated): " + json.dumps(state.get("executed", []))
        instruction = "Confirm to the analyst exactly which actions ran and against what, and that nothing else was done."
    else:
        brief = "Workflow findings: " + _findings(state)
        instruction = REPORT_PROMPT
    text = await _writer.ainvoke(
        [SystemMessage(content=PERSONA + "\n\n" + instruction + (state.get("context") or "")), HumanMessage(content=brief)],
        config=config,
    )
    return {"reply": text.content if isinstance(text.content, str) else "".join(b.get("text", "") for b in text.content if isinstance(b, dict))}


async def converse(state: State, config: RunnableConfig) -> State:
    text = await _writer.ainvoke(
        [SystemMessage(content=PERSONA + "\n\n" + CONVERSE_PROMPT + (state.get("context") or "")), HumanMessage(content=state["alert_text"])],
        config=config,
    )
    return {"reply": text.content if isinstance(text.content, str) else "".join(b.get("text", "") for b in text.content if isinstance(b, dict))}


_builder = StateGraph(State)
for name, node in (
    ("classify", classify), ("triage", triage), ("threat_intel", threat_intel), ("investigate", investigate),
    ("respond", respond), ("record", record), ("approve", approve), ("report", report), ("converse", converse),
):
    _builder.add_node(name, node)
_builder.add_edge(START, "classify")
_builder.add_conditional_edges("classify", route_after_classify, ["triage", "approve", "converse"])
_builder.add_conditional_edges("triage", route_after_triage, ["report", "threat_intel", "investigate"])
_builder.add_conditional_edges("threat_intel", route_after_threat_intel, ["investigate", "respond"])
_builder.add_conditional_edges("investigate", route_after_investigate, ["threat_intel", "respond"])
_builder.add_edge("respond", "record")
_builder.add_edge("record", "report")
_builder.add_edge("approve", "report")
_builder.add_edge("report", END)
_builder.add_edge("converse", END)
workflow = _builder.compile()


agent_app = AgentApp(
    # create_incident and execute_response check in_evaluation() (store.py), so a
    # sandboxed suite can run against the deployment that handles real alerts
    eval_per_request=True,
    name="Cyber Guardian",
    description="Incident response for raw alerts: triage against the asset inventory and incident "
    "ledger, threat intelligence, log investigation, the response playbook, and approval before "
    "anything disruptive (LangGraph workflow over Hopsworks feature-store lookups).",
    framework="langgraph",
    welcome_message="Paste a raw alert — an EDR detection, an IOC match or a phishing report — and "
    "I'll triage it, check the indicators, investigate the logs and propose the playbook.",
    suggested_prompts=[
        "What can you do?",
        "SCH_SR_Crowdstrike_IOC_Match/Host:winws1045\nUser m.chen on Host winws1045 (WINDOWS 11 "
        "Workstation, IP 10.12.0.101) alerted: an IP address matched a Custom Intelligence Indicator "
        "(LockbitC2), critical severity. IOC: 198.51.100.42. Process: teams.exe PID 7100, parent explorer.exe.",
    ],
    placeholder="Paste the raw alert text...",
    memory=ManagedMemoryService(summarize=anthropic_summarizer()),
    tool_events=True,
    graph=workflow,
)


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints: the report streams to the analyst and every
    tool the workflow calls shows as a progress chip."""
    if not request.text:
        raise AgentError("Paste the raw alert text.", code="invalid_request", status_code=400)
    context = ctx.system_context()
    final: dict = {}

    async def _analyst_calls_stay_inside(events):
        async for event in events:
            if event.get("event") == "on_chat_model_stream" and ANALYST_TAG in (event.get("tags") or []):
                continue
            if event.get("event") == "on_chain_end" and event.get("name") in ("report", "converse", "approve"):
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict) and output.get("reply"):
                    final["reply"] = output["reply"]
            yield event

    streamed = False
    async for delta in ctx.stream_langchain(
        _analyst_calls_stay_inside(
            workflow.astream_events(
                {"alert_text": request.text, "context": ("\n\n" + context) if context else ""}, version="v2"
            )
        )
    ):
        streamed = True
        yield delta
    if not streamed:
        # the approval and duplicate paths answer from state, not from a streamed model call
        yield final.get("reply") or "The workflow finished without a report; see the log."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
