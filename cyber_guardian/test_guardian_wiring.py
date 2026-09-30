"""That the workflow routes the way the recipe's plan says, without a model or a cluster.

    python -m pytest cyber_guardian/test_guardian_wiring.py
"""

from __future__ import annotations

import json
import sys
import types
from unittest import mock

import pytest

STUBBED = (
    "hopsworks_agents", "hopsworks_agents.protocol", "hopsworks_agents.protocol.autoevents",
    "hopsworks_agents.protocol.evaluation", "hopsworks", "pandas",
    "langchain_anthropic", "langchain_core", "langchain_core.messages", "langchain_core.runnables",
    "langchain_core.tools", "langgraph", "langgraph.graph",
)


@pytest.fixture
def stubbed(monkeypatch):
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["langchain_core.tools"].tool = lambda fn: fn
    sys.modules["hopsworks"].login = mock.MagicMock()
    sys.modules["hopsworks_agents.protocol.evaluation"].in_evaluation = lambda: False
    pandas = sys.modules["pandas"]
    pandas.isna = lambda v: v is None
    pandas.DataFrame = lambda rows: rows
    pandas.Timestamp = __import__("datetime").datetime.fromisoformat
    graph = sys.modules["langgraph.graph"]
    graph.START, graph.END = "__start__", "__end__"
    graph.StateGraph = lambda state: mock.MagicMock()
    monkeypatch.syspath_prepend(__file__.rsplit("/", 1)[0])
    for name in ("store", "prompts", "guardian_agent"):
        sys.modules.pop(name, None)
    yield


def test_the_agent_imports_and_wires_every_node(stubbed):
    import guardian_agent as g  # noqa: PLC0415

    assert g.agent_app is not None
    added = [call.args[0] for call in g._builder.add_node.call_args_list]
    assert added == ["classify", "triage", "threat_intel", "investigate", "respond", "record", "approve", "report", "converse"]


def test_routing_follows_the_recipes_plan(stubbed):
    import guardian_agent as g  # noqa: PLC0415

    assert g.route_after_classify({"kind": "IOC_MATCH"}) == "triage"
    assert g.route_after_classify({"kind": "APPROVAL"}) == "approve"
    assert g.route_after_classify({"kind": "OTHER"}) == "converse"
    # a duplicate stops everything
    assert g.route_after_triage({"kind": "IOC_MATCH", "triage": {"is_duplicate": True}}) == "report"
    # IOC-heavy: intel first, then investigate, then respond
    assert g.route_after_triage({"kind": "PHISHING_EMAIL", "triage": {}}) == "threat_intel"
    assert g.route_after_threat_intel({"kind": "IOC_MATCH"}) == "investigate"
    assert g.route_after_investigate({"kind": "IOC_MATCH", "entities": {}}) == "respond"
    # EDR: investigate first, then intel on what it derived, then respond
    assert g.route_after_triage({"kind": "EDR_DETECTION", "triage": {}}) == "investigate"
    state = {"kind": "EDR_DETECTION", "entities": {"iocs": []}, "investigation": {"derived_iocs": ["45.146.164.110"]}}
    assert g.route_after_investigate(state) == "threat_intel"
    assert g.route_after_threat_intel({"kind": "EDR_DETECTION"}) == "respond"
    # nothing derived and nothing to check: straight to respond
    assert g.route_after_investigate({"kind": "UNCATEGORIZED", "entities": {}, "investigation": {}}) == "respond"
    # an indicator already looked up is not looked up again
    state["threat_intel"] = [{"ioc": "45.146.164.110"}]
    assert g.route_after_investigate(state) == "respond"


def test_severity_follows_intel_and_criticality(stubbed):
    import guardian_agent as g  # noqa: PLC0415

    bad = [{"is_malicious": True, "threat_name": "LockbitC2"}]
    assert g._severity({"threat_intel": bad, "triage": {"asset_context": {"criticality": "Critical"}}, "entities": {}}) == "Critical"
    assert g._severity({"threat_intel": bad, "triage": {"asset_context": {"criticality": "Medium"}}, "entities": {}}) == "High"
    assert g._severity({"threat_intel": [], "triage": {"asset_context": "No asset context found."}, "entities": {"severity_hint": "low"}}) == "Low"


def test_triage_finds_a_duplicate_opened_in_this_conversation(stubbed):
    import store  # noqa: PLC0415

    state: dict = {}
    memory = mock.MagicMock()
    memory.get_state = lambda scope, owner, key: state.get(key)
    memory.set_state = lambda scope, owner, key, value: state.__setitem__(key, value)
    store.current_context = types.SimpleNamespace(get=lambda d=None: types.SimpleNamespace(memory=memory, conversation_id="c1"))
    store._lookup = lambda view, entry: {"owner": "Team-B", "business_criticality": "Medium", "os": "x", "asset_type": "Server", "ip_address": "10.20.3.16"} if view == store.ASSETS_FG else None
    first = json.loads(store.triage_query("winsrv0221", "EDR_DETECTION"))
    assert first == {"is_duplicate": False, "asset_context": {"owner": "Team-B", "criticality": "Medium", "os": "x", "asset_type": "Server", "ip_address": "10.20.3.16"}}
    store.in_evaluation = lambda: True
    opened = json.loads(store.create_incident("EDR_DETECTION", "winsrv0221", "system", "High", "x"))
    assert opened["status"] == "success" and opened["recorded"] is False
    second = json.loads(store.triage_query("winsrv0221", "EDR_DETECTION"))
    assert second == {"is_duplicate": True, "existing_incident": opened["incident_id"]}


def test_investigation_filters_the_logs_the_way_the_sql_did(stubbed):
    import store  # noqa: PLC0415

    events = [{"parent_process_name": "services.exe", "process_name": "lsass.exe"}, {"parent_process_name": "explorer.exe", "process_name": "chrome.exe"}]
    conns = [{"destination_ip": "198.51.100.42", "destination_port": 443}, {"destination_ip": "8.8.8.8", "destination_port": 53}]
    store._lookup = lambda view, entry: {"rows": json.dumps(events if view == store.PROCESS_EVENTS_FG else conns)}
    edr = json.loads(store.investigation_query("EDR_DETECTION", "h", parent_process="services.exe"))
    assert [e["process_name"] for e in edr["process_events"]] == ["lsass.exe"] and len(edr["network_connections"]) == 2
    ioc = json.loads(store.investigation_query("IOC_MATCH", "h", destination_ip="198.51.100.42"))
    assert [c["destination_ip"] for c in ioc["network_connections"]] == ["198.51.100.42"] and len(ioc["process_events"]) == 2


def test_unknown_indicators_come_back_as_unknown(stubbed):
    import store  # noqa: PLC0415

    store._lookup_many = lambda view, entries: [{"ioc_value": "198.51.100.42", "ioc_type": "IP", "is_malicious": True, "threat_name": "Cobalt Strike C2", "confidence": "High", "last_seen": "2025-07-20"}]
    report = json.loads(store.threat_intel_query(["198.51.100.42", "203.0.113.77", " 198.51.100.42 "]))
    assert [r["ioc"] for r in report] == ["198.51.100.42", "203.0.113.77"]
    assert report[0]["threat_name"] == "Cobalt Strike C2" and report[1] == {"ioc": "203.0.113.77", "is_malicious": False, "threat_name": "Unknown", "confidence": "Unknown", "note": "not in the knowledge base"}
    assert json.loads(store.threat_intel_query([])) == []


# ── the OpenAI Agents SDK entry point ────────────────────────────────────────

OPENAI_STUBBED = (
    "agents", "agents.stream_events", "agents.extensions", "agents.extensions.models",
    "agents.extensions.models.litellm_model", "openai", "openai.types", "openai.types.responses",
)


class _FakeAgent:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.tools = kw.get("tools", [])
        self.handoffs = kw.get("handoffs", [])

    def as_tool(self, tool_name, tool_description, **kw):
        return types.SimpleNamespace(name=tool_name, description=tool_description, agent=self)


@pytest.fixture
def openai_stubbed(stubbed, monkeypatch):
    for name in OPENAI_STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    agents = sys.modules["agents"]
    agents.Agent = _FakeAgent
    agents.function_tool = lambda fn: types.SimpleNamespace(name=fn.__name__, description=fn.__doc__, fn=fn)
    agents.ModelSettings = lambda **kw: kw
    agents.RunConfig = lambda **kw: kw
    agents.WebSearchTool = lambda: "web_search"
    yield


def test_the_openai_variant_is_the_recipes_orchestrator_with_analysts_as_tools(openai_stubbed):
    sys.modules.pop("guardian_agent_openai", None)
    import guardian_agent_openai as o  # noqa: PLC0415

    assert o.agent_app is not None
    names = [t.name for t in o.orchestrator.tools]
    assert names[:4] == ["triage_agent", "threat_intel_agent", "investigation_agent", "response_agent"]
    assert names[4:] == ["execute_response", "create_incident", "hold_for_approval", "pending_approvals", "approve_pending"]
    # each analyst wraps exactly its store lookup
    assert [t.name for t in o.ANALYSTS["triage_agent"].tools] == ["triage_query"]
    assert [t.name for t in o.ANALYSTS["response_agent"].tools] == ["get_playbook"]
    assert "Step 2, triage, always first" in o.ORCHESTRATOR_INSTR
