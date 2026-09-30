"""That the advisor module loads and the coordinator has its four analysts.

The dependencies are stubbed, so this checks the wiring and the working-memory
plumbing rather than any model: no cluster, no key, no network.

    python -m pytest financial_advisor/test_advisor_wiring.py
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

ANALYSTS = ("analyze_market", "propose_strategies", "plan_execution", "assess_risk")

STUBBED = (
    "hopsworks_agents", "hopsworks_agents.protocol", "hopsworks_agents.protocol.autoevents",
    "langchain_anthropic", "langchain_core", "langchain_core.messages",
    "langchain_core.tools", "langgraph", "langgraph.prebuilt",
)


@pytest.fixture
def stubbed(monkeypatch):
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    # `tool` returns the function, so the analysts stay plain coroutines here
    sys.modules["langchain_core.tools"].tool = lambda fn: fn
    sys.modules["langgraph.prebuilt"].create_react_agent = lambda llm, tools: {"tools": tools}
    monkeypatch.syspath_prepend(__file__.rsplit("/", 1)[0])
    for name in ("advisor_agent", "prompts"):
        sys.modules.pop(name, None)
    yield


def test_the_agent_imports_with_its_four_analysts(stubbed):
    import advisor_agent  # noqa: PLC0415

    assert advisor_agent.agent_app is not None
    assert [t.__name__ for t in advisor_agent.coordinator["tools"]] == list(ANALYSTS)
    for name in ANALYSTS:
        assert getattr(advisor_agent, name).__doc__, f"{name} has no description for the model"


def test_every_prompt_that_advises_carries_the_disclaimer():
    import prompts  # noqa: PLC0415

    for text in (
        prompts.COORDINATOR_PROMPT,
        prompts.TRADING_ANALYST_PROMPT,
        prompts.EXECUTION_ANALYST_PROMPT,
        prompts.RISK_ANALYST_PROMPT,
    ):
        assert prompts.DISCLAIMER in text
    assert "Google" not in prompts.DISCLAIMER


@pytest.mark.asyncio
async def test_later_analysts_refuse_without_the_earlier_report(stubbed):
    import advisor_agent  # noqa: PLC0415
    import advisor_memory  # noqa: PLC0415

    # no request context: nothing in working memory
    advisor_memory.current_context.get = lambda default=None: None
    assert "missing" in await advisor_agent.propose_strategies("moderate", "long-term")
    assert "missing" in await advisor_agent.plan_execution("", "")
    assert "missing" in await advisor_agent.assess_risk()


@pytest.mark.asyncio
async def test_reports_are_kept_in_this_conversations_working_memory(stubbed):
    import advisor_agent  # noqa: PLC0415
    import advisor_memory  # noqa: PLC0415

    store: dict[tuple, str] = {}
    memory = mock.MagicMock()
    memory.get_state = lambda scope, owner, key: store.get((scope, owner, key))
    memory.set_state = lambda scope, owner, key, value, **kw: store.__setitem__((scope, owner, key), value)
    ctx = mock.MagicMock(memory=memory, conversation_id="conv-1", turn_id="t1")
    advisor_memory.current_context.get = lambda default=None: ctx

    async def fake_analyst(system, brief, llm=None):
        return f"REPORT for: {brief.splitlines()[0]}"

    advisor_agent._analyst = fake_analyst
    report = await advisor_agent.analyze_market("nvda")
    assert report == "REPORT for: provided_ticker: NVDA"
    assert store[("session", "conv-1", advisor_agent.MARKET_ANALYSIS)] == report
    assert store[("session", "conv-1", advisor_agent.TICKER)] == "NVDA"

    strategies = await advisor_agent.propose_strategies("conservative", "long-term")
    assert strategies.startswith("REPORT")
    assert store[("session", "conv-1", advisor_agent.RISK_ATTITUDE)] == "conservative"
    # the next step now has what it needs, and the progress block says so
    assert "market analysis, trading strategies" in advisor_agent._progress_block()


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


def test_the_openai_variant_has_the_four_analysts_on_the_coordinator(openai_stubbed):
    for name in ("advisor_memory", "advisor_agent_openai"):
        sys.modules.pop(name, None)
    import advisor_agent_openai as o  # noqa: PLC0415

    assert o.agent_app is not None
    assert [t.name for t in o.coordinator.tools] == list(ANALYSTS)
    assert o.data_analyst.tools == ["web_search"] and o.trading_analyst.tools == []
    # the same working memory as the LangGraph entry point, so either can resume the other's plan
    import advisor_memory  # noqa: PLC0415

    assert o.mem is advisor_memory
