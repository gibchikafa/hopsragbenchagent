"""The demo's wiring, with the SDK and the model stubbed.

What is worth testing here is not the model's behaviour -- the evaluation
suites hold that -- but that the agent is actually wired to the memory
service: the four memory tools plus `identify` are registered, the memory is
configured with all three tiers, and `memory_report` reads the tiers rather
than describing them.
"""

from __future__ import annotations

import json
import sys
import types
from unittest import mock

import pytest

MEMORY_TOOL_NAMES = ("remember", "recall", "forget", "search")

STUBBED = (
    "hopsworks_agents",
    "hopsworks_agents.protocol",
    "hopsworks_agents.protocol.autoevents",
    "agents",
    "agents.stream_events",
    "agents.extensions",
    "agents.extensions.models",
    "agents.extensions.models.litellm_model",
    "openai",
    "openai.types",
    "openai.types.responses",
)


class _FakeAgent:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.tools = kw.get("tools", [])


def _tool(fn):
    return types.SimpleNamespace(name=fn.__name__, description=fn.__doc__, fn=fn)


@pytest.fixture
def stubbed(monkeypatch):
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    agents = sys.modules["agents"]
    agents.Agent = _FakeAgent
    agents.function_tool = _tool
    agents.ModelSettings = lambda **kw: kw
    protocol = sys.modules["hopsworks_agents.protocol"]
    # the SDK's own tools, as the framework wrapper would hand them over
    protocol.memory_tools = lambda framework: [
        types.SimpleNamespace(name=name) for name in MEMORY_TOOL_NAMES
    ]
    protocol.identity_tools = lambda framework: [
        types.SimpleNamespace(name="identify")
    ]
    protocol.ManagedMemoryService = lambda **kw: types.SimpleNamespace(**kw)
    protocol.AgentApp = lambda **kw: types.SimpleNamespace(
        **kw, stream=lambda fn: fn
    )
    monkeypatch.syspath_prepend(__file__.rsplit("/", 1)[0])
    for name in ("memory_agent", "prompts"):
        sys.modules.pop(name, None)
    yield


def test_every_memory_tool_is_registered(stubbed):
    import memory_agent  # noqa: PLC0415

    names = [tool.name for tool in memory_agent.agent.tools]
    # the SDK's four, then identify, then the demo's own reader
    assert names == [*MEMORY_TOOL_NAMES, "identify", "memory_report"]


def test_all_three_tiers_are_configured(stubbed):
    import memory_agent  # noqa: PLC0415

    memory = memory_agent.agent_app.memory
    # tier 2 and tier 3; tier 1 is the buffer and needs nothing
    assert memory.summarize is not None
    assert memory.long_term is True
    # folded early on purpose, so the summary tier shows up inside a demo chat
    assert memory.summarize_after_messages < 20
    assert memory.keep_recent_messages < memory.summarize_after_messages
    # vector search is off by default: no embedding model, no feature group
    assert "embedder" not in memory.__dict__


def test_vector_search_adds_an_embedder_when_asked(stubbed, monkeypatch):
    monkeypatch.setenv("MEMORY_DEMO_VECTOR_SEARCH", "true")
    protocol = sys.modules["hopsworks_agents.protocol"]
    embedder = object()
    protocol.sentence_transformer_embedder = lambda: embedder
    protocol.vector_store_for = lambda e: ("store-for", e)
    import memory_agent  # noqa: PLC0415

    memory = memory_agent.agent_app.memory
    assert memory.embedder is embedder
    assert memory.vector_store == ("store-for", embedder)


def test_memory_report_reads_the_tiers(stubbed):
    import memory_agent  # noqa: PLC0415

    ctx = mock.MagicMock(
        subject="dana@example.com",
        subject_source="client",
        history=[{"role": "user", "content": "I live in Oslo"}],
        summary="Dana introduced herself.",
    )
    ctx.state.side_effect = lambda scope: (
        {"home_city": "Oslo"} if scope == "user" else {"comparing": "laptops"}
    )
    memory_agent.current_context.get = lambda default=None: ctx

    report = json.loads(memory_agent.memory_report.fn())
    assert report["who_you_are_talking_to"]["subject"] == "dana@example.com"
    assert report["tier_1_conversation_buffer"]["turns_held"] == 1
    assert report["tier_2_rolling_summary"] == "Dana introduced herself."
    assert report["tier_3_durable_memory"] == {
        "about_this_person": {"home_city": "Oslo"},
        "this_conversation_only": {"comparing": "laptops"},
    }
    assert report["search_mode"] == "keyword"


def test_memory_report_says_so_when_there_is_no_memory(stubbed):
    import memory_agent  # noqa: PLC0415

    memory_agent.current_context.get = lambda default=None: None
    assert "No memory is configured" in memory_agent.memory_report.fn()


def test_nothing_folded_yet_is_said_not_shown_as_null(stubbed):
    import memory_agent  # noqa: PLC0415

    ctx = mock.MagicMock(
        subject="conv-1", subject_source="conversation", history=[], summary=None
    )
    ctx.state.side_effect = lambda scope: {}
    memory_agent.current_context.get = lambda default=None: ctx

    report = json.loads(memory_agent.memory_report.fn())
    assert report["tier_2_rolling_summary"] == "nothing folded yet"
    # nobody has identified themselves, so durable facts stay with the chat
    assert report["who_you_are_talking_to"]["known_from"] == "conversation"
    assert report["tier_1_conversation_buffer"]["oldest_turn_held"] is None


def test_the_prompt_forbids_inventing_and_claiming(stubbed):
    import prompts  # noqa: PLC0415

    assert "Never invent a memory" in prompts.SYSTEM_PROMPT
    assert "Never claim to have stored something you did not" in prompts.SYSTEM_PROMPT
    # the session-scope rule is what the evaluation suite holds it to
    assert "scope='session'" in prompts.SYSTEM_PROMPT
