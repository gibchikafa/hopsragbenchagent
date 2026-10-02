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


# ── the job that deploys the agent and talks to it ───────────────────────────


class FakeReply:
    def __init__(self, text, conversation_id):
        self.text, self.conversation_id = text, conversation_id


class FakeAgent:
    """An agent that answers from a script and records what it was asked."""

    name = "memorydemo"
    url = "https://gateway/v1/ns/memorydemo"

    def __init__(self, answers):
        self.answers = list(answers)
        self.asked: list[tuple[str, str | None, str | None]] = []
        self.started = self.stopped = 0
        self.running = False
        self.deployment = mock.MagicMock()
        self.deployment.predictor.env_vars = {}

    def chat(self, text, conversation_id=None, subject=None, timeout=None):
        self.asked.append((text, conversation_id, subject))
        answer = self.answers.pop(0) if self.answers else "ok"
        return FakeReply(answer, conversation_id or f"conv-{len(self.asked)}")

    def is_running(self):
        return self.running

    def start(self, await_running=None):
        self.started += 1
        self.running = True

    def stop(self):
        self.stopped += 1

    def _agent_get(self, path):
        return {
            "messages": [{"role": "user", "content": "hi"}],
            "summary": "Dana lives in Oslo and is vegetarian.",
            "summarized_through": 4,
            "subject": "dana@example.com",
        }


# the canned answers below are written for this persona, so the checks line up
PINNED = ["demo_job.py", "--persona", "Dana", "--city", "Oslo",
          "--subject", "dana@example.com"]

ANSWERS = [
    "Noted: Oslo, vegetarian.",          # the facts
    "Holding that for this chat.",       # session note
    "The 14-inch and the 16-inch.",      # from the buffer
    "Try a lentil stew.",
    "It keeps well for lunch.",
    "Try fårikål, without the lamb.",
    "You live in Oslo and you're vegetarian.",   # across conversations
    "I don't have any laptop sizes for you.",    # session note did not cross
    "Forgotten.",
    "I don't know where you live.",
]


@pytest.fixture
def job(stubbed, monkeypatch):
    sys.modules.pop("demo_job", None)
    import demo_job  # noqa: PLC0415

    agent = FakeAgent(ANSWERS)
    serving = types.SimpleNamespace(
        get_agent=lambda name: None,
        deploy_agent=mock.MagicMock(return_value=agent),
    )
    hopsworks = types.ModuleType("hopsworks")
    hopsworks.login = lambda project=None: types.SimpleNamespace(
        get_agent_serving=lambda: serving
    )
    hopsworks.get_secrets_api = lambda: types.SimpleNamespace(
        get=lambda name: "sk-from-secret"
    )
    monkeypatch.setitem(sys.modules, "hopsworks", hopsworks)
    monkeypatch.setenv("HOPSWORKS_API_KEY", "key")
    monkeypatch.setattr(demo_job.Path, "exists", lambda self: True)
    return demo_job, agent, serving


def test_the_job_deploys_starts_and_holds_two_conversations(job, monkeypatch, capsys):
    demo_job, agent, serving = job
    monkeypatch.setattr(sys, "argv", PINNED)
    demo_job.main()

    assert serving.deploy_agent.call_count == 1
    assert agent.started == 1 and agent.stopped == 0
    # the model key is lifted from the project secret onto the deployment
    assert agent.deployment.predictor.env_vars["OPENAI_API_KEY"] == "sk-from-secret"

    conversations = {cid for _text, cid, _subject in agent.asked if cid}
    # the second conversation is a different one: that is what makes tier 3 visible
    assert len(conversations) == 2
    # and both are with the same person, which is what carries memory between them
    assert {subject for _t, _c, subject in agent.asked} == {"dana@example.com"}

    out = capsys.readouterr().out
    assert "5 of 5 checks held" in out
    # the deterministic view of the tiers, not the model's own account of them
    assert "rolling summary (covers through 4)" in out


def test_a_tier_that_did_not_work_is_reported_and_can_fail_the_job(job, monkeypatch):
    demo_job, agent, _serving = job
    # the session note leaks into the second conversation, and the fact is not forgotten
    agent.answers = list(ANSWERS)
    agent.answers[7] = "You were comparing the 14-inch and the 16-inch."
    agent.answers[9] = "You live in Oslo."
    monkeypatch.setattr(sys, "argv", [*PINNED, "--strict"])

    with pytest.raises(SystemExit) as exit_info:
        demo_job.main()
    assert "2 of 5 checks" in str(exit_info.value)


def test_skip_deploy_needs_an_agent_that_exists(job, monkeypatch):
    demo_job, _agent, _serving = job
    monkeypatch.setattr(sys, "argv", [*PINNED, "--skip-deploy"])
    with pytest.raises(SystemExit) as exit_info:
        demo_job.main()
    assert "memorydemo" in str(exit_info.value)


def test_each_run_is_a_person_the_agent_has_never_met(job, monkeypatch):
    demo_job, agent, _serving = job
    # unpinned: the persona and the subject it is derived from are fresh, so a
    # later run cannot pass the cross-conversation check on an earlier one's facts
    monkeypatch.setattr(sys, "argv", ["demo_job.py"])
    demo_job.main()
    first = {s for _t, _c, s in agent.asked}

    agent.answers, agent.asked = list(ANSWERS), []
    demo_job.main()
    second = {s for _t, _c, s in agent.asked}

    assert len(first) == len(second) == 1
    assert first != second
    assert all(s.endswith("@example.com") for s in first | second)


def test_a_pinned_persona_keeps_the_same_subject(job, monkeypatch):
    demo_job, agent, _serving = job
    monkeypatch.setattr(sys, "argv", PINNED)
    demo_job.main()
    assert {s for _t, _c, s in agent.asked} == {"dana@example.com"}


def test_stop_leaves_nothing_running(job, monkeypatch):
    demo_job, agent, _serving = job
    monkeypatch.setattr(sys, "argv", [*PINNED, "--stop"])
    demo_job.main()
    assert agent.stopped == 1
