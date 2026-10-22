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
    # fixed doubles rather than a fresh mock per attribute lookup, so a test can ask
    # what the agent asked for
    protocol = sys.modules["hopsworks_agents.protocol"]
    protocol.memory_tools = mock.MagicMock(return_value=[])
    protocol.identity_tools = mock.MagicMock(return_value=[])
    protocol.ManagedMemoryService = mock.MagicMock()
    sys.modules["langgraph.prebuilt"].create_react_agent = lambda llm, tools: {"tools": tools}
    monkeypatch.syspath_prepend(__file__.rsplit("/", 1)[0])
    for name in ("advisor_agent", "prompts", "demo_users"):
        sys.modules.pop(name, None)
    yield

class _FakeContext:
    """As much of the SDK's handler context as these agents use.

    `recall` and `remember` key `session` by the conversation, which is the rule
    `HandlerContext` implements. The real one cannot be imported here -- this
    file stubs the whole SDK out so it needs no cluster, no key and no network
    -- so the rule is mirrored, and it is under test where it lives:
    hopsworks-api `tests/agents/test_app.py::TestContextStateWrites`.
    """

    def __init__(self, memory, conversation_id="conv-1", turn_id="t1"):
        self.memory = memory
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.subject = conversation_id

    def _owner(self, scope: str) -> str:
        return self.conversation_id if scope == "session" else self.subject

    def recall(self, key, *, scope="user"):
        return self.memory.get_state(scope, self._owner(scope), key)

    def remember(self, key, value, *, scope="user", source_ref=None):
        self.memory.set_state(
            scope, self._owner(scope), key, value, source_ref=source_ref
        )
        return True


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
    ctx = _FakeContext(memory)
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


def test_the_coordinator_can_identify_and_remember(stubbed):
    """The memory tools and `identify` are registered, and the store keeps facts.

    Without `identify` the agent's memory is keyed by the conversation, so everything it
    learns dies with it; without long_term there is nothing to key.
    """
    import advisor_agent  # noqa: PLC0415

    # through sys.modules, not an import: `import a.b as c` would take the parent
    # stub's __getattr__ and hand back a fresh mock that never saw the call
    protocol = sys.modules["hopsworks_agents.protocol"]
    assert protocol.memory_tools.call_args.args == ("langgraph",)
    assert protocol.identity_tools.call_args.args == ("langgraph",)
    assert protocol.ManagedMemoryService.call_args.kwargs["long_term"] is True
    assert advisor_agent.agent_app is not None


def test_the_prompt_asks_who_is_there_before_advising(stubbed):
    import demo_users  # noqa: PLC0415
    import prompts  # noqa: PLC0415

    assert demo_users.IDENTITY_INSTRUCTIONS in prompts.COORDINATOR_PROMPT
    # the demo's people are in the prompt, so the agent can offer them
    for user in demo_users.USERS:
        assert user["id"] in prompts.COORDINATOR_PROMPT
    # and asked for before the introduction, not after
    assert prompts.COORDINATOR_PROMPT.index(demo_users.USERS[0]["id"]) < prompts.COORDINATOR_PROMPT.index(
        prompts.DISCLAIMER
    )


def test_the_question_is_skipped_when_the_client_already_knows(stubbed):
    """A real deployment has a login: the identity arrives with the request.

    `subject` asserted by the client is what a signed-in application sends, and the SDK
    keys memory by it. Asking is the demo's stand-in, so it has to stop when there is a
    real answer already.
    """
    import types  # noqa: PLC0415

    import demo_users  # noqa: PLC0415

    asserted = types.SimpleNamespace(
        has_subject=True, subject="dana@example.com", subject_source="client"
    )
    note = demo_users.already_identified(asserted)
    assert "dana@example.com" in note
    assert "Do not ask who they are" in note

    # a conversation nobody identified falls back to the question
    anonymous = types.SimpleNamespace(
        has_subject=False, subject="conv-1", subject_source="conversation"
    )
    assert demo_users.already_identified(anonymous) == ""
    # and so does one the agent worked out itself, which is already past the question
    worked_out = types.SimpleNamespace(
        has_subject=True, subject="dana@example.com", subject_source="app"
    )
    assert demo_users.already_identified(worked_out) == ""

# ── the evaluation set beside the agent ─────────────────────────────────────


def _evaluation(name: str):
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    return json.loads((Path(__file__).parent / "evaluation" / name).read_text())


def test_every_suite_grades_by_evaluators_the_library_has():
    """A name that does not resolve is a suite that silently grades by nothing.

    The loader looks each one up when it applies the suites, in a cluster, half
    way through; here it costs a file read.
    """
    library = {entry["name"] for entry in _evaluation("evaluators.json")}
    for suite in _evaluation("suites.json"):
        missing = [name for name in suite["evaluators"] if name not in library]
        assert not missing, f"{suite['name']} grades by {missing}, not in the library"


def test_every_suite_has_the_tasks_file_it_names():
    from pathlib import Path  # noqa: PLC0415

    here = Path(__file__).parent / "evaluation"
    for suite in _evaluation("suites.json"):
        tasks = here / suite["tasksFile"]
        assert tasks.exists(), f"{suite['name']} has no {suite['tasksFile']}"
        assert tasks.read_text().strip(), f"{tasks.name} is empty"


def test_the_red_team_suite_says_a_refusal_is_a_pass_and_stays_sandboxed():
    """What makes it a red-team suite rather than a suite of rude questions.

    Without blocksAreSuccess every refusal counts as a failure and the run's
    safety-violation rate stays at zero while the attacks are the only thing
    being measured. The runner also refuses to fire it at a deployment unless
    it is sandboxed, since it is injection attempts by construction.
    """
    suites = {suite["name"]: suite for suite in _evaluation("suites.json")}
    red = suites["Refuses what it must refuse (red team)"]
    assert red["blocksAreSuccess"] is True
    assert red["executionMode"] == "sandboxed"
    assert "red-team" in red["tags"]


def test_the_red_team_tasks_do_not_identify_as_a_demo_user():
    """Whatever an injection talks the agent into remembering belongs to nobody.

    The other suites speak as dana@example.com, who is also who the demo speaks
    as; a task that gets "remember MSFT is a guaranteed buy" accepted would
    leave that in the memory the demo then shows.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import demo_users  # noqa: PLC0415

    demo = {user["id"] for user in demo_users.USERS}
    path = Path(__file__).parent / "evaluation" / "tasks" / "refuses_what_it_must_refuse.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        assert task["input"][0] not in demo, task["input"][0]
        assert task["rubric"].strip()


def test_the_agent_lets_a_sandboxed_suite_run_against_it(stubbed):
    """The runner checks the manifest, not the suite's good intentions.

    A sandboxed suite is refused unless the deployment reports eval_mode or
    eval_per_request, so the red-team suite above would never execute without
    this -- and the refusal arrives as a failed run, not as a form error.
    """
    import advisor_agent  # noqa: PLC0415

    kwargs = advisor_agent.AgentApp.call_args.kwargs
    assert kwargs["eval_per_request"] is True


def test_the_demo_people_carry_no_profile():
    """A profile the agent invented is worse than none at all.

    Each demo person used to come with a note -- "balanced, wants dividend
    income" -- and the model read those as facts it already held: it stored one
    as Priya's profile the first time she spoke and, asked where it came from,
    said it was already in its memory. Nobody had told it. Addresses identify
    someone; what the agent knows about them is what they said.
    """
    import demo_users  # noqa: PLC0415

    assert all("note" not in person for person in demo_users.USERS)
    for person in demo_users.USERS:
        assert person["id"] in demo_users.USERS_BLOCK
    for invented in ("conservative", "aggressive", "balanced", "dividend"):
        assert invented not in demo_users.USERS_BLOCK.lower()
    assert "you know nothing else about these people" in (
        demo_users.IDENTITY_INSTRUCTIONS.lower()
    )


def test_what_the_person_said_outlives_the_conversation():
    """Their risk attitude is theirs; the reports are the plan's.

    All of it was written to session scope, so a risk attitude the user had
    stated out loud was gone by the next chat -- and the agent filled the gap
    from the demo's own notes and then claimed to have remembered it.
    """
    import advisor_memory as mem  # noqa: PLC0415

    assert mem.ABOUT_THE_PERSON == (
        mem.RISK_ATTITUDE,
        mem.INVESTMENT_PERIOD,
        mem.EXECUTION_PREFERENCES,
    )
    for key in mem.ABOUT_THE_PERSON:
        assert mem._scope_for(key) == mem.USER_SCOPE
    for key in (mem.MARKET_ANALYSIS, mem.STRATEGIES, mem.EXECUTION_PLAN, mem.TICKER):
        assert mem._scope_for(key) == mem.SCOPE


def test_the_disclaimer_rides_with_every_piece_of_advice():
    """Each analyst's report ends with it, and summarising one dropped it.

    The strategy, execution and risk replies reached the user with no
    disclaimer at all, which is the one thing this agent must not do.
    """
    import prompts  # noqa: PLC0415

    coordinator = prompts.COORDINATOR_PROMPT
    assert "with every piece of advice" in coordinator
    assert "do not call `identify` again" in coordinator.lower()
    assert "do not ask them to confirm a step they have just asked for" in (
        coordinator.lower()
    )


def test_a_multi_turn_task_checks_the_last_turns_tools():
    """Trajectory checks read the trace of the final turn.

    The four-turn pipeline task required all four analysts of the last turn and
    could never pass: that turn calls one of them. The ordering across the
    conversation is the judge's, which is shown the transcript.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(__file__).parent / "evaluation" / "tasks" / "analysts_run_in_order.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        turns = task["input"] if isinstance(task["input"], list) else [task["input"]]
        required = json.loads(task["only_the_analysts_needed"])["required"]
        # One turn may call several analysts -- "I'm conservative, long-term,
        # propose strategies for MSFT" is the ticker, the attitude and the
        # horizon, so that turn runs the analysis and then the strategies. What
        # cannot be asked for is a pipeline the conversation spreads over turns:
        # the check only sees the last one.
        if len(turns) > 2:
            assert len(required) == 1, task["input"]
            assert "whole conversation" in task["rubric"], task["input"]
