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
    def function_tool(fn=None, **options):
        def wrap(f):
            return types.SimpleNamespace(name=f.__name__, description=f.__doc__, fn=f, **options)

        return wrap(fn) if fn is not None else wrap

    agents.function_tool = function_tool
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


def test_the_app_asks_who_is_there_and_the_prompt_never_does(stubbed):
    import demo_users  # noqa: PLC0415
    import prompts  # noqa: PLC0415

    assert demo_users.IDENTITY_INSTRUCTIONS in prompts.COORDINATOR_PROMPT
    # The application asks before the model runs, naming the demo's people. Told to ask as
    # well, a model asked a client-identified person for their address against the note
    # saying who they were.
    # One example address, not the list: two generic judges read the list as invented
    # facts and as a leak of other people's addresses.
    assert demo_users.USERS[0]["id"] in demo_users.ASK_WHO
    for user in demo_users.USERS[1:]:
        assert user["id"] not in demo_users.ASK_WHO
    assert "Never ask who they" in prompts.COORDINATOR_PROMPT


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
    # and one this demo bound from an address given earlier is told so every turn.
    # A rule the model reads once ("do not call `identify` again") loses to the
    # standing instruction to ask: it re-identified people on their third turn.
    bound = types.SimpleNamespace(
        has_subject=True, subject="dana@example.com", subject_source="app"
    )
    note = demo_users.already_identified(bound)
    assert "dana@example.com" in note
    assert "do not call `identify`" in note
    # and every turn after that one, which the SDK reports as "conversation": the
    # binding read back from an earlier turn. Missed, the agent re-identified on turn 3.
    later = types.SimpleNamespace(
        has_subject=True, subject="dana@example.com", subject_source="conversation"
    )
    assert "do not call `identify`" in demo_users.already_identified(later)


def test_an_address_binds_memory_to_the_person(stubbed):
    """The half `identify` deliberately does not do, which a demo has to supply.

    `identify` labels the conversation and never rebinds, so `user`-scope facts stay
    owned by the conversation id and the second conversation as the same person knows
    nothing -- which is the thing this demo exists to show.
    """
    import types  # noqa: PLC0415

    import demo_users  # noqa: PLC0415

    class Ctx(types.SimpleNamespace):
        def rebind_subject(self, subject):
            self.subject = subject
            self.has_subject = True
            self.subject_source = "app"

    ctx = Ctx(has_subject=False, subject="conv-1", subject_source="conversation")
    assert demo_users.bind_if_identified(ctx, "I'm dana@example.com") == "dana@example.com"
    assert ctx.subject == "dana@example.com"

    # the client's own assertion is never overridden by text in a message
    asserted = Ctx(has_subject=True, subject="marcus@example.com", subject_source="client")
    assert demo_users.bind_if_identified(asserted, "dana@example.com") == "marcus@example.com"

    # and a turn with no address in it changes nothing
    plain = Ctx(has_subject=False, subject="conv-1", subject_source="conversation")
    assert demo_users.bind_if_identified(plain, "Analyze MSFT for me.") == ""
    assert plain.subject_source == "conversation"

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
        mem.INVESTMENT_GOAL,
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
    assert "every piece of advice reaches the user with that disclaimer" in coordinator.lower()
    # carried by the reports themselves, which the app now shows in full
    for name in ("TRADING_ANALYST_PROMPT", "EXECUTION_ANALYST_PROMPT", "RISK_ANALYST_PROMPT"):
        assert prompts.DISCLAIMER in getattr(prompts, name), name
    assert "do not call `identify` again" in coordinator.lower()
    assert "do not ask them to confirm a step they have just asked for" in (
        coordinator.lower()
    )


def test_what_was_remembered_is_shown_without_being_asked_for():
    """A new conversation with someone the agent has met says what it knows.

    The profile used to be shown only alongside a finished report, so a fresh
    conversation said nothing -- and the model, told to `recall` but not told
    under which keys, guessed one, got nothing, and answered that it knew
    nothing about a person who had told it twice.
    """
    import advisor_memory as mem  # noqa: PLC0415

    stored = {
        (mem.RISK_ATTITUDE, mem.USER_SCOPE): "moderate",
        (mem.INVESTMENT_PERIOD, mem.USER_SCOPE): "long-term",
    }

    class Ctx:
        def recall(self, key, scope=None):
            return stored.get((key, scope), "")

    # The module attribute, not its .set(): an earlier test in this file stubs
    # the protocol package with a mock, so the real ContextVar is not there.
    class Holder:
        @staticmethod
        def get(_default=None):
            return Ctx()

    original = mem.current_context
    mem.current_context = Holder
    try:
        block = mem.progress_block()
    finally:
        mem.current_context = original

    # No report has run in this conversation, and it still says what it knows.
    assert "moderate" in block and "long-term" in block
    assert "stored long-term memory" in block
    # named, because the model guessed "risk_attitude" and was told nothing was stored
    assert "user_risk_attitude" in block
    assert "Progress in this conversation" not in block


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


def test_a_reply_written_across_two_model_calls_is_two_paragraphs(openai_stubbed):
    """output_index restarts at 0 for every model call.

    The sentence the coordinator writes before calling an analyst and the answer it
    writes after were both "item 0", so they streamed as one: "...which you've
    provided.The market analyst has completed...". A second item of the same call is
    still held back, and dropped when it only repeats what was said.
    """

    class Raw:
        def __init__(self, data):
            self.data = data

    class Delta:
        def __init__(self, delta, output_index=0):
            self.delta = delta
            self.output_index = output_index
            self.type = "response.output_text.delta"

    class ItemEvent:
        pass

    sys.modules["agents.stream_events"].RawResponsesStreamEvent = Raw
    sys.modules["agents.stream_events"].RunItemStreamEvent = ItemEvent
    sys.modules["openai.types.responses"].ResponseTextDeltaEvent = Delta
    sys.modules["hopsworks_agents.protocol"].AgentApp = lambda **kw: types.SimpleNamespace(
        stream=lambda fn: fn, chat=lambda fn: fn
    )
    for name in ("advisor_memory", "advisor_agent_openai"):
        sys.modules.pop(name, None)
    import advisor_agent_openai as o  # noqa: PLC0415

    created = types.SimpleNamespace(type="response.created")
    events = [
        Raw(created), Raw(Delta("Thanks, that's what ")), Raw(Delta("you've provided.")),
        Raw(Delta("Thanks, that's what you've provided.", output_index=1)),  # a restatement
        Raw(created), Raw(Delta("The market analyst has completed the report.")),
    ]

    class Result:
        final_output = ""

        async def stream_events(self):
            for event in events:
                yield event

    o.Runner = types.SimpleNamespace(run_streamed=lambda *a, **kw: Result())
    ctx = types.SimpleNamespace(
        has_subject=True, subject_source="client", subject="dana@example.com",
        history=[], system_context=lambda: "", rebind_subject=lambda s: None,
    )
    request = types.SimpleNamespace(text="Analyze MSFT for me.")

    import asyncio  # noqa: PLC0415

    async def collect():
        return "".join([d async for d in o.stream(request, ctx)])

    reply = asyncio.run(collect())

    assert reply == (
        "Thanks, that's what you've provided.\n\n"
        "The market analyst has completed the report."
    )


# ── what a turn may do, decided in code ─────────────────────────────────────


class _Memory:
    """advisor_memory's context: one conversation's memory, and this turn's message."""

    def __init__(self, text="", **reports):
        self.store = dict(reports)
        self.request = types.SimpleNamespace(text=text)

    def recall(self, key, scope=None):
        return self.store.get(key, "")

    def remember(self, key, value, scope=None):
        self.store[key] = value


def _with(ctx):
    import advisor_memory as mem  # noqa: PLC0415

    class Holder:
        @staticmethod
        def get(_default=None):
            return ctx

    original = mem.current_context
    mem.current_context = Holder
    # per-turn state starts empty, as it does in each request's own context
    mem._STEP_AT_START.set(None)
    mem._PROFILE_GIVEN.set(False)
    return mem, lambda: setattr(mem, "current_context", original)


def test_a_profile_stated_with_the_request_is_captured_before_the_model_runs():
    # "I'm a conservative, long-term investor. Propose trading strategies for MSFT"
    # ran the analysis and then asked for the attitude and the horizon it was given
    ctx = _Memory()
    mem, restore = _with(ctx)
    try:
        found = mem.capture_profile(
            "I'm a conservative, long-term investor. Propose trading strategies for MSFT."
        )
    finally:
        restore()
    assert found == {"risk attitude": "conservative", "investment period": "long-term"}
    assert ctx.store[mem.RISK_ATTITUDE] == "conservative"
    assert ctx.store[mem.INVESTMENT_PERIOD] == "long-term"


def test_a_message_that_states_no_profile_writes_none():
    ctx = _Memory()
    mem, restore = _with(ctx)
    try:
        assert mem.capture_profile("Analyze MSFT for me.") == {}
        # "high-risk" is the profile, not a request for the risk evaluation
        assert mem.capture_profile("I'm a high-risk investor") == {"risk attitude": "aggressive"}
    finally:
        restore()


def test_asked_for_the_plan_the_risk_evaluation_does_not_run():
    # "Plan the second strategy" ran assess_risk instead of plan_execution
    import advisor_memory as mem  # noqa: PLC0415

    reports = {mem.MARKET_ANALYSIS: "r", mem.STRATEGIES: "s"}
    mem_, restore = _with(_Memory("Plan the second strategy.", **reports))
    try:
        assert mem_.not_this_turn("plan") == ""
        refusal = mem_.not_this_turn("risk")
    finally:
        restore()
    assert refusal.startswith("Not run")
    assert "the execution plan" in refusal and "the risk evaluation" in refusal


def test_asked_for_strategies_the_analysis_before_them_may_run():
    mem, restore = _with(_Memory("Propose trading strategies for MSFT."))
    try:
        assert mem.not_this_turn("analysis") == ""
        assert mem.not_this_turn("strategies") == ""
        assert mem.not_this_turn("plan").startswith("Not run")
    finally:
        restore()


def test_a_go_ahead_runs_the_next_step_and_only_that():
    import advisor_memory as mem  # noqa: PLC0415

    reports = {mem.MARKET_ANALYSIS: "r", mem.STRATEGIES: "s"}
    mem_, restore = _with(_Memory("Yes please.", **reports))
    try:
        assert mem_.not_this_turn("plan") == ""
        assert mem_.not_this_turn("risk").startswith("Not run")
    finally:
        restore()


def test_with_no_report_yet_the_next_step_is_the_ticker():
    # The recipe asks for the ticker first. After the introduction the agent asked
    # for the risk profile instead, which the strategies need, two steps later.
    mem, restore = _with(_Memory())
    try:
        line = mem.next_step_block()
    finally:
        restore()
    assert "ask for the ticker" in line
    assert "Do not ask for a risk attitude" in line


def test_a_known_profile_is_not_asked_for_again():
    import advisor_memory as mem  # noqa: PLC0415

    ctx = _Memory(**{mem.MARKET_ANALYSIS: "r", mem.RISK_ATTITUDE: "moderate",
                     mem.INVESTMENT_PERIOD: "long-term"})
    mem_, restore = _with(ctx)
    try:
        line = mem_.next_step_block()
    finally:
        restore()
    assert "moderate, long-term" in line and "do not ask for it" in line


def test_a_report_is_shown_under_its_analysts_name_and_a_refusal_is_not():
    # The coordinator was asked to present each report and often did not: "the market
    # analyst has completed the analysis" with none of it on screen, or "I'm calling
    # the risk analyst now" and no call. A report that exists is now shown by the app.
    import advisor_memory as mem  # noqa: PLC0415

    shown = mem.presented("analyze_market", "MSFT trades at ...")
    assert shown.startswith("### Market analysis") and "MSFT trades at" in shown
    assert mem.presented("assess_risk", "Not run: this turn asked for the plan") == ""
    assert mem.presented("assess_risk", "Error: the foundational input(s) ...") == ""
    assert mem.presented("recall", "moderate") == ""


def test_the_stream_shows_the_analysts_report_between_the_two_calls(openai_stubbed):
    class Raw:
        def __init__(self, data):
            self.data = data

    class Delta:
        def __init__(self, delta, output_index=0):
            self.delta = delta
            self.output_index = output_index
            self.type = "response.output_text.delta"

    class ItemEvent:
        def __init__(self, name, raw_item, output=None):
            self.name = name
            self.item = types.SimpleNamespace(raw_item=raw_item, output=output)

    sys.modules["agents.stream_events"].RawResponsesStreamEvent = Raw
    sys.modules["agents.stream_events"].RunItemStreamEvent = ItemEvent
    sys.modules["openai.types.responses"].ResponseTextDeltaEvent = Delta
    sys.modules["hopsworks_agents.protocol"].AgentApp = lambda **kw: types.SimpleNamespace(
        stream=lambda fn: fn, chat=lambda fn: fn
    )
    for name in ("advisor_memory", "advisor_agent_openai"):
        sys.modules.pop(name, None)
    import advisor_agent_openai as o  # noqa: PLC0415

    created = types.SimpleNamespace(type="response.created")
    called = types.SimpleNamespace(name="analyze_market", call_id="c1", arguments='{"ticker": "MSFT"}')
    events = [
        Raw(created), Raw(Delta("Let's start with the market analysis.")),
        ItemEvent("tool_called", called),
        ItemEvent("tool_output", {"call_id": "c1"}, output="MSFT report body."),
        Raw(created), Raw(Delta("Next: your risk attitude and horizon.")),
    ]

    class Result:
        final_output = ""

        async def stream_events(self):
            for event in events:
                yield event

    async def emit_event(*_a, **_kw):
        return None

    o.Runner = types.SimpleNamespace(run_streamed=lambda *a, **kw: Result())
    ctx = types.SimpleNamespace(
        has_subject=True, subject_source="client", subject="dana@example.com",
        history=[], system_context=lambda: "", rebind_subject=lambda s: None,
        emit_event=emit_event,
    )
    request = types.SimpleNamespace(text="Analyze MSFT for me.")

    import asyncio  # noqa: PLC0415

    async def collect():
        return "".join([d async for d in o.stream(request, ctx)])

    reply = asyncio.run(collect())
    assert reply == (
        "Let's start with the market analysis.\n\n"
        "### Market analysis — from the data analyst\n\nMSFT report body.\n\n"
        "Next: your risk attitude and horizon."
    )


def _openai_advisor(monkeypatch, model=None):
    if model is None:
        monkeypatch.delenv("ADVISOR_OPENAI_MODEL", raising=False)
    else:
        monkeypatch.setenv("ADVISOR_OPENAI_MODEL", model)
    sys.modules["hopsworks_agents.protocol"].AgentApp = lambda **kw: types.SimpleNamespace(
        stream=lambda fn: fn, chat=lambda fn: fn
    )
    for name in ("advisor_memory", "advisor_agent_openai"):
        sys.modules.pop(name, None)
    import advisor_agent_openai as o  # noqa: PLC0415

    return o


def test_no_temperature_is_sent_to_a_model_that_refuses_it(openai_stubbed, monkeypatch):
    # gpt-6-luna answers a request carrying `temperature` with a 400, and the
    # default model moved to it with temperature still set: every chat 500'd.
    o = _openai_advisor(monkeypatch)
    assert o.coordinator.model_settings == {}
    assert o.data_analyst.model_settings == {}


def test_a_model_that_takes_a_temperature_still_gets_it(openai_stubbed, monkeypatch):
    o = _openai_advisor(monkeypatch, "gpt-5.4-mini-2026-03-17")
    assert o.coordinator.model_settings == {"temperature": 0.0}
    assert o.trading_analyst.model_settings == {"temperature": 0.2}


def test_nobody_identified_is_asked_who_they_are_and_nothing_runs(openai_stubbed, monkeypatch):
    # "Analyze NVDA for me." with no address went to identify("NVDA") and a full
    # analysis for nobody, every time. The application answers that turn itself.
    o = _openai_advisor(monkeypatch)
    o.Runner = types.SimpleNamespace(
        run_streamed=mock.MagicMock(side_effect=AssertionError("the model ran"))
    )
    ctx = types.SimpleNamespace(
        has_subject=False, subject_source="conversation", subject="conv-1",
        history=[], system_context=lambda: "", rebind_subject=lambda s: None,
    )

    import asyncio  # noqa: PLC0415

    async def collect():
        request = types.SimpleNamespace(text="Analyze NVDA for me.")
        return "".join([d async for d in o.stream(request, ctx)])

    reply = asyncio.run(collect())
    assert "email address" in reply and "dana@example.com" in reply


def test_an_analyst_the_turn_did_not_ask_for_is_not_offered(openai_stubbed, monkeypatch):
    # Offered and refused, the call stayed in the trace, and "only the analysts the
    # step needs" failed a turn that behaved correctly.
    o = _openai_advisor(monkeypatch)
    ctx = _Memory(text="Analyze MSFT for me.")
    _mem, restore = _with(ctx)
    try:
        tools = {t.name: t for t in o.coordinator.tools}
        assert tools["propose_strategies"].is_enabled(None, None) is False
        assert tools["assess_risk"].is_enabled(None, None) is False
        ctx.request.text = "Propose strategies for MSFT."
        assert tools["propose_strategies"].is_enabled(None, None) is True
    finally:
        restore()


# ── round 7: the gate keys on the plan, not only on words ──────────────────


def _gate(text, **reports):
    ctx = _Memory(text=text, **reports)
    mem, restore = _with(ctx)
    try:
        mem.capture_profile(text)
        mem.begin_turn()
        return mem, ctx, restore
    except Exception:
        restore()
        raise


def test_a_profile_given_in_answer_runs_the_strategies():
    # "Aggressive, short-term." names no step; it answers the question asked for the
    # strategies, and the strategies analyst was not offered, so nothing ran.
    mem, _ctx, restore = _gate("Aggressive, short-term.", market_data_analysis_output="r")
    try:
        assert mem.not_this_turn("strategies") == ""
        assert mem.not_this_turn("plan") != ""
        assert "call propose_strategies now" in mem.next_step_block()
    finally:
        restore()


def test_a_request_that_needs_an_earlier_step_says_to_run_both():
    # "One analyst per turn" read alone stopped after the analysis and asked the user to
    # request the strategies again.
    mem, _ctx, restore = _gate("Propose strategies for AAPL.")
    try:
        assert mem.not_this_turn("analysis") == ""
        assert mem.not_this_turn("strategies") == ""
        note = mem.next_step_block()
        assert "the market analysis, then the trading strategies" in note
    finally:
        restore()


def test_should_i_buy_is_not_an_analysis_request():
    # It went straight to a full report; the process introduces and offers first.
    for text in (
        "Can you tell me whether to buy Tesla right now?",
        "Should I buy TSLA?",
        "Hi, I want help with my investments.",
        "Hello. What can you do for me?",
    ):
        mem, _ctx, restore = _gate(text)
        try:
            assert mem.not_this_turn("analysis") != "", text
        finally:
            restore()
    # while a ticker given in answer, or an explicit request, still runs it
    for text in ("MSFT", "Analyze NVDA for me.", "Yes please."):
        mem, _ctx, restore = _gate(text)
        try:
            assert mem.not_this_turn("analysis") == "", text
        finally:
            restore()


def test_who_this_is_goes_into_the_instructions(openai_stubbed, monkeypatch):
    # As a user message, gpt-6-luna asked a client-identified person for their address.
    o = _openai_advisor(monkeypatch)
    ctx = _Memory(text="Hi")
    ctx.has_subject = True
    ctx.subject = "marcus@example.com"
    ctx.subject_source = "client"
    _mem, restore = _with(ctx)
    try:
        instructions = o._coordinator_instructions(None, None)
    finally:
        restore()
    assert "already identified as marcus@example.com" in instructions


# ── round 9 ─────────────────────────────────────────────────────────────────


def test_a_risk_level_in_a_profile_is_not_a_request_for_the_risk_evaluation():
    # "Moderate risk, medium-term. What strategies fit?" was read as asking for the
    # risk evaluation too, so the plan and the risk analysts were offered on a turn
    # that asked for the strategies, and the model ran all three.
    import advisor_memory as mem  # noqa: PLC0415

    assert mem.asked_steps("Moderate risk, medium-term. What strategies fit?") == ["strategies"]
    assert mem.asked_steps("low-risk please, long term") == []
    assert mem.asked_steps("Now evaluate the risk") == ["risk"]

    mem_, ctx, restore = _gate(
        "Moderate risk, medium-term. What strategies fit?", market_data_analysis_output="r"
    )
    try:
        assert mem_.not_this_turn("strategies") == ""
        assert mem_.not_this_turn("plan") != ""
        assert mem_.not_this_turn("risk") != ""
    finally:
        restore()


def test_an_investment_goal_stated_with_the_profile_is_kept():
    # Left to the model's `remember`, the goal was stored in one conversation and
    # not in the next.
    mem, ctx, restore = _gate("Moderate, long-term, focused on dividend income.")
    try:
        assert ctx.store[mem.INVESTMENT_GOAL] == "dividend income"
    finally:
        restore()
