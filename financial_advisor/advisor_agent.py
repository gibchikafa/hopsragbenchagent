"""
Financial advisor — Google's ADK financial-advisor recipe on LangGraph and the
Hopsworks Agent Protocol.

    coordinator ─┬─▶ analyze_market      (data_analyst: web search → market analysis)
                 ├─▶ propose_strategies  (trading_analyst: ≥5 strategies for the profile)
                 ├─▶ plan_execution      (execution_analyst: entry, holding, exits)
                 └─▶ assess_risk         (risk_analyst: the plan against the profile)

The recipe is a coordinator ``LlmAgent`` with four analysts attached as
``AgentTool``s, each reading its predecessor's report from session state and
writing its own under an ``output_key``. Here the coordinator is a LangGraph
ReAct agent and the analysts are its tools; each one is its own model call
with the analyst's prompt. Three things differ, and each is deliberate:

**Reports live in the conversation's working memory.** An analyst's report is
several thousand tokens. The coordinator never copies one into a tool call:
each tool writes its report to session-scoped state (this conversation only)
and the later tools read it there, which is what the recipe's state keys did.
Working memory is the SDK's, so it outlives the pod and is shared across
replicas.

**The data analyst searches the web through the model.** The recipe's
``google_search`` is Gemini's built-in grounding. Its analogue on Claude is
the server-side web search tool: one call, and the model runs the searches
itself and cites what it read. Nothing here fetches a page.

**The SDK owns the serving surface.** Manifest, ``/v1/chat``,
``/v1/chat/stream``, ``/v1/feedback``, health, CORS, tracing, and the memory
tiers are all ``AgentApp``; the code here is the domain.

Deploy:
    hops agent create advisor_agent.py --name financialadvisor \\
        --requirements requirements.txt --environment python-agent-pipeline
    hops agent start financialadvisor --wait 600
"""

from __future__ import annotations

import os
from typing import Any

from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
)
from hopsworks_agents.protocol.autoevents import current_context
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from prompts import (
    COORDINATOR_PROMPT,
    DATA_ANALYST_PROMPT,
    EXECUTION_ANALYST_PROMPT,
    RISK_ANALYST_PROMPT,
    TRADING_ANALYST_PROMPT,
)

MODEL = os.environ.get("ADVISOR_MODEL", "claude-sonnet-4-5")
#: how many searches the data analyst may run per report; each is a paid call
MAX_SEARCHES = int(os.environ.get("ADVISOR_MAX_SEARCHES", "8"))
#: a report is a few thousand tokens; the default 4k characters would cut one
MAX_REPORT_CHARS = 32_000

# ── the reports: working memory for this conversation ────────────────────────
#
# The recipe's state keys. `session` scope is this conversation only, which is
# what a plan is: a new chat starts over, as the recipe's session would.
SCOPE = "session"
MARKET_ANALYSIS = "market_data_analysis_output"
STRATEGIES = "proposed_trading_strategies_output"
EXECUTION_PLAN = "execution_plan_output"
RISK_EVALUATION = "risk_evaluation_output"
TICKER = "ticker"
RISK_ATTITUDE = "user_risk_attitude"
INVESTMENT_PERIOD = "user_investment_period"
EXECUTION_PREFERENCES = "user_execution_preferences"


def _memory_and_owner():
    """The store and this conversation's id, resolved the way the SDK's own tools do.

    A tool is called by the model and handed neither; the request context is a
    contextvar the SDK sets for the turn.
    """
    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        return None, None
    return ctx.memory, ctx.conversation_id


def _read(key: str) -> str:
    memory, owner = _memory_and_owner()
    if memory is None:
        return ""
    return memory.get_state(SCOPE, owner, key) or ""


def _write(key: str, value: str) -> None:
    memory, owner = _memory_and_owner()
    if memory is None:
        return
    ctx = current_context.get(None)
    memory.set_state(
        SCOPE,
        owner,
        key,
        value,
        source_ref=f'{{"conversation_id": "{owner}", "turn_id": "{ctx.turn_id}"}}',
    )


# ── the analysts ─────────────────────────────────────────────────────────────
#
# Tagged so the turn's stream can tell an analyst's tokens from the
# coordinator's: only the coordinator speaks to the user, and its reply must
# not carry a report streamed through it mid-tool-call.
ANALYST_TAG = "analyst"
_analyst_llm = ChatAnthropic(model=MODEL, max_tokens=8192, temperature=0.2)
_researcher_llm = _analyst_llm.bind_tools(
    [{"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCHES}]
)


def _text_of(message: Any) -> str:
    """The text of a reply whose content may be blocks: search results, citations, text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


async def _analyst(system: str, brief: str, llm: Any = None) -> str:
    reply = await (llm or _analyst_llm).ainvoke(
        [SystemMessage(content=system), HumanMessage(content=brief)],
        config={"tags": [ANALYST_TAG]},
    )
    return _text_of(reply)


def _missing(*needed: tuple[str, str]) -> str:
    """The recipe's prerequisite check, for every analyst after the first."""
    absent = [name for name, key in needed if not _read(key)]
    if not absent:
        return ""
    return (
        "Error: the foundational input(s) "
        + ", ".join(absent)
        + " are missing. The earlier step(s) must be completed first; run them, "
        "then call this analyst again."
    )


@tool
async def analyze_market(ticker: str) -> str:
    """The data analyst: a market analysis report for one ticker, from recent web sources.

    Args:
        ticker: The stock market ticker symbol to analyze, e.g. AAPL, GOOGL, MSFT.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return "Error: a ticker symbol is required."
    report = await _analyst(
        DATA_ANALYST_PROMPT,
        f"provided_ticker: {ticker}\nmax_data_age_days: 7\ntarget_results_count: 10",
        _researcher_llm,
    )
    _write(TICKER, ticker)
    _write(MARKET_ANALYSIS, report[:MAX_REPORT_CHARS])
    return report


@tool
async def propose_strategies(risk_attitude: str, investment_period: str) -> str:
    """The trading analyst: five or more strategies for the analyzed ticker and this profile.

    Reads the market analysis itself; call analyze_market first.

    Args:
        risk_attitude: The user's attitude to risk, e.g. conservative, moderate, aggressive.
        investment_period: The user's timeframe, e.g. short-term, medium-term, long-term.
    """
    if problem := _missing(("market analysis (market_data_analysis_output)", MARKET_ANALYSIS)):
        return problem
    _write(RISK_ATTITUDE, risk_attitude.strip())
    _write(INVESTMENT_PERIOD, investment_period.strip())
    report = await _analyst(
        TRADING_ANALYST_PROMPT,
        f"user_risk_attitude: {risk_attitude}\n"
        f"user_investment_period: {investment_period}\n\n"
        f"market_data_analysis_output:\n{_read(MARKET_ANALYSIS)}",
    )
    _write(STRATEGIES, report[:MAX_REPORT_CHARS])
    return report


@tool
async def plan_execution(selected_strategy: str = "", execution_preferences: str = "") -> str:
    """The execution analyst: how to enter, hold, scale and exit the chosen strategy.

    Reads the proposed strategies, the risk attitude and the period itself; call
    propose_strategies first.

    Args:
        selected_strategy: The strategy the user chose, by name or description; empty
            to plan for the proposed strategies as a set.
        execution_preferences: The user's execution preferences, such as broker or order
            types; empty when they stated none.
    """
    if problem := _missing(
        ("trading strategies (proposed_trading_strategies_output)", STRATEGIES)
    ):
        return problem
    _write(EXECUTION_PREFERENCES, execution_preferences.strip())
    chosen = selected_strategy.strip() or "(none selected: plan for the proposed strategies as a set)"
    report = await _analyst(
        EXECUTION_ANALYST_PROMPT,
        f"provided_trading_strategy: {chosen}\n"
        f"user_risk_attitude: {_read(RISK_ATTITUDE)}\n"
        f"user_investment_period: {_read(INVESTMENT_PERIOD)}\n"
        f"user_execution_preferences: {execution_preferences or '(none stated)'}\n\n"
        f"proposed_trading_strategies_output:\n{_read(STRATEGIES)}",
    )
    _write(EXECUTION_PLAN, report[:MAX_REPORT_CHARS])
    return report


@tool
async def assess_risk() -> str:
    """The risk analyst: the whole plan, analysis to execution, against the user's profile.

    Takes no input; reads the three earlier reports and the profile itself. Call it
    last, after plan_execution.
    """
    if problem := _missing(
        ("market analysis (market_data_analysis_output)", MARKET_ANALYSIS),
        ("trading strategies (proposed_trading_strategies_output)", STRATEGIES),
        ("execution plan (execution_plan_output)", EXECUTION_PLAN),
    ):
        return problem
    report = await _analyst(
        RISK_ANALYST_PROMPT,
        f"user_risk_attitude: {_read(RISK_ATTITUDE)}\n"
        f"user_investment_period: {_read(INVESTMENT_PERIOD)}\n"
        f"user_execution_preferences: {_read(EXECUTION_PREFERENCES) or '(none stated)'}\n\n"
        f"market_data_analysis_output:\n{_read(MARKET_ANALYSIS)}\n\n"
        f"provided_trading_strategy:\n{_read(STRATEGIES)}\n\n"
        f"provided_execution_strategy:\n{_read(EXECUTION_PLAN)}",
    )
    _write(RISK_EVALUATION, report[:MAX_REPORT_CHARS])
    return report


# ── the coordinator ──────────────────────────────────────────────────────────

ANALYSTS = [analyze_market, propose_strategies, plan_execution, assess_risk]
coordinator = create_react_agent(
    ChatAnthropic(model=MODEL, max_tokens=8192, temperature=0.0),
    ANALYSTS,
)

# What the chat panel's Graph tab shows. The compiled graph would draw
# "agent → tools" and hide the analysts inside the tools node; this is the
# recipe's picture, with the order the reports flow in.
GRAPH = {
    "nodes": [
        {"id": "__start__", "label": "__start__"},
        {"id": "coordinator", "label": "financial_coordinator"},
        {"id": "analyze_market", "label": "data_analyst"},
        {"id": "web_search", "label": "web_search"},
        {"id": "propose_strategies", "label": "trading_analyst"},
        {"id": "plan_execution", "label": "execution_analyst"},
        {"id": "assess_risk", "label": "risk_analyst"},
        {"id": "__end__", "label": "__end__"},
    ],
    "edges": [
        {"source": "__start__", "target": "coordinator"},
        {"source": "coordinator", "target": "analyze_market", "conditional": True},
        {"source": "analyze_market", "target": "web_search"},
        {"source": "coordinator", "target": "propose_strategies", "conditional": True},
        {"source": "coordinator", "target": "plan_execution", "conditional": True},
        {"source": "coordinator", "target": "assess_risk", "conditional": True},
        {"source": "analyze_market", "target": "propose_strategies", "label": "market analysis"},
        {"source": "propose_strategies", "target": "plan_execution", "label": "strategies"},
        {"source": "plan_execution", "target": "assess_risk", "label": "execution plan"},
        {"source": "coordinator", "target": "__end__", "conditional": True},
    ],
}

agent_app = AgentApp(
    name="Financial advisor",
    description="Guides you from a market ticker to a risk-evaluated plan: market analysis, "
    "trading strategies, an execution plan and a risk evaluation, each by its own analyst "
    "(LangGraph coordinator; Claude with web search).",
    framework="langgraph",
    welcome_message="Hello! I'm here to help you navigate the world of financial "
    "decision-making, one step at a time: a ticker, strategies for your profile, an "
    "execution plan, and the risks. Ready to get started?",
    suggested_prompts=[
        "Hello. What can you do for me?",
        "Analyze NVDA for me.",
        "I'm a conservative, long-term investor. What strategies fit MSFT?",
    ],
    placeholder="A ticker to analyze, or the next step...",
    memory=ManagedMemoryService(
        summarize=anthropic_summarizer(),
        max_state_value_chars=MAX_REPORT_CHARS,
    ),
    tool_events=True,
    graph=GRAPH,
)


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints: the coordinator's tokens stream to the
    user and each analyst it calls shows as a progress chip."""
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.", code="invalid_request", status_code=400
        )

    # The prompt, the rolling summary of folded turns, and where the process
    # stands: which reports exist, so a resumed conversation continues rather
    # than starting over.
    system = COORDINATOR_PROMPT + ctx.system_context() + _progress_block()
    messages = [{"role": "system", "content": system}]
    messages += ctx.history
    messages.append({"role": "user", "content": request.text})

    async def _coordinator_only(events):
        # An analyst's model call streams too; those tokens are its report,
        # returned to the coordinator, not words for the user.
        async for event in events:
            if event.get("event") == "on_chat_model_stream" and ANALYST_TAG in (
                event.get("tags") or []
            ):
                continue
            yield event

    async for delta in ctx.stream_langchain(
        _coordinator_only(coordinator.astream_events({"messages": messages}, version="v2"))
    ):
        yield delta


def _progress_block() -> str:
    done = [
        label
        for label, key in (
            ("market analysis", MARKET_ANALYSIS),
            ("trading strategies", STRATEGIES),
            ("execution plan", EXECUTION_PLAN),
            ("risk evaluation", RISK_EVALUATION),
        )
        if _read(key)
    ]
    if not done:
        return ""
    profile = ", ".join(
        f"{name}: {_read(key)}"
        for name, key in (
            ("ticker", TICKER),
            ("risk attitude", RISK_ATTITUDE),
            ("investment period", INVESTMENT_PERIOD),
        )
        if _read(key)
    )
    return (
        "\n\nProgress in this conversation: the "
        + ", ".join(done)
        + " report(s) are already done and kept for the later analysts"
        + (f" ({profile})" if profile else "")
        + ". Continue from the next step; do not repeat a completed one unless asked."
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
