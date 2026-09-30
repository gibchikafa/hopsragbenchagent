"""
Financial advisor on the OpenAI Agents SDK: the same prompts and working memory, a
different agent runtime.

The coordinator is an `Agent` whose tools run the four analysts, each an
`Agent` with the recipe's prompt; the data analyst carries the SDK's web
search tool, the analogue of the recipe's Google Search grounding. Reports go
through the conversation's working memory exactly as in `advisor_agent.py`,
so the two entry points are interchangeable on the same deployment history.

Deploy (git-backed, so `prompts.py` and `advisor_memory.py` come along):
    agents.deploy_agent("financial_advisor/advisor_agent_openai.py", name="financialadvisor-openai", ...)
"""

from __future__ import annotations

import os
from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner, WebSearchTool, function_tool
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
)
from openai.types.responses import ResponseTextDeltaEvent

import advisor_memory as mem
from prompts import (
    COORDINATOR_PROMPT,
    DATA_ANALYST_PROMPT,
    EXECUTION_ANALYST_PROMPT,
    RISK_ANALYST_PROMPT,
    TRADING_ANALYST_PROMPT,
)

MODEL_NAME = os.environ.get("ADVISOR_OPENAI_MODEL", "")


def _model_kwargs() -> dict[str, Any]:
    if "/" in MODEL_NAME:
        from agents.extensions.models.litellm_model import LitellmModel  # noqa: PLC0415

        return {"model": LitellmModel(model=MODEL_NAME)}
    return {"model": MODEL_NAME} if MODEL_NAME else {}


def _analyst(name: str, instructions: str, tools: list | None = None) -> Agent:
    return Agent(name=name, instructions=instructions, tools=tools or [],
                 model_settings=ModelSettings(temperature=0.2), **_model_kwargs())


data_analyst = _analyst("data_analyst", DATA_ANALYST_PROMPT, [WebSearchTool()])
trading_analyst = _analyst("trading_analyst", TRADING_ANALYST_PROMPT)
execution_analyst = _analyst("execution_analyst", EXECUTION_ANALYST_PROMPT)
risk_analyst = _analyst("risk_analyst", RISK_ANALYST_PROMPT)
_inner = RunConfig(tool_not_found_behavior="return_error_to_model")


async def _report(analyst: Agent, brief: str) -> str:
    result = await Runner.run(analyst, brief, max_turns=12, run_config=_inner)
    return str(result.final_output or "")


@function_tool
async def analyze_market(ticker: str) -> str:
    """The data analyst: a market analysis report for one ticker, from recent web sources.

    Args:
        ticker: The stock market ticker symbol to analyze, e.g. AAPL, GOOGL, MSFT.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return "Error: a ticker symbol is required."
    report = await _report(data_analyst, f"provided_ticker: {ticker}\nmax_data_age_days: 7\ntarget_results_count: 10")
    mem.write(mem.TICKER, ticker)
    mem.write(mem.MARKET_ANALYSIS, report)
    return report


@function_tool
async def propose_strategies(risk_attitude: str, investment_period: str) -> str:
    """The trading analyst: five or more strategies for the analyzed ticker and this profile.
    Reads the market analysis itself; call analyze_market first.

    Args:
        risk_attitude: The user's attitude to risk, e.g. conservative, moderate, aggressive.
        investment_period: The user's timeframe, e.g. short-term, medium-term, long-term.
    """
    if problem := mem.missing(("market analysis (market_data_analysis_output)", mem.MARKET_ANALYSIS)):
        return problem
    mem.write(mem.RISK_ATTITUDE, risk_attitude.strip())
    mem.write(mem.INVESTMENT_PERIOD, investment_period.strip())
    report = await _report(
        trading_analyst,
        f"user_risk_attitude: {risk_attitude}\nuser_investment_period: {investment_period}\n\n"
        f"market_data_analysis_output:\n{mem.read(mem.MARKET_ANALYSIS)}",
    )
    mem.write(mem.STRATEGIES, report)
    return report


@function_tool
async def plan_execution(selected_strategy: str = "", execution_preferences: str = "") -> str:
    """The execution analyst: how to enter, hold, scale and exit the chosen strategy. Reads the
    proposed strategies, the risk attitude and the period itself; call propose_strategies first.

    Args:
        selected_strategy: The strategy the user chose, by name or description; empty to plan for the set.
        execution_preferences: The user's execution preferences, such as broker or order types; empty when none.
    """
    if problem := mem.missing(("trading strategies (proposed_trading_strategies_output)", mem.STRATEGIES)):
        return problem
    mem.write(mem.EXECUTION_PREFERENCES, execution_preferences.strip())
    chosen = selected_strategy.strip() or "(none selected: plan for the proposed strategies as a set)"
    report = await _report(
        execution_analyst,
        f"provided_trading_strategy: {chosen}\nuser_risk_attitude: {mem.read(mem.RISK_ATTITUDE)}\n"
        f"user_investment_period: {mem.read(mem.INVESTMENT_PERIOD)}\n"
        f"user_execution_preferences: {execution_preferences or '(none stated)'}\n\n"
        f"proposed_trading_strategies_output:\n{mem.read(mem.STRATEGIES)}",
    )
    mem.write(mem.EXECUTION_PLAN, report)
    return report


@function_tool
async def assess_risk() -> str:
    """The risk analyst: the whole plan, analysis to execution, against the user's profile.
    Takes no input; reads the three earlier reports and the profile itself. Call it last."""
    if problem := mem.missing(
        ("market analysis (market_data_analysis_output)", mem.MARKET_ANALYSIS),
        ("trading strategies (proposed_trading_strategies_output)", mem.STRATEGIES),
        ("execution plan (execution_plan_output)", mem.EXECUTION_PLAN),
    ):
        return problem
    report = await _report(
        risk_analyst,
        f"user_risk_attitude: {mem.read(mem.RISK_ATTITUDE)}\nuser_investment_period: {mem.read(mem.INVESTMENT_PERIOD)}\n"
        f"user_execution_preferences: {mem.read(mem.EXECUTION_PREFERENCES) or '(none stated)'}\n\n"
        f"market_data_analysis_output:\n{mem.read(mem.MARKET_ANALYSIS)}\n\n"
        f"provided_trading_strategy:\n{mem.read(mem.STRATEGIES)}\n\n"
        f"provided_execution_strategy:\n{mem.read(mem.EXECUTION_PLAN)}",
    )
    mem.write(mem.RISK_EVALUATION, report)
    return report


ANALYSTS = [analyze_market, propose_strategies, plan_execution, assess_risk]


def _coordinator_instructions(_ctx, _agent) -> str:
    return COORDINATOR_PROMPT + mem.progress_block()


coordinator = Agent(
    name="financial_coordinator",
    instructions=_coordinator_instructions,
    tools=ANALYSTS,
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
)
_run_config = RunConfig(tool_not_found_behavior="return_error_to_model")

agent_app = AgentApp(
    name="Financial advisor (OpenAI Agents)",
    description="Guides you from a market ticker to a risk-evaluated plan: market analysis, "
    "trading strategies, an execution plan and a risk evaluation, each by its own analyst "
    "(OpenAI Agents SDK, web search).",
    framework="openai_agents",
    welcome_message="Hello! I'm here to help you navigate the world of financial "
    "decision-making, one step at a time: a ticker, strategies for your profile, an "
    "execution plan, and the risks. Ready to get started?",
    suggested_prompts=[
        "Hello. What can you do for me?",
        "Analyze NVDA for me.",
        "I'm a conservative, long-term investor. What strategies fit MSFT?",
    ],
    placeholder="A ticker to analyze, or the next step...",
    memory=ManagedMemoryService(summarize=anthropic_summarizer(), max_state_value_chars=mem.MAX_REPORT_CHARS),
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
    """One handler serves both endpoints: the coordinator's text streams to the user and each
    analyst it calls shows as a progress chip; the analysts' own runs are inner and never stream."""
    if not request.text:
        raise AgentError("The message content cannot be empty.", code="invalid_request", status_code=400)
    result = Runner.run_streamed(coordinator, _input_items(ctx, request), max_turns=10, run_config=_run_config)
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
        yield final or "Sorry — I wasn't able to put together a reply just then. Could you try rephrasing?"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
