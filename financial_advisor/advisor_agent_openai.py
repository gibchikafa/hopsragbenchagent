"""
Financial advisor on the OpenAI Agents SDK: the same prompts and working memory, a
different agent runtime.

The coordinator is an `Agent` whose tools run the four analysts, each an
`Agent` with the recipe's prompt; the data analyst carries the SDK's web
search tool, the analogue of the recipe's Google Search grounding. Reports go
through the conversation's working memory exactly as in `advisor_agent.py`,
so the two entry points are interchangeable on the same deployment history.

Deploy (git-backed, so `prompts.py` and `advisor_memory.py` come along):
    agents.deploy_agent("financial_advisor/advisor_agent_openai.py", name="financialadvisor-openai",
                        # Priced, or every traced turn costs 0.0, which reads as free
                        # rather than as unpriced. ADVISOR_OPENAI_MODEL picks the model,
                        # so set the pair that goes with it.
                        tracing={"enabled": True,
                                 "input_token_price_per_million_tokens": 2.0,
                                 "output_token_price_per_million_tokens": 8.0}, ...)
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
    identity_tools,
    memory_tools,
    openai_summarizer,
)
from openai.types.responses import ResponseTextDeltaEvent

import advisor_memory as mem
import demo_users
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
    # the memory tools the model calls itself, and `identify`, which says whose
    # memory this is
    tools=[*ANALYSTS, *memory_tools("openai_agents"), *identity_tools("openai_agents")],
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
    "execution plan, and the risks. What is your email address?",
    suggested_prompts=[
        "dana@example.com",
        "Analyze NVDA for me.",
        "I'm a conservative, long-term investor. What strategies fit MSFT?",
    ],
    placeholder="A ticker to analyze, or the next step...",
    memory=ManagedMemoryService(
        summarize=openai_summarizer(),
        long_term=True,
        max_state_value_chars=mem.MAX_REPORT_CHARS,
    ),
    # Lets a red-team suite run against this deployment. The runner refuses a
    # sandboxed suite -- one full of injection attempts -- unless the agent
    # says its tools are safe to point it at. Here they are: the analysts only
    # read (web search and the reports they pass each other), and the only
    # writes are to this agent's own memory, under whatever subject the trial
    # identifies as. The red-team suite identifies as redteam@example.com for
    # exactly that reason, so nothing it is talked into remembering lands on a
    # demo user.
    eval_per_request=True,
    tool_events=True,
)


def _input_items(ctx, request) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    # the identity the client asserted, when there is one: a real deployment has a login
    # and does not ask
    summary = ctx.system_context() + demo_users.already_identified(ctx)
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
    # Before the prompt is built, so this turn is already bound: `identify` labels the
    # conversation but never keys memory, which is the platform's decision and the right
    # one -- so binding is the application's job, and in this demo that is here. See
    # demo_users.bind_if_identified for why taking the address as typed is acceptable
    # only because these three people are fictional.
    demo_users.bind_if_identified(ctx, request.text)
    result = Runner.run_streamed(coordinator, _input_items(ctx, request), max_turns=10, run_config=_run_config)
    streamed = False
    # One turn can produce several message items, and this model often produces
    # two that say the same thing -- the reply arrived doubled, as
    # "...identified.Before we get started...identified.", in the stream, in
    # what the judges were shown and in the stored assistant turn.
    #
    # Grouped by item, because that is the only boundary available while the
    # deltas are arriving: RunItemStreamEvents for a turn come after its raw
    # events, so waiting for message_output_created to separate two messages
    # cannot work -- both messages' deltas have been yielded by then.
    #
    # An item is (model call, output_index), not output_index alone: the index
    # restarts at 0 for every model call, so the sentence written before a tool
    # call and the answer written after it were both "item 0" and ran together
    # mid-sentence -- "...which you've provided.The market analyst has completed".
    # Each model call's first item streams live, after a blank line when text came
    # before it; any further item of the same call is held back and emitted at the
    # end only if it says something the reply does not already contain.
    call = 0
    live: dict[int, int] = {}  # model call -> the output_index that streams live
    spoken = ""
    held: dict[tuple[int, int], str] = {}
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if getattr(event.data, "type", "") == "response.created":
                call += 1
            elif isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                index = getattr(event.data, "output_index", 0)
                if call not in live:
                    live[call] = index
                    if spoken.strip():
                        spoken += "\n\n"
                        yield "\n\n"
                if index == live[call]:
                    streamed = True
                    spoken += event.data.delta
                    yield event.data.delta
                else:
                    held[(call, index)] = held.get((call, index), "") + event.data.delta
        elif isinstance(event, RunItemStreamEvent):
            raw = getattr(event.item, "raw_item", None)
            if event.name == "tool_called":
                await ctx.emit_event(getattr(raw, "name", "tool"), status="running",
                                     message=str(getattr(raw, "arguments", "") or "")[:200],
                                     event_id=getattr(raw, "call_id", None))
            elif event.name == "tool_output":
                await ctx.emit_event("tool", status="done",
                                     event_id=raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None))
    for _key, text in sorted(held.items()):
        # Said once: a held-back item that repeats what was streamed is this model
        # restating itself; one that adds something is a continuation.
        if not text.strip() or text.strip() in spoken:
            continue
        spoken += "\n\n" + text.strip()
        yield "\n\n" + text.strip()

    if not streamed:
        final = str(result.final_output or "").strip()
        yield final or "Sorry — I wasn't able to put together a reply just then. Could you try rephrasing?"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
