"""
Financial advisor on the Claude Agent SDK: the same prompts and working memory, a
different agent runtime.

The coordinator is one Agent SDK run per turn. Its tools -- the four analysts, the
memory tools and `identify` -- sit on an in-process MCP server, and every built-in
Claude Code tool is switched off: this is an advisor, not a repo-editing agent.
Each analyst is its own Agent SDK run with the recipe's prompt for that analyst; the
data analyst carries the SDK's built-in `WebSearch` tool, the analogue of the
recipe's Google Search grounding, capped by a hook at ADVISOR_MAX_SEARCHES.

Reports go through the conversation's working memory exactly as in
`advisor_agent.py`, so the three entry points are interchangeable on the same
deployment history: an analyst's report is several thousand tokens and never
travels through a tool call.

Deploy (git-backed, so `prompts.py`, `advisor_memory.py` and `demo_users.py` come
along). The runtime is the Claude Code CLI, which the `claude-agent-sdk` wheel bundles;
it signs in with `ANTHROPIC_API_KEY`, or with the Claude Code login the deployment
form connects (`CLAUDE_CODE_OAUTH_TOKEN`):
    agents.deploy_agent("financial_advisor/advisor_agent_claude.py", name="financialadvisor-claude",
                        git_url=..., git_provider="GitHub", git_branch="main",
                        environment="python-agent-pipeline",
                        # Priced, or every traced turn costs 0.0, which reads as free
                        # rather than as unpriced. List prices per million tokens for
                        # claude-opus-5-5, the default ADVISOR_CLAUDE_MODEL, when this
                        # was written; change them when you change the model.
                        tracing={"enabled": True,
                                 "input_token_price_per_million_tokens": 4.0,
                                 "output_token_price_per_million_tokens": 20.0}).start()
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
    identity_tools,
    memory_tools,
)

import advisor_memory as mem
import demo_users
from prompts import (
    COORDINATOR_PROMPT,
    DATA_ANALYST_PROMPT,
    EXECUTION_ANALYST_PROMPT,
    RISK_ANALYST_PROMPT,
    TRADING_ANALYST_PROMPT,
)

#: The coordinator and the analysts. An alias the CLI knows ("opus", "sonnet") works too.
MODEL = os.environ.get("ADVISOR_CLAUDE_MODEL", "claude-opus-5-5")
#: how many web searches the data analyst may run per report; each is a paid call
MAX_SEARCHES = int(os.environ.get("ADVISOR_MAX_SEARCHES", "8"))
#: The in-process MCP server the coordinator's tools live on. The SDK names a tool
#: on it `mcp__<server>__<tool>`, which is how it shows in a trace.
SERVER = "advisor"


# ── running the SDK ──────────────────────────────────────────────────────────


def _options(system: str, *, tools: list[str] = (), **more: Any) -> ClaudeAgentOptions:
    """Options for one run: the prompt, the model, and no Claude Code tools or settings.

    `tools=[]` removes every built-in (Bash, Read, Edit, ...), and `setting_sources=[]`
    keeps a CLAUDE.md or a hook from the pod's filesystem out of the run. What a run
    may call is added by the caller: a built-in by name here, an MCP server in `more`.
    """
    return ClaudeAgentOptions(
        system_prompt=system, model=MODEL, tools=list(tools), setting_sources=[], **more
    )


def _search_cap(limit: int):
    """A PreToolUse hook that lets `WebSearch` run `limit` times and then refuses it.

    `max_turns` would also stop the run, but by ending it: the report is then the
    error. A refused search tells the model why and leaves it to write from what
    it has read.
    """
    used = 0

    async def hook(_input: dict, _tool_use_id: str | None, _context: dict) -> dict:
        nonlocal used
        used += 1
        if used <= limit:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"The budget of {limit} searches for this report is spent. "
                    "Write the report from what you have already read."
                ),
            }
        }

    return hook


def _researcher_options() -> ClaudeAgentOptions:
    """The data analyst's run: the web search tool, and nothing else."""
    return _options(
        DATA_ANALYST_PROMPT,
        tools=["WebSearch"],
        allowed_tools=["WebSearch"],
        hooks={"PreToolUse": [HookMatcher(matcher="WebSearch", hooks=[_search_cap(MAX_SEARCHES)])]},
        # a turn per search, the report, and room for a refused one
        max_turns=MAX_SEARCHES + 4,
    )


def _text_of(blocks: Any) -> str:
    """The text of a message or a tool result whose content may be blocks."""
    if isinstance(blocks, str):
        return blocks
    parts = []
    for block in blocks or ():
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


async def _run(prompt: str, options: ClaudeAgentOptions) -> str:
    """One run to completion, returning its final text.

    The result message carries the final answer; a run that ended in an error (the
    turn cap, a refused request) carries the error there instead, so the last thing
    the model wrote is what is returned then.
    """
    final = ""
    last = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            text = _text_of(message.content).strip()
            if text:
                last = text
        elif isinstance(message, ResultMessage) and not message.is_error:
            final = str(message.result or "")
    return (final or last).strip()


async def _analyst(system: str, brief: str, options: ClaudeAgentOptions | None = None) -> str:
    return await _run(brief, options or _options(system))


# ── the analysts ─────────────────────────────────────────────────────────────
#
# Plain coroutines, so the wiring test can call them without the SDK; `_as_tool`
# below puts each on the MCP server with its docstring as the description.


async def analyze_market(ticker: str, focus: str = "") -> str:
    """The data analyst: a market analysis report for one ticker, from recent web sources.

    Args:
        ticker: The stock market ticker symbol to analyze, e.g. AAPL, GOOGL, MSFT.
        focus: What the user wants the analysis to look at in particular, in their words, e.g. "trading around earnings"; empty for a general analysis.
    """
    if problem := mem.not_this_turn("analysis"):
        return problem
    ticker = ticker.strip().upper()
    if not ticker:
        return "Error: a ticker symbol is required."
    brief = f"provided_ticker: {ticker}\nmax_data_age_days: 7\ntarget_results_count: 10"
    if focus.strip():
        brief += f"\nuser_focus: {focus.strip()} -- cover this specifically"
    report = await _analyst(DATA_ANALYST_PROMPT, brief, _researcher_options())
    mem.write(mem.TICKER, ticker)
    mem.write(mem.MARKET_ANALYSIS, report)
    return report


async def propose_strategies(risk_attitude: str = "", investment_period: str = "") -> str:
    """The trading analyst: five or more strategies for the analyzed ticker and this profile.
    Reads the market analysis itself; call analyze_market first.

    Args:
        risk_attitude: The user's attitude to risk, e.g. conservative, moderate, aggressive; empty to use the one already known.
        investment_period: The user's timeframe, e.g. short-term, medium-term, long-term; empty to use the one already known.
    """
    if problem := mem.not_this_turn("strategies"):
        return problem
    if problem := mem.missing(("market analysis (market_data_analysis_output)", mem.MARKET_ANALYSIS)):
        return problem
    # a profile stated this turn was captured before the model ran; one from before is in memory
    risk_attitude = risk_attitude.strip() or mem.read(mem.RISK_ATTITUDE)
    investment_period = investment_period.strip() or mem.read(mem.INVESTMENT_PERIOD)
    if not (risk_attitude and investment_period):
        return ("Not run: the risk attitude and the investment period are both needed. "
                "Ask the user for whichever is missing.")
    mem.write(mem.RISK_ATTITUDE, risk_attitude)
    mem.write(mem.INVESTMENT_PERIOD, investment_period)
    report = await _analyst(
        TRADING_ANALYST_PROMPT,
        f"user_risk_attitude: {risk_attitude}\nuser_investment_period: {investment_period}\n\n"
        f"market_data_analysis_output:\n{mem.read(mem.MARKET_ANALYSIS)}",
    )
    mem.write(mem.STRATEGIES, report)
    return report


async def plan_execution(selected_strategy: str = "", execution_preferences: str = "") -> str:
    """The execution analyst: how to enter, hold, scale and exit the chosen strategy. Reads the
    proposed strategies, the risk attitude and the period itself; call propose_strategies first.

    Args:
        selected_strategy: The strategy the user chose, by name or description; empty to plan for the set.
        execution_preferences: Only what the user said about how to trade -- broker, order types, how to scale in or out -- in their words; empty when they said nothing. Not the strategy, not their goals: this is stored as their preference for next time.
    """
    if problem := mem.not_this_turn("plan"):
        return problem
    if problem := mem.missing(("trading strategies (proposed_trading_strategies_output)", mem.STRATEGIES)):
        return problem
    mem.write(mem.EXECUTION_PREFERENCES, execution_preferences.strip())
    chosen = selected_strategy.strip() or "(none selected: plan for the proposed strategies as a set)"
    report = await _analyst(
        EXECUTION_ANALYST_PROMPT,
        f"provided_trading_strategy: {chosen}\nuser_risk_attitude: {mem.read(mem.RISK_ATTITUDE)}\n"
        f"user_investment_period: {mem.read(mem.INVESTMENT_PERIOD)}\n"
        f"user_execution_preferences: {execution_preferences or '(none stated)'}\n\n"
        f"proposed_trading_strategies_output:\n{mem.read(mem.STRATEGIES)}",
    )
    mem.write(mem.EXECUTION_PLAN, report)
    return report


async def assess_risk() -> str:
    """The risk analyst: the whole plan, analysis to execution, against the user's profile.
    Takes no input; reads the three earlier reports and the profile itself. Call it last."""
    if problem := mem.not_this_turn("risk"):
        return problem
    if problem := mem.missing(
        ("market analysis (market_data_analysis_output)", mem.MARKET_ANALYSIS),
        ("trading strategies (proposed_trading_strategies_output)", mem.STRATEGIES),
        ("execution plan (execution_plan_output)", mem.EXECUTION_PLAN),
    ):
        return problem
    report = await _analyst(
        RISK_ANALYST_PROMPT,
        f"user_risk_attitude: {mem.read(mem.RISK_ATTITUDE)}\nuser_investment_period: {mem.read(mem.INVESTMENT_PERIOD)}\n"
        f"user_execution_preferences: {mem.read(mem.EXECUTION_PREFERENCES) or '(none stated)'}\n\n"
        f"market_data_analysis_output:\n{mem.read(mem.MARKET_ANALYSIS)}\n\n"
        f"provided_trading_strategy:\n{mem.read(mem.STRATEGIES)}\n\n"
        f"provided_execution_strategy:\n{mem.read(mem.EXECUTION_PLAN)}",
    )
    mem.write(mem.RISK_EVALUATION, report)
    return report


def _as_tool(fn):
    """An analyst as an SDK MCP tool: its docstring is the description, its Args the schema.

    Every argument is a string and the optional ones default to "", so the schema
    requires only what the signature does; the model may leave the rest out.
    """
    doc = inspect.getdoc(fn) or ""
    description, _, args = doc.partition("Args:")
    descriptions = {}
    for line in args.splitlines():
        name, sep, text = line.strip().partition(":")
        if sep and name.isidentifier():
            descriptions[name] = text.strip()
    params = inspect.signature(fn).parameters
    schema = {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": descriptions.get(name, name)}
            for name in params
        },
        "required": [n for n, p in params.items() if p.default is inspect.Signature.empty],
        "additionalProperties": False,
    }

    async def invoke(args: dict[str, Any]) -> dict[str, Any]:
        values = {name: str(args[name]) for name in params if name in args}
        try:
            text = await fn(**values)
        except Exception as err:  # noqa: BLE001 -- the model is told, and explains
            return {"content": [{"type": "text", "text": f"Error: {err}"}], "is_error": True}
        return {"content": [{"type": "text", "text": text}]}

    invoke.__name__ = fn.__name__
    return tool(fn.__name__, " ".join(description.split()), schema)(invoke)


#: Each analyst with the step it runs, in plan order.
ANALYSTS = [
    ("analysis", _as_tool(analyze_market)),
    ("strategies", _as_tool(propose_strategies)),
    ("plan", _as_tool(plan_execution)),
    ("risk", _as_tool(assess_risk)),
]
# The memory tools the model calls itself, and `identify`, which says whose memory this
# is: until it is called, what the agent remembers belongs to the conversation alone.
MEMORY_TOOLS = [*memory_tools("claude_agents"), *identity_tools("claude_agents")]


def _tools_this_turn() -> list:
    """The analysts this turn may call, and the memory tools.

    Offered to the model only on a turn that asked for their step, as the OpenAI
    entry point's `is_enabled` does: the guard in each analyst still refuses, but a
    refusal is a call, and it stays in the trace. Decided once at the start of the
    turn, which is when `not_this_turn` has everything it needs -- what the message
    asks for and where the plan stood -- so the set does not change mid-turn.
    """
    return [analyst for step, analyst in ANALYSTS if not mem.not_this_turn(step)] + MEMORY_TOOLS


# ── the coordinator ──────────────────────────────────────────────────────────

# What the chat panel's Graph tab shows: the recipe's picture, with the order the
# reports flow in.
GRAPH = {
    "nodes": [
        {"id": "__start__", "label": "__start__"},
        {"id": "coordinator", "label": "financial_coordinator"},
        {"id": "analyze_market", "label": "data_analyst"},
        {"id": "web_search", "label": "WebSearch"},
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
    name="Financial advisor (Claude Agent SDK)",
    description="Guides you from a market ticker to a risk-evaluated plan: market analysis, "
    "trading strategies, an execution plan and a risk evaluation, each by its own analyst "
    "(Claude Agent SDK coordinator; analysts on in-process MCP tools, web search).",
    framework="claude_agents",
    # No question about who they are: in the chat panel and the chat app the person is
    # already identified by Hopsworks (a member by their username, a visitor by their
    # own id), and anyone else is asked by the agent itself (demo_users.who_first).
    welcome_message="Hello! I'm here to help you navigate the world of financial "
    "decision-making, one step at a time: a ticker, strategies for your profile, an "
    "execution plan, and the risks. Which ticker shall we start with?",
    suggested_prompts=[
        "Analyze NVDA for me.",
        "I'm a conservative, long-term investor. What strategies fit MSFT?",
        "How would I execute a covered-call strategy on AAPL, and what are the risks?",
    ],
    placeholder="A ticker to analyze, or the next step...",
    memory=ManagedMemoryService(
        summarize=anthropic_summarizer(),
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
    graph=GRAPH,
)


def _prompt(ctx, request) -> str:
    """This turn's message, after the turns before it.

    Every run is a new session for the SDK, so the conversation goes in with the
    message: the kept turns as a transcript, the folded ones as the summary the
    SDK puts in the system prompt (`ctx.system_context`).
    """
    lines = []
    for turn in ctx.history:
        content = str(turn.get("content") or "").strip()
        if content:
            role = "user" if turn.get("role") == "user" else "assistant"
            lines.append(f"{role}: {content}")
    if not lines:
        return request.text
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nuser: {request.text}"


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints: the coordinator's text streams to the user and each
    analyst it calls shows as a progress chip; the analysts' own runs are separate and never stream."""
    if not request.text:
        raise AgentError("The message content cannot be empty.", code="invalid_request", status_code=400)
    # Before the prompt is built, so this turn is already bound: `identify` labels the
    # conversation but never keys memory, which is the platform's decision and the right
    # one -- so binding is the application's job, and in this demo that is here. See
    # demo_users.bind_if_identified for why taking the address as typed is acceptable
    # only because these three people are fictional.
    demo_users.bind_if_identified(ctx, request.text)
    if question := demo_users.who_first(ctx):
        yield question
        return
    # After binding, so a profile stated here is filed under the person: the strategies
    # step then reads it rather than the model asking for what it was just told.
    mem.capture_profile(request.text)
    mem.begin_turn()

    # Who this is goes in the system prompt, beside the rule it overrides; and where the
    # process stands, so a resumed conversation continues rather than starting over.
    system = (COORDINATOR_PROMPT + ctx.system_context()
              + demo_users.already_identified(ctx) + mem.progress_block())
    tools = _tools_this_turn()
    options = _options(
        system,
        mcp_servers={SERVER: create_sdk_mcp_server(name=SERVER, version="1.0.0", tools=tools)},
        allowed_tools=[f"mcp__{SERVER}__{item.name}" for item in tools],
        strict_mcp_config=True,
        max_turns=10,
        include_partial_messages=True,
    )

    spoken = ""
    streamed = False
    # The model's text before a tool call and after it are two messages; a blank
    # line between them, or they run together mid-sentence.
    paragraph_break = False
    called: dict[str, str] = {}  # tool use id -> tool name, to name a report when it returns
    final = ""
    async for message in query(prompt=_prompt(ctx, request), options=options):
        if isinstance(message, StreamEvent):
            event = message.event or {}
            if event.get("type") == "message_start":
                paragraph_break = bool(spoken.strip())
                continue
            delta = event.get("delta") or {}
            if event.get("type") != "content_block_delta" or delta.get("type") != "text_delta":
                continue
            text = delta.get("text") or ""
            if not text:
                continue
            if paragraph_break and not spoken.endswith("\n\n"):
                spoken += "\n\n"
                yield "\n\n"
            paragraph_break = False
            spoken += text
            streamed = True
            yield text
        elif isinstance(message, AssistantMessage):
            # its text streamed above; what is left to read here is the tool calls
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    name = block.name.rsplit("__", 1)[-1]
                    called[block.id] = name
                    await ctx.emit_event(name, status="running",
                                         message=str(block.input or "")[:200], event_id=block.id)
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                name = called.get(block.tool_use_id, "")
                await ctx.emit_event(name or "tool", status="done", event_id=block.tool_use_id)
                # The report itself, on the screen as soon as the analyst returns: see
                # advisor_memory.presented for why this is not left to the model.
                report = mem.presented(name, _text_of(block.content))
                if report:
                    block_text = ("\n\n" if spoken.strip() and not spoken.endswith("\n\n") else "") + report
                    spoken += block_text
                    streamed = True
                    paragraph_break = True
                    yield block_text
        elif isinstance(message, ResultMessage) and not message.is_error:
            final = str(message.result or "").strip()

    if not streamed:
        yield final or "Sorry — I wasn't able to put together a reply just then. Could you try rephrasing?"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
