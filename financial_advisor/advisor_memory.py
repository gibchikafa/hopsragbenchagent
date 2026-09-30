"""The reports, in the conversation's working memory, shared by both entry points.

The recipe passes each analyst's report to the next through session-state
keys (``output_key``). Here the same keys live in the SDK's ``session`` scope:
this conversation only, which is what a plan is; a new chat starts over, as
the recipe's session would. Both the LangGraph and the OpenAI Agents SDK
entry points read and write them through this module.
"""

from __future__ import annotations

from hopsworks_agents.protocol.autoevents import current_context

#: a report is a few thousand tokens; the default 4k characters would cut one
MAX_REPORT_CHARS = 32_000
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
    """The store and this conversation's id, resolved the way the SDK's own tools do."""
    ctx = current_context.get(None)
    if ctx is None or ctx.memory is None:
        return None, None
    return ctx.memory, ctx.conversation_id


def read(key: str) -> str:
    memory, owner = _memory_and_owner()
    if memory is None:
        return ""
    return memory.get_state(SCOPE, owner, key) or ""


def write(key: str, value: str) -> None:
    memory, owner = _memory_and_owner()
    if memory is None:
        return
    ctx = current_context.get(None)
    memory.set_state(
        SCOPE, owner, key, value[:MAX_REPORT_CHARS],
        source_ref=f'{{"conversation_id": "{owner}", "turn_id": "{ctx.turn_id}"}}',
    )


def missing(*needed: tuple[str, str]) -> str:
    """The recipe's prerequisite check, for every analyst after the first."""
    absent = [name for name, key in needed if not read(key)]
    if not absent:
        return ""
    return (
        "Error: the foundational input(s) " + ", ".join(absent)
        + " are missing. The earlier step(s) must be completed first; run them, then call this analyst again."
    )


def progress_block() -> str:
    """Which reports exist, so a resumed conversation continues rather than starting over."""
    done = [
        label
        for label, key in (
            ("market analysis", MARKET_ANALYSIS), ("trading strategies", STRATEGIES),
            ("execution plan", EXECUTION_PLAN), ("risk evaluation", RISK_EVALUATION),
        )
        if read(key)
    ]
    if not done:
        return ""
    profile = ", ".join(
        f"{name}: {read(key)}"
        for name, key in (("ticker", TICKER), ("risk attitude", RISK_ATTITUDE), ("investment period", INVESTMENT_PERIOD))
        if read(key)
    )
    return (
        "\n\nProgress in this conversation: the " + ", ".join(done)
        + " report(s) are already done and kept for the later analysts"
        + (f" ({profile})" if profile else "")
        + ". Continue from the next step; do not repeat a completed one unless asked."
    )
