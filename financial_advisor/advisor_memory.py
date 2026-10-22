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
#: What the person said about themselves, rather than about this plan. Kept at
#: `user` scope so the next conversation starts knowing it: written to `session`
#: with the reports, a risk attitude the user had stated out loud was gone by
#: the next chat, and the agent filled the gap from the demo's own notes and
#: then said it had remembered it.
USER_SCOPE = "user"
MARKET_ANALYSIS = "market_data_analysis_output"
STRATEGIES = "proposed_trading_strategies_output"
EXECUTION_PLAN = "execution_plan_output"
RISK_EVALUATION = "risk_evaluation_output"
TICKER = "ticker"
RISK_ATTITUDE = "user_risk_attitude"
INVESTMENT_PERIOD = "user_investment_period"
EXECUTION_PREFERENCES = "user_execution_preferences"
#: The three above are the person's, not the plan's. One tuple, so a reader can
#: see which keys outlive the conversation, and so the two scopes cannot drift.
ABOUT_THE_PERSON = (RISK_ATTITUDE, INVESTMENT_PERIOD, EXECUTION_PREFERENCES)


def _scope_for(key: str) -> str:
    return USER_SCOPE if key in ABOUT_THE_PERSON else SCOPE


def read(key: str) -> str:
    """One report, or "" when the step that writes it has not run.

    A fact about the person is read from `user` scope, and from `session` as
    well: a conversation that stated it before this change, or one where the
    person was never identified, still has it there.
    """
    ctx = current_context.get(None)
    if ctx is None:
        return ""
    scope = _scope_for(key)
    value = ctx.recall(key, scope=scope) or ""
    if not value and scope == USER_SCOPE:
        value = ctx.recall(key, scope=SCOPE) or ""
    return value


def write(key: str, value: str) -> None:
    """Store one report for the steps after it.

    Through the context rather than through the store: the scope each key
    belongs to is decided in one place, and `session` scope is keyed by the
    conversation. The provenance this fact came from -- the conversation, the
    turn, the message -- is the context's default, so a report's own row says
    which exchange produced it.
    """
    ctx = current_context.get(None)
    if ctx is not None:
        ctx.remember(key, value[:MAX_REPORT_CHARS], scope=_scope_for(key))


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
    """What is already known: about the person, and about this conversation.

    Both halves, and each on its own. The profile used to be shown only
    alongside a finished report, so a new conversation with someone the agent
    had met before said nothing about them -- and the model, told to `recall`
    but not told under which keys, guessed `investment_profile`, got nothing
    and answered that it knew nothing about a person who had told it twice.
    What is remembered is put in front of it instead of left to be found.
    """
    known = ", ".join(
        f"{name}: {value}"
        for name, value in (
            ("risk attitude", read(RISK_ATTITUDE)),
            ("investment period", read(INVESTMENT_PERIOD)),
            ("execution preferences", read(EXECUTION_PREFERENCES)),
        )
        if value
    )
    done = [
        label
        for label, key in (
            ("market analysis", MARKET_ANALYSIS), ("trading strategies", STRATEGIES),
            ("execution plan", EXECUTION_PLAN), ("risk evaluation", RISK_EVALUATION),
        )
        if read(key)
    ]
    block = ""
    if known:
        block += (
            "\n\nWhat this person has told you before, which you remembered: "
            + known
            + ". Treat it as theirs and say where it came from if asked; do not ask "
            "for it again, and do not state anything else about them as remembered."
        )
    if done:
        ticker = read(TICKER)
        block += (
            "\n\nProgress in this conversation: the " + ", ".join(done)
            + " report(s) are already done and kept for the later analysts"
            + (f" (ticker: {ticker})" if ticker else "")
            + ". Continue from the next step; do not repeat a completed one unless asked."
        )
    return block
