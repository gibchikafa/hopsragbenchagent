"""The reports, in the conversation's working memory, shared by both entry points.

The recipe passes each analyst's report to the next through session-state
keys (``output_key``). Here the same keys live in the SDK's ``session`` scope:
this conversation only, which is what a plan is; a new chat starts over, as
the recipe's session would. Both the LangGraph and the OpenAI Agents SDK
entry points read and write them through this module.
"""

from __future__ import annotations

import re

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
            "\n\nThis is your stored long-term memory of this person, read back for you: "
            + known
            + ". You have it; you do not need to call `recall` to get it, and you may say "
            "you remember it. Treat it as theirs and say where it came from if asked; do "
            "not ask for it again, and do not state anything else about them as remembered."
        )
    # Named, because the model guessed. Told to `recall` but not told the keys, it asked
    # for "risk_attitude" and "investment_period", got nothing back, and told a returning
    # user it had nothing stored about them -- in the same turn the block above was telling
    # it exactly what it had stored. Either half alone produces that; both together do not.
    block += (
        "\n\nIf you do call `recall`, the keys this agent stores under are exactly: "
        + ", ".join((*ABOUT_THE_PERSON, TICKER, MARKET_ANALYSIS, STRATEGIES,
                     EXECUTION_PLAN, RISK_EVALUATION))
        + ". A key you invent returns nothing, which does not mean nothing is stored."
    )
    if done:
        ticker = read(TICKER)
        block += (
            "\n\nProgress in this conversation: the " + ", ".join(done)
            + " report(s) are already done and kept for the later analysts"
            + (f" (ticker: {ticker})" if ticker else "")
            + ". Continue from the next step; do not repeat a completed one unless asked."
        )
    return block + next_step_block()


# ── what this turn may do, decided in code ──────────────────────────────────
#
# Three rules the prompt stated and the model did not keep at temperature 0: ask
# only for what is missing (it asked again for a risk attitude it had just been
# given), one step per turn (it ran the risk evaluation when asked for the plan),
# and the ticker comes first (it asked for the profile after the introduction).
# A rule read once in a long prompt loses to whatever else the model weighs, so
# each is now enforced where the decision is made: the profile is captured from
# the message before the model runs, an analyst refuses a step this turn did not
# ask for, and the prompt is told the one next step for this state of the plan.


STEPS = ("analysis", "strategies", "plan", "risk")
_REPORT_OF = {
    "analysis": MARKET_ANALYSIS,
    "strategies": STRATEGIES,
    "plan": EXECUTION_PLAN,
    "risk": RISK_EVALUATION,
}
_STEP_NAME = {
    "analysis": "the market analysis",
    "strategies": "the trading strategies",
    "plan": "the execution plan",
    "risk": "the risk evaluation",
}
_ASKS_FOR = {
    "analysis": re.compile(r"\banaly[sz]", re.I),
    "strategies": re.compile(r"\bstrateg", re.I),
    "plan": re.compile(r"\bplan\b|\bexecut", re.I),
    # "risk attitude", "risk tolerance" and the like are the profile, not the step
    "risk": re.compile(
        r"(?<!-)\brisk\b(?!-|\s+(attitude|tolerance|appetite|profile|level))"
        r"|\bassess|\bevaluat", re.I),
}
_GO_ON = re.compile(
    r"^\W*(yes|yeah|yep|sure|ok(ay)?|please|go\s+ahead|continue|proceed|next|do\s+it|"
    r"sounds\s+good|let'?s\s+(go|do\s+it|continue))\b", re.I)

_ATTITUDES = (
    ("very conservative", r"very\s+conservative"), ("very aggressive", r"very\s+aggressive"),
    ("conservative", r"conservative|low[\s-]risk|risk[\s-]averse"),
    ("moderate", r"moderate|balanced|medium[\s-]risk"),
    ("aggressive", r"aggressive|high[\s-]risk"),
)
_PERIODS = (
    ("intraday", r"intraday|day[\s-]trad"),
    ("short-term", r"short[\s-]term"),
    ("medium-term", r"medium[\s-]term|mid[\s-]term|(a\s+few|several|\d+)\s+months"),
    ("long-term", r"long[\s-]term|(\d+|several|many)\s+years|retirement"),
)


def _turn_text() -> str:
    ctx = current_context.get(None)
    request = getattr(ctx, "request", None)
    return str(getattr(request, "text", "") or "")


def _first_match(text: str, table) -> str:
    for label, pattern in table:
        if re.search(rf"\b(?:{pattern})", text, re.I):
            return label
    return ""


def capture_profile(text: str) -> dict[str, str]:
    """Store the risk attitude and horizon the user just stated, before the model runs.

    So "I'm a conservative, long-term investor. Propose strategies for MSFT" is the
    profile and the request together: the strategies step reads it from memory, and
    the prompt says it is known, rather than the model asking for what it was told.
    Only what the message states plainly is taken; nothing is inferred.
    """
    found = {}
    attitude = _first_match(text, _ATTITUDES)
    period = _first_match(text, _PERIODS)
    if attitude:
        write(RISK_ATTITUDE, attitude)
        found["risk attitude"] = attitude
    if period:
        write(INVESTMENT_PERIOD, period)
        found["investment period"] = period
    return found


def next_step() -> str:
    """The first step of the plan whose report does not exist yet, or "" when all are done."""
    return next((step for step in STEPS if not read(_REPORT_OF[step])), "")


def not_this_turn(step: str, text: str | None = None) -> str:
    """Why an analyst must not run now, or "" when this turn asked for it.

    A step may run when the message asks for it, when it is the prerequisite of a
    step the message asks for (strategies for a ticker nobody analysed yet), or when
    the message is a go-ahead and it is the next step. Anything else is the agent
    running ahead -- the plan asked for and the risk evaluation run as well.
    """
    text = _turn_text() if text is None else text
    asked = {s for s in STEPS if _ASKS_FOR[s].search(text)}
    if asked:
        furthest = max(STEPS.index(s) for s in asked)
        if STEPS.index(step) <= furthest:
            return ""
    elif _GO_ON.search(text) and step == next_step():
        return ""
    elif not text.strip():
        return ""
    wanted = ", ".join(_STEP_NAME[s] for s in sorted(asked, key=STEPS.index)) or "nothing that needs an analyst"
    return (
        f"Not run: this turn asked for {wanted}, not {_STEP_NAME[step]}. One step per turn: "
        f"present what was asked, and offer {_STEP_NAME[step]} as the next step."
    )


def next_step_block() -> str:
    """The one thing to do next, for this state of the plan, told every turn."""
    step = next_step()
    profile = read(RISK_ATTITUDE) and read(INVESTMENT_PERIOD)
    if step == "analysis":
        return (
            "\n\nNext step: the market analysis. If the user has not given a ticker, ask for "
            "the ticker -- and only the ticker. Do not ask for a risk attitude or an "
            "investment period yet: they are needed for the strategies, which come after."
        )
    if step == "strategies":
        if profile:
            return (
                f"\n\nNext step: the trading strategies. The profile is known ({read(RISK_ATTITUDE)}, "
                f"{read(INVESTMENT_PERIOD)}), so do not ask for it: call propose_strategies when "
                "the user wants the strategies."
            )
        return (
            "\n\nNext step: the trading strategies. Ask for whichever of the risk attitude and "
            "the investment period you do not have yet, and nothing else."
        )
    if step == "plan":
        return "\n\nNext step: the execution plan, for the strategy the user chooses."
    if step == "risk":
        return "\n\nNext step: the risk evaluation of the whole plan."
    return ""
