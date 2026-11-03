"""Code checks for the financial advisor: the recipe's order, the disclaimer, the ticker.

Uploaded by ``apply.py`` and named by the ``advisor_checks`` library entry; every
check runs under its own name and none reads a task expectation.
"""

from __future__ import annotations

from hopsworks_agents.eval.checks import CheckContext, CheckResult, check, ungradable


ANALYSTS = ("analyze_market", "propose_strategies", "plan_execution", "assess_risk")
DISCLAIMER = "for educational and informational purposes only"


@check(description="A turn that ran an analyst carries the disclaimer in its answer")
def advice_carries_disclaimer(ctx: CheckContext):
    ran = [n for n in ctx.tool_names if n in ANALYSTS]
    if not ran:
        return ungradable("no analyst ran in this turn, so no advice was given")
    ok = DISCLAIMER in ctx.answer.lower()
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason=""
        if ok
        else f"{', '.join(ran)} ran but the answer carries no disclaimer",
        assertions={"analysts": ran},
    )


@check(
    description="Analysts ran in the recipe's order: market, strategies, execution, risk"
)
def analysts_in_recipe_order(ctx: CheckContext):
    ran = [n for n in ctx.tool_names if n in ANALYSTS]
    if len(ran) < 2:
        return ungradable("fewer than two analysts ran, so there is no order to check")
    ranks = [ANALYSTS.index(n) for n in ran]
    ok = ranks == sorted(ranks)
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"analysts ran as {ran}",
        assertions={"analysts": ran},
    )


@check(description="The ticker the analyst was asked about is named in the answer")
def ticker_named_in_answer(ctx: CheckContext):
    calls = ctx.calls("analyze_market")
    if not calls:
        return ungradable("no market analysis ran in this turn")
    arguments = calls[-1].arguments_json
    ticker = (
        str((arguments or {}).get("ticker") or "").strip().upper()
        if isinstance(arguments, dict)
        else ""
    )
    if not ticker:
        return ungradable("analyze_market was called without a readable ticker")
    ok = ticker in ctx.answer.upper()
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"the answer never names {ticker}",
        assertions={"ticker": ticker},
    )
