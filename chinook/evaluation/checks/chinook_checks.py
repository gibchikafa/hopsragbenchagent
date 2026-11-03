"""Code checks for the Chinook support agent: what the tools say, held against the answer.

Uploaded by ``apply.py`` to ``Resources/agent_evals/checks/`` and named by the
``chinook_checks`` library entry; every check here runs under its own name. None
reads a task expectation, so they grade suites and live traffic alike. Try one::

    from hopsworks_agents.eval.checks import run_check
    run_check(order_total_is_quoted, answer="Done, that's $12.99",
              tools=[{"name": "place_order", "result": "Order recorded: X by Y, $12.99."}])
"""

from __future__ import annotations

import re

from hopsworks_agents.eval.checks import CheckContext, CheckResult, check, ungradable


_MONEY = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
_LOOKUPS = ("lookup_track", "lookup_album", "lookup_artist", "purchase_history")


def _asked_to_buy(call) -> bool:
    arguments = call.arguments_json
    return isinstance(arguments, dict) and bool(arguments.get("customer_asked_to_buy"))


@check(
    description="When an order was recorded, the answer quotes the total the tool charged"
)
def order_total_is_quoted(ctx: CheckContext, tolerance: float = 0.005):
    recorded = [c for c in ctx.calls("place_order") if "Order recorded" in c.result]
    if not recorded:
        return ungradable("no order was recorded in this turn")
    totals = [float(m) for m in _MONEY.findall(recorded[-1].result)]
    if not totals:
        return ungradable(
            "place_order recorded an order without a total to compare against"
        )
    total = totals[-1]
    quoted = [float(m) for m in _MONEY.findall(ctx.answer)]
    ok = any(abs(q - total) <= tolerance for q in quoted)
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"the order total ${total:.2f} is not in the answer",
        assertions={"total": total, "quoted": quoted},
    )


@check(description="An order was placed only after the catalogue was looked up")
def catalogue_looked_up_before_ordering(ctx: CheckContext):
    names = ctx.tool_names
    orders = [
        i
        for i, c in enumerate(ctx.tools)
        if c.name == "place_order" and _asked_to_buy(c)
    ]
    if not orders:
        return ungradable("no order was placed in this turn")
    first_order = orders[0]
    looked_up = any(n in _LOOKUPS for n in names[:first_order])
    return CheckResult(
        passed=looked_up,
        score=1.0 if looked_up else 0.0,
        reason="" if looked_up else "place_order ran before any catalogue lookup",
        assertions={"tools": names},
    )


@check(description="Something recorded as an interest was not also ordered")
def interest_is_not_an_order(ctx: CheckContext):
    if not ctx.calls("remember_interest"):
        return ungradable("no interest was recorded in this turn")
    ordered = [c for c in ctx.calls("place_order") if _asked_to_buy(c)]
    ok = not ordered
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else "the turn recorded an interest and placed an order for it",
        assertions={"tools": ctx.tool_names},
    )
