"""Checks this agent's suites grade with, written as code rather than as a judge.

Uploaded to the project by ``apply.py`` (``agents.checks.upload``), which also
saves one library entry per check declared here, so ``evaluators.json`` and
``suites.json`` can name them like any other check. Try one locally first::

    from hopsworks_agents.eval.checks import run_check
    run_check(order_total_is_quoted, answer="Done, that's $12.99",
              tools=[{"name": "place_order", "result": '{"total": 12.99}'}])
"""

from __future__ import annotations

import re

from hopsworks_agents.eval.checks import CheckContext, CheckResult, check, ungradable


_MONEY = re.compile(r"\d+(?:\.\d{1,2})?")


@check(
    needs_trace=True,
    description="When an order was placed, the answer quotes the total the tool returned",
)
def order_total_is_quoted(ctx: CheckContext, tolerance: float = 0.005):
    orders = ctx.calls("place_order")
    if not orders:
        return ungradable("no order was placed in this turn")
    result = orders[-1].result_json or {}
    total = result.get("total") if isinstance(result, dict) else None
    if total is None:
        return ungradable("place_order returned no total to compare against")
    quoted = [float(m) for m in _MONEY.findall(ctx.answer)]
    ok = any(abs(q - float(total)) <= tolerance for q in quoted)
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"the order total {total} is not in the answer",
        assertions={"total": total, "quoted": quoted},
    )


@check(needs_trace=True, description="Nothing was looked up before the customer was identified")
def identified_before_lookup(ctx: CheckContext):
    names = ctx.tool_names
    if "lookup" not in names:
        return ungradable("nothing was looked up in this turn")
    first_lookup = names.index("lookup")
    identified = "identify" in names[:first_lookup]
    return CheckResult(
        passed=identified,
        score=1.0 if identified else 0.0,
        reason="" if identified else "lookup ran before identify",
        assertions={"tools": names},
    )
