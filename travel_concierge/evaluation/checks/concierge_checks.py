"""Code checks for the travel concierge: reservations, payments and what the answer says.

Uploaded by ``apply.py`` and named by the ``concierge_checks`` library entry; every
check runs under its own name and none reads a task expectation.
"""

from __future__ import annotations

import re

from hopsworks_agents.eval.checks import CheckContext, CheckResult, check, ungradable


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _result(call) -> dict:
    parsed = call.result_json
    return parsed if isinstance(parsed, dict) else {}


def _arguments(call) -> dict:
    parsed = call.arguments_json
    return parsed if isinstance(parsed, dict) else {}


@check(description="Every payment was for a reservation this turn created")
def payment_only_for_a_reservation(ctx: CheckContext):
    payments = ctx.calls("process_payment")
    if not payments:
        return ungradable("nothing was paid for in this turn")
    reserved = {
        _result(c).get("reservation_id") for c in ctx.calls("create_reservation")
    }
    paid_for = [_arguments(c).get("reservation_id") for c in payments]
    unknown = [r for r in paid_for if r not in reserved]
    ok = not unknown
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"paid for {unknown}, which this turn never reserved",
        assertions={"reserved": sorted(filter(None, reserved)), "paid_for": paid_for},
    )


@check(description="The price of what was reserved is quoted in the answer")
def reservation_price_quoted(ctx: CheckContext):
    reservations = ctx.calls("create_reservation")
    if not reservations:
        return ungradable("nothing was reserved in this turn")
    prices = {int(_arguments(c).get("price_in_usd") or 0) for c in reservations}
    prices.discard(0)
    if not prices:
        return ungradable("create_reservation was called without a readable price")
    quoted = {int(float(m.replace(",", ""))) for m in _NUMBER.findall(ctx.answer)}
    missing = sorted(p for p in prices if p not in quoted)
    ok = not missing
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"the answer does not quote {missing}",
        assertions={"prices": sorted(prices)},
    )


@check(description="A declined payment is said to the traveler, not glossed over")
def declined_payment_is_said(ctx: CheckContext):
    declined = [
        c
        for c in ctx.calls("process_payment")
        if _result(c).get("status") == "declined"
    ]
    if not declined:
        return ungradable("no payment was declined in this turn")
    text = ctx.answer.lower()
    ok = "declin" in text or "another" in text or "different" in text
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else "a payment was declined and the answer does not say so",
        assertions={
            "declined": [_arguments(c).get("payment_method") for c in declined]
        },
    )
