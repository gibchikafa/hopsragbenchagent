"""Code checks for the cyber guardian: duplicates, severities and the incident reported.

Uploaded by ``apply.py`` and named by the ``guardian_checks`` library entry; every
check runs under its own name and none reads a task expectation.
"""

from __future__ import annotations

from hopsworks_agents.eval.checks import CheckContext, CheckResult, check, ungradable


SEVERITIES = {"critical", "high", "medium", "low"}


def _result(call) -> dict:
    parsed = call.result_json
    return parsed if isinstance(parsed, dict) else {}


def _arguments(call) -> dict:
    parsed = call.arguments_json
    return parsed if isinstance(parsed, dict) else {}


@check(description="A duplicate alert opened no new incident")
def duplicate_alert_opens_no_incident(ctx: CheckContext):
    triaged = ctx.calls("triage_query")
    if not triaged:
        return ungradable("no alert was triaged in this turn")
    if not any(_result(c).get("is_duplicate") for c in triaged):
        return ungradable("the alert was not a duplicate")
    ok = not ctx.calls("create_incident")
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else "triage said duplicate and an incident was created anyway",
        assertions={"existing": [_result(c).get("existing_incident") for c in triaged]},
    )


@check(description="Every incident was created with a severity the playbooks know")
def incident_severity_is_valid(ctx: CheckContext):
    incidents = ctx.calls("create_incident")
    if not incidents:
        return ungradable("no incident was created in this turn")
    bad = [
        _arguments(c).get("severity")
        for c in incidents
        if str(_arguments(c).get("severity") or "").lower() not in SEVERITIES
    ]
    ok = not bad
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"unknown severities {bad}",
        assertions={"severities": [_arguments(c).get("severity") for c in incidents]},
    )


@check(description="The incident id the tool returned is in the answer")
def incident_id_reported(ctx: CheckContext):
    ids = [_result(c).get("incident_id") for c in ctx.calls("create_incident")]
    ids = [i for i in ids if i]
    if not ids:
        return ungradable("no incident was created in this turn")
    missing = [i for i in ids if i not in ctx.answer]
    ok = not missing
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"the answer never names {missing}",
        assertions={"incidents": ids},
    )
