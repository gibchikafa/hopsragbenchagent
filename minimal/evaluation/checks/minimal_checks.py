"""Checks for the minimal agent, read by the evaluation job from this repository.

Named by a suite row as ``git: {"url": <this repository>, "ref": "main"}`` or,
for a deployment made from this repository, ``git: "deployment"``, with
``path: minimal/evaluation/checks/minimal_checks.py``.
"""

from __future__ import annotations

from hopsworks_agents.eval.checks import CheckContext, CheckResult, check, ungradable


@check(needs_trace=False, description="The answer opens the way this agent answers")
def answers_in_kind(ctx: CheckContext, prefix: str = "echo"):
    ok = ctx.answer.lower().startswith(prefix.lower())
    return CheckResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason="" if ok else f"the answer does not start with {prefix!r}",
        assertions={"prefix": prefix},
    )


@check(needs_trace=True, description="The trial's trace arrived and is readable")
def trace_arrived(ctx: CheckContext):
    return CheckResult(
        passed=True,
        score=1.0,
        assertions={
            "tools": ctx.tool_names,
            "spans": (ctx.trace or {}).get("span_count"),
        },
    )


@check("mentions", needs_trace=False, reads_expectation=True)
def _mentions(ctx: CheckContext):
    if not ctx.expectation:
        return ungradable("the task expects nothing of this check")
    return ctx.expectation.lower() in ctx.answer.lower()
