"""Grade a sample of what the advisor actually told people.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m financial_advisor.evaluation.sample --agent financialadvisor

The online half. `apply.py` creates the suites — cases someone wrote down, with
an expected answer each — and this grades conversations that already happened,
where there is no expected answer at all.

Both matter and neither substitutes for the other. A suite cannot contain the
ticker nobody thought to ask about, which is most of what people bring. And
production cannot tell you whether a fix held, because the conversation that
broke may never come back. The loop between them is the point: something goes
wrong in production, you promote that trace to a task, and from then on a suite
defends against it.

Two kinds of judge are used here and the difference is the whole design. The
rubric judge in `rubric.md` reads the *reply*: is this advice, framed as
education, built on figures that came from a report? The evaluators already in
`evaluators.json` -- `Every figure came from a report`, `Steps were not
skipped`, `Advice stayed educational`, `Answered what was asked`, `Asked who it
is talking to` -- read one thing each, which is what makes a bad score point at
something. Run the broad one for a number to watch and a narrow one when the
number moves:

    python -m financial_advisor.evaluation.sample --agent financialadvisor
    python -m financial_advisor.evaluation.sample --agent financialadvisor \\
        --evaluator "Every figure came from a report"

The score this produces is **not** comparable with a suite's pass rate. One says
how the agent does on cases with declared answers, graded by the same checks
every time; the other is one judge's opinion on whatever traffic arrived.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUBRIC = Path(__file__).with_name("rubric.md")

#: The model the judge runs on. Not `ADVISOR_MODEL`: the agent's own model
#: grading the agent's own answers is the one judge whose agreement means
#: nothing, and a judge is a cheaper call than an analyst anyway.
JUDGE_MODEL = "claude-sonnet-5"
JUDGE_NAME = "Financial advisor reply quality"


def rubric() -> str:
    """The rubric, without its explanatory preamble.

    Everything above the horizontal rule explains why the file exists, which is
    for whoever edits it and not for the judge -- sending it would spend tokens
    telling the model about offline evaluation.
    """
    text = RUBRIC.read_text()
    _, _, body = text.partition("\n---\n")
    return (body or text).strip()


def ensure_judge(agents):
    """The rubric as a saved judge in the project's library.

    Saved under one name, so the library holds one of these and an edit to
    `rubric.md` lands on the next run rather than forking a second judge nobody
    can tell from the first. The rubric is its one criterion: production carries
    no expected answer, so the judge has to reach a verdict from the
    conversation and the rubric alone -- which is exactly what lets it grade
    live traffic, and why it is refused as a suite's judge.
    """
    existing = agents.evaluators.find(JUDGE_NAME)
    checks = [
        {
            "type": "llm_judge",
            "name": "reply_quality",
            "provider": "anthropic",
            "model": JUDGE_MODEL,
            "temperature": 0,
            "score_range": [1, 5],
            # No `expected_result` among the inputs: a live trace has no task to
            # read one from, and a judge configured to be shown one describes a
            # different run from the one that happened.
            "inputs": ["user_request", "agent_response", "rubric"],
            "criteria": {
                "follows_the_rubric": {"weight": 1, "description": rubric()},
            },
            "thresholds": {"pass_score": 4.0},
        },
    ]
    description = (
        "What a good reply from the financial advisor looks like, from "
        "financial_advisor/evaluation/rubric.md. Grades live traffic; no expected answer."
    )
    if existing is not None:
        # An edit, not a second entry under the same name.
        return agents.evaluators.update(
            existing, checks=checks, description=description
        )
    return agents.evaluators.save(JUDGE_NAME, checks, description=description)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True,
                        help="the agent deployment, by name or id")
    parser.add_argument("--evaluator", default=None,
                        help="a saved evaluator to grade with instead of the "
                             "rubric judge, e.g. 'Every figure came from a report'")
    parser.add_argument("--sample", type=int, default=25,
                        help="how many conversations to grade. Each one costs a "
                             "judge call, which is why this samples at all")
    parser.add_argument("--since-hours", type=float, default=24.0,
                        help="omit to continue from the last successful sample")
    parser.add_argument("--project", default=None,
                        help="project name; HOPSWORKS_PROJECT when omitted")
    args = parser.parse_args()

    if not os.environ.get("HOPSWORKS_API_KEY") and not os.environ.get("REST_ENDPOINT"):
        sys.exit("set HOPSWORKS_HOST, HOPSWORKS_PROJECT and HOPSWORKS_API_KEY, "
                 "or run this inside a Hopsworks job")

    import hopsworks

    agents = hopsworks.login(project=args.project).get_agent_serving()
    agent = agents.get_agent(int(args.agent) if args.agent.isdigit() else args.agent)
    if agent is None:
        sys.exit(f"no agent deployment called {args.agent!r} in the project")

    if args.evaluator:
        judge = agents.evaluators.find(args.evaluator)
        if judge is None:
            sys.exit(f"no saved evaluator called {args.evaluator!r}; run apply.py first")
    else:
        judge = ensure_judge(agents)

    run = agent.sample(
        evaluator=judge,
        since=datetime.now(timezone.utc) - timedelta(hours=args.since_hours),
        sample=args.sample,
    )
    print(f"started {run.run_id} against {agent.name}, grading with {judge.name!r}")
    print(f"  up to {args.sample} conversations from the last {args.since_hours:g}h")
    print("  results appear under Evals → Runs on the deployment, badged "
          "as a production sample")
    # A judge with no key is skipped rather than failed, and a sample whose judge
    # was skipped grades nothing -- which reads as a clean run.
    print("  needs ANTHROPIC_API_KEY in your account's environment variables")


if __name__ == "__main__":
    main()
