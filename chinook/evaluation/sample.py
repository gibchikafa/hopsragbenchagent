"""Grade a sample of what this agent actually told customers.

    export HOPSWORKS_API_KEY=...
    python -m chinook.evaluation.sample --deployment-id 12

The online half. `apply.py` creates the suites — cases someone wrote down, with
an expected answer each — and this grades conversations that already happened,
where there is no expected answer at all.

Both matter and neither substitutes for the other. A suite cannot contain a
question nobody thought to write, which is most of what customers ask. And
production cannot tell you whether a fix held, because the conversation that
broke may never come back. The loop between them is the point: something goes
wrong in production, you promote that trace to a task, and from then on a suite
defends against it.

The rubric is the whole input, since it is the only thing the judge has to grade
against. It lives in `rubric.md` beside this file rather than being typed in,
because it is a statement about this agent that will be argued over, and a change
to it changes what every future score means.

The score this produces is **not** comparable with a suite's pass rate. One says
how the agent does on cases with declared answers, graded by the same checks
every time; the other is one judge's opinion on whatever traffic arrived.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RUBRIC = Path(__file__).with_name("rubric.md")


def rubric() -> str:
    """The rubric, without its explanatory preamble.

    Everything above the horizontal rule explains why the file exists, which is
    for whoever edits it and not for the judge — sending it would spend tokens
    telling the model about offline evaluation.
    """
    text = RUBRIC.read_text()
    _, _, body = text.partition("\n---\n")
    return (body or text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-id", type=int, required=True)
    parser.add_argument("--sample", type=int, default=25,
                        help="how many conversations to grade. Each one costs a "
                             "judge call, which is why this samples at all")
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument("--project-id", type=int, default=None)
    args = parser.parse_args()

    try:
        from hopsworks_agent_eval.api import EvalApi
    except ImportError:
        sys.exit(
            "needs hopsworks-agent-protocol[eval]: "
            "pip install 'hopsworks-agent-protocol[eval]'"
        )

    if not os.environ.get("HOPSWORKS_API_KEY") and not os.environ.get("SECRETS_DIR"):
        sys.exit("set HOPSWORKS_API_KEY, or run this inside a Hopsworks job")

    api = EvalApi.from_env(project_id=args.project_id)
    run = api.sample_production(
        deployment_id=args.deployment_id,
        sample=args.sample,
        since_hours=args.since_hours,
        rubric=rubric(),
    )
    print(f"started {run['runId']}")
    print(f"  {args.sample} conversations from the last {args.since_hours:g}h")
    print("  results appear under Evaluation → Runs on the deployment, badged "
          "as a production sample")
    # A judge with no key is skipped rather than failed, and a sample whose judge
    # was skipped grades nothing but tool errors -- which reads as a clean run.
    print("  needs ANTHROPIC_API_KEY in your account's environment variables; "
          "without it only the tool-error check runs")


if __name__ == "__main__":
    main()
