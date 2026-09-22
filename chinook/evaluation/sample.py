"""Grade a sample of what this agent actually told customers.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m chinook.evaluation.sample --agent customeragent

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
from datetime import datetime, timedelta, timezone
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


JUDGE_NAME = "Chinook conversation quality"


def ensure_judge(agents):
    """The rubric as a saved judge in the project's library.

    Saved by name, so the library holds one of these and a change to `rubric.md`
    lands on the next run. The rubric is the judge's one criterion: production
    carries no expected answer, so the judge must reach a verdict from the
    conversation and the rubric alone, which is what lets it grade live traffic.
    """
    existing = agents.evaluators.find(JUDGE_NAME)
    if existing is not None:
        agents.evaluators.delete(existing)
    return agents.evaluators.save(
        JUDGE_NAME,
        [
            {
                "type": "llm_judge",
                "name": "conversation_quality",
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "temperature": 0,
                "score_range": [1, 5],
                "criteria": {
                    "follows_the_rubric": {"weight": 1, "description": rubric()},
                },
            },
        ],
        description="What a good answer from the Chinook support agent looks like, "
        "from chinook/evaluation/rubric.md. Grades live traffic; no expected answer.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True,
                        help="the agent deployment, by name or id")
    parser.add_argument("--sample", type=int, default=25,
                        help="how many conversations to grade. Each one costs a "
                             "judge call, which is why this samples at all")
    parser.add_argument("--since-hours", type=float, default=24.0)
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

    judge = ensure_judge(agents)
    run = agent.sample(
        evaluator=judge,
        since=datetime.now(timezone.utc) - timedelta(hours=args.since_hours),
        sample=args.sample,
    )
    print(f"started {run.run_id} against {agent.name}")
    print(f"  up to {args.sample} conversations from the last {args.since_hours:g}h")
    print("  results appear under Evals → Runs on the deployment, badged "
          "as a production sample")
    # A judge with no key is skipped rather than failed, and a sample whose judge
    # was skipped grades nothing -- which reads as a clean run.
    print("  needs ANTHROPIC_API_KEY in your account's environment variables")


if __name__ == "__main__":
    main()
