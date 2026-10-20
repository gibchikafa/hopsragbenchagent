"""Grade a sample of what this agent actually said to people.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m data_analyst.evaluation.sample --agent dataanalyst

The online half. `apply.py` creates the suites -- cases with an expected answer
each -- and this grades conversations that already happened, where there is
none. The evaluator is the library's "Reply is one of the agent's shapes": a
pattern over the sentences this agent is written to say, so a reply outside
them is an exception that reached a person, or the protocol returning
something the agent did not. No model, so it grades every conversation it is
given rather than a sample of them, and costs nothing per trace.

Its score is not comparable with a suite's pass rate: one says the agent gave
the declared answer, this says it gave an answer at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LIBRARY = Path(__file__).with_name("evaluators.json")
EVALUATOR_NAME = "Reply is one of the agent's shapes"


def ensure_evaluator(agents):
    """The shape check as a saved evaluator in the project's library, from the file."""
    entry = next(
        e for e in json.loads(LIBRARY.read_text()) if e["name"] == EVALUATOR_NAME
    )
    existing = agents.evaluators.find(EVALUATOR_NAME)
    if existing is not None:
        return existing
    return agents.evaluators.save(entry["name"], entry["checks"], entry["description"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", required=True, help="the agent deployment, by name or id"
    )
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument(
        "--project", default=None, help="project name; HOPSWORKS_PROJECT when omitted"
    )
    args = parser.parse_args()

    if not os.environ.get("HOPSWORKS_API_KEY") and not os.environ.get("REST_ENDPOINT"):
        sys.exit(
            "set HOPSWORKS_HOST, HOPSWORKS_PROJECT and HOPSWORKS_API_KEY, "
            "or run this inside a Hopsworks job"
        )

    import hopsworks

    agents = hopsworks.login(project=args.project).get_agent_serving()
    agent = agents.get_agent(int(args.agent) if args.agent.isdigit() else args.agent)
    if agent is None:
        sys.exit(f"no agent deployment called {args.agent!r} in the project")
    run = agent.sample(
        evaluator=ensure_evaluator(agents),
        since=datetime.now(timezone.utc) - timedelta(hours=args.since_hours),
    )
    print(f"started {run.run_id} against {agent.name}")
    print(f"  every conversation from the last {args.since_hours:g}h")
    print(
        "  results appear under Evals -> Runs on the deployment, badged as a production sample"
    )


if __name__ == "__main__":
    main()
