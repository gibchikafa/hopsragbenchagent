"""Recreate the data analyst's evaluation suites in a Hopsworks project.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m data_analyst.evaluation.apply            # the library, the suites, their tasks
    python -m data_analyst.evaluation.apply --publish  # and freeze them, so they can run

The same loader as the Chinook agent's, pointed at this directory: the files
beside this one are the library (`evaluators.json`), the suites (`suites.json`)
and the cases (`tasks/*.jsonl`), in the columns the UI's Import understands.

No judge anywhere: the agent calls no model, so every reply to a case is a
sentence this file can state in advance, and `contains` and `regex` grade it.
That is also why the bar is every case -- a deterministic agent that fails one
is wrong, not unlucky. Each case names its own person (`case-NN@eval.example`)
in the question, the way the chat panel would be answered, so the identity
gate is passed inside the trial and no case meets another's long-term memory.

The tasks carry the sample table inlined the way the chat panel attaches a
file; `test_evaluation_cases.py` beside the agent runs every case in-process,
so a change to the agent that breaks a case is caught before a run is paid for.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from chinook.evaluation.apply import apply

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="freeze each suite, which is what makes it runnable",
    )
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

    project = hopsworks.login(project=args.project)
    apply(project.get_agent_serving(), publish=args.publish, directory=HERE)


if __name__ == "__main__":
    main()
