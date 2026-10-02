"""Recreate the memory demo's evaluation suites in a Hopsworks project.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m agent_memory.evaluation.apply            # the library, the suites, their tasks
    python -m agent_memory.evaluation.apply --publish  # and freeze them, so they can run

The same loader as the Chinook agent's, pointed at this directory: the files
beside this one are the library (`evaluators.json`), the suites (`suites.json`)
and the cases (`tasks/*.jsonl`), in the columns the UI's Import understands.

What the suites hold the agent to is what makes a memory agent trustworthy
rather than merely talkative: that it stored what it claims to have stored,
that it never reports a memory it was not given, and that it reads the turns
already in front of it before paying for a lookup. The tasks are
conversations, not single prompts, because memory only shows itself across
turns.
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
    parser.add_argument("--publish", action="store_true",
                        help="freeze each suite, which is what makes it runnable")
    parser.add_argument("--project", default=None,
                        help="project name; HOPSWORKS_PROJECT when omitted")
    args = parser.parse_args()

    if not os.environ.get("HOPSWORKS_API_KEY") and not os.environ.get("REST_ENDPOINT"):
        sys.exit("set HOPSWORKS_HOST, HOPSWORKS_PROJECT and HOPSWORKS_API_KEY, "
                 "or run this inside a Hopsworks job")

    import hopsworks

    project = hopsworks.login(project=args.project)
    apply(project.get_agent_serving(), publish=args.publish, directory=HERE)


if __name__ == "__main__":
    main()
