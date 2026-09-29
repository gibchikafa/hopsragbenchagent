"""Recreate the shopping agent's evaluation suites in a Hopsworks project.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m personalized_shopping.evaluation.apply            # the library, the suites, their tasks
    python -m personalized_shopping.evaluation.apply --publish  # and freeze them, so they can run

The same loader as the Chinook agent's, pointed at this directory. The recipe's
own cases (`eval/eval_data/simple.test.json`, `tests/tools/tools.test.json`)
are the first two suites; the other two check what the recipe's website made
hard to test: that a product is read before it is described, and that an order
needs an instruction, chosen options, and never claims a payment.
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
