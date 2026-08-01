"""Recreate this agent's evaluation suites in a Hopsworks project.

    python -m chinook.evaluation.apply            # create anything missing
    python -m chinook.evaluation.apply --publish  # and freeze them, so they can run

`suites.json` beside this file is the definition. It is data rather than code so
it can be diffed: a suite changing is a review comment, not a paragraph of Python
to read past.

Suites are versioned and frozen on publish, so this creates and never edits. A
suite that already exists by name is left exactly as it is — re-running after
changing the file gives you a new version to publish, not a silent rewrite of the
one your last run was measured against.

The four suites here are built from what this agent can actually get wrong, taken
from its own tools and system prompt:

  - a catalogue question must not touch the customer's account
  - being interested in an album is not asking to buy it
  - an order is recorded, never charged
  - nothing about a customer's orders before all three identity fields

Every expected string is a real Chinook value, so a failure means the agent was
wrong and not that the test was.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFINITION = Path(__file__).with_name("suites.json")

# Two of these make the agent place orders, so they are sandboxed and the runner
# refuses them unless the deployment reports eval_mode. That means EVAL_MODE=true
# on a deployment running this agent's eval-mode code, which suppresses the
# writes — see EVAL_MODE in support_agent.py.
SANDBOXED_NOTE = (
    "sandboxed — needs a deployment with EVAL_MODE=true, or the runner refuses it"
)


def load() -> list[dict]:
    return json.loads(DEFINITION.read_text())


def apply(api, publish: bool = False) -> None:
    existing = {suite["name"]: suite for suite in api.suites()}

    for definition in load():
        name = definition["name"]
        if name in existing:
            print(f"= {name} (exists, left alone)")
            suite = existing[name]
        else:
            suite = api.create_suite(
                name,
                description=definition["description"],
                tags=definition["tags"],
                execution_mode=definition["executionMode"],
                pass_policy=definition["passPolicy"],
                pass_threshold=definition["passThreshold"],
                evaluators=[
                    {
                        "type": check["type"],
                        "name": check["name"],
                        "config": json.dumps(check["config"]),
                    }
                    for check in definition["evaluators"]
                ],
            )
            print(f"+ {name}  {definition['executionMode']}")
            for task in definition["tasks"]:
                api.add_task(suite, task["asks"], task["expectations"])
                print(f"    {task['asks'][:68]}")

        if publish and suite.get("status") != "PUBLISHED":
            api.publish(suite)
            print(f"  published {name}")
        if definition["executionMode"] == "sandboxed":
            print(f"  {SANDBOXED_NOTE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true",
                        help="freeze each suite, which is what makes it runnable")
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

    apply(EvalApi.from_env(project_id=args.project_id), publish=args.publish)


if __name__ == "__main__":
    main()
