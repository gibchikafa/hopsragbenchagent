"""Recreate this agent's evaluation suites in a Hopsworks project.

    python -m chinook.evaluation.apply            # create anything missing
    python -m chinook.evaluation.apply --publish  # and freeze them, so they can run

Two files beside this one, both data rather than code so they can be diffed — a
change to what the agent is held to is a review comment, not a paragraph of
Python to read past.

`evaluators.json` is the library: one named check each, written once. Several
suites hold the agent to "place_order was not called", and writing that judge's
criteria into each of them is how they drift apart.

`suites.json` names the ones it wants. A suite copies them in when it is created
and never points back, which is what keeps a published suite meaning exactly what
it meant when it was published — so editing the library later does not rewrite a
suite that has already been run against.

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

LIBRARY = Path(__file__).with_name("evaluators.json")
SUITES = Path(__file__).with_name("suites.json")

# Two of these make the agent place orders, so they are sandboxed and the runner
# refuses them unless the deployment reports eval_mode. That means EVAL_MODE=true
# on a deployment running this agent's eval-mode code, which suppresses the
# writes — see EVAL_MODE in support_agent.py.
SANDBOXED_NOTE = (
    "sandboxed — needs a deployment with EVAL_MODE=true, or the runner refuses it"
)


def load() -> tuple[list[dict], list[dict]]:
    return json.loads(LIBRARY.read_text()), json.loads(SUITES.read_text())


def apply(api, publish: bool = False) -> None:
    library, suites = load()

    # The library first: a suite copies its checks in, so they have to exist as
    # something to copy. Saving is by name, so re-running updates an entry rather
    # than making a second one called the same thing.
    checks_by_name = {}
    saved = {entry["name"] for entry in api.evaluators()}
    for entry in library:
        checks_by_name[entry["name"]] = entry["checks"]
        if entry["name"] in saved:
            print(f"= {entry['name']} (library, exists)")
            continue
        api.save_evaluator(entry["name"], entry["checks"], entry["description"])
        print(f"+ {entry['name']} (library)")

    existing = {suite["name"]: suite for suite in api.suites()}

    for definition in suites:
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
                    # Copied in, not referenced. The suite is the record of what
                    # a run executed, and a reference would let the library
                    # change it after the fact.
                    {
                        "type": check.pop("type"),
                        "name": check.pop("name"),
                        "config": json.dumps(check),
                    }
                    for entry_name in definition["evaluators"]
                    for check in (dict(one) for one in checks_by_name[entry_name])
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
