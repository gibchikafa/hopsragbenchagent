"""Recreate this agent's evaluation suites in a Hopsworks project.

    export HOPSWORKS_HOST=... HOPSWORKS_PROJECT=... HOPSWORKS_API_KEY=...
    python -m chinook.evaluation.apply            # the library, the suites, their tasks
    python -m chinook.evaluation.apply --publish  # and freeze them, so they can run

Everything goes through the project's agent-serving API, `project.get_agent_serving()`,
which is also where the suites are run against the deployed agent.

The tasks are files under `tasks/`, one per suite, in the columns the UI's Import
understands. They are imported here the same way Import would — a column named
after a check goes to that check; `expected`, `rubric`, `requiredTools` and
`forbiddenTools` go to the first check of the shape that reads them — so a file
that imports cleanly in the UI imports cleanly here, and the other way round.
Keeping the cases in files is what lets someone add twenty of them without
touching this; a suite has to exist before its tasks have anywhere to go, because
the checks decide what a task must declare, so the suite comes first.

Two files beside this one, both data rather than code so they can be diffed — a
change to what the agent is held to is a review comment, not a paragraph of
Python to read past.

`evaluators.json` is the library: one named check each, written once. Several
suites hold the agent to "place_order was not called", and writing that judge's
criteria into each of them is how they drift apart.

`suites.json` names the ones it wants, and names the task file that belongs to
it. A suite copies its checks in when it is created and never points back, which
is what keeps a published suite meaning exactly what it meant when it was
published — so editing the library later does not rewrite a suite that has
already been run against.

Suites are versioned and frozen on publish, so this creates and never edits. A
suite that already exists by name is left exactly as it is — re-running after
changing the file gives you a new version to publish, not a silent rewrite of the
one your last run was measured against. Tasks follow the same rule: a draft suite
gets the cases from its file that it does not already hold, and a published one
gets nothing, because the server would refuse.

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

# What each kind of check reads from a task, which decides where a familiar column
# goes. The same table the UI's Import uses (evaluatorFields.ts), so a task file
# means the same thing whichever way it is loaded. Checks not listed read nothing
# from the task.
SHAPES = {
    "contains": "text",
    "exact_match": "text",
    "regex": "text",
    "pairwise": "text",
    "llm_judge": "rubric",
    "tool_call": "tools",
    "tool_order": "list",
    "no_unnecessary_tools": "list",
}


def load() -> tuple[list[dict], list[dict]]:
    return json.loads(LIBRARY.read_text()), json.loads(SUITES.read_text())


def load_tasks(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _tool_list(value) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else str(value).split(",")
    return [str(tool).strip() for tool in items if str(tool).strip()]


def turns_of(row: dict) -> list[str]:
    """The user's messages, in order. A string is one turn; a list is a conversation."""
    raw = row.get("input") or row.get("question") or row.get("prompt") or ""
    turns = raw if isinstance(raw, list) else [raw]
    return [str(turn) for turn in turns if str(turn).strip()]


def expectations_of(row: dict, checks: list[dict]) -> dict[str, str]:
    """Columns to checks, the way Import does it.

    A column named after a check goes to it. The familiar names go to the first
    check of the shape that reads them and has not been claimed, so a suite with
    two checks of the same shape needs the column named after the check — which is
    the same rule the UI applies, and the reason it is the same rule.
    """
    expectations: dict[str, str] = {}

    def claim(name: str, value: str) -> None:
        if value and name not in expectations:
            expectations[name] = value

    for check in checks:
        claim(check["name"], str(row.get(check["name"], "")).strip())

    def first(shape: str) -> str | None:
        return next(
            (check["name"] for check in checks
             if SHAPES.get(check["type"]) == shape
             and check["name"] not in expectations),
            None,
        )

    expected = str(row.get("expected") or row.get("expectedOutput") or "").strip()
    if expected and (target := first("text")):
        claim(target, expected)
    rubric = str(row.get("rubric") or "").strip()
    if rubric and (target := first("rubric")):
        claim(target, rubric)
    required = _tool_list(row.get("requiredTools") or row.get("required_tools"))
    forbidden = _tool_list(row.get("forbiddenTools") or row.get("forbidden_tools"))
    if required or forbidden:
        if target := first("tools"):
            # a call check judges what must be called and what must not, so its
            # one expectation holds both
            claim(target, json.dumps({"required": required, "forbidden": forbidden}))
        elif required and (target := first("list")):
            claim(target, ", ".join(required))
    return expectations


def _held_turns(task) -> list[str]:
    return [str(m.get("content", "")) for m in task.messages if m.get("role") == "user"]


def import_tasks(suite, checks: list[dict], path: Path) -> int:
    """Add the cases in `path` that the suite does not already hold.

    Matched on the user's turns, so re-running adds nothing twice and a case
    appended to the file arrives without disturbing the ones before it.
    """
    held = {tuple(_held_turns(task)) for task in suite.tasks()}
    added = 0
    for row in load_tasks(path):
        turns = turns_of(row)
        if not turns:
            print(f"  skipped a row with no input in {path.name}")
            continue
        if tuple(turns) in held:
            continue
        suite.add_task(turns if len(turns) > 1 else turns[0],
                       expectations=expectations_of(row, checks))
        held.add(tuple(turns))
        added += 1
    return added


def apply(agents, publish: bool = False) -> None:
    library, suites = load()

    # The library first: a suite copies its checks in, so they have to exist as
    # something to copy. Saving is by name, so re-running updates an entry rather
    # than making a second one called the same thing.
    checks_by_name = {}
    saved = {entry.name for entry in agents.evaluators.list()}
    for entry in library:
        checks_by_name[entry["name"]] = entry["checks"]
        if entry["name"] in saved:
            print(f"= {entry['name']} (library, exists)")
            continue
        agents.evaluators.save(entry["name"], entry["checks"], entry["description"])
        print(f"+ {entry['name']} (library)")

    existing = {}
    for suite in agents.suites.list():
        # the newest version of each name is the one tasks go to
        if suite.name not in existing or suite.version > existing[suite.name].version:
            existing[suite.name] = suite

    for definition in suites:
        name = definition["name"]
        if name in existing:
            print(f"= {name} (exists, left alone)")
            suite = existing[name]
        else:
            suite = agents.suites.create(
                name,
                description=definition["description"],
                tags=definition["tags"],
                execution_mode=definition["executionMode"],
                pass_policy=definition["passPolicy"],
                pass_threshold=definition["passThreshold"],
                # A tag is descriptive and nothing reads it. What a suite does
                # is stated: `golden` used to imply a release gate, and now the
                # gate names its own metric and bar, which is the difference
                # between a rule someone can see and one hidden in a category.
                gate_metric=definition.get("gateMetric", ""),
                gate_threshold=definition.get("gateThreshold"),
                checks=[
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

        # The cases, from the file beside the suite. A published suite is frozen
        # and refuses them, which is the point of publishing; the file is still
        # the record of what the next version should hold.
        tasks_file = Path(__file__).parent / definition["tasksFile"]
        task_count = suite.task_count or 0
        if suite.published:
            print(f"  published, tasks left as they are ({tasks_file.name})")
        else:
            checks = [
                check
                for entry_name in definition["evaluators"]
                for check in checks_by_name[entry_name]
            ]
            added = import_tasks(suite, checks, tasks_file)
            task_count = len(suite.tasks())
            print(f"  tasks: {added} added, {task_count} in the suite")

        # A suite with no tasks cannot be published — there would be nothing to
        # run — so this says so rather than failing with the server's refusal.
        if publish and not suite.published:
            if task_count:
                suite.publish()
                print(f"  published {name}")
            else:
                print(f"  not published: {tasks_file.name} gave it no tasks")
        if definition["executionMode"] == "sandboxed":
            print(f"  {SANDBOXED_NOTE}")
        if definition.get("gateMetric"):
            print(f"  gates on {definition['gateMetric']} "
                  f">= {definition['gateThreshold']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true",
                        help="freeze each suite, which is what makes it runnable")
    parser.add_argument("--project", default=None,
                        help="project name; HOPSWORKS_PROJECT when omitted, and the "
                             "job's own project inside Hopsworks")
    args = parser.parse_args()

    if not os.environ.get("HOPSWORKS_API_KEY") and not os.environ.get("REST_ENDPOINT"):
        sys.exit("set HOPSWORKS_HOST, HOPSWORKS_PROJECT and HOPSWORKS_API_KEY, "
                 "or run this inside a Hopsworks job")

    import hopsworks

    project = hopsworks.login(project=args.project)
    apply(project.get_agent_serving(), publish=args.publish)


if __name__ == "__main__":
    main()
