"""Every evaluation case, run through the agent in-process and graded the same way.

The suites in `evaluation/` are data: a case is a question and the string the
reply must contain, or a pattern it must match. This runs each case against
the agent with FastAPI's test client, threading a multi-turn case through one
conversation the way the evaluation runner does, and applies the suite's
`contains` and `regex` checks to the final reply. A change to the agent that
would fail a suite fails here first, without a deployment or a run.

Cases carry no `subject`: the person is named in the question, the way the
runner sends them, so the identity gate is exercised in every case.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest


pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from fastapi.testclient import TestClient  # noqa: E402


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the repository root, for the shared loader

from chinook.evaluation.apply import expectations_of, load, load_tasks, turns_of  # noqa: E402


@pytest.fixture(scope="module")
def client():
    memory_dir = tempfile.mkdtemp(prefix="data_analyst_eval_")
    os.environ["DATA_ANALYST_MEMORY_URL"] = f"sqlite:///{memory_dir}/memory.db"
    spec = importlib.util.spec_from_file_location(
        "data_analyst_agent", HERE / "data_analyst_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return TestClient(module.app)


def cases() -> list[tuple[str, dict, list[dict]]]:
    """(suite name, case row, the suite's checks with their config) for every case."""
    library, suites = load(HERE / "evaluation")
    checks_by_name = {entry["name"]: entry["checks"] for entry in library}
    out = []
    for suite in suites:
        checks = [dict(c) for name in suite["evaluators"] for c in checks_by_name[name]]
        for row in load_tasks(HERE / "evaluation" / suite["tasksFile"]):
            out.append((suite["name"], row, checks))
    return out


def case_id(case) -> str:
    suite, row, _ = case
    return f"{suite} :: {turns_of(row)[0].splitlines()[0][:60]}"


def final_reply(client, turns: list[str]) -> str:
    conversation_id = None
    text = ""
    for turn in turns:
        body = {
            "message": {"role": "user", "content": [{"type": "text", "text": turn}]}
        }
        if conversation_id:
            body["conversation_id"] = conversation_id
        response = client.post("/v1/chat", json=body)
        assert response.status_code == 200, f"{turn[:60]!r}: {response.text}"
        reply = response.json()
        conversation_id = reply["conversation_id"]
        text = "".join(
            p["text"] for p in reply["message"]["content"] if p["type"] == "text"
        )
    return text


def grade(check: dict, expectation: str | None, answer: str) -> tuple[bool, str]:
    """The suite's check on the final answer, as the evaluation runner applies it."""
    if check["type"] == "contains":
        assert expectation, f"{check['name']} has no expectation for this case"
        # case-insensitive, as the SDK's contains check is
        return expectation.lower() in answer.lower(), f"contains {expectation!r}"
    if check["type"] == "regex":
        matched = re.search(check["pattern"], answer) is not None
        wanted = check.get("should_match", True)
        return (
            matched == wanted,
            f"{'matches' if wanted else 'does not match'} /{check['pattern'][:40]}.../",
        )
    raise AssertionError(f"no in-process grader for a {check['type']} check")


@pytest.mark.parametrize("case", cases(), ids=case_id)
def test_the_case_passes_its_suites_checks(client, case):
    suite, row, checks = case
    expectations = expectations_of(row, checks)
    answer = final_reply(client, turns_of(row))
    failures = []
    for check in checks:
        passed, what = grade(check, expectations.get(check["name"]), answer)
        if not passed:
            failures.append(f"{check['name']}: {what}")
    assert not failures, f"{suite}: {failures}\n--- final reply ---\n{answer}"


def test_every_suite_names_checks_the_library_has():
    library, suites = load(HERE / "evaluation")
    names = {entry["name"] for entry in library}
    for suite in suites:
        missing = set(suite["evaluators"]) - names
        assert not missing, (
            f"{suite['name']} uses evaluators not in the library: {missing}"
        )
        assert (HERE / "evaluation" / suite["tasksFile"]).exists(), suite["tasksFile"]


def test_the_library_is_valid_json_with_one_check_shape_each():
    library, _ = load(HERE / "evaluation")
    for entry in library:
        for check in entry["checks"]:
            assert check["type"] in ("contains", "regex"), (
                f"{entry['name']}: a {check['type']} check needs a judge or a trace, "
                "and this agent has neither"
            )
            if check["type"] == "regex":
                re.compile(check["pattern"])
    json.dumps(library)
