"""The data analyst agent, run in-process with FastAPI's test client.

The agent is the multimodal template: a table in as a file part or as the
chat panel's inlined block, a text summary, a PNG and a CSV out. These cases
pin what a copy of it can rely on -- the manifest, both ways a file arrives,
the parts of a reply, the follow-up without a re-attach, the refusal by name
and the streamed shape -- so a change to the protocol that breaks the
template breaks here, not in a deployment.

Memory runs on SQLite here (`DATA_ANALYST_MEMORY_URL`), the same managed
service the deployment points at the project database. Most cases assert a
`subject` on the request, the way a client with a login does, so the agent
does not ask who it is talking to; the identity cases send none and go
through the email question.

Needs the `hopsworks[agents,agents-memory]` package plus pandas and matplotlib:

    pip install "hopsworks[agents,agents-memory]" matplotlib pytest httpx
    pytest test_data_analyst_agent.py
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from fastapi.testclient import TestClient  # noqa: E402


AGENT = Path(__file__).resolve().parent / "data_analyst_agent.py"
SUBJECT = "tester@example.com"

SALES = (
    "region,product,revenue,units,order_date\n"
    "North,widget,500,5,2026-01-03\n"
    "North,gadget,700,2,2026-02-10\n"
    "South,widget,300,3,2026-01-15\n"
    "East,gadget,450,1,2026-02-20\n"
    "East,widget,150,2,2026-03-01\n"
)


@pytest.fixture(scope="module")
def example():
    # One SQLite database for the module: the agent builds its memory at import.
    memory_dir = tempfile.mkdtemp(prefix="data_analyst_memory_")
    os.environ["DATA_ANALYST_MEMORY_URL"] = f"sqlite:///{memory_dir}/memory.db"
    spec = importlib.util.spec_from_file_location("data_analyst_agent", AGENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(example):
    return TestClient(example.app)


def message(
    text: str,
    *parts: dict,
    conversation_id: str | None = None,
    subject: str | None = SUBJECT,
) -> dict:
    body = {
        "message": {"role": "user", "content": [{"type": "text", "text": text}, *parts]}
    }
    if conversation_id:
        body["conversation_id"] = conversation_id
    if subject:
        body["subject"] = subject
    return body


def csv_file(text: str, name: str = "sales.csv") -> dict:
    return {
        "type": "file",
        "name": name,
        "media_type": "text/csv",
        "data": base64.b64encode(text.encode()).decode(),
    }


def inlined(text: str, name: str = "sales.csv") -> str:
    """How the Hopsworks chat panel attaches a text file."""
    return f"--- Attached file: {name} ---\n{text}\n--- End of {name} ---"


def parts_by_type(reply: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for part in reply["message"]["content"]:
        grouped.setdefault(part["type"], []).append(part)
    return grouped


def decoded_csv(part: dict) -> list[list[str]]:
    rows = base64.b64decode(part["data"]).decode().strip().splitlines()
    return [row.split(",") for row in rows]


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((event, data))
    return events


def test_the_manifest_declares_a_file_in_and_an_image_and_a_file_out(client):
    manifest = client.get("/.well-known/hopsworks-agent.json").json()
    capabilities = manifest["capabilities"]
    assert capabilities["input_modalities"] == ["text", "file"]
    assert capabilities["output_modalities"] == ["text", "image", "file"]
    # What makes the chat panel show the attachment picker at all.
    assert capabilities["attachments"] is True
    assert capabilities["streaming"] is True
    assert capabilities["tool_events"] is True
    # Memory is served by the SDK: the panel's Memory tab reads these.
    assert capabilities["conversation_history"] is True
    memory = capabilities["memory"]
    assert memory["summary"] is True, "short-term memory: the rule-based fold"
    assert memory["state"] is True, "long-term memory: remember/recall under the person"


def test_a_csv_file_part_is_answered_with_text_a_chart_and_the_numbers(client):
    response = client.post(
        "/v1/chat", json=message("total revenue by region", csv_file(SALES))
    )
    assert response.status_code == 200, response.text
    reply = response.json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text", "image", "file"}, reply["message"]["content"]

    text = parts["text"][0]["text"]
    assert text.startswith(
        "Sum of revenue by region, from 5 rows of sales.csv: 3 groups."
    )
    assert "Highest: North (1200)" in text
    assert "Lowest: South (300)" in text

    image = parts["image"][0]
    assert image["media_type"] == "image/png"
    assert base64.b64decode(image["data"])[:8] == b"\x89PNG\r\n\x1a\n"

    file = parts["file"][0]
    assert file["name"] == "sum_revenue_by_region.csv"
    assert file["media_type"] == "text/csv"
    assert decoded_csv(file) == [
        ["region", "revenue"],
        ["North", "1200"],
        ["East", "600"],
        ["South", "300"],
    ]

    # The turn reports its own usage, and the steps it took.
    assert reply["usage"]["output_tokens"] > 0
    assert [e["name"] for e in reply["metadata"]["tool_events"]] == [
        "load_table",
        "aggregate",
        "aggregate",
        "render_chart",
        "render_chart",
    ]


def test_the_chat_panels_inlined_attachment_is_read_the_same_way(client):
    body = message(f"average units per product\n\n{inlined(SALES)}")
    reply = client.post("/v1/chat", json=body).json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text", "image", "file"}
    rows = decoded_csv(parts["file"][0])
    assert rows[0] == ["product", "units"]
    # The CSV carries what pandas computed, at full precision; the text rounds.
    assert [(r[0], float(r[1])) for r in rows[1:]] == [
        ("widget", pytest.approx(10 / 3)),
        ("gadget", 1.5),
    ]
    text = parts["text"][0]["text"]
    assert "Average of units by product" in text
    assert "| widget | 3.33 |" in text
    # The question the agent answered is the text without the attachment.
    assert "Attached file" not in text


def test_a_follow_up_in_the_conversation_needs_no_re_attach(client):
    first = client.post(
        "/v1/chat", json=message("describe the data", csv_file(SALES))
    ).json()
    parts = parts_by_type(first)
    assert set(parts) == {"text", "file"}, (
        "describe answers with text and a CSV, no chart"
    )
    assert "5 rows, 5 columns" in parts["text"][0]["text"]

    follow_up = message(
        "how many units by region", conversation_id=first["conversation_id"]
    )
    reply = client.post("/v1/chat", json=follow_up).json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text", "image", "file"}
    assert decoded_csv(parts["file"][0])[0] == ["region", "units"]
    assert "Count of units by region" in parts["text"][0]["text"]


def test_a_date_column_gives_a_line_in_time_order(client):
    reply = client.post(
        "/v1/chat", json=message("total revenue by order_date", csv_file(SALES))
    ).json()
    rows = decoded_csv(parts_by_type(reply)["file"][0])
    assert [r[0] for r in rows[1:]] == [
        "2026-01-03",
        "2026-01-15",
        "2026-02-10",
        "2026-02-20",
        "2026-03-01",
    ], "time on the x axis is sorted by time, not by value"


def test_by_month_buckets_a_date_column(client):
    """ "By month" names no column; with a date column it is a period of that."""
    reply = client.post(
        "/v1/chat", json=message("how many orders by month", csv_file(SALES))
    ).json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text", "image", "file"}
    rows = decoded_csv(parts["file"][0])
    assert rows[0][0] == "month"
    assert [(r[0], r[1]) for r in rows[1:]] == [
        ("2026-01", "2"),
        ("2026-02", "2"),
        ("2026-03", "1"),
    ]


def test_a_question_with_no_group_picks_the_coarsest_category(client):
    """ "total revenue" on its own groups by the non-numeric column with fewest values."""
    reply = client.post(
        "/v1/chat", json=message("total revenue", csv_file(SALES))
    ).json()
    text = parts_by_type(reply)["text"][0]["text"]
    # product has two values, region three, order_date five.
    assert text.startswith("Sum of revenue by product"), text


def test_a_column_the_table_does_not_have_is_refused_by_name(client):
    response = client.post(
        "/v1/chat", json=message("total revenue by planet", csv_file(SALES))
    )
    assert response.status_code == 400
    error = response.json()["detail"]
    assert error["code"] == "unknown_column"
    assert "planet" in error["message"]
    assert "region" in error["message"], "the refusal lists the columns there are"


def test_a_question_before_any_table_gets_the_help_text(client):
    reply = client.post(
        "/v1/chat",
        json=message("total revenue by region", subject="nobody@example.com"),
    ).json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text"}
    assert parts["text"][0]["text"].startswith("Attach a CSV"), parts["text"][0]["text"]


# ── identity and memory ───────────────────────────────────────────────────


def test_without_a_subject_the_agent_asks_for_an_email_and_keeps_the_table(client):
    """The other examples ask who they are talking to; so does this one.

    The table that came with the question is already in session memory, so
    the answer to the question needs no re-attach once the address is given.
    """
    first = client.post(
        "/v1/chat",
        json=message("total revenue by region", csv_file(SALES), subject=None),
    ).json()
    parts = parts_by_type(first)
    assert set(parts) == {"text"}, "nothing is analysed before the person is known"
    assert "email address" in parts["text"][0]["text"]
    assert "I have sales.csv ready" in parts["text"][0]["text"]

    second = client.post(
        "/v1/chat",
        json=message(
            "it's Dana.Smith@Example.com -- total revenue by region",
            conversation_id=first["conversation_id"],
            subject=None,
        ),
    ).json()
    parts = parts_by_type(second)
    assert set(parts) == {"text", "image", "file"}, second["message"]["content"]
    text = parts["text"][0]["text"]
    assert text.startswith(
        "Thanks, dana.smith@example.com. Sum of revenue by region"
    ), text

    # And the conversation stays bound: a third turn is not asked again.
    third = client.post(
        "/v1/chat",
        json=message(
            "average units per product",
            conversation_id=first["conversation_id"],
            subject=None,
        ),
    ).json()
    assert "Average of units by product" in parts_by_type(third)["text"][0]["text"]


def test_a_client_that_asserts_the_subject_is_not_asked(client):
    reply = client.post(
        "/v1/chat", json=message("describe the data", csv_file(SALES))
    ).json()
    text = parts_by_type(reply)["text"][0]["text"]
    assert "email" not in text.lower()
    assert text.startswith("sales.csv: 5 rows, 5 columns.")


def test_a_returning_person_gets_their_last_table_back(client):
    """Long-term memory: a new conversation, the address, and the table is there."""
    who = "returning@example.com"
    client.post(
        "/v1/chat",
        json=message("total revenue by region", csv_file(SALES), subject=who),
    )

    back = client.post("/v1/chat", json=message(who, subject=None)).json()
    text = parts_by_type(back)["text"][0]["text"]
    assert text.startswith(
        "Welcome back, returning@example.com. Last time you analysed sales.csv "
        "(5 rows); I still have it."
    ), text

    reply = client.post(
        "/v1/chat",
        json=message(
            "max revenue by product",
            conversation_id=back["conversation_id"],
            subject=None,
        ),
    ).json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text", "image", "file"}, "the remembered table was analysed"
    assert decoded_csv(parts["file"][0]) == [
        ["product", "revenue"],
        ["gadget", "700"],
        ["widget", "500"],
    ]


def test_remember_recall_and_forget_are_the_sdks_own_calls(client):
    who = "notes@example.com"

    def say(text: str) -> str:
        reply = client.post("/v1/chat", json=message(text, subject=who)).json()
        return parts_by_type(reply)["text"][0]["text"]

    assert (
        say("remember fiscal year: starts in April")
        == "Remembered fiscal_year: starts in April"
    )
    # Each call is a new conversation, under the same person.
    assert say("recall fiscal year") == "fiscal_year: starts in April"
    say("remember that the quarter closes on Friday")
    everything = say("what do you remember")
    assert "- fiscal_year: starts in April" in everything
    assert "- the_quarter_closes_on: the quarter closes on Friday" in everything
    assert say("forget fiscal year") == "Forgot fiscal_year."
    assert say("recall fiscal year") == "Nothing stored under 'fiscal_year'."

    # Another person sees none of it.
    other = client.post(
        "/v1/chat", json=message("what do you remember", subject="other@example.com")
    ).json()
    assert (
        parts_by_type(other)["text"][0]["text"] == "I have nothing stored for you yet."
    )


def test_search_finds_what_was_said_in_an_earlier_conversation(client):
    who = "searcher@example.com"
    client.post(
        "/v1/chat",
        json=message("remember the warehouse: moved to Malmo in March", subject=who),
    )
    found = client.post(
        "/v1/chat", json=message("search warehouse", subject=who)
    ).json()
    text = parts_by_type(found)["text"][0]["text"]
    assert text.startswith("From earlier conversations, about 'warehouse':"), text
    assert "Malmo" in text


def test_a_streamed_reply_carries_the_text_as_deltas_and_the_binary_parts_last(client):
    response = client.post(
        "/v1/chat/stream", json=message("total revenue by region", csv_file(SALES))
    )
    assert response.status_code == 200, response.text
    events = parse_sse(response.text)
    kinds = [event for event, _ in events]
    assert kinds[-1] == "message.completed"
    assert "tool_event" in kinds

    deltas = "".join(
        data["delta"]["text"] for event, data in events if event == "message.delta"
    )
    completed = next(data for event, data in events if event == "message.completed")
    parts = parts_by_type(completed)
    assert set(parts) == {"text", "image", "file"}
    assert deltas == parts["text"][0]["text"], (
        "joining the deltas must give exactly the completed reply's text"
    )
    # Nothing binary went out as a delta.
    assert all(
        set(data["delta"]) == {"text"}
        for event, data in events
        if event == "message.delta"
    )


# ── the agent decides what to remember and what to recall ─────────────────


def test_an_introduction_beats_the_subject_the_client_asserted(client):
    """The chat panel asserts the logged-in username; a person who says who
    they are has said it more precisely, and that is what memory is keyed by."""
    reply = client.post(
        "/v1/chat", json=message("Hi am dana@example.com", subject="gibson")
    ).json()
    text = parts_by_type(reply)["text"][0]["text"]
    assert text.startswith("Thanks, dana@example.com."), text
    # Reported to the client, which is how the panel's memory view follows it
    # and how the trace list shows a subject for the conversation.
    assert reply["metadata"]["subject"] == "dana@example.com"


def test_facts_said_in_passing_are_kept_and_answered_from(client):
    who = "dana.facts@example.com"
    first = client.post(
        "/v1/chat",
        json=message(f"Hi, I'm Dana ({who}), I work at Hopsworks", subject="gibson"),
    ).json()
    text = parts_by_type(first)["text"][0]["text"]
    assert text.startswith(
        "Thanks, Dana (dana.facts@example.com). Noted: name Dana, employer Hopsworks."
    ), text

    def ask(question: str, conversation_id: str | None = None) -> str:
        reply = client.post(
            "/v1/chat",
            json=message(question, conversation_id=conversation_id, subject="gibson"),
        ).json()
        return parts_by_type(reply)["text"][0]["text"]

    cid = first["conversation_id"]
    assert ask("where do I work?", cid) == "You work at Hopsworks."
    assert ask("who am I?", cid) == "You are Dana (dana.facts@example.com)."
    assert ask("my fiscal year is April", cid).startswith("Noted: fiscal year April. ")
    assert ask("what is my fiscal year?", cid) == "Your fiscal year is April."
    assert ask("where do I live?", cid) == "You haven't told me where you live yet."

    # A new conversation, the address alone: the facts came with the person.
    again = ask(f"{who} -- what do you know about me?")
    assert again.startswith("Thanks, Dana (dana.facts@example.com). You are Dana"), (
        again
    )
    assert "- employer: Hopsworks" in again
    assert "- fiscal year: April" in again


def test_a_name_given_before_the_address_is_kept_for_the_person(client):
    first = client.post(
        "/v1/chat", json=message("Hello, I'm Priya", subject=None)
    ).json()
    text = parts_by_type(first)["text"][0]["text"]
    assert text.startswith(
        "Nice to meet you, Priya. Before we start, what is your email"
    ), text

    second = client.post(
        "/v1/chat",
        json=message(
            "priya@example.com", conversation_id=first["conversation_id"], subject=None
        ),
    ).json()
    text = parts_by_type(second)["text"][0]["text"]
    assert text.startswith("Thanks, Priya (priya@example.com). Noted: name Priya."), (
        text
    )

    later = client.post(
        "/v1/chat", json=message("priya@example.com -- who am I?", subject=None)
    ).json()
    assert parts_by_type(later)["text"][0]["text"].endswith(
        "You are Priya (priya@example.com)."
    )


def test_an_introduction_and_a_question_in_one_message(client):
    reply = client.post(
        "/v1/chat",
        json=message(
            "Hi, I'm Sam (sam@example.com), I prefer line charts. Total revenue by region",
            csv_file(SALES),
            subject=None,
        ),
    ).json()
    parts = parts_by_type(reply)
    assert set(parts) == {"text", "image", "file"}, "the question was answered"
    text = parts["text"][0]["text"]
    assert text.startswith(
        "Thanks, Sam (sam@example.com). Noted: name Sam, preference line charts. Sum of revenue by region"
    ), text


def test_two_facts_in_one_clause_and_a_bare_introduction(client):
    who = "two.facts@example.com"
    first = client.post(
        "/v1/chat", json=message(f"Hi am {who}", subject="gibson")
    ).json()
    text = parts_by_type(first)["text"][0]["text"]
    assert text.startswith(f"Thanks, {who}. Attach a CSV"), (
        "a bare introduction is not a question"
    )

    second = client.post(
        "/v1/chat",
        json=message(
            "I'm Dana, I work at Hopsworks and I prefer line charts",
            conversation_id=first["conversation_id"],
            subject="gibson",
        ),
    ).json()
    text = parts_by_type(second)["text"][0]["text"]
    assert text.startswith(
        "Noted: name Dana, employer Hopsworks, preference line charts."
    ), text
