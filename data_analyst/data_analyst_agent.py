"""A data analyst agent: a CSV in, a chart and the numbers out.

Multimodal both ways, and no model in it. The user attaches a table and asks
an aggregate question -- "total revenue by region", "average price per
category", "describe the data" -- and the agent answers with three parts:
a text summary, a PNG chart and a CSV of the aggregated rows. Every part is
computed from the data, so the reply is the same for the same question, which
is what makes it gradable with `contains` and `regex` checks and usable as a
template: swap `_answer` for a model-backed analysis and the plumbing stays.

What the example shows of the protocol:

- `input_modalities` / `output_modalities` in the manifest, which is what
  makes the chat panel offer an attachment picker and render the reply's
  image inline and its file as a download.
- A file arriving two ways. A client posting to `/v1/chat` sends it as a
  `file` part (base64). The Hopsworks chat panel inlines text files into the
  message instead, as `--- Attached file: <name> --- ... --- End of <name> ---`
  blocks, so an agent that wants a CSV from the panel reads both.
- `AgentResponse.parts(...)` mixing text, image and file, from a `@stream`
  handler: the text goes out as deltas, the binary parts travel only in the
  completed reply, and the completed reply carries the whole text again.
- Progress events (`ctx.emit_event`) around each step, which the panel shows
  as chips and the trace records on the turn.

Memory, in three tiers, with no model driving any of it:

- Working memory is the conversation, kept by `ManagedMemoryService` in the
  project database; the chat panel's Memory tab shows it. The table attached
  to a conversation is kept beside it in session scope (`ctx.remember(...,
  scope="session")`), so a follow-up question needs no re-attach and a
  restart or a second replica sees the same table.
- Short-term memory is a rolling summary of older turns, folded by a rule
  (`fold`) that keeps the questions and drops the inlined attachments, which
  would otherwise sit in the transcript at full length.
- Long-term memory is keyed by the person, so the agent first asks for an
  email address, as the other examples do, and binds the conversation to it
  with `ctx.rebind_subject`. Under that subject it remembers the last table
  (the CSV itself, when small) and the last question, so a returning user is
  greeted with what they analysed last time and can ask straight away. A
  client that already knows who it is speaking to asserts `subject` on the
  request and the question is skipped.
- The memory operations the other examples give a model as tools --
  `remember`, `recall`, `forget`, `search` -- are commands here, since there
  is no model to call them: "remember fiscal year: starts in April",
  "recall fiscal year", "what do you remember", "forget fiscal year",
  "search revenue". Each is the SDK call the tool makes (`ctx.remember`,
  `ctx.recall`, `ctx.forget`, `memory.search`), keyed by the person.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re

import matplotlib


matplotlib.use("Agg")  # headless: there is no display in a serving pod

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from hopsworks_agents.protocol import (  # noqa: E402
    AgentApp,
    AgentError,
    AgentResponse,
    FileContent,
    ImageContent,
    ManagedMemoryService,
    TextContent,
)


#: How the Hopsworks chat panel inlines a text attachment into the message.
ATTACHED_FILE = re.compile(
    r"--- Attached file: (?P<name>[^\n]+?) ---\n(?P<body>.*?)\n--- End of (?P=name) ---",
    re.DOTALL,
)
TABLE_SUFFIXES = (".csv", ".tsv", ".txt")
TABLE_MEDIA_TYPES = ("text/csv", "text/tab-separated-values", "text/plain")

#: Words in a question that pick the aggregate. Checked in this order; "sum"
#: when none is present.
AGGREGATES: list[tuple[str, tuple[str, ...]]] = [
    ("mean", ("average", "mean", "avg")),
    ("median", ("median",)),
    ("count", ("count", "how many", "number of")),
    ("max", ("max", "maximum", "highest", "largest", "biggest")),
    ("min", ("min", "minimum", "lowest", "smallest")),
    ("sum", ("sum", "total")),
]
AGGREGATE_LABELS = {
    "sum": "Sum",
    "mean": "Average",
    "median": "Median",
    "count": "Count",
    "max": "Maximum",
    "min": "Minimum",
}
DESCRIBE_WORDS = ("describe", "summar", "what is in", "what's in", "columns", "schema")
#: "by month" on a table with a date column groups by that period of the date.
PERIODS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
MAX_GROUPS_IN_CHART = 20
MAX_GROUPS_IN_TEXT = 10

HELP = (
    "Attach a CSV and ask me about it. I answer aggregate questions -- "
    "'total revenue by region', 'average price per category', 'how many orders "
    "by month' -- with a summary, a chart and the numbers as a CSV. "
    "'Describe the data' lists the columns."
)
ASK_EMAIL = (
    "Before we start, what is your email address? It is how I keep what you "
    "analysed for next time."
)
MEMORY_HELP = (
    "I also keep notes for you: 'remember <key>: <value>', 'recall <key>', "
    "'what do you remember', 'forget <key>', and 'search <words>' looks through "
    "our earlier conversations."
)
#: The memory commands, matched at the start of the question.
REMEMBER = re.compile(
    r"^remember\s+(?:that\s+)?(?P<rest>.+)$", re.IGNORECASE | re.DOTALL
)
RECALL = re.compile(
    r"^(?:recall|what do you remember(?: about)?|what do you know(?: about me)?)\s*(?P<key>.*)$",
    re.IGNORECASE,
)
FORGET = re.compile(r"^forget\s+(?P<key>.+)$", re.IGNORECASE)
SEARCH = re.compile(r"^search\s+(?:for\s+)?(?P<query>.+)$", re.IGNORECASE)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
#: Keys in durable memory. `session` scope is this conversation; `user` scope
#: is the person, across conversations.
SESSION_EMAIL = "email"
LAST_TABLE = "last_table"
TABLE_PREFIX = "table:"
#: A table this size or smaller is kept in memory whole -- for the rest of the
#: conversation, and for the person's next one; bigger ones are remembered by
#: name and shape only and have to be attached with each question. The cap is
#: the state table's TEXT column.
TABLE_MEMORY_CHARS = 60_000
SESSION_TABLE = "table"
SESSION_TABLE_NAME = "table_name"


def _is_agent_key(key: str) -> bool:
    """The keys the agent writes for itself, as opposed to a person's notes."""
    return key == LAST_TABLE or key.startswith(TABLE_PREFIX)


#: Short-term memory: fold the conversation past this many messages, keeping
#: the latest few verbatim.
SUMMARIZE_AFTER_MESSAGES = 12
KEEP_RECENT_MESSAGES = 6


def fold(previous: str | None, turns: list[dict]) -> str:
    """Short-term memory by rule: the questions asked, without the attachments.

    A model would compress; this keeps one line per user turn. What matters is
    what it drops: the chat panel inlines a CSV into the message, so without
    this the summary would carry every table ever attached, in full.
    """
    said = []
    for turn in turns:
        if turn.get("role") != "user":
            continue
        text = str(turn.get("content") or "")
        names = [m["name"].strip() for m in ATTACHED_FILE.finditer(text)]
        text = ATTACHED_FILE.sub("", text).strip()
        if names:
            text = f"{text} [attached {', '.join(names)}]".strip()
        if text:
            said.append(f"earlier: {text}")
    return " | ".join(([previous.strip()] if previous else []) + said)


memory = ManagedMemoryService(
    # The project database in a deployment (MYSQL_* is injected); anything
    # SQLAlchemy opens locally, e.g. sqlite:///memory.db for the tests.
    url=os.environ.get("DATA_ANALYST_MEMORY_URL"),
    summarize=fold,
    summarize_after_messages=SUMMARIZE_AFTER_MESSAGES,
    keep_recent_messages=KEEP_RECENT_MESSAGES,
    long_term=True,
    max_state_value_chars=TABLE_MEMORY_CHARS,
)

app = AgentApp(
    name="Data analyst",
    description=(
        "Reads a CSV you attach, answers aggregate questions about it, and "
        "replies with a chart and the numbers. Remembers your last table."
    ),
    input_modalities=["text", "file"],
    output_modalities=["text", "image", "file"],
    tool_events=True,
    memory=memory,
    welcome_message=f"Hello! {HELP} {ASK_EMAIL}",
    suggested_prompts=[
        "dana@example.com",
        "describe the data",
        "total revenue by region",
        "average price by category",
    ],
    placeholder="Attach a CSV, then ask: total <column> by <column>",
)

# ── reading the table ─────────────────────────────────────────────────────


def _is_table(name: str | None, media_type: str | None) -> bool:
    lowered = (name or "").lower()
    return lowered.endswith(TABLE_SUFFIXES) or (media_type or "") in TABLE_MEDIA_TYPES


def _parse_table(name: str, text: str) -> pd.DataFrame:
    try:
        # sep=None lets pandas sniff commas, tabs and semicolons
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 — anything pandas cannot read
        raise AgentError(
            f"I could not read {name} as a table: {exc}", "bad_table", 400
        ) from exc
    if frame.empty or frame.shape[1] == 0:
        raise AgentError(f"{name} has no rows to analyse.", "empty_table", 400)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _attached_tables(request) -> list[tuple[str, pd.DataFrame, str]]:
    """Every table the message carries, with its text: file parts, then inlined blocks."""
    found: list[tuple[str, pd.DataFrame, str]] = []
    for part in request.files:
        if _is_table(part.name, part.media_type):
            name = part.name or "attachment.csv"
            text = base64.b64decode(part.data).decode("utf-8", errors="replace")
            found.append((name, _parse_table(name, text), text))
    for match in ATTACHED_FILE.finditer(request.text):
        name = match["name"].strip()
        if _is_table(name, None):
            found.append((name, _parse_table(name, match["body"]), match["body"]))
    return found


def _question(request) -> str:
    """The message without the inlined attachments and without an email address.

    "dana@example.com -- total revenue by region" is the question after the
    separator, so what the address leaves behind -- dashes, a colon, a comma,
    "it's" -- is trimmed from the front as well.
    """
    text = EMAIL.sub("", ATTACHED_FILE.sub("", request.text)).strip()
    return re.sub(
        r"^(?:it'?s|i'?m|this is)?\s*[-:,;.]*\s*", "", text, flags=re.IGNORECASE
    ).strip()


# ── understanding the question ────────────────────────────────────────────


def _normalise(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _columns_mentioned(question: str, frame: pd.DataFrame) -> list[tuple[int, str]]:
    """Columns named in the question, by where they appear; longest names win."""
    haystack = _normalise(question)
    hits: list[tuple[int, str]] = []
    for column in sorted(frame.columns, key=len, reverse=True):
        needle = _normalise(column)
        at = haystack.find(needle)
        if at >= 0:
            # the needle starts with the padding space; the name is one in
            hits.append((at + 1, column))
            haystack = haystack.replace(needle, " " * len(needle), 1)
    return sorted(hits)


def _aggregate_in(question: str) -> str:
    text = _normalise(question)
    for name, words in AGGREGATES:
        if any(f" {w} " in text for w in words):
            return name
    return "sum"


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def _is_time(frame: pd.DataFrame, column: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(frame[column]):
        return True
    return any(
        w in column.lower() for w in ("date", "time", "month", "week", "day", "year")
    )


def _time_column(frame: pd.DataFrame) -> str | None:
    return next((c for c in frame.columns if _is_time(frame, c)), None)


def _plan(question: str, frame: pd.DataFrame) -> tuple[str, str, str, pd.DataFrame]:
    """(aggregate, value column, group column, frame) for a question.

    The column after "by" / "per" / "for each" groups; the other named column
    is the value. "By month" (day, week, quarter, year) on a table with a
    date column groups by that period of it, added as a column of the frame
    returned. Unnamed ones fall back to the first numeric column and the
    non-numeric column with the fewest distinct values, which is what "total
    revenue" means on most tables. A "by" with nothing recognisable after it
    is refused by name, so the user learns the columns rather than getting
    the wrong chart.
    """
    aggregate = _aggregate_in(question)
    mentioned = _columns_mentioned(question, frame)
    text = _normalise(question)
    numeric = _numeric_columns(frame)

    group = None
    by = re.search(r" (?:by|per|for each|for every|across) (?P<rest>.+)$", text)
    if by:
        start = by.start("rest")
        after = [c for at, c in mentioned if at >= start]
        period = by["rest"].split()[0]
        time_column = _time_column(frame)
        if after:
            group = after[0]
        elif period in PERIODS and time_column is not None:
            dates = pd.to_datetime(frame[time_column], errors="coerce")
            frame = frame.assign(
                **{period: dates.dt.to_period(PERIODS[period]).astype(str)}
            )
            group = period
        else:
            raise AgentError(
                f"I don't know a column called {by['rest'].strip()!r}. "
                f"The table has: {', '.join(frame.columns)}.",
                "unknown_column",
                400,
            )
    value_candidates = [c for _, c in mentioned if c != group]
    value = next((c for c in value_candidates if c in numeric), None)
    if value is None and aggregate != "count":
        value = next((c for c in numeric if c != group), None)
    if value is None:
        if aggregate == "count":
            value = next((c for c in frame.columns if c != group), frame.columns[0])
        else:
            raise AgentError(
                f"The table has no numeric column to take the {aggregate} of. "
                f"It has: {', '.join(frame.columns)}.",
                "no_numeric_column",
                400,
            )
    if group is None:
        categorical = [c for c in frame.columns if c != value and c not in numeric]
        group = (
            min(categorical, key=lambda c: frame[c].nunique())
            if categorical
            else next((c for c in frame.columns if c != value), value)
        )
    return aggregate, value, group, frame


# ── who this is, and what they analysed before ────────────────────────────


def _who(ctx, request) -> tuple[str | None, bool]:
    """(the person, whether this turn is where they said who they are).

    A client with a login asserts ``subject`` on the request and nothing is
    asked. Otherwise the address given earlier in this conversation is in
    session memory, and failing that the message itself is searched for one.
    Either way ``rebind_subject`` points durable memory at the person rather
    than at the conversation.
    """
    if ctx.has_subject and ctx.subject_source == "client":
        return ctx.subject, False
    known = ctx.state("session").get(SESSION_EMAIL)
    if known:
        ctx.rebind_subject(known)
        return known, False
    match = EMAIL.search(request.text)
    if match is None:
        return None, False
    email = match.group(0).lower()
    ctx.rebind_subject(email)
    ctx.remember(SESSION_EMAIL, email, scope="session")
    return email, True


def _keep_for_conversation(ctx, name: str, raw: str) -> bool:
    """Working memory of the table: session scope, beside the conversation's turns."""
    ctx.remember(SESSION_TABLE_NAME, name, scope="session")
    if len(raw) > TABLE_MEMORY_CHARS:
        ctx.forget(SESSION_TABLE, scope="session")
        return False
    ctx.remember(SESSION_TABLE, raw, scope="session")
    return True


def _keep_for_person(
    ctx, name: str, frame: pd.DataFrame, raw: str, question: str | None = None
) -> None:
    """Long-term memory of what was attached: the shape always, the table when small."""
    ctx.remember(
        LAST_TABLE,
        json.dumps(
            {
                "name": name,
                "rows": int(len(frame)),
                "columns": [c for c in frame.columns if c not in PERIODS],
                "question": question,
            }
        ),
    )
    if len(raw) <= TABLE_MEMORY_CHARS:
        ctx.remember(TABLE_PREFIX + name, raw)


def _conversation_table(ctx) -> tuple[str, pd.DataFrame, str] | None:
    """The table this conversation is about, from session memory."""
    state = ctx.state("session")
    raw = state.get(SESSION_TABLE)
    if not raw:
        return None
    name = state.get(SESSION_TABLE_NAME) or "attachment.csv"
    try:
        return name, _parse_table(name, raw), raw
    except AgentError:
        return None


def _remembered(ctx) -> tuple[dict | None, tuple[str, pd.DataFrame, str] | None]:
    """The last table this person analysed, and the table itself when it was kept."""
    state = ctx.state("user")
    try:
        last = json.loads(state.get(LAST_TABLE) or "")
    except ValueError:
        return None, None
    name = str(last.get("name"))
    raw = state.get(TABLE_PREFIX + name)
    if not raw:
        return last, None
    try:
        return last, (name, _parse_table(name, raw), raw)
    except AgentError:
        return last, None


# ── remember, recall, forget, search ──────────────────────────────────────


def _slug(text: str, words: int = 4) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", text.lower())[:words]) or "note"


def _memory_command(ctx, question: str) -> str | None:
    """A memory operation, if the question is one; None when it is a data question.

    The same four operations the SDK registers as tools for a model, called
    the way the tools call them, under the person the turn is bound to.
    """
    match = REMEMBER.match(question)
    if match:
        rest = match["rest"].strip()
        key, sep, value = rest.partition(":")
        if sep and value.strip():
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
        else:
            # "remember that the fiscal year starts in April": keyed by its words
            key, value = _slug(rest), rest
        ctx.remember(key, value)
        return f"Remembered {key}: {value}"

    match = RECALL.match(question)
    if match:
        key = match["key"].strip().strip("?").lower().replace(" ", "_")
        if key:
            value = ctx.recall(key)
            return f"{key}: {value}" if value else f"Nothing stored under {key!r}."
        notes = {k: v for k, v in ctx.state("user").items() if not _is_agent_key(k)}
        last, _table = _remembered(ctx)
        lines = []
        if last:
            asked = f', last asked "{last["question"]}"' if last.get("question") else ""
            lines.append(
                f"Your last table: {last['name']} ({last['rows']} rows{asked})."
            )
        if notes:
            lines.append("Your notes:")
            lines += [f"- {k}: {v}" for k, v in sorted(notes.items())]
        return "\n".join(lines) if lines else "I have nothing stored for you yet."

    match = FORGET.match(question)
    if match:
        key = match["key"].strip().strip("?").lower()
        if key in ("everything", "all", "me", "it all"):
            removed = ctx.forget()
            return f"Forgot everything I had for you ({removed} entries)."
        key = key.replace(" ", "_")
        removed = ctx.forget(key)
        return f"Forgot {key}." if removed else f"Nothing stored under {key!r}."

    match = SEARCH.match(question)
    if match:
        query = match["query"].strip()
        hits = ctx.memory.search(query, subject=ctx.subject) if ctx.memory else []
        if not hits:
            return f"Nothing in our earlier conversations about {query!r}."
        lines = [f"From earlier conversations, about {query!r}:"]
        for hit in hits[:5]:
            content = ATTACHED_FILE.sub(
                "[attached table]", str(hit.get("content") or "")
            )
            lines.append(f"- ({hit.get('role', '?')}) {content[:200]}")
        return "\n".join(lines)
    return None


# ── answering ─────────────────────────────────────────────────────────────


def _fmt(number) -> str:
    try:
        return f"{float(number):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(number)


def _aggregate(
    frame: pd.DataFrame, aggregate: str, value: str, group: str
) -> pd.DataFrame:
    grouped = frame.groupby(group, dropna=False, sort=False)[value]
    result = getattr(grouped, aggregate)().reset_index()
    result.columns = [group, value]
    if _is_time(frame, group):
        order = pd.to_datetime(result[group], errors="coerce")
        if order.notna().all():
            return (
                result.assign(_order=order).sort_values("_order").drop(columns="_order")
            )
    return result.sort_values(value, ascending=False)


def _chart_png(result: pd.DataFrame, title: str, time_axis: bool) -> bytes:
    group, value = result.columns[0], result.columns[1]
    shown = result.head(MAX_GROUPS_IN_CHART)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    labels = [str(v) for v in shown[group]]
    if time_axis:
        ax.plot(labels, shown[value], marker="o")
    else:
        ax.bar(labels, shown[value])
    ax.set_title(title)
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    if len(labels) > 6 or max(len(label) for label in labels) > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _summary(
    name: str,
    frame: pd.DataFrame,
    aggregate: str,
    value: str,
    group: str,
    result: pd.DataFrame,
) -> str:
    label = AGGREGATE_LABELS[aggregate]
    by_value = result.sort_values(value, ascending=False)
    highest, lowest = by_value.iloc[0], by_value.iloc[-1]
    lines = [
        f"{label} of {value} by {group}, from {len(frame)} rows of {name}: "
        f"{len(result)} group{'' if len(result) == 1 else 's'}. "
        f"Highest: {highest[group]} ({_fmt(highest[value])}). "
        f"Lowest: {lowest[group]} ({_fmt(lowest[value])}).",
        "",
        f"| {group} | {label.lower()} of {value} |",
        "| --- | --- |",
    ]
    lines += [
        f"| {row[group]} | {_fmt(row[value])} |"
        for _, row in result.head(MAX_GROUPS_IN_TEXT).iterrows()
    ]
    if len(result) > MAX_GROUPS_IN_TEXT:
        lines.append(
            f"| ... | {len(result) - MAX_GROUPS_IN_TEXT} more in the attached CSV |"
        )
    return "\n".join(lines)


def _describe(name: str, frame: pd.DataFrame) -> tuple[str, FileContent]:
    lines = [
        f"{name}: {len(frame)} rows, {frame.shape[1]} columns.",
        "",
        "| column | type | example |",
        "| --- | --- | --- |",
    ]
    for column in frame.columns:
        first = frame[column].dropna()
        example = str(first.iloc[0]) if len(first) else ""
        lines.append(f"| {column} | {frame[column].dtype} | {example[:40]} |")
    numeric = _numeric_columns(frame)
    if numeric:
        lines.append("")
        lines.append(
            "Numeric columns I can total, average, count, max or min: "
            + ", ".join(numeric)
            + "."
        )
    stats = (
        frame.describe(include="all")
        .transpose()
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return "\n".join(lines), _csv_part(stats, "describe.csv")


def _csv_part(frame: pd.DataFrame, name: str) -> FileContent:
    data = frame.to_csv(index=False).encode("utf-8")
    return FileContent(
        name=name, media_type="text/csv", data=base64.b64encode(data).decode("ascii")
    )


def _usage(request, text: str) -> dict[str, int]:
    # No model here, so the turn's own figures are the only usage a reader finds.
    return {
        "input_tokens": len(request.text.split()),
        "output_tokens": len(text.split()),
    }


@app.stream
async def stream(request, ctx):
    attached = _attached_tables(request)
    kept = True
    for name, frame, raw in attached:
        await ctx.emit_event(
            "load_table",
            status="done",
            message=f"read {name}: {len(frame)} rows, {frame.shape[1]} columns",
            data={
                "file": name,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            },
            event_id=f"load-{name}",
        )
        kept = _keep_for_conversation(ctx, name, raw)

    email, just_identified = _who(ctx, request)
    if email is None:
        # Ask first, as the other examples do; a table that came with the
        # question is already in session memory, so it need not be attached again.
        text = ASK_EMAIL
        if attached:
            text = f"I have {attached[-1][0]} ready. {ASK_EMAIL}"
        yield AgentResponse.text(text, usage=_usage(request, text))
        return

    for name, frame, raw in attached:
        _keep_for_person(ctx, name, frame, raw)

    greeting = ""
    table = attached[-1] if attached else _conversation_table(ctx)
    if table is None:
        last, table = _remembered(ctx)
        if table is not None:
            _keep_for_conversation(ctx, table[0], table[2])
        if just_identified and last is not None:
            still = (
                "I still have it"
                if table is not None
                else "attach it again to continue"
            )
            greeting = (
                f"Welcome back, {email}. Last time you analysed {last['name']} "
                f"({last['rows']} rows); {still}. "
            )
    if not greeting and just_identified:
        greeting = f"Thanks, {email}. "
    if attached and not kept:
        greeting += (
            f"{attached[-1][0]} is too big for me to keep between turns, so "
            "attach it with each question. "
        )

    question = _question(request)
    answer = _memory_command(ctx, question) if question else None
    if answer is not None:
        text = greeting + answer
        yield AgentResponse.text(text, usage=_usage(request, text))
        return
    if table is None:
        text = f"{greeting}{HELP} {MEMORY_HELP}"
        yield AgentResponse.text(text, usage=_usage(request, text))
        return
    name, frame, raw = table
    if not question:
        if just_identified:
            text = f"{greeting}What would you like to know about {name}? {HELP} {MEMORY_HELP}"
            yield AgentResponse.text(text, usage=_usage(request, text))
            return
        question = "describe the data"

    if any(w in question.lower() for w in DESCRIBE_WORDS):
        text, csv_part = _describe(name, frame)
        text = greeting + text
        yield text
        yield AgentResponse.parts(
            TextContent(text=text), csv_part, usage=_usage(request, text)
        )
        return

    aggregate, value, group, frame = _plan(question, frame)
    await ctx.emit_event(
        "aggregate",
        status="running",
        message=f"{aggregate}({value}) by {group}",
        event_id="aggregate",
    )
    try:
        result = _aggregate(frame, aggregate, value, group)
    except Exception as exc:  # noqa: BLE001 — e.g. a text column under sum
        await ctx.emit_event(
            "aggregate", status="failed", message=str(exc), event_id="aggregate"
        )
        raise AgentError(
            f"I could not take the {aggregate} of {value!r} by {group!r}: {exc}",
            "aggregation_failed",
            400,
        ) from exc
    await ctx.emit_event(
        "aggregate",
        status="done",
        message=f"{len(result)} groups",
        event_id="aggregate",
    )

    text = greeting + _summary(name, frame, aggregate, value, group, result)
    # Text first, as deltas: a client renders the numbers while the chart renders.
    for line in text.splitlines(keepends=True):
        yield line

    title = f"{AGGREGATE_LABELS[aggregate]} of {value} by {group}"
    await ctx.emit_event(
        "render_chart", status="running", message=title, event_id="chart"
    )
    png = _chart_png(result, title, time_axis=_is_time(frame, group))
    await ctx.emit_event("render_chart", status="done", message=title, event_id="chart")
    # What they asked, for the greeting next time.
    _keep_for_person(ctx, name, frame, raw, question)

    stem = (
        re.sub(r"[^A-Za-z0-9]+", "_", f"{aggregate}_{value}_by_{group}")
        .strip("_")
        .lower()
    )
    # Binary parts do not stream: they arrive with the completed reply, which
    # carries the whole text again so joining the deltas and reading the
    # completed message agree.
    yield AgentResponse.parts(
        TextContent(text=text),
        ImageContent(
            media_type="image/png", data=base64.b64encode(png).decode("ascii")
        ),
        _csv_part(result, f"{stem}.csv"),
        usage=_usage(request, text),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
