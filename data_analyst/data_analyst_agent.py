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

The table is kept in this process per conversation, so a follow-up question
needs no re-attach; the transcript the memory tiers store records only that a
file arrived, never its contents. A restart or a second replica starts empty,
and the agent says so rather than guessing.
"""

from __future__ import annotations

import base64
import io
import re
from collections import OrderedDict

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
#: Tables remembered, one per conversation, oldest dropped first.
MAX_CONVERSATIONS = 100

HELP = (
    "Attach a CSV and ask me about it. I answer aggregate questions -- "
    "'total revenue by region', 'average price per category', 'how many orders "
    "by month' -- with a summary, a chart and the numbers as a CSV. "
    "'Describe the data' lists the columns."
)

app = AgentApp(
    name="Data analyst",
    description=(
        "Reads a CSV you attach, answers aggregate questions about it, and "
        "replies with a chart and the numbers."
    ),
    input_modalities=["text", "file"],
    output_modalities=["text", "image", "file"],
    tool_events=True,
    welcome_message=HELP,
    suggested_prompts=[
        "describe the data",
        "total revenue by region",
        "average price by category",
    ],
    placeholder="Attach a CSV, then ask: total <column> by <column>",
)

#: The last table each conversation attached: (file name, frame).
_tables: OrderedDict[str, tuple[str, pd.DataFrame]] = OrderedDict()


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


def _attached_tables(request) -> list[tuple[str, pd.DataFrame]]:
    """Every table the message carries: file parts first, then inlined blocks."""
    found: list[tuple[str, pd.DataFrame]] = []
    for part in request.files:
        if _is_table(part.name, part.media_type):
            name = part.name or "attachment.csv"
            text = base64.b64decode(part.data).decode("utf-8", errors="replace")
            found.append((name, _parse_table(name, text)))
    for match in ATTACHED_FILE.finditer(request.text):
        name = match["name"].strip()
        if _is_table(name, None):
            found.append((name, _parse_table(name, match["body"])))
    return found


def _question(request) -> str:
    """The message without the inlined attachments."""
    return ATTACHED_FILE.sub("", request.text).strip()


def _remember(conversation_id: str, name: str, frame: pd.DataFrame) -> None:
    _tables[conversation_id] = (name, frame)
    _tables.move_to_end(conversation_id)
    while len(_tables) > MAX_CONVERSATIONS:
        _tables.popitem(last=False)


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
        f"{len(result)} groups. Highest: {highest[group]} ({_fmt(highest[value])}). "
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
    for name, frame in attached:
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
        _remember(ctx.conversation_id, name, frame)

    table = _tables.get(ctx.conversation_id)
    question = _question(request)
    if table is None:
        yield AgentResponse.text(
            HELP
            if not question
            else f"I have no table for this conversation yet. {HELP}",
            usage=_usage(request, HELP),
        )
        return
    name, frame = table
    if not question:
        question = "describe the data"

    if any(w in question.lower() for w in DESCRIBE_WORDS):
        text, csv_part = _describe(name, frame)
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

    text = _summary(name, frame, aggregate, value, group, result)
    # Text first, as deltas: a client renders the numbers while the chart renders.
    for line in text.splitlines(keepends=True):
        yield line

    title = f"{AGGREGATE_LABELS[aggregate]} of {value} by {group}"
    await ctx.emit_event(
        "render_chart", status="running", message=title, event_id="chart"
    )
    png = _chart_png(result, title, time_axis=_is_time(frame, group))
    await ctx.emit_event("render_chart", status="done", message=title, event_id="chart")

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
