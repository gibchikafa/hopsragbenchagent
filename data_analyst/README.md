# Data analyst agent

An agent that takes a file and gives back a picture. You attach a CSV and ask an
aggregate question -- "total revenue by region", "average units per product", "how
many orders by month", "describe the data" -- and it answers with three parts: a text
summary, a PNG chart and a CSV of the aggregated rows. It asks who it is talking to
first, remembers your table and your notes for next time, and takes `remember`,
`recall`, `forget` and `search` as commands.

No model in it, on purpose. Every example here that calls a model makes the reply
depend on what the model felt like saying; this one is about the protocol between a
caller and an agent when the message is more than text, and a reply that is the same
every time is what makes that protocol checkable: by the tests in this folder, and by
an evaluation suite with checks that call no judge. Swap `_plan` and `_summary` for a
model-backed analysis and the plumbing stays.

## Files

| File | Purpose |
|---|---|
| `data_analyst_agent.py` | The agent: reads the table, answers the question, returns text + PNG + CSV, served by `AgentApp` with the SDK's managed memory |
| `requirements.txt` | Deployment requirements: matplotlib (pandas and the memory service's SQLAlchemy are already in the agent image) |
| `sample_sales.csv` | A table to try it with: twenty orders across four regions, three products and three months |
| `test_data_analyst_agent.py` | The agent in-process, with FastAPI's test client and memory on SQLite: the manifest, both ways a file arrives, the reply's parts, the stream, the email question, a returning person, remember/recall/forget/search |
| `evaluation/` | Four suites and an online evaluator, no judge anywhere: the numbers, the table across turns, the email question, the memory commands. See [its README](evaluation/README.md) |
| `test_evaluation_cases.py` | Every evaluation case run through the agent in-process and graded with its suite's checks, so a change that would fail a suite fails here first |

## What it shows of the protocol

- **Modalities are declared on the app** -- `input_modalities=["text", "file"]`,
  `output_modalities=["text", "image", "file"]` -- and the manifest carries them.
  That is what makes the chat panel offer an attachment picker and render a returned
  image inline and a returned file as a download; a client of your own reads the same
  manifest. The panel only offers kinds the agent declared.
- **Parts are read off the request**: `request.text` joins the text parts;
  `request.files`, `request.images` and `request.audio_clips` are the rest, each
  base64 in `.data` with a `.media_type` (a file part also has a `.name`).
- **Parts are returned** with `AgentResponse.parts(TextContent(...), ImageContent(...),
  FileContent(...))`. From a `@stream` handler the text goes out as deltas and the
  binary parts travel only in the completed reply, so the completed reply carries the
  whole text again: a client that joins the deltas and one that reads the completed
  message have to agree.
- **Steps are reported** with `ctx.emit_event` around reading, aggregating and
  rendering, which the panel shows as chips and the trace records on the turn.

### How a file reaches the agent

Two ways, and the agent reads both:

| Sender | A CSV arrives as |
|---|---|
| A client posting to `/v1/chat`, an evaluation task, `agent.chat` from the SDK | a `file` part: `{"type": "file", "name": "sales.csv", "media_type": "text/csv", "data": "<base64>"}` |
| The Hopsworks chat panel | text, inlined into the message: `--- Attached file: sales.csv ---` … `--- End of sales.csv ---` |

The panel inlines anything with a text extension (`.csv`, `.tsv`, `.json`, `.md`,
`.py`, …, up to 256 KB) so that an agent which declared only `text` can still be
handed a file; images, audio and other binaries go as parts, up to 5 MB each, and
only when the agent declared that modality. An agent that wants a table from the
panel therefore parses the inlined block too; `ATTACHED_FILE` in the agent is the
panel's format.

## Memory

The agent uses the SDK's memory service, `ManagedMemoryService`, in the three tiers
the other examples use, with no model driving any of them:

| Tier | What it holds here | How |
|---|---|---|
| Working memory | The conversation, and the table it is about | The turns are the SDK's; the table is kept beside them in `session` scope with `ctx.remember(..., scope="session")`, so "and by product?" needs no re-attach and a restart or a second replica sees the same table |
| Short-term memory | Older turns, folded into a rolling summary | `summarize=fold`, a rule that keeps one line per question and drops the inlined attachments, which would otherwise sit in the summary at full length |
| Long-term memory | The person's last table (the CSV itself when under 60 KB), their last question, and their notes | `user` scope, keyed by the person: `ctx.remember` / `ctx.recall` / `ctx.forget`; `memory.search` reads their earlier conversations |

### Who it is talking to

Long-term memory is keyed by a person, so the agent asks for an email address before
it analyses anything, as the other examples do, and binds the conversation to it with
`ctx.rebind_subject`. The address can come with the question ("dana@example.com --
total revenue by region"); a table attached before the address is given is kept, so it
need not be attached again. A client with a login asserts `subject` on the request
instead and is never asked: `agent.chat(text, subject="dana@example.com")`.

A returning person is greeted with what they analysed last time, and the table is
there to ask about straight away:

```
> dana@example.com
Welcome back, dana@example.com. Last time you analysed sales.csv (20 rows); I still
have it. What would you like to know about sales.csv? ...
> max revenue by product
```

### remember, recall, forget, search

The four operations the SDK registers as tools for a model are commands here, each the
same call the tool makes, under the same person:

| Say | Does |
|---|---|
| `remember fiscal year: starts in April` | `ctx.remember("fiscal_year", "starts in April")` |
| `remember that the quarter closes on Friday` | the same, keyed by the first words |
| `recall fiscal year` | `ctx.recall("fiscal_year")` |
| `what do you remember` | every note, plus the last table |
| `forget fiscal year` / `forget everything` | `ctx.forget("fiscal_year")` / `ctx.forget()` |
| `search warehouse` | `memory.search("warehouse", subject=...)` over earlier conversations; keyword matching, since no embedder is configured (add `sentence_transformer_embedder()` and `vector_store_for(...)` for semantic search) |

The transcript itself records that a file arrived, never its bytes: the panel inlines
a CSV into the message, and the fold above is what keeps it out of the summary.

## Deploy

```python
import hopsworks

project = hopsworks.login()
agents = project.get_agent_serving()
agent = agents.deploy_agent(
    "data_analyst_agent.py",
    name="dataanalyst",             # a serving name is alphanumeric only
    requirements="requirements.txt",
    environment="data_analyst",
    tracing={"enabled": True},
)
agent.start(await_running=900)   # the first start builds the environment
```

Or with the CLI: `hops agent create data_analyst_agent.py --name dataanalyst
--requirements requirements.txt --environment data_analyst && hops agent start dataanalyst`.

Then open the deployment's chat panel, attach `sample_sales.csv` and ask
"total revenue by region".

## Talk to it from Python

```python
import base64

csv = open("sample_sales.csv").read()
# The chat panel's form, which agent.chat can send as text. `subject` is the
# person; without it the agent asks for an email address first.
reply = agent.chat(
    f"total revenue by region\n\n--- Attached file: sales.csv ---\n{csv}\n--- End of sales.csv ---",
    subject="dana@example.com",
)
print(reply.text)                                  # the summary and the table
[p["type"] for p in reply.parts]                   # ["text", "image", "file"]
png = base64.b64decode(next(p for p in reply.parts if p["type"] == "image")["data"])
open("chart.png", "wb").write(png)
agent.chat("and by product?", conversation_id=reply.conversation_id, subject="dana@example.com")
agent.chat("remember fiscal year: starts in April", subject="dana@example.com")
agent.chat("what do you remember", subject="dana@example.com")
```

A client of your own posts the file as a part instead, to `agent.url` with the
serving credential:

```python
requests.post(f"{agent.url}/v1/chat", headers=auth, json={
    "subject": "dana@example.com",
    "message": {"role": "user", "content": [
        {"type": "text", "text": "average units per product"},
        {"type": "file", "name": "sales.csv", "media_type": "text/csv",
         "data": base64.b64encode(csv.encode()).decode()},
    ]},
})
```

## Evaluate it

The suites are in [`evaluation/`](evaluation/): four of them plus an evaluator for
live traffic, all `contains` and `regex`, since the agent is deterministic and
calls no model. Create and freeze them, then run one against the deployment:

```bash
export HOPSWORKS_HOST=https://...  HOPSWORKS_PROJECT=...  HOPSWORKS_API_KEY=...
python -m data_analyst.evaluation.apply --publish            # from the repository root
python -m data_analyst.evaluation.sample --agent dataanalyst   # grade real traffic
```

```python
agents = project.get_agent_serving()
agent = agents.get_agent("dataanalyst")
run = agent.run(agents.suites.find("Answers the numbers")).wait()
```

A task's question is text, so a case carries the table inlined the way the chat
panel attaches it, and names its person in the question (`case-01@eval.example --
total revenue by region`) so the identity gate is passed inside the trial. Every
case also runs in-process in `test_evaluation_cases.py`.

## Run the tests

```bash
pip install "hopsworks[agents,agents-memory]" -r requirements.txt pytest httpx
pytest test_data_analyst_agent.py
```

Memory runs on SQLite in the tests (`DATA_ANALYST_MEMORY_URL`); in a deployment the
service reads the project database from the `MYSQL_*` variables the platform injects.
