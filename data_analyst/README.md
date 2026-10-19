# Data analyst agent

An agent that takes a file and gives back a picture. You attach a CSV and ask an
aggregate question -- "total revenue by region", "average units per product", "how
many orders by month", "describe the data" -- and it answers with three parts: a text
summary, a PNG chart and a CSV of the aggregated rows.

No model in it, on purpose. Every example here that calls a model makes the reply
depend on what the model felt like saying; this one is about the protocol between a
caller and an agent when the message is more than text, and a reply that is the same
every time is what makes that protocol checkable: by the tests in this folder, and by
an evaluation suite with checks that call no judge. Swap `_plan` and `_summary` for a
model-backed analysis and the plumbing stays.

## Files

| File | Purpose |
|---|---|
| `data_analyst_agent.py` | The agent: reads the table, answers the question, returns text + PNG + CSV, served by `AgentApp` |
| `requirements.txt` | Deployment requirements: matplotlib (pandas is already in the agent image) |
| `sample_sales.csv` | A table to try it with: twenty orders across four regions, three products and three months |
| `test_data_analyst_agent.py` | The agent in-process, with FastAPI's test client: the manifest, both ways a file arrives, the reply's parts, a follow-up, a refusal, the stream |

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

### What memory keeps

The transcript the memory tiers store records that a file arrived (`[file]`), never
its bytes: an image is megabytes of base64 and does not belong in a chat log. An
agent that needs the attachment on a later turn keeps it itself. This one holds the
last table per conversation in the process, bounded, so "and by product?" works
without a re-attach; a restart or a second replica starts empty, and the agent says
so rather than guessing. Put parsed data in your own store (`ctx.remember` for small
facts, a feature group or the project database for tables) when it has to survive
either.

## Deploy

```python
import hopsworks

project = hopsworks.login()
agents = project.get_agent_serving()
agent = agents.deploy_agent(
    "data_analyst_agent.py",
    name="data_analyst",
    requirements="requirements.txt",
    environment="data_analyst",
    tracing={"enabled": True},
)
agent.start(await_running=900)   # the first start builds the environment
```

Or with the CLI: `hops agent create data_analyst_agent.py --name data_analyst
--requirements requirements.txt --environment data_analyst && hops agent start data_analyst`.

Then open the deployment's chat panel, attach `sample_sales.csv` and ask
"total revenue by region".

## Talk to it from Python

```python
import base64

csv = open("sample_sales.csv").read()
# The chat panel's form, which agent.chat can send as text:
reply = agent.chat(
    f"total revenue by region\n\n--- Attached file: sales.csv ---\n{csv}\n--- End of sales.csv ---"
)
print(reply.text)                                  # the summary and the table
[p["type"] for p in reply.parts]                   # ["text", "image", "file"]
png = base64.b64decode(next(p for p in reply.parts if p["type"] == "image")["data"])
open("chart.png", "wb").write(png)
agent.chat("and by product?", conversation_id=reply.conversation_id)   # no re-attach
```

A client of your own posts the file as a part instead, to `agent.url` with the
serving credential:

```python
requests.post(f"{agent.url}/v1/chat", headers=auth, json={
    "message": {"role": "user", "content": [
        {"type": "text", "text": "average units per product"},
        {"type": "file", "name": "sales.csv", "media_type": "text/csv",
         "data": base64.b64encode(csv.encode()).decode()},
    ]},
})
```

## Evaluate it

The reply is deterministic, so a suite grades it with checks that call no model. A
task's question is text, so the inlined form is what a task carries:

```python
suite = agents.suites.create(
    "data-analyst", checks=[{"type": "contains", "name": "states the total"}]
)
question = f"total revenue by region\n\n--- Attached file: sales.csv ---\n{csv}\n--- End of sales.csv ---"
task = agents.tasks.create(question)
agents.tasks.add_to_suite(task, suite, expectations={"states the total": "Highest: East (2775)"})
run = agents.runs.start(agents.suites.publish(suite), agent.id)
```

A `regex` check on `^(Sum|Average|Count|Maximum|Minimum|Median) of ` needs no
expectation, so it grades live traffic too, through
`agents.runs.sample(agent.id, evaluator=...)`.

## Run the tests

```bash
pip install "hopsworks[agents]" -r requirements.txt pytest httpx
pytest test_data_analyst_agent.py
```
