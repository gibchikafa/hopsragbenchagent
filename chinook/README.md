# Chinook customer-support agent

A customer-support supervisor over the
[Chinook](https://github.com/lerocha/chinook-database) sample music store,
ported from a LangGraph/LangSmith notebook. The notebook's evaluation half is
deliberately not carried over — this is the agent only.

```
intent_classifier ─┬─▶ refund_agent               (gather_info → lookup | refund)
                   └─▶ question_answering_agent   (ReAct over catalogue lookups)
                                 │
                           compile_followup
```

| File | Purpose |
|---|---|
| `feature_pipeline.py` | Embeds every artist/album/track name into an embedding feature group (fuzzy name → canonical name) |
| `migrate_to_feature_store.py` | One-off migration of Chinook out of SQLite into keyed feature groups |
| `support_agent.py` | LangGraph support agent, served by `AgentApp` |
| `support_agent_llamaindex.py` | LlamaIndex support agent, served by `AgentApp` |
| `support_agent_openai.py` | OpenAI Agents SDK support agent, served by `AgentApp` |
| `support_agent_claude.py` | Claude Agent SDK support agent, served by `AgentApp` |
| `requirements.txt` | Deployment requirements |

### Feature groups

| Feature group | Key | Answers |
|---|---|---|
| `chinook_catalog_embeddings` | *(vector)* | "what is the canonical name for what the customer typed?" |
| `chinook_artist_catalog` | `artist_name` | "what albums and tracks does this artist have?" |
| `chinook_customers` | `customer_key` | "who is this, and which lines are theirs?" |
| `chinook_purchases` | `invoice_line_id` | one row per purchased line |
| `chinook_refunds` | `invoice_line_id` | "has this line already been refunded?" |

The online store is a keyed lookup, not a query engine, so the migration shapes
the data around the questions the agent actually asks.

Purchases are one row per invoice line. Because the online store cannot scan for
"every line belonging to this customer", the customer row carries the ids of
their lines and that list is the index: read the customer, then batch-read the
lines with `get_feature_vectors`. Two round trips instead of one, in exchange
for rows that are small and independently writable — adding a purchase rewrites
one short id list rather than the customer's entire history, which is what the
earlier single-JSON-blob shape required. `customer_key` is a deterministic hash of the first name,
last name and phone — the same three fields the refund flow already asks for, so
the agent computes it from the conversation rather than looking it up.

### Memory that outlives the conversation

`remember_interest` records two things the customer says: that they **want to
buy** something, or simply that they **like** it. On a later conversation the
agent is shown them again, each want reconciled against their actual orders:

```
From earlier conversations with this customer:
- wanted to buy 'Coda' — still not purchased
- wanted to buy 'Led Zeppelin I' — HAS SINCE BOUGHT IT
- likes 'Physical Graffiti'
```

That reconciliation is the interesting part: the *intention* lives in agent
memory, the *purchase* lives in the feature store, and neither knows about the
other until this step joins them. A fulfilled want is reported as fulfilled
rather than nagged about.

These are keyed on `customer_key` — the same hash of name, surname and phone the
purchases feature group uses — rather than on the conversation, which is what
makes them survive a new chat. The agent already asks for those three fields, so
the application supplies its own notion of identity instead of waiting for the
platform. It is *identification, not authentication*: knowing someone's name and
phone is enough to see their memories, exactly as it is already enough to see
their order history. Fine for a demo; do not put anything sensitive behind it.

### The agent can see the customer's own orders

`purchase_history` gives the question-answering agent read access to what the
signed-in customer has already bought, grouped by album, with the tracks, dates
and how many lines have since been refunded. Without it the agent could describe
the whole catalogue but not answer "what have I bought before" — purchase lookup
lived only inside the refund sub-graph and was unreachable from a general
question.

A LangChain tool is called by the model and gets no graph state, so the handler
publishes the turn's customer into a `ContextVar` that the tool reads — a
ContextVar rather than a global because two turns can be in flight at once.

### Identity is asked once per conversation

The graph will not route to refunds or catalogue questions until it knows the
customer's first name, last name and phone. The `identify` node asks for them
once, writes them to `session`-scoped memory, and the stream handler seeds the
graph from that on every later turn — so it asks once per chat, not once per
message, and the refund flow never has to ask again either.

Once per *conversation* rather than once per *customer* is a deliberate limit,
not an oversight. The chat transport authenticates with a project-wide serving
key, so the agent has no trustworthy way to tell one end user from another:
`subject` is whatever the client claims, defaulting to the conversation id.
Storing identity in `session` scope states that honestly. When real end-user
identity exists — the deployment-scoped chat token the panel design calls for —
flipping `IDENTITY_SCOPE` to `"user"` is the entire change needed to make it
once per customer, across conversations.

### Refunds are events, not deletions

The original deleted Invoice and InvoiceLine rows. That cannot be ported: **a
feature group has no row-level delete reachable from a Python client.**
`FeatureGroup.delete()` drops the entire feature group, `commit_delete_record()`
needs Spark and a Hudi/Delta/Iceberg table, and the Python engine has no per-row
delete at all.

So `chinook_refunds` is an append-only ledger and a line counts as refunded once
a row exists for it. Better modelling regardless of the platform — destroying
the record of a sale loses the audit trail — but it changes behaviour: refunded
purchases still appear in a customer's history, marked `refunded`, rather than
vanishing.

**What changed from the notebook.** The original built three in-process
`InMemoryVectorStore`s at import time to disambiguate what a customer types
("prince") against what the database stores ("Prince"). That re-embeds the whole
catalogue on every pod start, keeps three copies per replica, and can't be
rebuilt without redeploying. Here it's a feature pipeline writing one embedding
feature group that the agent queries online.

There is no SQLite at runtime. `migrate_to_feature_store.py` moves Chinook into
the feature groups above; the agent reads only from the feature store, so
nothing is pod-local and state survives restarts.

```bash
python feature_pipeline.py            # catalogue name embeddings
python migrate_to_feature_store.py    # invoices, catalogue, refund ledger

hops agent create support_agent.py --name chinooksupport \
    --requirements requirements.txt \
    --environment python-agent-pipeline-meb10000-v1
hops agent start chinooksupport --wait 600
```

| Variable | Default | Description |
|---|---|---|
| `CHINOOK_DB_PATH` | `chinook.db` | Source SQLite file — **migration scripts only**; the agent never reads it |
| `CHINOOK_ROUTER_MODEL` | `claude-haiku-4-5` | Intent routing + purchase-info extraction |
| `CHINOOK_ANSWER_MODEL` | `claude-haiku-4-5` | Catalogue question answering |
| `CHINOOK_OPENAI_ROUTER_MODEL` | OpenAI Agents SDK default | OpenAI identity extraction |
| `CHINOOK_OPENAI_ANSWER_MODEL` | OpenAI Agents SDK default | OpenAI catalogue/refund answering |
| `CHINOOK_CLAUDE_ROUTER_MODEL` | `CHINOOK_ANSWER_MODEL` | Claude Agent SDK identity extraction |
| `CHINOOK_CLAUDE_ANSWER_MODEL` | `CHINOOK_ANSWER_MODEL` | Claude Agent SDK catalogue/refund answering |

## Four agents, one store

| file | framework | entrypoint |
|---|---|---|
| `support_agent.py` | LangGraph supervisor | `chinook/support_agent.py` |
| `support_agent_llamaindex.py` | LlamaIndex `FunctionAgent` | `chinook/support_agent_llamaindex.py` |
| `support_agent_openai.py` | OpenAI Agents SDK `Agent` | `chinook/support_agent_openai.py` |
| `support_agent_claude.py` | Claude Agent SDK + in-process MCP tools | `chinook/support_agent_claude.py` |

All four import the store and all seven tools from `store.py`, and use a byte-identical
system prompt. That is deliberate: an evaluation suite written against one should
hold against the others, so a difference in results is a difference in the
framework rather than in what the agent was told to do.

They differ in wiring only. The LangGraph agent routes through sub-agents — one
to identify the customer, one to answer, one to refund; the LlamaIndex and
OpenAI entrypoints give a single agent all seven tools; the Claude entrypoint
exposes those tools through an in-process MCP server. The rules that matter are
enforced in the tools either way:

- `purchase_history` and `place_order` refuse without a first name, last name and
  the phone on the account, whoever asks them
- `place_order` records an interest instead of an order unless it is told the
  customer said to buy

A rule that only holds because a supervisor routed correctly is a rule that holds
until the routing changes, which is why they are in the tools.

Deploy any one by pointing a deployment's entrypoint at that file. The suites in
`evaluation/` apply to all four.
