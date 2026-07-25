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
| `support_agent.py` | The agent, served by `AgentApp` |
| `requirements.txt` | Deployment requirements |

### Feature groups

| Feature group | Key | Answers |
|---|---|---|
| `chinook_catalog_embeddings` | *(vector)* | "what is the canonical name for what the customer typed?" |
| `chinook_artist_catalog` | `artist_name` | "what albums and tracks does this artist have?" |
| `chinook_customer_purchases` | `customer_key` | "what did this customer buy?" |
| `chinook_refunds` | `invoice_line_id` | "has this line already been refunded?" |

The online store is a keyed lookup, not a query engine, so the migration
denormalises around the two questions the agent actually asks and makes each one
a single keyed read. `customer_key` is a deterministic hash of the first name,
last name and phone — the same three fields the refund flow already asks for, so
the agent computes it from the conversation rather than looking it up.

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
