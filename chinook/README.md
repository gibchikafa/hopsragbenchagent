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
| `feature_pipeline.py` | Embeds every artist/album/track name into one Hopsworks embedding feature group |
| `support_agent.py` | The agent, served by `AgentApp` |
| `requirements.txt` | Deployment requirements |

**What changed from the notebook.** The original built three in-process
`InMemoryVectorStore`s at import time to disambiguate what a customer types
("prince") against what the database stores ("Prince"). That re-embeds the whole
catalogue on every pod start, keeps three copies per replica, and can't be
rebuilt without redeploying. Here it's a feature pipeline writing one embedding
feature group that the agent queries online.

Chinook itself stays in SQLite — the refund path deletes invoice rows, which is
not what a feature store is for. The file is pod-local, so **refunds do not
survive a restart** unless `CHINOOK_DB_PATH` points at a mounted dataset.

```bash
python feature_pipeline.py        # once, before first deploy

hops agent create support_agent.py --name chinooksupport \
    --requirements requirements.txt \
    --environment python-agent-pipeline-meb10000-v1
hops agent start chinooksupport --wait 600
```

| Variable | Default | Description |
|---|---|---|
| `CHINOOK_DB_PATH` | `chinook.db` | SQLite location; downloaded on first use if absent |
| `CHINOOK_ROUTER_MODEL` | `claude-haiku-4-5` | Intent routing + purchase-info extraction |
| `CHINOOK_ANSWER_MODEL` | `claude-haiku-4-5` | Catalogue question answering |
