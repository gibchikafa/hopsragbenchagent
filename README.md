# RAGBench Agent

A RAG system built on the [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) dataset, deployed on Hopsworks.

## Architecture

- **Feature pipeline** — downloads the RAGBench dataset, generates 384-dim embeddings using `all-MiniLM-L6-v2`, and stores them in a Hopsworks vector-search feature group (`ragbench_embeddings`).
- **LlamaIndex agent** — ReActAgent powered by Claude Haiku, queries the feature group for relevant passages, and answers questions.
- **LangGraph agent** — equivalent agent built with LangGraph's `create_react_agent` and Claude Haiku.
- Both agents expose a `/query` REST endpoint and persist multi-turn conversation history in MySQL.

## Files

| File | Description |
|---|---|
| `feature_pipeline.py` | Hopsworks feature pipeline — embeds RAGBench and writes to the feature store |
| `ragbench_llamaindex_agent.py` | LlamaIndex ReActAgent deployment |
| `ragbench_langchain_agent.py` | LangGraph ReActAgent deployment |
| `requirements_pipeline.txt` | Dependencies for the feature pipeline |
| `ragbench_llamaindex_requirements.txt` | Dependencies for the LlamaIndex agent |
| `ragbench_langchain_requirements.txt` | Dependencies for the LangGraph agent |

## Deployment

### 1. Feature pipeline

1. Upload `feature_pipeline.py` and `requirements_pipeline.txt` to your Hopsworks project.
2. In the Hopsworks UI, go to **Jobs** and create a new Python job pointing to `feature_pipeline.py`. Select `python-feature-pipeline` as the environment.
3. Run the job and wait for it to complete. This populates the `ragbench_embeddings` feature group with vector embeddings.

### 2. LlamaIndex agent

**Create the environment**

Before deploying the agent, create a dedicated Python environment with the required libraries:

1. In the Hopsworks UI, go to **Environments** and clone the base agent environment (e.g. `python-agent-pipeline`).
2. Name the cloned environment (e.g. `ragbench-agent`) and install `ragbench_llamaindex_requirements.txt` into it.
3. Wait for the installation to complete.

**Deploy the agent**

1. In the Hopsworks UI, go to **Deployments** and create a new agent deployment.
2. Set the predictor script to `ragbench_llamaindex_agent.py` and select the `ragbench-agent` environment created above.
3. Set the environment variables listed in the [Environment variables](#environment-variables-agent-deployments) section below.
4. Start the deployment and wait for it to reach the **Running** state.

### 3. LangGraph agent

**Create the environment**

1. In the Hopsworks UI, go to **Environments** and clone the base agent environment (e.g. `python-agent-pipeline`).
2. Name the cloned environment (e.g. `ragbench-langgraph-agent`) and install `ragbench_langchain_requirements.txt` into it.
3. Wait for the installation to complete.

**Deploy the agent**

1. In the Hopsworks UI, go to **Deployments** and create a new agent deployment.
2. Set the predictor script to `ragbench_langchain_agent.py` and select the `ragbench-langgraph-agent` environment created above.
3. Set the environment variables listed in the [Environment variables](#environment-variables-agent-deployments) section below.
4. Start the deployment and wait for it to reach the **Running** state.

## Query

```bash
# Single-turn
curl -s -X POST http://<istio-ip>/v1/g2/ragbenchagent/query \
  -H "Authorization: ApiKey <api-key>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is chain-of-thought prompting?"}'

# Multi-turn (echo back session_id to continue the conversation)
curl -s -X POST http://<istio-ip>/v1/g2/ragbenchagent/query \
  -H "Authorization: ApiKey <api-key>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Can you give an example?", "session_id": "<session_id from previous response>"}'
```

## Environment variables (agent deployments)

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | API key for Claude (set as a global user environment variable) |



---

## Chinook customer-support agent

A second, unrelated agent in this repo: a customer-support supervisor over the
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
| `chinook_feature_pipeline.py` | Embeds every artist/album/track name into one Hopsworks embedding feature group |
| `chinook_support_agent.py` | The agent, served by `AgentApp` |
| `chinook_support_requirements.txt` | Deployment requirements |

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
python chinook_feature_pipeline.py        # once, before first deploy

hops agent create chinook_support_agent.py --name chinooksupport \
    --requirements chinook_support_requirements.txt \
    --environment python-agent-pipeline-meb10000-v1
hops agent start chinooksupport --wait 600
```

| Variable | Default | Description |
|---|---|---|
| `CHINOOK_DB_PATH` | `chinook.db` | SQLite location; downloaded on first use if absent |
| `CHINOOK_ROUTER_MODEL` | `claude-haiku-4-5` | Intent routing + purchase-info extraction |
| `CHINOOK_ANSWER_MODEL` | `claude-haiku-4-5` | Catalogue question answering |
