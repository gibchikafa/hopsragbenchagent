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
| `requirements_pipeline.txt` | Dependencies for the feature pipeline |
| `llamaindex_agent.py` | LlamaIndex ReActAgent, hand-rolled `/query` endpoint |
| `langchain_agent.py` | LangGraph ReActAgent, hand-rolled `/query` endpoint |
| `llamaindex_agent_native.py` | Same agent on the Hopsworks Agent Protocol SDK — manifest, streaming, memory tiers |
| `langchain_agent_native.py` | Same, LangGraph flavour |
| `langchain_agent_hap.py` | Thin SDK wrapper reusing `langchain_agent.py`'s predictor |
| `*_requirements.txt` | Per-agent dependencies (`_hap_` variants pull in the SDK) |

The `_native` agents are the ones to read if you want the SDK: they show the
manifest, streaming, conversation memory, summarization and durable per-user
memory. The plain ones predate it and hand-roll their own HTTP surface.

## Deployment

### 1. Feature pipeline

1. Upload `feature_pipeline.py` and `requirements_pipeline.txt` to your Hopsworks project.
2. In the Hopsworks UI, go to **Jobs** and create a new Python job pointing to `feature_pipeline.py`. Select `python-feature-pipeline` as the environment.
3. Run the job and wait for it to complete. This populates the `ragbench_embeddings` feature group with vector embeddings.

### 2. LlamaIndex agent

**Create the environment**

Before deploying the agent, create a dedicated Python environment with the required libraries:

1. In the Hopsworks UI, go to **Environments** and clone the base agent environment (e.g. `python-agent-pipeline`).
2. Name the cloned environment (e.g. `ragbench-agent`) and install `llamaindex_requirements.txt` into it.
3. Wait for the installation to complete.

**Deploy the agent**

1. In the Hopsworks UI, go to **Deployments** and create a new agent deployment.
2. Set the predictor script to `llamaindex_agent.py` and select the `ragbench-agent` environment created above.
3. Set the environment variables listed in the [Environment variables](#environment-variables-agent-deployments) section below.
4. Start the deployment and wait for it to reach the **Running** state.

### 3. LangGraph agent

**Create the environment**

1. In the Hopsworks UI, go to **Environments** and clone the base agent environment (e.g. `python-agent-pipeline`).
2. Name the cloned environment (e.g. `ragbench-langgraph-agent`) and install `langchain_requirements.txt` into it.
3. Wait for the installation to complete.

**Deploy the agent**

1. In the Hopsworks UI, go to **Deployments** and create a new agent deployment.
2. Set the predictor script to `langchain_agent.py` and select the `ragbench-langgraph-agent` environment created above.
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
