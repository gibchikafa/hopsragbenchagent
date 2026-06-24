# RAGBench Agent

A RAG system built on the [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) dataset, deployed on Hopsworks.

## Architecture

- **Feature pipeline** — downloads the RAGBench dataset, generates 384-dim embeddings using `all-MiniLM-L6-v2`, and stores them in a Hopsworks vector-search feature group (`ragbench_embeddings`).
- **LlamaIndex agent** — ReActAgent powered by Claude Haiku, queries the feature group for relevant passages, and answers questions.
- **LangChain agent** — equivalent agent built with LangGraph's `create_react_agent` and Claude Haiku.
- Both agents expose a `/query` REST endpoint and persist multi-turn conversation history in MySQL.

## Files

| File | Description |
|---|---|
| `feature_pipeline.py` | Hopsworks feature pipeline — embeds RAGBench and writes to the feature store |
| `ragbench_llamaindex_agent.py` | LlamaIndex ReActAgent deployment |
| `ragbench_langchain_agent.py` | LangChain + LangGraph ReActAgent deployment |
| `requirements_pipeline.txt` | Dependencies for the feature pipeline |
| `ragbench_llamaindex_requirements.txt` | Dependencies for the LlamaIndex agent |
| `ragbench_langchain_requirements.txt` | Dependencies for the LangChain agent |

## Deployment

### Feature pipeline

```bash
hops job deploy ragbench-feature-pipeline feature_pipeline.py \
  --env python-feature-pipeline --run --wait --overwrite
```

### LlamaIndex agent

```bash
hops agent create ragbench_llamaindex_agent.py --name ragbenchagent \
  --requirements ragbench_llamaindex_requirements.txt --environment ragbench-agent
hops agent start ragbenchagent --wait 600
```

### LangChain agent

```bash
hops agent create ragbench_langchain_agent.py --name ragbenchlcagent \
  --requirements ragbench_langchain_requirements.txt \
  --environment python-agent-pipeline-meb10000-v1
hops agent start ragbenchlcagent --wait 600
```

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
| `MYSQL_USER` | MySQL username |
| `MYSQL_PASSWORD_SECRET_NAME` | Name of the Hopsworks secret holding the MySQL password |
| `MYSQL_HOST` | MySQL host |
| `MYSQL_PORT` | MySQL port (default: 3306) |
| `MYSQL_DB` | MySQL database name |
| `DEPLOYMENT_ID` | Appended to the chat history table name to isolate deployments |
| `ANTHROPIC_API_KEY` | API key for Claude (set as a global user environment variable) |
