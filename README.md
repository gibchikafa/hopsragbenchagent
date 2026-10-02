# Hopsworks agent examples

Reference agents deployed on Hopsworks. Most pair a **feature pipeline** that writes
an embedding feature group with an **agent deployment** that queries it; a couple need
no data at all.

| Example | What it is |
|---|---|
| [`ragbench/`](ragbench/) | RAG over the [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) paper corpus. LlamaIndex and LangGraph flavours, each in a hand-rolled and an SDK-native variant. |
| [`chinook/`](chinook/) | Customer support for the [Chinook](https://github.com/lerocha/chinook-database) music store: a supervisor routing between refunds and catalogue questions. |
| [`personalized_shopping/`](personalized_shopping/) | Google's [ADK personalized-shopping recipe](https://github.com/google/adk-recipes/tree/main/python/agents/personalized-shopping) on LangGraph, with the WebShop catalogue in the feature store instead of a website to click through: search, read a product, order it. |
| [`cyber_guardian/`](cyber_guardian/) | Google's [ADK cyber-guardian recipe](https://github.com/google/adk-recipes/tree/main/python/agents/cyber-guardian-agent) as a LangGraph workflow: triage, threat intel, investigation and playbook over keyed feature-store lookups, with disruptive actions held for approval. |
| [`travel_concierge/`](travel_concierge/) | Google's [ADK travel-concierge recipe](https://github.com/google/adk-recipes/tree/main/python/agents/travel-concierge) on the OpenAI Agents SDK: seven agents handing a traveler to one another from inspiration to booking and through the trip, with working memory on Hopsworks. |
| [`financial_advisor/`](financial_advisor/) | Google's [ADK financial-advisor recipe](https://github.com/google/adk-recipes/tree/main/python/agents/financial-advisor) on LangGraph: a coordinator calling four analysts in turn, with reports passed through the SDK's working memory. No feature pipeline; the data analyst searches the web. |
| [`agent_memory/`](agent_memory/) | The memory service on its own: a small assistant that stores what you tell it, recalls it in your next conversation, searches what you said before, and prints what is in each tier when you ask. No feature pipeline, one file. |

Each folder has its own README with deployment steps.

## The shape they share

Both examples split the same way, and it is the point of the examples:

- **Indexing is a pipeline, not agent startup.** Embeddings are built once by a
  job and written to a feature group. Agents query it online, so a pod start
  costs nothing, replicas share one index, and the index can be rebuilt without
  redeploying.
- **The SDK owns the serving surface.** In the `_native` agents and the Chinook
  one, `hopsworks_agents.protocol` (in the [`hopsworks`](https://github.com/logicalclocks/hopsworks-api) package)
  provides the manifest, `/v1/chat` and `/v1/chat/stream`, health and readiness,
  CORS, tracing, and the memory tiers. What is left is the domain logic.

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | API key for Claude. Set it as a global user environment variable so every deployment inherits it. |

Per-example variables are documented in each folder's README.
