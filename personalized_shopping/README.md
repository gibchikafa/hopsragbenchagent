# Personalized shopping

Google's [ADK personalized-shopping recipe](https://github.com/google/adk-recipes/tree/main/python/agents/personalized-shopping)
on LangGraph and the Hopsworks Agent Protocol, with the feature store in place
of the website. A single agent finds products, explores them with the shopper,
and orders when told to.

```
shopper ──▶ agent ─┬─▶ search_products   vector search over the catalogue
                   ├─▶ product_details   the item page and its tabs, in one read
                   └─▶ buy_now           an order in the ledger; never a charge
```

| File | Purpose |
|---|---|
| `feature_pipeline.py` | Loads WebShop's 1,000-product catalogue into three feature groups: embeddings for search, products for lookup, an empty order ledger |
| `store.py` | The three tools over the feature store, and the `in_evaluation()` guard on the order write |
| `shopping_agent.py` | The LangGraph agent (`create_react_agent`) served by `AgentApp`, text and image input |
| `shopping_agent_openai.py` | The same agent on the OpenAI Agents SDK: the store's functions as `function_tool`s |
| `prompts.py` | The prompt both entry points use |
| `evaluation/` | The recipe's evaluation cases as Hopsworks suites, plus two the website made hard to test |

## How the recipe maps

| ADK | Here |
|---|---|
| WebShop, a simulated website of 1.18M products browsed by clicking | The same product data (its 1,000-item subset) in the feature store; nothing is browsed |
| `search[keywords]` over a Lucene index | `search_products`: `find_neighbors` on a `all-MiniLM-L6-v2` embedding of title, first bullet and description |
| `click[ASIN]`, then Description / Features / Reviews, then `< Prev` | `product_details`: one keyed read returning all of it, options included |
| Click a colour, a size, then `Buy Now` | `buy_now(asin, options, shopper_confirmed)`: refuses until every offered option is chosen and the shopper said to buy; records a row in `webshop_orders` |
| Image upload analysed by Gemini | `input_modalities=["text","image"]`; the photo goes to Claude as an image block, and the model searches for what it sees |
| Conversational memory in the context window | `ManagedMemoryService` with a rolling summary |

The recipe's prompt is two-thirds button handling: click only what is on the
current page, use `< Prev` to go back, use "Back to Search" when there is no
"Search". None of that survives, because there are no pages. What survives is
the interaction flow: ask, search, present, explore on request, confirm before
buying, then say the order is recorded and nothing was charged.

### Feature groups

| Feature group | Key | Answers |
|---|---|---|
| `webshop_product_embeddings` | *(vector)* | "which products match what the shopper said?" |
| `webshop_products` | `asin` | "everything the item page showed about this one" |
| `webshop_orders` | `order_id` | "what was ordered, by whom, in what options" |

The recipe fetched the catalogue from Google Drive, where the files no longer
resolve; the pipeline reads the same `items_shuffle_1000.json` from a Hugging
Face mirror (set `WEBSHOP_ITEMS` to a local copy to skip the download). The
data has no review text, only a rating and a count, so "Reviews" is the rating.

## Two entry points

`shopping_agent.py` runs the agent on LangGraph, `shopping_agent_openai.py` on the OpenAI Agents
SDK. They share the store, the prompt and the feature groups; only the runtime differs. The
OpenAI entry point runs on OpenAI's models by default (`OPENAI_API_KEY`), or on any LiteLLM
model name in `WEBSHOP_OPENAI_MODEL`, e.g. `anthropic/claude-sonnet-4-5`.

## Who it is shopping for

The agent asks for an email address before it searches, and calls `identify` with it. `demo_users.py` is the list it offers, because a demo has no login to read an
identity from; nothing checks the names.
In a real deployment the agent would not ask at all. The client asserts `subject` on each
request — the user id or address its own session already knows — and the SDK keys memory by
it; `demo_users.py` stands in for that login, and the question is skipped whenever an
identity arrives with the request.

That first question is what makes the rest of the memory work: `remember` and `recall` are
keyed by the subject, so until someone has identified themselves a size or an allergy
belongs to the conversation and is lost with it. Come back with the same address and the shop remembers.

## Deploy

```bash
python feature_pipeline.py            # the catalogue, ~2 minutes with the download
```

The pipeline also puts the embedding model (`all-MiniLM-L6-v2`) in the project's model
registry, once. The agent loads it from there with `load_sentence_transformer`, so a serving
pod never downloads it from the internet.

The agent is two files, so deploy it git-backed and the repository comes along:

```python
agents = hopsworks.login().get_agent_serving()
shop = agents.deploy_agent(
    "personalized_shopping/shopping_agent.py", name="webshop",
    git_url="https://github.com/<you>/hopsragbenchagent.git", git_provider="GitHub",
    git_branch="main",
    environment="python-agent-pipeline", tracing={"enabled": True},
)
shop.start()
shop.chat("Hello, who are you?").text
```

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | `shopping_agent.py`: the agent and the memory summariser run on Claude. |
| `OPENAI_API_KEY` | `shopping_agent_openai.py`: the agent and the memory summariser (`openai_summarizer`). |
| `WEBSHOP_MODEL` | Model for the agent. Default `claude-sonnet-4-5`. |
| `WEBSHOP_ITEMS` | Pipeline only: a local path to `items_shuffle_1000.json`. |

## Evaluation

```bash
python -m personalized_shopping.evaluation.apply --publish
```

- **Introduces itself, searches nothing** — the recipe's `simple.test.json`.
- **Searches when asked, and shows what it found** — the recipe's tool test and
  its example session: the results shown are the ones returned, with prices.
- **Explores a product before pitching it** — read, then describe, in one go.
- **Buys only when told, and never charges** — sandboxed; `buy_now` records
  nothing under evaluation. An interest is not an order; options are the
  shopper's to choose; the confirmation says recorded, not paid. Gates at 100%.
