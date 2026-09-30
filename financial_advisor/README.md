# Financial advisor

Google's [ADK financial-advisor recipe](https://github.com/google/adk-recipes/tree/main/python/agents/financial-advisor)
on LangGraph and the Hopsworks Agent Protocol. A coordinator walks the user
from a ticker to a risk-evaluated plan, calling four analysts in turn:

```
coordinator ─┬─▶ analyze_market      data_analyst:      web search → market analysis report
             ├─▶ propose_strategies  trading_analyst:   ≥5 strategies for the risk attitude and period
             ├─▶ plan_execution      execution_analyst: entry, holding, scaling, exits
             └─▶ assess_risk         risk_analyst:      the whole plan against the profile
```

| File | Purpose |
|---|---|
| `advisor_agent.py` | The coordinator (a LangGraph ReAct agent) with the analysts as its tools, served by `AgentApp` |
| `advisor_agent_openai.py` | The same on the OpenAI Agents SDK: the analysts are `Agent`s run by the coordinator's tools, the data analyst with the SDK's web search |
| `advisor_memory.py` | The reports in working memory, shared by both entry points |
| `prompts.py` | The recipe's prompts, ported; what changed and why is in its docstring |
| `requirements.txt` | Deployment requirements |
| `evaluation/` | The recipe's evaluation cases as Hopsworks suites, plus two the recipe could not express |

## How the recipe maps

| ADK | Here |
|---|---|
| `LlmAgent` coordinator with `AgentTool`s | `create_react_agent` with four `@tool`s, each one model call with the analyst's prompt |
| `output_key` → session state, read by the next analyst | Each tool writes its report to the conversation's working memory (`session` scope) and the next tool reads it there |
| `google_search` (Gemini grounding) | Claude's server-side web search tool, bound to the data analyst's model |
| `adk web`, `adk run` | The Hopsworks chat panel, `agent.chat()`, or any client on the gateway |
| `eval/data/financial-advisor.test.json` | `evaluation/tasks/introduces_itself_before_advising.jsonl`, run as a suite |

The coordinator never copies a report into a tool call: an analyst's report is
several thousand tokens, and the recipe's state keys existed for the same
reason. Working memory is the SDK's, so it survives a pod restart and is shared
across replicas, and a new conversation starts the process over, as the recipe's
session would. A resumed conversation is told which reports exist so it
continues from the next step.

Only the coordinator speaks to the user. The analysts' model calls are tagged
and their tokens filtered out of the stream, so a report reaches the user once,
explained by the coordinator, rather than twice.

## Two entry points

`advisor_agent.py` runs on LangGraph with Claude and its web search; `advisor_agent_openai.py`
on the OpenAI Agents SDK with OpenAI's models and the SDK's `WebSearchTool` (`OPENAI_API_KEY`),
or any LiteLLM model name in `ADVISOR_OPENAI_MODEL`. Both keep the reports in the same
working-memory keys, so either can continue a plan the other started.

## Deploy

```bash
hops agent create advisor_agent.py --name financialadvisor \
    --requirements requirements.txt --environment python-agent-pipeline
hops agent start financialadvisor --wait 600
```

Then open the deployment's chat panel, or from Python:

```python
agents = hopsworks.login().get_agent_serving()
advisor = agents.get_agent("financialadvisor")
reply = advisor.chat("Hello. What can you do for me?")
advisor.chat("Analyze NVDA for me.", conversation_id=reply.conversation_id)
```

`prompts.py` must sit beside `advisor_agent.py`. A git-backed deployment brings the
whole repository, so that is the simplest way to ship both:

```python
agents.deploy_agent("financial_advisor/advisor_agent.py", name="financialadvisor",
                    git_url="https://github.com/<you>/hopsragbenchagent.git", git_provider="GitHub",
                    git_branch="main", requirements="financial_advisor/requirements.txt",
                    environment="python-agent-pipeline", tracing={"enabled": True}).start()
```

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | `advisor_agent.py`: the analysts, the coordinator and the memory summariser all run on Claude. Web search must be enabled for the key's organisation. |
| `OPENAI_API_KEY` | `advisor_agent_openai.py`: the agents, the web search and the memory summariser (`openai_summarizer`). |
| `ADVISOR_MODEL` | Model for the coordinator and the analysts. Default `claude-sonnet-4-5`. |
| `ADVISOR_MAX_SEARCHES` | Web searches the data analyst may run per report. Default 8; each is a paid call. |

## Evaluation

```bash
python -m financial_advisor.evaluation.apply --publish
```

Three suites, all read-only (this agent writes nothing):

- **Introduces itself before advising** — the recipe's own cases: a first hello
  gets the introduction and the full disclaimer, a "yes" gets asked for a
  ticker, and no analyst runs before there is one.
- **Analysts run in order, each on the last one's report** — the pipeline the
  coordinator is told to follow, checked with `tool_order` and a rubric per
  case; gates at 80%.
- **Every recommendation carries the disclaimer** — strategies and plans come
  framed as an AI model's educational output, tailored to the stated profile,
  never as an instruction to buy or sell; gates at 100%.

The data analyst searches the web, so a run's cost is the searches plus the
generations; a task with all four analysts is a few minutes.
