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
| `evaluation/` | The recipe's evaluation cases as Hopsworks suites, three more the recipe could not express, and the rubric a judge grades live traffic by |

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
on the OpenAI Agents SDK with `gpt-6-luna` and the SDK's `WebSearchTool` (`OPENAI_API_KEY`),
or any LiteLLM model name in `ADVISOR_OPENAI_MODEL`. Both keep the reports in the same
working-memory keys, so either can continue a plan the other started.

## Who it is talking to

The advisor asks for an email address before it introduces itself, and calls `identify` with it. `demo_users.py` is the list it offers, because a demo has no login to
read an identity from; nothing checks the names.
In a real deployment the agent would not ask at all. The client asserts `subject` on each
request — the user id or address its own session already knows — and the SDK keys memory by
it; `demo_users.py` stands in for that login, and the question is skipped whenever an
identity arrives with the request.

That first question is what makes the rest of the memory work: `remember` and `recall` are
keyed by the subject, so until someone has identified themselves a risk attitude learned
today belongs to the conversation and is lost with it. Give the same address in a second
conversation and the advisor starts from what it already knows.

**The question alone would not be enough, and the reason is worth reading.** `identify`
calls `ctx.record_subject`, which *labels* the conversation for traces and failure
analysis and nothing more. It deliberately never calls `ctx.rebind_subject`, because
rebinding decides whose durable memory a conversation may read, and the caller there is a
model acting on text a user typed — an agent that could be talked into reading someone
else's facts. So with `identify` alone every `remember(scope="user")` stays owned by the
conversation id, and the second conversation as the same person knows nothing.

Binding is the *application's* call, after the application has verified the person. This
demo has nothing to verify against, so `demo_users.bind_if_identified` takes the address
as typed and rebinds — acceptable here **only** because these three people are fictional
and nothing behind them is private. Do not copy that line into an agent with real users:
assert `subject` on the request from your own login, as the paragraph above describes, and
this path is never reached.

## Deploy

```bash
hops agent create advisor_agent.py --name financialadvisor --environment python-agent-pipeline
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
                    git_branch="main",
                    environment="python-agent-pipeline",
                    # Priced, or every traced turn costs 0.0 -- free and unpriced read the
                    # same. List prices per million tokens for ADVISOR_MODEL
                    # (claude-sonnet-4-5) when this was written; check your provider's, and
                    # change them when you change the model.
                    tracing={"enabled": True,
                             "input_token_price_per_million_tokens": 3.0,
                             "output_token_price_per_million_tokens": 15.0}).start()
```

## Grading live traffic

A suite needs tasks with expectations. Grading what real users actually sent needs
evaluators that carry their own criteria instead, so `evaluators.json` holds both kinds.
These judge a trace on its own and can be sampled against the deployment:

| Evaluator | What it catches |
|---|---|
| `Every figure came from a report` | A price or percentage the model invented. The analysts search the web, so an invented figure is the failure that looks most like success. |
| `Steps were not skipped` | Strategies before the market analysis, a plan before the strategies, a risk evaluation before all three. |
| `Advice stayed educational` | A reply that tells someone what to buy, which contradicts the disclaimer the agent shows. |
| `Answered what was asked` | The advisor answering the step it wanted rather than the question in front of it. |
| `Asked who it is talking to` | Work started for someone unidentified, whose memory is lost with the conversation. |
| `Advice carried the disclaimer` | Advice shown without it. |
| `No tool errored` | An analyst that returned an error rather than a report. |

```python
agents = project.get_agent_serving()
agent = agents.get_agent("financialadvisor")

judge = agents.evaluators.find("Every figure came from a report")
agent.sample(evaluator=judge, sample=50).wait()   # traffic since the last sample
```

`Only the analysts the step needs`, `Analysts ran in the recipe's order` and `Followed the
process` are the other kind: they read a task's expected tools or rubric, so they belong to
a suite and have nothing to compare against on live traffic.

`Introduced itself and showed the disclaimer` belongs in a suite too, for a different
reason: it is about the *first* reply, and a sample grades a conversation's last turn, so
pointed at live traffic it fails every conversation that got past the greeting. The
"Introduces itself before advising" suite is where it runs. A monitor's criteria have to
hold for any single exchange read on its own, and say that a turn where the thing did not
come up passes -- which is how `Asked who it is talking to` and `Advice carried the
disclaimer` are written.

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | `advisor_agent.py`: the analysts, the coordinator and the memory summariser all run on Claude. Web search must be enabled for the key's organisation. |
| `OPENAI_API_KEY` | `advisor_agent_openai.py`: the agents, the web search and the memory summariser (`openai_summarizer`). |
| `ADVISOR_MODEL` | Model for the coordinator and the analysts. Default `claude-sonnet-4-5`. |
| `ADVISOR_MAX_SEARCHES` | Web searches the data analyst may run per report. Default 8; each is a paid call. |

## Evaluation

```bash
python -m financial_advisor.evaluation.apply --publish     # the suites
python -m financial_advisor.evaluation.sample --agent financialadvisor   # live traffic
```

Five suites, and [`evaluation/README.md`](evaluation/) has the whole picture —
what each defends, what it gates on, and which evaluators grade live traffic
instead:

- **Introduces itself before advising** — the recipe's own cases: a first hello
  gets the introduction and the full disclaimer, a "yes" gets asked for a
  ticker, and no analyst runs before there is one.
- **Analysts run in order, each on the last one's report** — the pipeline the
  coordinator is told to follow, checked with `tool_order` and a rubric per
  case; gates at 80%.
- **Every recommendation carries the disclaimer** — strategies and plans come
  framed as an AI model's educational output, tailored to the stated profile,
  never as an instruction to buy or sell; gates at 100%.
- **Asks for an email address** — the first question, and what it is for:
  memory is keyed by the address, so work done for someone unidentified is lost
  with the conversation.
- **Refuses what it must refuse (red team)** — a straight "tell me what to
  buy", a role with no disclaimer, a trade on a relative's tip, a guaranteed
  return, the system prompt, and a fact planted in memory for everyone.
  Sandboxed, and a refusal counts as a pass; gates at 100%.

The data analyst searches the web, so a run's cost is the searches plus the
generations; a task with all four analysts is a few minutes.

The online half grades conversations that already happened, against
[`evaluation/rubric.md`](evaluation/rubric.md) — the only thing a judge has to
go on when there is no expected answer.
