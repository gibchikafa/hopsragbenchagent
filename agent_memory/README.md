# Agent memory demo

A small agent whose subject is the memory service itself. Every other example here
uses memory in passing; this one exists to make it visible. It stores what you tell
it, recalls it in your next conversation, searches what you said before, forgets on
request, and will print what is in each tier when you ask.

No feature store, no data pipeline, one file to deploy.

## Files

| File | Purpose |
|---|---|
| `memory_agent.py` | The agent: the SDK's memory tools on an OpenAI Agents SDK loop, served by `AgentApp` |
| `prompts.py` | The prompt, including the two rules the evaluation suites hold it to |
| `demo_job.py` | Deploys the agent and talks to it, as a Hopsworks job: the job log is the demonstration |
| `requirements.txt` | Deployment requirements |
| `evaluation/` | Three suites: what it stored, what it must never invent, and when not to look things up |

## The three tiers

| Tier | What it holds | Turned on by | Seen in the demo as |
|---|---|---|---|
| 1. Conversation buffer | The turns of this chat | nothing; it is the default | "What's my cat called?" answered from the turn before |
| 2. Rolling summary | Older turns, folded instead of dropped | `summarize=` | `memory_report` showing a summary once the chat passes six messages |
| 3. Durable memory | Facts about the person, across conversations | `long_term=True` | A new chat that still knows you are vegetarian |

Tier 3 is reached by four tools the model calls itself, `remember`, `recall`,
`forget` and `search`, registered with one line:

```python
tools = [*memory_tools("openai_agents"), *identity_tools("openai_agents")]
```

`identify` is separate on purpose. What it writes is the person an audit surface
will show on the other end of the conversation, so an agent opts into it rather
than receiving it alongside the other four.

Two scopes matter. `remember` defaults to `user` scope, keyed by `ctx.subject`, and
outlives the conversation. `scope='session'` is for working notes that should not
follow someone around: what they are shopping for today is not who they are.

## Run it as a job

`demo_job.py` does the whole demonstration in one run: it deploys the agent, waits for
it to answer, then holds two conversations as the same person and prints what memory
did between them. The job's log is the demo.

```bash
python demo_job.py                 # deploy, start, run the script
python demo_job.py --skip-deploy   # talk to an agent that is already up
python demo_job.py --strict        # exit 1 if a tier did not do its job
```

Create it as a job with `python-agent-pipeline` as the environment. Upload
`demo_job.py` alongside `memory_agent.py`, `prompts.py` and `requirements.txt`, or
upload only the script and pass `--git-url` so the deployment pulls the agent from the
repository.

It ends with a line per tier, so a failure says which one:

```
  [ok] buffer answered from this conversation
  [ok] older turns folded into a rolling summary
  [ok] a durable fact crossed into a new conversation
  [ok] the session note did not cross
  [ok] the forgotten fact is gone
```

The second conversation is the one worth reading. It is a new `conversation_id` with no
shared history, so anything the agent still knows came from durable memory, and anything
it has correctly lost was session-scoped. Between the turns the script reads the
server's own view of the conversation, `/v1/conversations/{id}/messages`, which carries
the rolling summary and how far it reaches. That view does not go through a language
model, so it is evidence rather than the agent's account of itself.

The model key comes from the project's secret store: the script copies the secret named
by `--model-secret` (default `OPENAI_API_KEY`) onto the deployment, so the key stays in
the secret store and the pod can still reach a model. Pass `--model-secret ''` to skip
that if the deployment already has one.

## Try it by hand

Say these in order, in one chat, then start a second chat and ask the last one again:

1. `I'm Dana, I live in Oslo and I'm vegetarian.` — watch the `remember` chip appear.
2. `Just for now: I'm comparing the 14-inch and the 16-inch.` — session scope, not user.
3. `Show me what's in each memory tier.` — `memory_report` prints all three.
4. Six or so messages in, ask again: the summary tier is no longer empty.
5. In a **new conversation**: `What do you remember about me?`

Until someone identifies themselves, durable facts are keyed by the conversation, so
step 5 only works across chats once `identify` has run. Saying your name is what does
it.

## Deploy

```bash
hops agent create memory_agent.py --name memorydemo \
    --requirements requirements.txt --environment python-agent-pipeline
hops agent start memorydemo --wait 600
```

`prompts.py` must sit beside `memory_agent.py`, so a git-backed deployment is the
simplest way to ship both:

```python
agents = hopsworks.login().get_agent_serving()
agents.deploy_agent("agent_memory/memory_agent.py", name="memorydemo",
                    git_url="https://github.com/<you>/hopsragbenchagent.git", git_provider="GitHub",
                    git_branch="main", requirements="agent_memory/requirements.txt",
                    environment="python-agent-pipeline", tracing={"enabled": True}).start()
```

From Python, the second conversation is the interesting one:

```python
demo = agents.get_agent("memorydemo")
first = demo.chat("I'm Dana, I live in Oslo and I'm vegetarian.")
demo.chat("What do you remember about me?")   # new conversation, same person
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | The agent and the memory summariser (`openai_summarizer`). One key covers the deployment. |
| `MEMORY_DEMO_MODEL` | Model name. Empty uses the SDK's default; a name with a slash, such as `anthropic/claude-sonnet-4-5`, goes through LiteLLM and needs that provider's key. |
| `MEMORY_DEMO_FOLD_AFTER` | Messages before older turns fold into the summary. Default 6, so folding happens inside a demo chat; the SDK's own default is 20. |
| `MEMORY_DEMO_KEEP_RECENT` | Messages kept in the buffer after a fold. Default 4. |
| `MEMORY_DEMO_VECTOR_SEARCH` | `true` adds an embedder and an embedding feature group, which makes `search` semantic instead of keyword. Off by default: the first three tiers need neither. |

An LLM deployed in this project can serve the summaries instead of OpenAI, with no
external key at all — replace `openai_summarizer()` with
`hopsworks_summarizer("<deployment name>")`.

## Evaluation

```bash
python -m agent_memory.evaluation.apply --publish
```

Three read-only suites. The tasks are conversations rather than single prompts,
because memory only shows itself across turns:

- **Keeps what it is told** — the fact is written with `remember`, under a key that
  names it, in the scope it deserves, and a correction ends up as the stored value;
  gates at 80%.
- **Remembers nothing it was not told** — the failure that makes a memory agent worse
  than no memory is an answer that sounds recalled and was invented. Gates at 100%.
- **Reads the conversation before reaching for memory** — a lookup for something two
  turns old is a round trip that buys nothing; gates at 80%.
