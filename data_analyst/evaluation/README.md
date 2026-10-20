# Evaluation

What the data analyst is measured by, kept here so it survives a project being
rebuilt. Same layout as the other examples' `evaluation/` folders and the same
loader; what differs is that nothing here needs a judge.

| | **Offline** -- the suites | **Online** -- the sample |
|---|---|---|
| Asks | Does it give the figures the table gives, keep the table across turns, ask who it is talking to, and recall what it was told? | Did every reply in production come out as one of the agent's own sentences? |
| Input | `suites.json` + `tasks/*.jsonl` | Traces the deployment already served |
| Graded by | `contains` and `regex` -- the agent is deterministic, so every expected string is one it produces | one reference-free `regex`, so it grades every conversation at no cost |
| Run it | `apply.py --publish`, then `agent.run(suite)` or the deployment's Evals tab | `sample.py`, or `agent.sample(evaluator=...)` |

```bash
export HOPSWORKS_HOST=https://...  HOPSWORKS_PROJECT=...  HOPSWORKS_API_KEY=...
python -m data_analyst.evaluation.apply --publish        # offline: library, suites, tasks, frozen
python -m data_analyst.evaluation.sample --agent dataanalyst   # online: real traffic
```

Run from the repository root. The five suites, from what this agent can get wrong:

- **Answers the numbers** -- sum, average, median, count, max, min, by a column and by a
  period of the date column, and describe, on `sample_sales.csv`. Every case is its own
  person, named in the question the way the chat panel would be answered.
- **Keeps the table across turns** -- attached once, asked about again; a table attached
  before the person was known; the last question in what the agent remembers.
- **Asks who it is talking to** -- no address, no asserted subject: the email question,
  and nothing analysed before it, whether the turn brought a table, a question or a
  memory command.
- **Remembers, recalls and forgets** -- `remember`, `recall`, `what do you remember`,
  `forget`, `forget everything`, `search`, across the turns of one conversation.
- **Knows who it is talking to without being told** -- an introduction becomes the
  subject; a name, an employer, a place or "my fiscal year is April" is kept as a fact
  and answered from on a later turn, with no `remember` or `recall` said.

Two things cannot be graded this way. A refusal (`total revenue by planet`) is an error
envelope, which the runner records as an agent error rather than an answer, so the
refusal cases live in `test_data_analyst_agent.py` instead. And the chart itself: checks
read the reply's text, so the suites grade the numbers the chart is drawn from, and the
in-process tests check that a PNG came back.

`test_evaluation_cases.py` beside the agent runs every case here through the agent
in-process and applies the same checks, so a change to the agent that would fail a suite
fails there first, without a deployment.
