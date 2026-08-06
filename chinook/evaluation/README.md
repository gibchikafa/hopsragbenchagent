# Evaluation

What this agent is measured by, kept here so it survives a project being rebuilt
— the suites live in Hopsworks, and the migration that creates their tables drops
them.

Two halves, and neither substitutes for the other:

| | **Offline** — the suites | **Online** — the sample |
|---|---|---|
| Asks | Does it pass the cases we wrote down? | How is it doing on what customers actually ask? |
| Input | `suites.json` + `tasks/*.jsonl` | Traces this deployment already served |
| Expected answer | Yes, per task per check | None — `rubric.md` is all the judge has |
| Run it | `apply.py`, then the deployment's Evaluation tab | `sample.py` |

A suite cannot contain a question nobody thought to write, which is most of what
customers ask. Production cannot tell you whether a fix held, because the
conversation that broke may never come back. The loop between them is the point:
production surfaces a failure, you promote that trace to a task, and a suite
defends against it from then on.

**Their scores are not comparable.** One is measured against declared answers
with the same checks on every task; the other is one judge's opinion on whatever
arrived. Both are useful; one average of the two is useful for nothing.

```bash
export HOPSWORKS_API_KEY=...
python -m chinook.evaluation.apply --publish          # offline: the suites
python -m chinook.evaluation.sample --deployment-id 12  # online: real traffic
```

- **`evaluators.json`** — the library. One named check each, written once.
  Several suites hold the agent to "`place_order` was not called", and writing a
  judge's criteria into each of them separately is how they drift apart.
- **`suites.json`** — the suites: which library entries each uses, and which task
  file belongs to it.
- **`tasks/*.jsonl`** — the cases, one file per suite, uploaded with **Import**
  on the suite page.
- **`apply.py`** — saves the library, then creates any suite that is missing.
- **`rubric.md`** — what a good answer looks like, for grading real traffic. The
  only input online evaluation has, since production carries no expected
  answers.
- **`sample.py`** — starts one online sample against a deployment.

`apply.py` creates suites, not tasks. A suite has to exist before its tasks have
anywhere to go — its checks decide what a task must declare — and keeping the
cases in files is what lets someone add twenty of them without touching any of
this.

The task files use the column names the importer already understands:

```json
{"input": "Do you have anything by Queen?", "expected": "News Of The World",
 "forbiddenTools": "place_order, purchase_history, remember_interest"}
```

`expected` goes to the check that reads an answer, `rubric` to the judge, and
`requiredTools` / `forbiddenTools` to the tool check. A column named after a
check goes to that one instead, which is what to use if a suite ever has two
checks of the same shape.

A suite **copies** its checks in when it is created and never points back. That
is what keeps a published suite meaning exactly what it meant when it was
published: editing the library afterwards cannot rewrite a suite that has already
been run against. The library is for not retyping, not for editing every suite at
once.

For the same reason `apply.py` creates and never edits a suite. Re-running after
a change gives you a new version to publish, not a rewrite of the one your last
results were measured against.

`--publish` freezes a suite only once it has tasks: an empty one has nothing to
run, and the server refuses it.

## Running one

From the **deployment**, not from the suite: open the agent under Deployments,
go to **Evaluation**, and use **Run a suite**. That is also where **Sample
production** is, and where this deployment's evaluation job — its resources,
environment and alerts — is configured. Suites are project-level because running
one against two deployments and comparing them is the point of freezing them;
runs start from the agent because that is the thing you actually have a question
about.

Each deployment gets its own evaluation job, `<deployment>_eval_runner`, created
the first time you start a run against it. To grade production on a schedule
rather than by hand, set a schedule on that job under **Jobs** — there is no
second job and no separate switch.

## The four suites

Each is built from something this agent can actually get wrong, taken from its
own tools and system prompt rather than from a template.

| Suite | Mode | Gates | What it holds the agent to |
|---|---|---|---|
| Catalogue answers stay out of the account | `read_only` | — | A question about the shop is answered from the catalogue; nothing reaches the asker's orders, history or interests |
| Interest is not an order | `sandboxed` | — | Liking an album, wanting it, or thinking about it is an interest — recorded as one and followed with a question, never with an order |
| Orders are recorded, never charged | `sandboxed` | `pass_rate` ≥ 1.0 | A plain instruction to buy places the order, and the reply says it is recorded with nothing charged |
| Nobody's account before we know who they are | `read_only` | — | First name, last name and the phone on the account, before anything about that customer's orders |

Every expected string is a real Chinook value — `Houses Of The Holy`,
`Let There Be Rock`, `We Will Rock You`, `Motörhead` — pulled from the source
database rather than remembered. Aaron Mitchell and `+1 (204) 452-6452` are a
real customer and the phone on their account, so the identity gate genuinely
opens. A failure means the agent was wrong, not that the test was.

## The two sandboxed suites

They make the agent call `place_order`, which writes against a real customer's
account. The runner refuses a sandboxed suite unless the deployment reports
`eval_mode`, so they need a deployment with:

```
EVAL_MODE=true
```

`AgentApp` reads that and reports it in the manifest;
[`support_agent.py`](../support_agent.py) reads it too and skips the three
writes — order lines, the customer's line index, the refund ledger. Everything
else is identical, because a deployment that behaved differently under
evaluation would be measuring something other than the agent that serves
customers.

Setting the variable without that code would make the manifest claim something
untrue and place real orders.

## Gates

One suite states a gate: **Orders are recorded, never charged** must have a
`pass_rate` of 1.0. Claiming a card was billed is the one failure with a real
consequence for a customer, so a single failing task should block a promotion.

Being tagged `golden` does **not** gate anything — the tag is descriptive and
nothing reads it. Gates name their own metric and bar, which is the difference
between a rule someone can see in the suite and one hidden inside a category
name. The other three suites gate nothing on purpose: they are worth watching,
not worth blocking a release over.

## Judges

Three suites score with an LLM judge. The key comes from the provider's own
variable — `ANTHROPIC_API_KEY` — set once in your account's environment
variables, which reaches every job. A judge with no key is skipped rather than
failed, so a run can report a suite fully passed while the check that would have
tested the behaviour never ran; the job log says when that happens.
