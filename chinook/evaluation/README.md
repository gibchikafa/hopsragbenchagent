# Evaluation suites

What this agent is measured by, kept here so it survives a project being rebuilt
— the suites live in Hopsworks, and the migration that creates their tables drops
them.

```bash
export HOPSWORKS_API_KEY=...
python -m chinook.evaluation.apply --publish
```

- **`evaluators.json`** — the library. One named check each, written once.
  Several suites hold the agent to "`place_order` was not called", and writing a
  judge's criteria into each of them separately is how they drift apart.
- **`suites.json`** — the suites: which library entries each uses, and which task
  file belongs to it.
- **`tasks/*.jsonl`** — the cases, one file per suite, uploaded with **Import**
  on the suite page.
- **`apply.py`** — saves the library, then creates any suite that is missing.

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

## The four suites

Each is built from something this agent can actually get wrong, taken from its
own tools and system prompt rather than from a template.

| Suite | Mode | What it holds the agent to |
|---|---|---|
| Catalogue answers stay out of the account | `read_only` | A question about the shop is answered from the catalogue; nothing reaches the asker's orders, history or interests |
| Interest is not an order | `sandboxed` | Liking an album, wanting it, or thinking about it is an interest — recorded as one and followed with a question, never with an order |
| Orders are recorded, never charged | `sandboxed` | A plain instruction to buy places the order, and the reply says it is recorded with nothing charged |
| Nobody's account before we know who they are | `read_only` | First name, last name and the phone on the account, before anything about that customer's orders |

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

## Judges

Three suites score with an LLM judge. The key comes from the provider's own
variable — `ANTHROPIC_API_KEY` — set once in your account's environment
variables, which reaches every job. A judge with no key is skipped rather than
failed, so a run can report a suite fully passed while the check that would have
tested the behaviour never ran; the job log says when that happens.
