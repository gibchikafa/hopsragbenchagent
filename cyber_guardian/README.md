# Cyber Guardian

Google's [ADK cyber-guardian recipe](https://github.com/google/adk-recipes/tree/main/python/agents/cyber-guardian-agent)
as a LangGraph workflow on the Hopsworks Agent Protocol, with the feature store
in place of BigQuery. Raw alert text in; triage, threat intelligence, log
investigation, a response playbook and an incident out, with disruptive actions
held for the analyst's approval.

```
classify ─▶ triage ─┬─ duplicate ──────────────────────────────▶ report
                    ├─ IOC_MATCH / PHISHING ─▶ threat_intel ─▶ investigate ─▶ respond ─▶ record ─▶ report
                    └─ EDR / UNCATEGORIZED ──▶ investigate ─▶ threat_intel ─▶ respond ─▶ record ─▶ report
approval ─▶ approve ─▶ report
other    ─▶ converse
```

| File | Purpose |
|---|---|
| `feature_pipeline.py` | The recipe's six SecOps CSVs into seven keyed feature groups, each shaped for the question a tool asks |
| `store.py` | The recipe's tools over the feature store, with the `in_evaluation()` guard on the two that write |
| `guardian_agent.py` | The workflow as a LangGraph `StateGraph`, served by `AgentApp` |
| `guardian_agent_openai.py` | The recipe's own shape on the OpenAI Agents SDK: an orchestrator whose plan is its prompt, with the analysts as tools |
| `prompts.py` | The three things the model is asked; what changed from the recipe and why is in its docstring |
| `sample_alerts.txt` | The recipe's three sample alerts, to paste into the chat |
| `requirements.txt` | Deployment requirements |
| `evaluation/` | Suites built from the sample alerts, plus the two-turn cases the recipe could not test |

## How the recipe maps

| ADK | Here |
|---|---|
| Orchestrator `Agent` whose prompt is an execution plan with conditional routing | A `StateGraph`: the routing is edges, the plan is the graph, the Graph tab shows it |
| Four sub-agent LLMs, each wrapping one tool and told to hand control back | Nodes that call the tool and, where analysis is needed, ask the model for a structured report |
| `BuiltInPlanner` thinking on the orchestrator | Structured output for classification and investigation; free text only for the report |
| BigQuery SQL: dedup scan, asset lookup, log queries, intel `IN`, playbook match | Keyed lookups: an incident index by `host|alert_type`, assets by host, logs per host as JSON filtered in code, intel by indicator (batched), playbooks by condition |
| `createIncidentTool` (present, never called by the plan) | `record` node opens the incident, so the next identical alert is a duplicate |
| `responseExecutionTool` after HITL approval | Steps without `requires_approval` run at once (simulated); the rest wait in session memory for an "approved" turn |

Two things the model decides, structured: what kind of alert this is and what
is in it, and what the logs say happened, including decoding encoded PowerShell
and deriving new indicators. One thing it writes: the report, step by step,
ending with the JSON incident log the recipe asked its orchestrator for.

### Feature groups

| Feature group | Key | Answers |
|---|---|---|
| `secops_assets` | `hostname` | owner and business criticality |
| `secops_host_process_events` | `hostname` | the host's process events, newest first |
| `secops_host_network_connections` | `hostname` | the host's connections, newest first |
| `secops_threat_intel` | `ioc_value` | is this indicator known bad, and as what |
| `secops_playbooks` | `triggering_condition` | the playbook's ordered steps |
| `secops_incidents` | `incident_id` | the incident ledger; the agent appends |
| `secops_incident_index` | `dedup_key` | the latest incident per host and alert type |

The online store is a keyed lookup, not a query engine, so each table is
shaped for its question: the logs are one row per host holding its events as
JSON, and the filtering the recipe's SQL did (parent process, destination IP,
last 24 hours) happens in Python on that row.

## Two entry points

`guardian_agent.py` makes the recipe's execution plan a graph: the routing is edges and the
model is asked three things. `guardian_agent_openai.py` keeps the recipe's own shape: an
orchestrator told the plan in its prompt, with the four analysts attached as tools, each
wrapping one store lookup; held actions wait in working memory for `approve_pending`. Same
store, same feature groups, same evaluation suites. The OpenAI entry point runs on OpenAI's
models (`OPENAI_API_KEY`) or any LiteLLM model name in `GUARDIAN_OPENAI_MODEL`.

## Deploy

```bash
python feature_pipeline.py            # downloads the recipe's CSVs, ~1 minute
```

The agent is three files, so deploy it git-backed and the repository comes along:

```python
agents = hopsworks.login().get_agent_serving()
guardian = agents.deploy_agent(
    "cyber_guardian/guardian_agent.py", name="cyberguardian",
    git_url="https://github.com/<you>/hopsragbenchagent.git", git_provider="GitHub",
    git_branch="main", requirements="cyber_guardian/requirements.txt",
    environment="python-agent-pipeline", tracing={"enabled": True},
)
guardian.start()
print(guardian.chat(open("cyber_guardian/sample_alerts.txt").read().split("Sample Input 2")[0]).text)
```

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | The workflow's model calls and the memory summariser run on Claude. |
| `GUARDIAN_MODEL` | Model for classification, investigation and the report. Default `claude-sonnet-4-5`. |
| `CYBER_GUARDIAN_DATA` | Pipeline only: a directory holding the recipe's CSVs, to skip the download. |

## Evaluation

```bash
python -m cyber_guardian.evaluation.apply --publish
```

- **A question runs no workflow** — a greeting or a stray "approved" gets the
  orchestrator's description and a request for alert text; nothing runs.
- **Each alert type takes the recipe's path** — the three sample alerts, with
  `tool_order` checking the recipe's routing (intel before investigation for
  the IOC match and the phishing report, after it for the EDR detection) and a
  rubric per alert naming the threat, the playbook and the approval. Gates at 100%.
- **Duplicates stop, approvals execute** — the same alert twice is a duplicate
  after triage; "approved" executes exactly the held steps; "no" executes nothing.

The write-guarded tools record nothing under evaluation, but an incident opened
earlier in the same conversation still counts as a duplicate, which is what
lets the duplicate case be tested without a write.
