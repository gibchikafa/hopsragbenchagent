# Travel concierge

Google's [ADK travel-concierge recipe](https://github.com/google/adk-recipes/tree/main/python/agents/travel-concierge)
on the OpenAI Agents SDK and the Hopsworks Agent Protocol. A cohort of agents
takes a traveler from an idea to a booked trip and then supports them before,
during and after it, handing the conversation to one another.

```
root_agent ──▶ inspiration_agent ──▶ planning_agent ──▶ booking_agent     before booking
    └────────▶ pre_trip_agent ──▶ in_trip_agent ──▶ post_trip_agent         after: the trip has an itinerary
```

| File | Purpose |
|---|---|
| `concierge_agent.py` | The root and the six agents with their hand-offs, served by `AgentApp` |
| `tools.py` | The recipe's agent-tools as agents-as-tools with structured output; its mocks as function tools |
| `prompts.py` | The recipe's prompts, ported; what changed and why is in its docstring |
| `memory.py` | Working memory: the recipe's session state on the SDK's memory tiers; the scenario loader; the day-of segment finder |
| `types_.py` | The shapes the tools answer in (flights, hotels, the itinerary) |
| `profiles/` | The recipe's two scenarios: an empty itinerary, and a booked Seattle trip |
| `requirements.txt` | Deployment requirements |
| `evaluation/` | The recipe's three evaluation sets as suites, plus one it did not have |

## Why the OpenAI Agents SDK

The recipe's whole mechanism is the hand-off: a root agent with six
sub-agents that transfer the conversation to one another, and whichever agent
last had it keeps it on the next turn. That is the OpenAI Agents SDK's own
shape, so the port is nearly one-to-one:

| ADK | OpenAI Agents SDK |
|---|---|
| `Agent(sub_agents=[...])`, `transfer_to_agent` | `Agent(handoffs=[...])`; the hand-off tools are named `transfer_to_<agent>` too |
| `AgentTool(agent=place_agent)` with `output_schema` | `place_agent.as_tool(...)` with `output_type` |
| `output_key` writing the tool's result to session state | the tool's `custom_output_extractor` writes it to working memory |
| the active agent persists across turns | the run's `last_agent` is remembered; the next turn starts there |
| `before_agent_callback` seeding session state from a scenario file | `memory.load_scenario()` on a conversation's first turn |
| `{itinerary}`, `{user_profile}`, `{origin}` in instructions | dynamic `instructions` filled from working memory each call |

On LangGraph the same thing needs a classifier to pick the agent and a loop to
carry a transfer through the turn; on the Claude Agent SDK, sub-agents are a
harness feature rather than a conversation one.

## What is real and what is simulated

As in the recipe: destinations, points of interest, flights, seats, hotels,
rooms, the itinerary and the packing list are model calls asked for a shape.
Flight status, event bookings and weather are mocks (the Space Needle is
closed on the sample trip). Reservations and payments are mocks with the
recipe's rules: Apple Pay is declined, Google Pay and the card go through.
Pre-trip information comes from a real web search. Google Maps grounding has
no equivalent here, so points of interest carry coordinates as the model knows
them, unverified.

## Memory

The recipe keeps everything in ADK session state. Here the same keys live in
the SDK's `session` scope, this conversation only, which is what a trip plan
is; `itinerary_datetime` is the "current time on the trip", memorized to move
the clock as the recipe's sample sessions do. Preferences the post-trip agent
learns go to `user` scope, keyed on who is talking, so they are part of the
profile in later conversations, which is what the recipe said it wanted from
them. The stored itinerary is a few thousand characters, so the state value
limit is raised.

## Deploy

```python
agents = hopsworks.login().get_agent_serving()
concierge = agents.deploy_agent(
    "travel_concierge/concierge_agent.py", name="concierge",
    git_url="https://github.com/<you>/hopsragbenchagent.git", git_provider="GitHub",
    git_branch="main", requirements="travel_concierge/requirements.txt",
    environment="python-agent-pipeline", tracing={"enabled": True},
)
concierge.start()
reply = concierge.chat("Inspire me about the Americas")
concierge.chat("Tell me more what I can do in Peru", conversation_id=reply.conversation_id)
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | The agents run on OpenAI models by default (the SDK's default model); the web search tool and the memory summariser (`openai_summarizer`) use it too, so one key covers the deployment. |
| `CONCIERGE_MODEL` | A model name for OpenAI, or a LiteLLM name such as `anthropic/claude-sonnet-4-5` (then that provider's key is needed). |
| `TRAVEL_CONCIERGE_SCENARIO` | The scenario file loaded on a conversation's first turn. Default `profiles/itinerary_empty_default.json`; the pre-trip and in-trip flows need `profiles/itinerary_seattle_example.json`. |

## Evaluation

```bash
python -m travel_concierge.evaluation.apply --publish
```

- **Inspiration, then hand over to planning** — the recipe's inspire set:
  destinations, then Peru's points of interest, then "start planning" moves the
  conversation to the planning agent.
- **Planning selects but never books** — the recipe's "worth trying" prompt:
  told to choose everything without input, planning does, memorizes each
  choice, and stops before the itinerary and any booking. Gates at 100%.
- **Pre-trip briefing from the itinerary** and **In-trip monitoring and day-of
  logistics** — the recipe's pretrip and intrip sets, on the Seattle scenario.
  Deploy with `TRAVEL_CONCIERGE_SCENARIO` pointing at it before running them.
