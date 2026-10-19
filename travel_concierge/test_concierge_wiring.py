"""That the cohort wires up the recipe's way, and that working memory does what session state did.

The SDK, the models and the network are stubbed; the day-of segment finder and
the phase rule run for real on the recipe's Seattle scenario.

    python -m pytest travel_concierge/test_concierge_wiring.py
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = ("inspiration_agent", "planning_agent", "booking_agent", "pre_trip_agent", "in_trip_agent", "post_trip_agent")

STUBBED = (
    "hopsworks_agents", "hopsworks_agents.protocol", "hopsworks_agents.protocol.autoevents",
    "agents", "agents.stream_events", "agents.extensions", "agents.extensions.models",
    "agents.extensions.models.litellm_model", "openai", "openai.types", "openai.types.responses",
)


class FakeAgent:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.handoffs = kw.get("handoffs", [])
        self.tools = kw.get("tools", [])

    def as_tool(self, tool_name, tool_description, custom_output_extractor=None, **kw):
        return types.SimpleNamespace(name=tool_name, description=tool_description, extractor=custom_output_extractor, agent=self)


@pytest.fixture
def stubbed(monkeypatch):
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    agents = sys.modules["agents"]
    agents.Agent = FakeAgent
    agents.function_tool = lambda fn: types.SimpleNamespace(name=fn.__name__, description=fn.__doc__, fn=fn)
    agents.ModelSettings = lambda **kw: kw
    agents.RunConfig = lambda **kw: kw
    agents.WebSearchTool = lambda: "web_search"
    # fixed doubles rather than a fresh mock per attribute lookup, so a test can ask
    # what the agent asked for
    protocol = sys.modules["hopsworks_agents.protocol"]
    protocol.memory_tools = mock.MagicMock(return_value=[])
    protocol.identity_tools = mock.MagicMock(return_value=[])
    protocol.ManagedMemoryService = mock.MagicMock()
    monkeypatch.syspath_prepend(HERE)
    for name in ("memory", "prompts", "types_", "tools", "concierge_agent", "demo_users"):
        sys.modules.pop(name, None)
    yield


def fake_memory(monkeypatch, subject="conv-1"):
    """A store standing in for the SDK's, keyed (scope, owner, key)."""
    import memory  # noqa: PLC0415

    store: dict[tuple, str] = {}
    mem = mock.MagicMock()
    mem.get_state = lambda scope, owner, key: store.get((scope, owner, key))
    mem.set_state = lambda scope, owner, key, value, **kw: store.__setitem__((scope, owner, key), value)
    mem.list_state = lambda scope, owner: [{"key": k, "value": v} for (s, o, k), v in store.items() if s == scope and o == owner]
    # `recall` and `remember` key `session` by the conversation and `user` by the
    # traveler, which is the rule `HandlerContext` implements. The real one
    # cannot be imported here -- this file stubs the whole SDK out so it needs
    # no cluster, no key and no network -- so the rule is mirrored, and it is
    # under test where it lives: hopsworks-api
    # `tests/agents/test_app.py::TestContextStateWrites`.
    def owner_of(scope):
        return "conv-1" if scope == "session" else subject

    ctx = types.SimpleNamespace(
        memory=mem, conversation_id="conv-1", subject=subject, turn_id="t1",
        state=lambda scope="user": {
            r["key"]: r["value"] for r in mem.list_state(scope, owner_of(scope))
        },
        recall=lambda key, *, scope="user": mem.get_state(scope, owner_of(scope), key),
        remember=lambda key, value, *, scope="user", source_ref=None: (
            mem.set_state(scope, owner_of(scope), key, value, source_ref=source_ref)
            or True
        ),
    )
    monkeypatch.setattr(memory.current_context, "get", lambda default=None: ctx)
    return store


def test_the_cohort_hands_off_the_recipes_way(stubbed):
    import concierge_agent as c  # noqa: PLC0415

    assert c.agent_app is not None
    assert list(c.SUB_AGENTS) == list(AGENTS)
    assert [a.name for a in c.root_agent.handoffs] == list(AGENTS)
    for name, agent in c.SUB_AGENTS.items():
        # every sub-agent can go back to the root and to any peer, as ADK allows by default
        assert [a.name for a in agent.handoffs] == ["root_agent", *[n for n in AGENTS if n != name]]
    assert [t.name for t in c.SUB_AGENTS["inspiration_agent"].tools] == ["place_agent", "poi_agent"]
    assert [t.name for t in c.SUB_AGENTS["in_trip_agent"].tools] == ["trip_monitor", "day_of", "memorize"]
    assert [t.name for t in c.SUB_AGENTS["booking_agent"].tools] == ["create_reservation", "payment_choice", "process_payment"]


def test_the_active_agent_keeps_the_conversation(stubbed, monkeypatch):
    import concierge_agent as c  # noqa: PLC0415
    import memory  # noqa: PLC0415

    fake_memory(monkeypatch)
    assert c.starting_agent().name == "root_agent"
    memory.put(memory.ACTIVE_AGENT, "in_trip_agent")
    assert c.starting_agent().name == "in_trip_agent"


def test_the_scenario_seeds_the_conversation_once_and_sets_the_phase(stubbed, monkeypatch):
    import memory  # noqa: PLC0415

    store = fake_memory(monkeypatch)
    memory.load_scenario(os.path.join(HERE, "profiles", "itinerary_seattle_example.json"))
    assert memory.get(memory.ITIN_KEY)["trip_name"] == "San Diego to Seattle Getaway"
    assert memory.get(memory.ITIN_DATETIME) == "2025-06-15 00:00"
    assert memory.phase() == "in_trip"
    assert memory.phase("2025-06-01 09:00") == "pre_trip" and memory.phase("2025-07-01 09:00") == "post_trip"
    # a second load does not reset what the conversation has since memorized
    memory.put(memory.ITIN_DATETIME, "2025-06-16 12:45:00")
    memory.load_scenario(os.path.join(HERE, "profiles", "itinerary_seattle_example.json"))
    assert memory.get(memory.ITIN_DATETIME) == "2025-06-16 12:45:00"
    values = memory.prompt_context()
    assert "Pike Place Market" in values["itinerary"] and values["phase"] == "in_trip"
    assert store[("session", "conv-1", "_itin_initialized")] == "true"


def test_day_of_finds_the_leg_from_lunch_to_the_space_needle(stubbed):
    import memory  # noqa: PLC0415

    with open(os.path.join(HERE, "profiles", "itinerary_seattle_example.json"), encoding="utf-8") as handle:
        state = json.load(handle)["state"]
    travel_from, travel_to, leave_by, arrive_by = memory.find_segment(state["user_profile"], state["itinerary"], "2025-06-16 12:45:00")
    assert travel_from.startswith("Lunch at Ivar's Acres of Clams") and leave_by == "13:30"
    assert travel_to.startswith("Visit the Space Needle") and arrive_by == "14:30"
    # before the trip: home to the airport, an hour before boarding
    travel_from, travel_to, _, arrive_by = memory.find_segment(state["user_profile"], state["itinerary"], "2025-06-15 05:00:00")
    assert travel_from.startswith("drive from") and travel_to == "SAN Airport" and arrive_by == "An hour before 07:30"


def test_the_mocks_keep_the_recipes_rules(stubbed, monkeypatch):
    import tools  # noqa: PLC0415

    fake_memory(monkeypatch)
    assert "closed" in json.loads(tools.event_booking_check.fn("Visit the Space Needle", "2025-06-16", "Seattle"))["status"]
    assert "open" in json.loads(tools.event_booking_check.fn("Pike Place Market", "2025-06-16", "Seattle"))["status"]
    reservation = json.loads(tools.create_reservation.fn("Flights AA1234/UA5678", 900))
    rid = reservation["reservation_id"]
    assert json.loads(tools.process_payment.fn(rid, "Apple Pay"))["status"] == "declined"
    paid = json.loads(tools.process_payment.fn(rid, "Google Pay"))
    assert paid["status"] == "approved" and paid["paid_in_usd"] == 900
    assert "Google Pay" in json.loads(tools.payment_choice.fn())["note"]
    assert json.loads(tools.process_payment.fn("RSV-NOPE", "Google Pay"))["status"] == "error"


def test_memorize_and_preferences_land_in_their_scopes(stubbed, monkeypatch):
    import memory  # noqa: PLC0415
    import tools  # noqa: PLC0415

    store = fake_memory(monkeypatch, subject="alice")
    tools.memorize.fn("itinerary_datetime", "2025-06-16 12:45:00")
    tools.memorize.fn("outbound_flight_selection", json.dumps({"flight_number": "AA1234"}))
    assert store[("session", "conv-1", "itinerary_datetime")] == "2025-06-16 12:45:00"
    assert memory.get("outbound_flight_selection") == {"flight_number": "AA1234"}
    tools.remember_preference.fn("food_preference", "vegan, loves ceviche")
    assert store[("user", "alice", "food_preference")] == "vegan, loves ceviche"
    assert "learned_preferences" in memory.prompt_context()["user_profile"]


def test_the_concierge_can_identify_and_remember(stubbed):
    """The memory tools and `identify` are on the root agent.

    The preferences the post-trip agent learns go to `user` scope, which is keyed by the
    subject, so without `identify` they have nowhere to go.
    """
    import concierge_agent  # noqa: PLC0415

    protocol = sys.modules["hopsworks_agents.protocol"]
    assert protocol.memory_tools.call_args.args == ("openai_agents",)
    assert protocol.identity_tools.call_args.args == ("openai_agents",)
    assert protocol.ManagedMemoryService.call_args.kwargs["long_term"] is True
    assert concierge_agent.root_agent is not None


def test_the_root_instructions_ask_who_is_travelling(stubbed):
    from string import Formatter  # noqa: PLC0415

    import concierge_agent  # noqa: PLC0415
    import demo_users  # noqa: PLC0415
    import prompts  # noqa: PLC0415

    # the context block's own placeholders, so this test does not have to know them
    fields = {name for _, name, _, _ in Formatter().parse(prompts.CONTEXT_BLOCK) if name}
    with (
        mock.patch.object(concierge_agent.memory, "prompt_context",
                          return_value=dict.fromkeys(fields, "")),
        mock.patch.object(concierge_agent.memory, "itinerary_summary", return_value=""),
    ):
        instructions = concierge_agent._root_instructions(None, None)

    assert demo_users.IDENTITY_INSTRUCTIONS in instructions
    for user in demo_users.USERS:
        assert user["id"] in instructions


def test_the_question_is_skipped_when_the_client_already_knows(stubbed):
    """A real deployment has a login: the identity arrives with the request.

    `subject` asserted by the client is what a signed-in application sends, and the SDK
    keys memory by it. Asking is the demo's stand-in, so it has to stop when there is a
    real answer already.
    """
    import types  # noqa: PLC0415

    import demo_users  # noqa: PLC0415

    asserted = types.SimpleNamespace(
        has_subject=True, subject="dana@example.com", subject_source="client"
    )
    note = demo_users.already_identified(asserted)
    assert "dana@example.com" in note
    assert "Do not ask who they are" in note

    # a conversation nobody identified falls back to the question
    anonymous = types.SimpleNamespace(
        has_subject=False, subject="conv-1", subject_source="conversation"
    )
    assert demo_users.already_identified(anonymous) == ""
    # and so does one the agent worked out itself, which is already past the question
    worked_out = types.SimpleNamespace(
        has_subject=True, subject="dana@example.com", subject_source="app"
    )
    assert demo_users.already_identified(worked_out) == ""
