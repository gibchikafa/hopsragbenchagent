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
    monkeypatch.syspath_prepend(HERE)
    for name in ("memory", "prompts", "types_", "tools", "concierge_agent"):
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
    ctx = types.SimpleNamespace(memory=mem, conversation_id="conv-1", subject=subject, turn_id="t1",
                                state=lambda scope="user": {r["key"]: r["value"] for r in mem.list_state(scope, "conv-1" if scope == "session" else subject)})
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
