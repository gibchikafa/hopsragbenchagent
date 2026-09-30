"""The concierge's working memory: the recipe's session state on the SDK's memory tiers.

The recipe keeps everything in ADK's session state: the user profile, the
itinerary, the trip's dates and selections, the active agent, the "current
time on the trip". Here the same keys live in the SDK's ``session`` scope,
this conversation only, which is what a trip plan is; and the preferences the
post-trip agent learns go to ``user`` scope, keyed on who is talking, which is
where the recipe said they belonged ("useful in future interactions").

A conversation starts by loading a scenario file, as the recipe's
``before_agent_callback`` did: the profile and, for the post-booking agents,
an itinerary. ``TRAVEL_CONCIERGE_SCENARIO`` names the file; the default is the
empty itinerary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from hopsworks_agents.protocol.autoevents import current_context

SCOPE = "session"
USER_SCOPE = "user"
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
SCENARIO_PATH = os.environ.get(
    "TRAVEL_CONCIERGE_SCENARIO", os.path.join(PROFILES_DIR, "itinerary_empty_default.json")
)

#: the recipe's constants: keys into the state
ITIN_KEY = "itinerary"
PROF_KEY = "user_profile"
ITIN_INITIALIZED = "_itin_initialized"
ITIN_START_DATE = "itinerary_start_date"
ITIN_END_DATE = "itinerary_end_date"
ITIN_DATETIME = "itinerary_datetime"
ACTIVE_AGENT = "active_agent"
#: what every prompt may reference; empty until memorized
TRIP_KEYS = (
    "origin", "destination", "start_date", "end_date",
    "outbound_flight_selection", "outbound_seat_number",
    "return_flight_selection", "return_seat_number",
    "hotel_selection", "room_selection", "poi",
)


def _ctx():
    return current_context.get(None)


def _memory_and_owner(scope: str = SCOPE):
    ctx = _ctx()
    if ctx is None or ctx.memory is None:
        return None, None
    return ctx.memory, (ctx.conversation_id if scope == SCOPE else ctx.subject)


def get(key: str, default: Any = "", scope: str = SCOPE) -> Any:
    """A value from working memory, JSON-decoded when it was stored as JSON."""
    memory, owner = _memory_and_owner(scope)
    if memory is None:
        return default
    raw = memory.get_state(scope, owner, key)
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def put(key: str, value: Any, scope: str = SCOPE) -> None:
    memory, owner = _memory_and_owner(scope)
    if memory is None:
        return
    text = value if isinstance(value, str) else json.dumps(value)
    ctx = _ctx()
    memory.set_state(
        scope, owner, key, text,
        source_ref=json.dumps({"conversation_id": ctx.conversation_id, "turn_id": ctx.turn_id}),
    )


def all_state(scope: str = SCOPE) -> dict[str, Any]:
    ctx = _ctx()
    if ctx is None or ctx.memory is None:
        return {}
    out = {}
    for key, raw in ctx.state(scope).items():
        try:
            out[key] = json.loads(raw)
        except (ValueError, TypeError):
            out[key] = raw
    return out


def load_scenario(path: str = SCENARIO_PATH) -> None:
    """The recipe's ``_load_precreated_itinerary``: seed the conversation once, from a file."""
    if get(ITIN_INITIALIZED, False) is True:
        return
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle).get("state", {})
    for key, value in state.items():
        if value not in ("", None):
            put(key, value)
    itinerary = state.get(ITIN_KEY) or {}
    if itinerary:
        put(ITIN_START_DATE, itinerary.get("start_date", ""))
        put(ITIN_END_DATE, itinerary.get("end_date", ""))
        put(ITIN_DATETIME, itinerary.get("start_date", "") + " 00:00")
    put(ITIN_INITIALIZED, True)


def phase(now: str | None = None) -> str:
    """pre_trip, in_trip, post_trip, or none: the recipe's root-agent rule on the itinerary's dates."""
    itinerary = get(ITIN_KEY, {})
    if not itinerary:
        return "none"
    start = str(get(ITIN_START_DATE) or itinerary.get("start_date") or "")
    end = str(get(ITIN_END_DATE) or itinerary.get("end_date") or "")
    current = str(now or get(ITIN_DATETIME) or start)[:10]
    if not start or not end:
        return "none"
    if current < start:
        return "pre_trip"
    if current > end:
        return "post_trip"
    return "in_trip"


def prompt_context() -> dict[str, Any]:
    """Everything a prompt placeholder may name, as text, from working memory."""
    state = all_state()
    profile = state.get(PROF_KEY) or {}
    learned = all_state(USER_SCOPE)
    if learned:
        profile = {**profile, "learned_preferences": learned}
    itinerary = state.get(ITIN_KEY) or {}
    values: dict[str, Any] = {
        "user_profile": json.dumps(profile, indent=1) if profile else "(unknown)",
        "itinerary": json.dumps(itinerary, indent=1) if itinerary else "",
        "_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "phase": phase(),
        ITIN_DATETIME: state.get(ITIN_DATETIME) or "",
    }
    for key in TRIP_KEYS:
        value = state.get(key, "")
        values[key] = value if isinstance(value, str) else json.dumps(value)
    return values


def itinerary_summary() -> str:
    itinerary = get(ITIN_KEY, {})
    if not itinerary:
        return "(none yet)"
    return (
        f"{itinerary.get('trip_name', 'a trip')}: {itinerary.get('origin', '?')} → "
        f"{itinerary.get('destination', '?')}, {itinerary.get('start_date', '?')} to "
        f"{itinerary.get('end_date', '?')}, {len(itinerary.get('days', []))} day(s)"
    )


# ── the day-of segment finder, ported from the recipe's in_trip/tools.py ────


def _event_time_as_destination(event: dict[str, Any], default: str) -> str:
    return {
        "flight": event.get("boarding_time"),
        "hotel": event.get("check_in_time"),
        "visit": event.get("start_time"),
    }.get(event.get("event_type"), None) or default


def _as_origin(event: dict[str, Any]) -> tuple[str, str]:
    kind = event.get("event_type")
    if kind == "flight":
        return event.get("arrival_airport", "") + " Airport", event.get("arrival_time", "any time")
    if kind == "hotel":
        return f"{event.get('description', '')} {event.get('address', '')}".strip(), "any time"
    if kind == "visit":
        return f"{event.get('description', '')} {event.get('address', '')}".strip(), event.get("end_time", "any time")
    if kind == "home":
        return f"{event.get('local_prefer_mode', '')} from {event.get('address', '')}".strip(), "any time"
    return "Local in the region", "any time"


def _as_destination(event: dict[str, Any]) -> tuple[str, str]:
    kind = event.get("event_type")
    if kind == "flight":
        return event.get("departure_airport", "") + " Airport", "An hour before " + event.get("boarding_time", "")
    if kind == "hotel":
        return f"{event.get('description', '')} {event.get('address', '')}".strip(), "any time"
    if kind == "visit":
        return f"{event.get('description', '')} {event.get('address', '')}".strip(), event.get("start_time", "as soon as possible")
    if kind == "home":
        return f"{event.get('local_prefer_mode', '')} to {event.get('address', '')}".strip(), "any time"
    return "Local in the region", "as soon as possible"


def find_segment(profile: dict[str, Any], itinerary: dict[str, Any], current_datetime: str) -> tuple[str, str, str, str]:
    """The leg the traveler is on now: (from, to, leave_by, arrive_by), from the itinerary and the time."""
    moment = datetime.fromisoformat(current_datetime.strip())
    current_date, current_time = moment.strftime("%Y-%m-%d"), moment.strftime("%H:%M")
    home = profile.get("home") or {"event_type": "home", "address": "", "local_prefer_mode": ""}
    origin, destination = home, home
    for day in itinerary.get("days", []):
        event_date = day.get("date", "")
        found = False
        for event in day.get("events", []):
            origin, destination = destination, event
            event_time = _event_time_as_destination(event, current_time)
            if event_date >= current_date and event_time >= current_time:
                found = True
                break
        if found:
            break
    travel_from, leave_by = _as_origin(origin)
    travel_to, arrive_by = _as_destination(destination)
    return travel_from, travel_to, leave_by, arrive_by
