"""The concierge's tools on the OpenAI Agents SDK: the recipe's agent-tools as agents-as-tools,
its mocks as function tools.

The recipe's ``place_agent``, ``poi_agent``, the flight and hotel searches, the
seat and room selections, the itinerary and packing agents are each an LLM
agent asked for a JSON shape and attached as an ``AgentTool``. Here each is an
``Agent`` with that shape as its ``output_type``, attached with ``as_tool``;
its output is stored in working memory by the extractor, as the recipe's
``output_key`` did. The recipe's mocks (flight status, event booking, weather,
the reservation and payment agents told to simulate) are function tools with
the same rules: Apple Pay is declined, the Space Needle is closed.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from agents import Agent, ModelSettings, RunResult, function_tool

import memory
from prompts import (
    FLIGHT_SEARCH_INSTR,
    FLIGHT_SEAT_SELECTION_INSTR,
    HOTEL_ROOM_SELECTION_INSTR,
    HOTEL_SEARCH_INSTR,
    ITINERARY_AGENT_INSTR,
    LOGISTIC_INSTR_TEMPLATE,
    NEED_ITIN_INSTR,
    PLACE_AGENT_INSTR,
    POI_AGENT_INSTR,
    SEARCH_GROUNDING_INSTR,
    TRIP_MONITOR_INSTR,
    WHATTOPACK_INSTR,
)
from types_ import (
    DestinationIdeas,
    FlightsSelection,
    HotelsSelection,
    Itinerary,
    PackingList,
    POISuggestions,
    RoomsSelection,
    SeatsSelection,
)

#: OpenAI by default (the SDK's own default model); "anthropic/claude-sonnet-4-5" and the like go
#: through LiteLLM, which the requirements bring in
MODEL_NAME = os.environ.get("CONCIERGE_MODEL", "")


def model() -> Any:
    """What the agents run on: a model name for OpenAI, a LiteLLM model for anyone else."""
    if "/" in MODEL_NAME:
        from agents.extensions.models.litellm_model import LitellmModel  # noqa: PLC0415

        return LitellmModel(model=MODEL_NAME)
    return MODEL_NAME or None


def _model_kwargs() -> dict[str, Any]:
    chosen = model()
    return {"model": chosen} if chosen is not None else {}


def _context_lines(_ctx=None, _agent=None) -> str:
    values = memory.prompt_context()
    return (
        f"\n\nCurrent user:\n<user_profile>\n{values['user_profile']}\n</user_profile>\n"
        f"Current time: {values['_time']}\n"
        f"Origin: {values['origin'] or '(unknown)'}; destination: {values['destination'] or '(unknown)'}; "
        f"dates: {values['start_date'] or '?'} to {values['end_date'] or '?'}"
    )


def _dynamic(instruction: str, with_trip: bool = False):
    """Instructions built per call: the recipe's placeholders filled from working memory."""

    def build(_ctx, _agent) -> str:
        text = instruction.format(**memory.prompt_context()) if with_trip else instruction
        return text + _context_lines()

    return build


def _stores(key: str | None, shape: type | None = None):
    """An output extractor that keeps the agent-tool's result in working memory, as output_key did."""

    async def extract(result: RunResult) -> str:
        output = result.final_output
        data = output.model_dump() if hasattr(output, "model_dump") else output
        if key:
            memory.put(key, data)
        return json.dumps(data) if not isinstance(data, str) else data

    return extract


def _agent_tool(name: str, description: str, instruction: str, shape: type, key: str | None,
                with_trip: bool = False, temperature: float = 0.3):
    agent = Agent(
        name=name,
        handoff_description=description,
        instructions=_dynamic(instruction, with_trip),
        output_type=shape,
        model_settings=ModelSettings(temperature=temperature),
        **_model_kwargs(),
    )
    return agent.as_tool(tool_name=name, tool_description=description, custom_output_extractor=_stores(key, shape))


# ── inspiration ─────────────────────────────────────────────────────────────

place_agent = _agent_tool(
    "place_agent", "Suggest up to three vacation destinations for a vague idea: a region, a mood, a kind of trip.",
    PLACE_AGENT_INSTR, DestinationIdeas, "place",
)
poi_agent = _agent_tool(
    "poi_agent", "Suggest up to five points of interest and things to do at a specific destination.",
    POI_AGENT_INSTR, POISuggestions, "poi",
)

# ── planning ────────────────────────────────────────────────────────────────

flight_search_agent = _agent_tool(
    "flight_search_agent", "Find flight choices between an origin and a destination on the dates given (simulated).",
    FLIGHT_SEARCH_INSTR, FlightsSelection, "flight",
)
flight_seat_selection_agent = _agent_tool(
    "flight_seat_selection_agent", "Show the seat map for a chosen flight, with prices and availability (simulated).",
    FLIGHT_SEAT_SELECTION_INSTR, SeatsSelection, "seat",
)
hotel_search_agent = _agent_tool(
    "hotel_search_agent", "Find hotel choices at the destination for the dates given (simulated).",
    HOTEL_SEARCH_INSTR, HotelsSelection, "hotel",
)
hotel_room_selection_agent = _agent_tool(
    "hotel_room_selection_agent", "Show the room types at the chosen hotel, with prices and availability (simulated).",
    HOTEL_ROOM_SELECTION_INSTR, RoomsSelection, "room",
)


async def _store_itinerary(result: RunResult) -> str:
    output = result.final_output
    data = output.model_dump() if hasattr(output, "model_dump") else output
    memory.put(memory.ITIN_KEY, data)
    memory.put(memory.ITIN_START_DATE, data.get("start_date", ""))
    memory.put(memory.ITIN_END_DATE, data.get("end_date", ""))
    if not memory.get(memory.ITIN_DATETIME):
        memory.put(memory.ITIN_DATETIME, data.get("start_date", "") + " 00:00")
    return "Itinerary stored: " + json.dumps(data)


itinerary_agent = Agent(
    name="itinerary_agent",
    handoff_description="Turn the agreed plan into the structured itinerary and store it as the trip's itinerary.",
    instructions=_dynamic(ITINERARY_AGENT_INSTR, with_trip=True),
    output_type=Itinerary,
    model_settings=ModelSettings(temperature=0.1),
    **_model_kwargs(),
).as_tool(
    tool_name="itinerary_agent",
    tool_description="Turn the agreed plan (days, flights, hotel, activities, times) into the structured itinerary and store it.",
    custom_output_extractor=_store_itinerary,
)


@function_tool
def memorize(key: str, value: str) -> str:
    """Memorize one piece of trip information under a key, for the rest of this conversation.

    Args:
        key: The label, e.g. origin, destination, start_date, outbound_flight_selection, itinerary_datetime.
        value: The information to store; a JSON entry from a search result is stored as it is.
    """
    key = key.strip()
    if not key:
        return "Nothing stored: the key was empty."
    try:
        parsed = json.loads(value)
        memory.put(key, parsed if isinstance(parsed, (dict, list)) else value)
    except ValueError:
        memory.put(key, value)
    return f'Stored "{key}": "{value[:200]}"'


@function_tool
def remember_preference(key: str, value: str) -> str:
    """Remember a preference the traveler revealed, across conversations: food, destinations,
    activities, businesses they would recommend.

    Args:
        key: The preference, e.g. food_preference, likes, dislikes, recommended_business.
        value: What they said.
    """
    key = key.strip()
    if not key:
        return "Nothing stored: the key was empty."
    memory.put(key, value.strip(), scope=memory.USER_SCOPE)
    return f'Remembered "{key}": "{value[:200]}" for next time.'


# ── booking (the recipe's simulated reservation and payment agents) ────────


@function_tool
def create_reservation(item: str, price_in_usd: int) -> str:
    """Reserve one bookable item from the itinerary: a flight pair, a hotel stay, a ticketed visit (simulated).

    Args:
        item: What is being reserved, e.g. "Flights AA1234 / UA5678 San Diego-Seattle return".
        price_in_usd: The total price to be paid for it.
    """
    reservation_id = "RSV-" + uuid.uuid4().hex[:6].upper()
    reservations = memory.get("reservations", [])
    reservations = reservations if isinstance(reservations, list) else []
    reservations.append({"reservation_id": reservation_id, "item": item, "price_in_usd": int(price_in_usd), "paid": False})
    memory.put("reservations", reservations)
    return json.dumps({"reservation_id": reservation_id, "item": item, "price_in_usd": int(price_in_usd),
                       "status": "reserved, awaiting payment"})


@function_tool
def payment_choice() -> str:
    """The payment methods on offer: Apple Pay, Google Pay, or the credit card on file. Ask the traveler to choose."""
    last = memory.get("last_payment_method", "")
    note = f"The traveler used {last} last time; ask whether to use it again." if last else ""
    return json.dumps({"choices": ["Apple Pay", "Google Pay", "Credit Card on file"], "note": note})


@function_tool
def process_payment(reservation_id: str, payment_method: str) -> str:
    """Pay for a reservation with the chosen method (simulated). Apple Pay is declined; Google Pay and the card go through.

    Args:
        reservation_id: The reservation from create_reservation.
        payment_method: Apple Pay, Google Pay, or Credit Card.
    """
    method = payment_method.strip().lower()
    reservations = memory.get("reservations", [])
    entry = next((r for r in reservations if r.get("reservation_id") == reservation_id), None) if isinstance(reservations, list) else None
    if entry is None:
        return json.dumps({"status": "error", "message": f"no reservation {reservation_id}; create it first"})
    if "apple" in method:
        return json.dumps({"status": "declined", "reservation_id": reservation_id,
                           "message": "Apple Pay declined the transaction; ask for another method."})
    if not ("google" in method or "card" in method or "credit" in method):
        return json.dumps({"status": "error", "message": f"unknown payment method {payment_method!r}"})
    order_id = "ORD-" + uuid.uuid4().hex[:8].upper()
    entry.update({"paid": True, "order_id": order_id, "payment_method": payment_method.strip()})
    memory.put("reservations", reservations)
    memory.put("last_payment_method", payment_method.strip())
    return json.dumps({"status": "approved", "reservation_id": reservation_id, "order_id": order_id,
                       "paid_in_usd": entry.get("price_in_usd"), "payment_method": payment_method.strip()})


# ── pre-trip ────────────────────────────────────────────────────────────────

search_grounding = Agent(
    name="search_grounding",
    handoff_description="Answer a travel question from a current web search.",
    instructions=SEARCH_GROUNDING_INSTR,
    tools=[__import__("agents").WebSearchTool()],
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
).as_tool(
    tool_name="search_grounding",
    tool_description="Answer a travel question from a current web search: visa requirements, medical requirements, storms, travel advisories. Put the origin, destination and dates in the question.",
)

what_to_pack = _agent_tool(
    "what_to_pack", "Suggest what to pack for the trip, given the origin, destination, dates and activities.",
    WHATTOPACK_INSTR, PackingList, "what_to_pack",
)

# ── in-trip: the recipe's mocks, and the day-of and monitor agents ──────────


@function_tool
def flight_status_check(flight_number: str, flight_date: str, checkin_time: str, departure_time: str) -> str:
    """Check the status of a flight (simulated: every flight is on time).

    Args:
        flight_number: e.g. AA1234.
        flight_date: YYYY-MM-DD.
        checkin_time: HH:MM.
        departure_time: HH:MM.
    """
    return json.dumps({"status": f"Flight {flight_number} on {flight_date} checked: on time"})


@function_tool
def event_booking_check(event_name: str, event_date: str, event_location: str) -> str:
    """Check the status of a booked event (simulated: the Space Needle is closed).

    Args:
        event_name: The event or venue.
        event_date: YYYY-MM-DD.
        event_location: Where it is.
    """
    if "space needle" in event_name.lower():
        return json.dumps({"status": f"{event_name} is closed on {event_date}."})
    return json.dumps({"status": f"{event_name} on {event_date} checked: open as booked"})


@function_tool
def weather_impact_check(activity_name: str, activity_date: str, activity_location: str) -> str:
    """Check whether the weather may affect an outdoor activity (simulated: it does not).

    Args:
        activity_name: The activity.
        activity_date: YYYY-MM-DD.
        activity_location: Where it is.
    """
    return json.dumps({"status": f"{activity_name} on {activity_date} checked: no weather impact expected"})


def _monitor_instructions(_ctx, _agent) -> str:
    values = memory.prompt_context()
    if not values["itinerary"]:
        return NEED_ITIN_INSTR
    return (TRIP_MONITOR_INSTR
            + f"\n<itinerary>\n{values['itinerary']}\n</itinerary>\n<user_profile>\n{values['user_profile']}\n</user_profile>")


async def _store_checks(result: RunResult) -> str:
    text = str(result.final_output or "")
    memory.put("daily_checks", text)
    return text


trip_monitor = Agent(
    name="trip_monitor_agent",
    handoff_description="Monitor the itinerary's bookings and flag what needs a change.",
    instructions=_monitor_instructions,
    tools=[flight_status_check, event_booking_check, weather_impact_check],
    model_settings=ModelSettings(temperature=0.0),
    **_model_kwargs(),
).as_tool(
    tool_name="trip_monitor",
    tool_description="Check every flight, booked event and weather-sensitive activity in the itinerary, and list what needs a change (simulated checks).",
    custom_output_extractor=_store_checks,
)


def _day_of_instructions(_ctx, _agent) -> str:
    """The recipe's transit_coordination: a dynamic instruction for the leg the traveler is on."""
    itinerary = memory.get(memory.ITIN_KEY, {})
    if not itinerary:
        return NEED_ITIN_INSTR
    profile = memory.get(memory.PROF_KEY, {})
    current = str(memory.get(memory.ITIN_DATETIME) or (itinerary.get("start_date", "") + " 00:00"))
    try:
        travel_from, travel_to, leave_by, arrive_by = memory.find_segment(profile, itinerary, current)
    except ValueError:
        return (f"The current time on the trip, {current!r}, is not a date and time you can read; "
                "tell the traveler to memorize it as 'YYYY-MM-DD HH:MM'.")
    return LOGISTIC_INSTR_TEMPLATE.format(
        CURRENT_TIME=current, TRAVEL_FROM=travel_from, LEAVE_BY_TIME=leave_by,
        TRAVEL_TO=travel_to, ARRIVE_BY_TIME=arrive_by,
    ) + _context_lines()


day_of = Agent(
    name="day_of_agent",
    handoff_description="Logistics for the leg the traveler is on right now.",
    instructions=_day_of_instructions,
    model_settings=ModelSettings(temperature=0.2),
    **_model_kwargs(),
).as_tool(
    tool_name="day_of",
    tool_description="Logistics for the leg the traveler is on right now: how and when to get from where they are to the next event, given the itinerary and the current time on the trip. Pass what they need, e.g. 'help'.",
)

# ── who gets what ───────────────────────────────────────────────────────────

TOOLS_BY_AGENT = {
    "inspiration_agent": [place_agent, poi_agent],
    "planning_agent": [flight_search_agent, flight_seat_selection_agent, hotel_search_agent,
                       hotel_room_selection_agent, itinerary_agent, memorize],
    "booking_agent": [create_reservation, payment_choice, process_payment],
    "pre_trip_agent": [search_grounding, what_to_pack],
    "in_trip_agent": [trip_monitor, day_of, memorize],
    "post_trip_agent": [remember_preference, memorize],
}
