"""The shapes the concierge's tools answer in, from the recipe's ``shared_libraries/types.py``.

Each search tool and the itinerary tool ask the model for one of these, so what
goes into working memory is a record with known fields rather than prose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Room(BaseModel):
    is_available: bool = Field(description="Whether the room type is available for selection.")
    price_in_usd: int = Field(description="The cost of the room selection.")
    room_type: str = Field(description="Type of room, e.g. Twin with Balcony, King with Ocean View.")


class RoomsSelection(BaseModel):
    rooms: list[Room]


class Hotel(BaseModel):
    name: str = Field(description="Name of the hotel")
    address: str = Field(description="Full address of the hotel")
    check_in_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    check_out_time: str = Field(description="Time in HH:MM format, e.g. 11:00")
    price: int = Field(description="Price of the room per night, in US dollars")


class HotelsSelection(BaseModel):
    hotels: list[Hotel]


class Seat(BaseModel):
    is_available: bool = Field(description="Whether the seat is available for selection.")
    price_in_usd: int = Field(description="The cost of the seat selection.")
    seat_number: str = Field(description="Seat number, e.g. 22A, 34F")


class SeatsSelection(BaseModel):
    seats: list[list[Seat]] = Field(description="Rows of seats, each row a list of seats")


class AirportEvent(BaseModel):
    city_name: str = Field(description="Name of the city")
    airport_code: str = Field(description="IATA code of the airport")
    timestamp: str = Field(description="ISO 8601 date and time")


class Flight(BaseModel):
    flight_number: str = Field(description="Unique identifier for the flight, like BA123, AA31")
    departure: AirportEvent
    arrival: AirportEvent
    airlines: list[str] = Field(description="Airline names, e.g. American Airlines, Emirates")
    price_in_usd: int = Field(description="Flight price in US dollars")
    number_of_stops: int = Field(description="Number of stops during the flight")


class FlightsSelection(BaseModel):
    flights: list[Flight]


class Destination(BaseModel):
    name: str = Field(description="The destination's name")
    country: str = Field(description="The destination's country")
    highlights: str = Field(description="Short description highlighting key features")
    rating: str = Field(description="Numerical rating from 1 to 5, e.g. 4.5")


class DestinationIdeas(BaseModel):
    places: list[Destination]


class POI(BaseModel):
    place_name: str = Field(description="Name of the attraction")
    address: str = Field(description="An address, or enough to find it on a map")
    lat: str = Field(description="Latitude, e.g. 20.6843")
    long: str = Field(description="Longitude, e.g. -88.5678")
    review_ratings: str = Field(description="Rating, e.g. 4.8")
    highlights: str = Field(description="Short description highlighting key features")


class POISuggestions(BaseModel):
    places: list[POI]


class AttractionEvent(BaseModel):
    event_type: str = Field(default="visit")
    description: str = Field(description="A title or description of the activity or the attraction visit")
    address: str = Field(default="", description="Full address of the attraction; empty for free time")
    start_time: str = Field(default="", description="Time in HH:MM format, e.g. 16:00")
    end_time: str = Field(default="", description="Time in HH:MM format, e.g. 18:00")
    booking_required: bool = Field(default=False)
    price: str = Field(default="", description="Some events cost money")
    booking_id: str = Field(default="")


class FlightEvent(BaseModel):
    event_type: str = Field(default="flight")
    description: str = Field(description="A title or description of the flight")
    booking_required: bool = Field(default=True)
    departure_airport: str = Field(description="Airport code, e.g. SEA")
    arrival_airport: str = Field(description="Airport code, e.g. SAN")
    flight_number: str = Field(description="Flight number, e.g. UA5678")
    boarding_time: str = Field(description="Time in HH:MM format, e.g. 15:30")
    seat_number: str = Field(default="", description="Seat row and position, e.g. 32A")
    departure_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    arrival_time: str = Field(description="Time in HH:MM format, e.g. 20:00")
    price: str = Field(default="", description="Total air fare")
    booking_id: str = Field(default="", description="Booking reference, e.g. LMN-012-STU")


class HotelEvent(BaseModel):
    event_type: str = Field(default="hotel")
    description: str = Field(description="The hotel's name")
    address: str = Field(description="Full address of the hotel")
    check_in_time: str = Field(description="Time in HH:MM format, e.g. 16:00")
    check_out_time: str = Field(description="Time in HH:MM format, e.g. 11:00")
    room_selection: str = Field(default="")
    booking_required: bool = Field(default=True)
    price: str = Field(default="", description="Total hotel price including all nights")
    booking_id: str = Field(default="", description="Booking reference, e.g. ABCD12345678")


class ItineraryDay(BaseModel):
    day_number: int = Field(description="Which day of the trip this is: 1, 2, 3...")
    date: str = Field(description="The date, YYYY-MM-DD")
    events: list[FlightEvent | HotelEvent | AttractionEvent] = Field(
        default_factory=list, description="The events of the day, in order"
    )


class Itinerary(BaseModel):
    trip_name: str = Field(description="One line describing the trip, e.g. 'San Diego to Seattle Getaway'")
    start_date: str = Field(description="Trip start date, YYYY-MM-DD")
    end_date: str = Field(description="Trip end date, YYYY-MM-DD")
    origin: str = Field(description="Trip origin, e.g. San Diego")
    destination: str = Field(description="Trip destination, e.g. Seattle")
    days: list[ItineraryDay] = Field(default_factory=list, description="The multi-day itinerary")


class PackingList(BaseModel):
    items: list[str] = Field(description="Things to pack, e.g. walking shoes, fleece, umbrella")


class Route(BaseModel):
    """Where the concierge sends a turn: which agent takes it."""

    agent: str = Field(description="One of the agent names offered")
    reason: str = Field(default="", description="One short clause on why")
