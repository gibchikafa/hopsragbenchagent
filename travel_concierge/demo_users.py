"""The travellers this demo knows, and the rule that makes the concierge ask who is there.

An agent with memory has to know whose memory it is: the preferences the post-trip agent
learns go to `user` scope, keyed by the subject, so until someone has identified
themselves what the concierge learns belongs to the conversation and is lost with it.
Hence the first question.

A demo has no login to read an identity from, so it ships a few people instead. They are
not credentials and nothing checks them; they exist so that a second trip can be planned
as the same traveller, which is the only way to see memory working.
"""

from __future__ import annotations

import re

USERS = [
    {
        "id": "dana@example.com",
        "name": "Dana",
        "note": "aisle seat, travels light, will not fly overnight",
    },
    {
        "id": "marcus@example.com",
        "name": "Marcus",
        "note": "two children, needs a kitchen, books far ahead",
    },
    {
        "id": "priya@example.com",
        "name": "Priya",
        "note": "vegetarian, walks everywhere, picks cities over beaches",
    },
]

USERS_BLOCK = "\n".join(f"- {u['name']} ({u['id']}) — {u['note']}" for u in USERS)

IDENTITY_INSTRUCTIONS = """The application knows who you are speaking with before you run: \
it asks for their email address itself and binds memory to them, so by the time you read \
this the person is identified. Never ask who they are or for an email address, and do not \
call `identify`.

What is stored about them is given to you with the conversation; `recall` only what you \
need and was not given. `remember` what they tell you that would still be true on their \
next trip: a seat they always take, a food they do not eat, who travels with them. This trip's dates and destination are not that."""

#: An email address in what the user typed. Deliberately loose: a demo takes the
#: address at face value, because there is nothing here to check it against.
_ADDRESS = re.compile(r"[^\s<>@,;:]+@[^\s<>@,;:]+\.[A-Za-z]{2,}")

#: What the agent says while nobody has said who they are: the question, the people this
#: demo knows, and what it will do once it knows.
ASK_WHO = (
    "Before we start, what is your email address? I use it to remember how you like to travel between trips; this demo knows "
    + ", ".join(u["id"] for u in USERS[:-1]) + " and " + USERS[-1]["id"]
    + ", and any other address works too. Then I can plan a trip with you: where to go, how to get there, and where to stay."
)


def claimed_address(text: str) -> str:
    """The email address the user just gave, if they gave one."""
    found = _ADDRESS.search(text or "")
    return found.group(0).strip(".,;:").lower() if found else ""


def bind_if_identified(ctx, text: str) -> str:
    """Make long-term memory follow the person, once they have said who they are.

    `identify`, the tool the model calls, only labels the conversation: it never decides
    whose memory the conversation reads, because the caller there is a model acting on
    text a user typed. With it alone every `remember(scope="user")` stayed with the
    conversation, and the next visit as the same person knew nothing. Binding is the
    application's call. Here the address is taken as typed, which is acceptable only
    because these people are fictional; a real deployment asserts `subject` from its own
    login and never reaches this path.
    """
    if getattr(ctx, "has_subject", False) and getattr(ctx, "subject_source", "") == "client":
        return ctx.subject
    address = claimed_address(text)
    if address:
        ctx.rebind_subject(address)
        return address
    return ctx.subject if getattr(ctx, "has_subject", False) else ""


def who_first(ctx) -> str:
    """The reply to give instead of running the agent, while nobody is identified.

    Asked of the application rather than left to the prompt: told to ask first, a model
    mostly does, and sometimes answers the request for nobody instead -- or identifies
    the conversation with whatever string it has. Nothing done before it knows who this
    is can be remembered for them, so until an address has been given the answer is the
    question.
    """
    return "" if getattr(ctx, "has_subject", False) else ASK_WHO


def already_identified(ctx) -> str:
    """What to tell the model when it does not need to ask who this is.

    Either the application said so -- a real deployment has a login, and the client
    asserts `subject` on each request -- or this demo bound it from an address given
    earlier in the conversation. Both are the application's doing, so neither needs the
    model to ask or to call `identify`.
    """
    if not getattr(ctx, "has_subject", False):
        return ""
    return (
        f"\n\nThe application has already identified this person as {ctx.subject}. "
        "Do not ask who they are and do not call `identify`. Greet them, call `recall` "
        "for what you already know, and carry on."
    )
