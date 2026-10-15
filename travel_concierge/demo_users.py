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

IDENTITY_INSTRUCTIONS = f"""Before you plan anything, ask who you are travelling with, and \
wait for the answer. These are the people this demo knows:

{USERS_BLOCK}

Take a first name, a full name or an address, and call `identify` with the address above \
that matches. If the name is not one of these, call `identify` with what they gave you \
anyway, and carry on; this is a demo, not a login.

Then call `recall` before asking what you may already know, and `remember` what they tell \
you that would still be true on their next trip: a seat they always take, a food they do \
not eat, who travels with them. This trip's dates and destination are not that."""


def already_identified(ctx) -> str:
    """What to tell the model when the application already said who this is.

    A real deployment has a login: the client asserts `subject` on each request — the user
    id or address its own session knows — and the SDK keys memory by it without anyone
    being asked anything. Asking is what a demo does instead, so the question is skipped
    whenever the identity arrived with the request.
    """
    if not getattr(ctx, "has_subject", False) or getattr(ctx, "subject_source", "") != "client":
        return ""
    return (
        f"\n\nThe application has already identified this person as {ctx.subject}. "
        "Do not ask who they are. Greet them, call `recall` for what you already know, "
        "and carry on."
    )
