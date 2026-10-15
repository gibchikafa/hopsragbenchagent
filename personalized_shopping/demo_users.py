"""The shoppers this demo knows, and the rule that makes the agent ask who is there.

An agent with memory has to know whose memory it is: `remember` and `recall` are keyed
by the subject, so until someone has identified themselves everything the agent stores
belongs to the conversation and is lost with it. Hence the first question.

A demo has no login to read an identity from, so it ships a few people instead. They are
not credentials and nothing checks them; they exist so that a second conversation can be
had as the same person, which is the only way to see memory working.
"""

from __future__ import annotations

USERS = [
    {
        "id": "dana@example.com",
        "name": "Dana",
        "note": "buys for a small flat, likes muted colours, avoids plastic",
    },
    {
        "id": "marcus@example.com",
        "name": "Marcus",
        "note": "runs most mornings, size 11, replaces kit rather than browses",
    },
    {
        "id": "priya@example.com",
        "name": "Priya",
        "note": "shops for a family of four, compares prices before buying",
    },
]

USERS_BLOCK = "\n".join(f"- {u['name']} ({u['id']}) — {u['note']}" for u in USERS)

IDENTITY_INSTRUCTIONS = f"""Before you search for anything, ask for the shopper's email address, \
and wait for the answer. These are the people this demo knows:

{USERS_BLOCK}

Ask for an email address, not a name: it is what identifies someone across visits, and \
two shoppers share a first name sooner than an address. Call `identify` with the address \
exactly as they typed it. If they answer with a name instead, say you need the address \
and ask again. An address that is not on the list is fine; this is a demo, not a login.

Then call `recall` before asking what you may already have been told, and `remember` what \
they tell you that would still be true on their next visit: a size, an allergy, a brand \
they will not buy. What they are looking for today is not that."""


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
