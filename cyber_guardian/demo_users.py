"""The analysts this demo knows, and how the agent works out which one is on.

An agent with memory has to know whose memory it is: `remember` and `recall` are keyed by
the subject, so until the analyst has said who they are, everything the agent stores
belongs to the conversation and is lost with it. Hence the first question.

A demo has no login to read an identity from, so it ships a few people instead. They are
not credentials and nothing checks them; they exist so that a second shift can be worked
as the same analyst, which is the only way to see memory working.

Unlike the agents that talk their way to an answer, this one is a workflow: it classifies
the message and routes it. So the identity is matched here, in code, rather than left to a
model that might call `identify` or might not.
"""

from __future__ import annotations

USERS = [
    {
        "id": "r.okafor@example.com",
        "name": "Rachel Okafor",
        "note": "tier 2, owns the Windows estate, wants parent processes in every brief",
    },
    {
        "id": "s.lindqvist@example.com",
        "name": "Sven Lindqvist",
        "note": "tier 3, threat intel, asks for the IOC provenance before anything else",
    },
    {
        "id": "a.mehta@example.com",
        "name": "Anika Mehta",
        "note": "on call at night, escalates rather than contains without approval",
    },
]

USERS_BLOCK = "\n".join(f"- {u['name']} ({u['id']}) — {u['note']}" for u in USERS)

WHO_IS_ON_SHIFT = (
    "Before we start: who is on shift? This demo knows\n\n"
    + USERS_BLOCK
    + "\n\nSay a name and I will pick up where that analyst left off. Any other name is "
    "fine too; nothing here is a login."
)


def match(text: str) -> str | None:
    """The analyst the message names, or None.

    Matches a full name, a first name, a surname or the address itself, because an analyst
    answering "who is on shift" types the shortest thing that identifies them.
    """
    said = (text or "").strip().lower()
    if not said:
        return None
    for user in USERS:
        names = user["name"].lower().split()
        candidates = {user["id"].lower(), user["name"].lower(), *names}
        if said in candidates or any(f" {c} " in f" {said} " for c in candidates if len(c) > 2):
            return user["id"]
    return None


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
