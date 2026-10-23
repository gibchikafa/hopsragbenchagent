"""Who the memory demo is talking to, asked before anything else.

Long-term memory is filed under the person, so a demo of it has to know who that is: until
someone has said, everything kept belongs to the conversation and is gone with it. A demo
has no login to read an identity from, so it asks for an email address and takes it as
typed. These people are fictional and nothing checks them; they exist so a second
conversation can be had as the same person, which is the only way to see memory follow them.
"""

from __future__ import annotations

import re

USERS = [
    {"id": "dana@example.com", "name": "Dana"},
    {"id": "marcus@example.com", "name": "Marcus"},
    {"id": "priya@example.com", "name": "Priya"},
]

#: An email address in what the user typed. Deliberately loose: a demo takes the
#: address at face value, because there is nothing here to check it against.
_ADDRESS = re.compile(r"[^\s<>@,;:]+@[^\s<>@,;:]+\.[A-Za-z]{2,}")

#: What the agent says while nobody has said who they are.
ASK_WHO = (
    "Before we start, what is your email address? Long-term memory is filed under it, so it "
    "is what lets me remember you in your next conversation; this demo knows "
    + ", ".join(u["id"] for u in USERS[:-1]) + " and " + USERS[-1]["id"]
    + ", and any other address works too. Then tell me something about yourself, and ask "
    "me what I remember."
)


def claimed_address(text: str) -> str:
    """The email address the user just gave, if they gave one."""
    found = _ADDRESS.search(text or "")
    return found.group(0).strip(".,;:").lower() if found else ""


def bind_if_identified(ctx, text: str) -> str:
    """Make long-term memory follow the person, once they have said who they are.

    `identify`, the tool the model calls, only labels the conversation: it never decides
    whose memory the conversation reads, because the caller there is a model acting on
    text a user typed. Binding is the application's call. Here the address is taken as
    typed, which is acceptable only because these people are fictional; a real deployment
    asserts `subject` from its own login and never reaches this path.
    """
    if getattr(ctx, "has_subject", False) and getattr(ctx, "subject_source", "") == "client":
        return ctx.subject
    address = claimed_address(text)
    if address:
        ctx.rebind_subject(address)
        return address
    return ctx.subject if getattr(ctx, "has_subject", False) else ""


def who_first(ctx) -> str:
    """The reply to give instead of running the agent, while nobody is identified."""
    return "" if getattr(ctx, "has_subject", False) else ASK_WHO


def already_identified(ctx) -> str:
    """What to tell the model when it does not need to ask who this is."""
    if not getattr(ctx, "has_subject", False):
        return ""
    return (
        f"\n\nThis conversation is already identified as {ctx.subject}, and long-term "
        "memory is filed under them. Do not ask who they are and do not call `identify`."
    )
