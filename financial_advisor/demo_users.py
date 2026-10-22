"""The people this demo knows, and the rule that makes the agent ask who is there.

An agent with memory has to know whose memory it is: `remember` and `recall` are keyed
by the subject, so until someone has identified themselves everything the agent stores
belongs to the conversation and is lost with it. Hence the first question.

A demo has no login to read an identity from, so it ships a few people instead. They are
not credentials and nothing checks them; they exist so that a second conversation can be
had as the same person, which is the only way to see memory working.

They come with nothing attached. What the agent knows about someone is what that person
told it, and seeding a profile here taught it to state one it had invented.
"""

from __future__ import annotations

import re

USERS = [
    {"id": "dana@example.com", "name": "Dana"},
    {"id": "marcus@example.com", "name": "Marcus"},
    {"id": "priya@example.com", "name": "Priya"},
]

# Names and addresses only. Each of these used to carry a note -- "conservative,
# long horizon", "balanced, wants dividend income" -- and the model read them as
# facts it already knew: it stored one as Priya's profile the first time she
# spoke, and when asked where that came from answered that it was already in its
# memory. Nobody had told it. A profile the agent made up is worse than no
# profile, because it is indistinguishable from one the person gave.
USERS_BLOCK = "\n".join(f"- {u['name']} ({u['id']})" for u in USERS)

IDENTITY_INSTRUCTIONS = f"""Before anything else, ask for the email address you are speaking with, and \
wait for the answer. These are the people this demo knows:

{USERS_BLOCK}

Ask for an email address, not a name: it is what identifies someone across \
conversations, and two people share a first name sooner than an address. Call `identify` \
with the address exactly as they typed it. If they answer with a name instead, say you \
need the address and ask again. An address that is not on the list is fine; this is a \
demo, not a login.

Once you know who is there, call `recall` before asking anything you may already have \
been told, and `remember` whatever they tell you about themselves that would still be \
true next week. What they say about one ticker today is not that; their risk attitude \
and what they are investing for is.

You know nothing else about these people. The list above is addresses, not profiles: \
until someone tells you their risk attitude, their horizon or what they are investing \
for, you do not have it, and `recall` returning nothing means exactly that. Never state \
a profile you were not given, and never say you remember something that was not stored."""


#: An email address in what the user typed. Deliberately loose: a demo takes the
#: address at face value, because there is nothing here to check it against.
_ADDRESS = re.compile(r"[^\s<>@,;:]+@[^\s<>@,;:]+\.[A-Za-z]{2,}")


def claimed_address(text: str) -> str:
    """The email address the user just gave, if they gave one."""
    found = _ADDRESS.search(text or "")
    return found.group(0).strip(".,;:") if found else ""


def bind_if_identified(ctx, text: str) -> str:
    """Make long-term memory follow the person, once they have said who they are.

    This is the half a demo has to supply itself, and it is worth being plain about why.
    `identify` — the tool the model calls — only *labels* the conversation: it calls
    `record_subject`, never `rebind_subject`, because rebinding decides whose durable
    memory a conversation may read and the caller there is a model acting on text a user
    typed. So with `identify` alone every `remember(scope="user")` is owned by the
    conversation id, and the second conversation as the same person knows nothing — which
    is exactly what this demo promises and could not deliver.

    `rebind_subject` is the application's call to make, after the application has verified
    the person. Here there is no verification: the address is taken as typed, which is
    acceptable **only** because these are three fictional people and nothing behind them is
    private. A real deployment asserts `subject` on the request from its own login and
    never reaches this path.
    """
    if getattr(ctx, "has_subject", False) and getattr(ctx, "subject_source", "") == "client":
        return ctx.subject
    address = claimed_address(text)
    if address:
        ctx.rebind_subject(address)
        return address
    return ctx.subject if getattr(ctx, "has_subject", False) else ""


def already_identified(ctx) -> str:
    """What to tell the model when it does not need to ask who this is.

    Either the application said so — a real deployment has a login, the client asserts
    `subject` on each request, and the SDK keys memory by it without anyone being asked
    anything — or this demo bound it from an address given earlier in the conversation.
    Both are the application's doing, so neither needs the model to ask again. Told every
    turn, because a rule it read once ("do not call `identify` again") lost to the
    standing instruction to ask, and the agent re-identified people on their third turn.
    """
    if not getattr(ctx, "has_subject", False):
        return ""
    if getattr(ctx, "subject_source", "") not in ("client", "app"):
        return ""
    return (
        f"\n\nThis conversation is already identified as {ctx.subject}, and memory is "
        "bound to them. Do not ask who they are and do not call `identify`. Greet them, "
        "call `recall` for what you already know, and carry on."
    )
