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
