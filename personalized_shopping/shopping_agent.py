"""
Personalized shopping — Google's ADK personalized-shopping recipe on LangGraph
and the Hopsworks Agent Protocol, without the website.

    shopper ──▶ agent ─┬─▶ search_products   (vector search over the catalogue)
                       ├─▶ product_details   (the item page and its tabs, in one read)
                       └─▶ buy_now           (an order in the ledger; never a charge)

The recipe's agent has two tools, `search` and `click`, and drives a simulated
web shop: click a result to open the item page, click Description, Features
and Reviews in turn, click "< Prev" to get back, click a colour and a size,
click Buy Now. The prompt is mostly about not clicking buttons that are not on
the current page.

Here the catalogue is in the feature store (see `feature_pipeline.py`) and
the tools are what the clicks were for: search, read one product, order it.
The prompt keeps the recipe's interaction flow — ask, search, present, explore
on request, confirm before buying — and drops the button handling, because
there are no buttons. Image search is kept: a shopper can send a photo and the
model describes what it sees before searching for it.

Deploy (git-backed, so `store.py` comes along):
    agents = hopsworks.login().get_agent_serving()
    agents.deploy_agent("personalized_shopping/shopping_agent.py", name="webshop",
                        git_url="https://github.com/<you>/hopsragbenchagent.git",
                        git_provider="GitHub", git_branch="main",
                        # Priced, or every traced turn costs 0.0 -- which reads as free
                        # rather than as unpriced. List prices per million tokens for
                        # claude-sonnet-4-5 when this was written; check your provider's, and
                        # change them when you change the model.
                        tracing={"enabled": True,
                                 "input_token_price_per_million_tokens": 3.0,
                                 "output_token_price_per_million_tokens": 15.0}).start()
"""

from __future__ import annotations

import os

from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    identity_tools,
    memory_tools,
    anthropic_summarizer,
)
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

import demo_users
from prompts import SYSTEM_PROMPT
from store import TOOLS

MODEL = os.environ.get("WEBSHOP_MODEL", "claude-sonnet-4-5")


# the shop's tools, the memory tools the model calls itself, and `identify`, which
# says whose memory this is: until it is called, what the agent remembers belongs to
# the conversation alone
SHOPPER_TOOLS = [*TOOLS, *memory_tools("langgraph"), *identity_tools("langgraph")]

agent = create_react_agent(
    ChatAnthropic(model=MODEL, max_tokens=4096, temperature=0.0),
    SHOPPER_TOOLS,
    prompt=SYSTEM_PROMPT,
)

agent_app = AgentApp(
    # buy_now checks in_evaluation() before writing (store.py), so a sandboxed
    # suite can run against the deployment that serves shoppers
    eval_per_request=True,
    name="Personalized shopping",
    description="Finds products in the shop's catalogue, explores them with you and places "
    "the order when you say so (LangGraph agent over Hopsworks feature-store search).",
    framework="langgraph",
    welcome_message="Hi! I can help you find something in our shop, tell you all about "
    "it, and order it for you when you're ready. What is your email address?",
    suggested_prompts=[
        "dana@example.com",
        "Can you help me find a summer dress? Something flowy and floral.",
        "I'm looking for a rustic console table for my entryway.",
    ],
    placeholder="What are you looking for? You can also send a photo...",
    input_modalities=["text", "image"],
    memory=ManagedMemoryService(summarize=anthropic_summarizer(), long_term=True),
    tool_events=True,
    graph=agent,
)


def _user_content(request):
    """The turn's message for the model: its text, and any image the shopper sent."""
    blocks = []
    if request.text:
        blocks.append({"type": "text", "text": request.text})
    for image in request.images:
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.data,
                },
            }
        )
    if not blocks:
        raise AgentError(
            "Send some text, or an image of what you are looking for.",
            code="invalid_request",
            status_code=400,
        )
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return request.text
    if not request.text:
        blocks.insert(0, {"type": "text", "text": "Find me this product."})
    return blocks


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints; ctx.stream_langchain yields token
    deltas and turns each tool call into a progress chip."""
    # Who this is, before anything runs: bound from an address in the message, or asked
    # for. See demo_users.who_first for why this is the application's job.
    demo_users.bind_if_identified(ctx, request.text)
    if question := demo_users.who_first(ctx):
        yield question
        return
    content = _user_content(request)
    # the identity the client asserted, when there is one: a real deployment has a
    # login and does not ask
    context = ctx.system_context() + demo_users.already_identified(ctx)
    messages = [{"role": "system", "content": context}] if context else []
    messages += ctx.history
    messages.append({"role": "user", "content": content})
    async for delta in ctx.stream_langchain(
        agent.astream_events({"messages": messages}, version="v2")
    ):
        yield delta


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
