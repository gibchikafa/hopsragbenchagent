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
                        requirements="personalized_shopping/requirements.txt").start()
"""

from __future__ import annotations

import os

from hopsworks_agents.protocol import (
    AgentApp,
    AgentError,
    ManagedMemoryService,
    anthropic_summarizer,
)
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from prompts import SYSTEM_PROMPT
from store import TOOLS

MODEL = os.environ.get("WEBSHOP_MODEL", "claude-sonnet-4-5")


agent = create_react_agent(
    ChatAnthropic(model=MODEL, max_tokens=4096, temperature=0.0),
    TOOLS,
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
    "it, and order it for you when you're ready. What are you looking for?",
    suggested_prompts=[
        "Hello, who are you?",
        "Can you help me find a summer dress? Something flowy and floral.",
        "I'm looking for a rustic console table for my entryway.",
    ],
    placeholder="What are you looking for? You can also send a photo...",
    input_modalities=["text", "image"],
    memory=ManagedMemoryService(summarize=anthropic_summarizer()),
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
    content = _user_content(request)
    messages = [{"role": "system", "content": ctx.system_context()}] if ctx.system_context() else []
    messages += ctx.history
    messages.append({"role": "user", "content": content})
    async for delta in ctx.stream_langchain(
        agent.astream_events({"messages": messages}, version="v2")
    ):
        yield delta


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
