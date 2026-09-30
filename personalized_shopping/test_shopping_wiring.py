"""That the shopping agent loads, wires its three tools, and never buys unasked.

The dependencies are stubbed, so this checks the wiring and the order rules
rather than the store: no cluster, no model, no network.

    python -m pytest personalized_shopping/test_shopping_wiring.py
"""

from __future__ import annotations

import json
import sys
import types
from unittest import mock

import pytest

TOOLS = ("search_products", "product_details", "buy_now")

STUBBED = (
    "hopsworks_agents", "hopsworks_agents.protocol", "hopsworks_agents.protocol.autoevents",
    "hopsworks_agents.protocol.evaluation", "hopsworks_agents.protocol.embeddings", "hopsworks", "pandas",
    "sentence_transformers",
    "langchain_anthropic", "langchain_core", "langchain_core.tools", "langgraph", "langgraph.prebuilt",
)


@pytest.fixture
def stubbed(monkeypatch):
    for name in STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["langchain_core.tools"].tool = lambda fn: fn
    sys.modules["langgraph.prebuilt"].create_react_agent = lambda llm, tools, prompt=None: {"tools": tools}
    sys.modules["hopsworks"].login = mock.MagicMock()
    sys.modules["hopsworks_agents.protocol.evaluation"].in_evaluation = lambda: False
    pandas = sys.modules["pandas"]
    pandas.isna = lambda v: v is None
    pandas.DataFrame = lambda rows: rows
    monkeypatch.syspath_prepend(__file__.rsplit("/", 1)[0])
    for name in ("store", "shopping_agent"):
        sys.modules.pop(name, None)
    yield


def test_the_agent_imports_with_its_three_tools(stubbed):
    import shopping_agent  # noqa: PLC0415

    assert shopping_agent.agent_app is not None
    assert [t.__name__ for t in shopping_agent.agent["tools"]] == list(TOOLS)


def test_every_tool_has_a_description_for_the_model(stubbed):
    import store  # noqa: PLC0415

    for name in TOOLS:
        assert getattr(store, name).__doc__, f"{name} has no description"


def _product(**overrides):
    row = {
        "asin": "B0TEST", "title": "Test dress", "brand": "Acme", "price": 20.0,
        "price_text": "$20.00", "rating": 4.5, "review_count": 12, "category": "fashion",
        "product_category": "Women › Dresses", "description": "A dress.",
        "bullet_points": json.dumps(["Flowy", "Floral"]),
        "options": json.dumps({"color": ["black", "blue"], "size": ["small", "medium"]}),
        "attributes": json.dumps({"Material": "cotton"}), "image_url": "",
    }
    row.update(overrides)
    return row


def test_buying_needs_the_options_and_the_shoppers_word(stubbed):
    import store  # noqa: PLC0415

    store._lookup = lambda view, entry: _product()
    store.current_context = types.SimpleNamespace(get=lambda default=None: None)
    inserted = []
    store._fs.get_feature_group.return_value.insert = lambda df, **kw: inserted.append(df)

    # options first
    reply = store.buy_now("b0test", "", True)
    assert reply.startswith("NOT ORDERED") and "color" in reply and "size" in reply and not inserted
    # then the shopper's word
    reply = store.buy_now("B0TEST", "color: black, size: medium", False)
    assert reply.startswith("NOT ORDERED") and "asked to buy" in reply and not inserted
    # an option the product does not offer
    reply = store.buy_now("B0TEST", "color: red, size: medium", True)
    assert reply.startswith("NOT ORDERED") and "not offered" in reply and not inserted
    # both given: one row, and the reply says nothing was charged
    reply = store.buy_now("B0TEST", "Color: Black, size: medium", True)
    assert reply.startswith("ORDER RECORDED") and "Nothing has been charged" in reply
    [rows] = inserted
    assert rows[0]["asin"] == "B0TEST" and json.loads(rows[0]["options"]) == {"color": "Black", "size": "medium"}
    assert rows[0]["customer"] == "anonymous"


def test_under_evaluation_an_order_is_confirmed_but_not_written(stubbed):
    import store  # noqa: PLC0415

    store._lookup = lambda view, entry: _product(options="{}")
    store.current_context = types.SimpleNamespace(get=lambda default=None: None)
    store.in_evaluation = lambda: True
    inserted = []
    store._fs.get_feature_group.return_value.insert = lambda df, **kw: inserted.append(df)
    assert store.buy_now("B0TEST", "", True).startswith("ORDER RECORDED")
    assert not inserted


def test_details_read_the_item_page_in_one_go(stubbed):
    import store  # noqa: PLC0415

    store._lookup = lambda view, entry: _product()
    text = store.product_details("b0test")
    for expected in ("Test dress", "$20.00", "4.5 / 5 from 12", "A dress.", "- Flowy", "color: black, blue", "Material: cotton"):
        assert expected in text
    store._lookup = lambda view, entry: None
    assert "No product with ASIN" in store.product_details("nope")


def test_an_image_turn_becomes_image_blocks(stubbed):
    import shopping_agent  # noqa: PLC0415

    image = types.SimpleNamespace(media_type="image/webp", data="QUJD")
    request = types.SimpleNamespace(text="", images=[image])
    content = shopping_agent._user_content(request)
    assert content[0] == {"type": "text", "text": "Find me this product."}
    assert content[1]["source"] == {"type": "base64", "media_type": "image/webp", "data": "QUJD"}
    assert shopping_agent._user_content(types.SimpleNamespace(text="hi", images=[])) == "hi"


# ── the OpenAI Agents SDK entry point ────────────────────────────────────────

OPENAI_STUBBED = (
    "agents", "agents.stream_events", "agents.extensions", "agents.extensions.models",
    "agents.extensions.models.litellm_model", "openai", "openai.types", "openai.types.responses",
)


class _FakeAgent:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.tools = kw.get("tools", [])
        self.handoffs = kw.get("handoffs", [])

    def as_tool(self, tool_name, tool_description, **kw):
        return types.SimpleNamespace(name=tool_name, description=tool_description, agent=self)


@pytest.fixture
def openai_stubbed(stubbed, monkeypatch):
    for name in OPENAI_STUBBED:
        module = types.ModuleType(name)
        module.__getattr__ = lambda _n: mock.MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    agents = sys.modules["agents"]
    agents.Agent = _FakeAgent
    agents.function_tool = lambda fn: types.SimpleNamespace(name=fn.__name__, description=fn.__doc__, fn=fn)
    agents.ModelSettings = lambda **kw: kw
    agents.RunConfig = lambda **kw: kw
    agents.WebSearchTool = lambda: "web_search"
    yield


def test_the_openai_variant_wraps_the_same_store_tools(openai_stubbed):
    sys.modules.pop("shopping_agent_openai", None)
    import shopping_agent_openai as o  # noqa: PLC0415

    assert o.agent_app is not None
    assert [t.name for t in o.agent.tools] == list(TOOLS)
    assert o.agent.instructions.startswith("You are a webshop agent")
