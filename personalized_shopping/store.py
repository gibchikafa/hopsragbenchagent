"""
The shop, as the agent sees it: the catalogue in the feature store, and the
three things the recipe's shopper could do on the website.

The recipe drives a simulated website: `search[keywords]` shows a results page,
`click[B09...]` opens an item page, `click[Description]` and its siblings open
tabs, `click[Buy Now]` ends the episode. Here each of those is a keyed read or
a vector search, and the tabs come back together: the recipe told the model to
open all three and summarise them anyway, so `product_details` returns them in
one call rather than making the model click through.

Buying records an order in the ledger feature group. It never charges anything,
and under evaluation it records nothing (`in_evaluation()`), so a sandboxed
suite can run against the deployment that serves shoppers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

import hopsworks
import pandas as pd
from hopsworks_agents.protocol.autoevents import current_context
from hopsworks_agents.protocol.evaluation import in_evaluation
from langchain_core.tools import tool

HF_CACHE_DIR = (
    os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    or os.environ.get("HF_HOME")
    or os.path.join(tempfile.gettempdir(), "huggingface")
)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)

from sentence_transformers import SentenceTransformer  # noqa: E402

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FG_VERSION = 1
PRODUCTS_FG = "webshop_products"
EMBEDDINGS_FG = "webshop_product_embeddings"
ORDERS_FG = "webshop_orders"
DEFAULT_RESULTS = 10
MAX_RESULTS = 20
#: a description longer than this is shown cut, with a marker
MAX_DESCRIPTION_CHARS = 2500

# ── data access ──────────────────────────────────────────────────────────────

_embed = SentenceTransformer(EMBEDDING_MODEL, cache_folder=HF_CACHE_DIR)
_project = hopsworks.login()
_fs = _project.get_feature_store()
_embeddings_fg = None
_embedding_features: list[str] = []
_views: dict[str, object] = {}


def _catalogue():
    global _embeddings_fg, _embedding_features
    if _embeddings_fg is None:
        _embeddings_fg = _fs.get_feature_group(EMBEDDINGS_FG, version=FG_VERSION)
        if _embeddings_fg is None:
            raise LookupError(
                f"Feature group {EMBEDDINGS_FG!r} v{FG_VERSION} does not exist — "
                "run personalized_shopping/feature_pipeline.py first"
            )
        _embedding_features = [f.name for f in _embeddings_fg.features]
    return _embeddings_fg


def _view(name: str):
    """A serving-initialised feature view, cached for the process."""
    if name not in _views:
        view = _fs.get_feature_view(name=name, version=FG_VERSION)
        if view is None:
            raise LookupError(
                f"Feature view {name!r} v{FG_VERSION} does not exist — "
                "run personalized_shopping/feature_pipeline.py first"
            )
        view.init_serving()
        _views[name] = view
    return _views[name]


def _lookup(view_name: str, entry: dict) -> dict | None:
    """One keyed online read, or None when the key is absent."""
    try:
        row = _view(view_name).get_feature_vector(entry, return_type="pandas")
    except Exception:  # noqa: BLE001 — a missing key raises on some versions
        log.exception("Lookup in %s failed for %s", view_name, entry)
        return None
    if row is None or getattr(row, "empty", False):
        return None
    record = row.iloc[0].to_dict()
    # an absent key comes back as one row of nulls, not as no row
    if all(pd.isna(value) for value in record.values()):
        return None
    return record


def _as_list(value) -> list:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _as_dict(value) -> dict:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _customer() -> str:
    """Who is shopping: the conversation's subject, or anonymous."""
    ctx = current_context.get(None)
    subject = getattr(ctx, "subject", None) if ctx else None
    return (subject or "").strip() or "anonymous"


# ── the tools ────────────────────────────────────────────────────────────────


@tool
def search_products(keywords: str, max_results: int = DEFAULT_RESULTS) -> str:
    """Search the shop's catalogue for products matching the shopper's words.

    Returns up to `max_results` products, one per line: ASIN, title, price. Use the
    ASIN with `product_details` to read everything the shop knows about one.

    Args:
        keywords: What the shopper is looking for, e.g. "flowy floral summer dress".
        max_results: How many results to return, at most 20.
    """
    keywords = (keywords or "").strip()
    if not keywords:
        return "Nothing to search for: give some keywords."
    try:
        catalogue = _catalogue()
        vector = _embed.encode(keywords, normalize_embeddings=True).tolist()
        hits = catalogue.find_neighbors(
            vector, col="embedding", k=max(1, min(int(max_results), MAX_RESULTS))
        )
    except Exception:  # noqa: BLE001
        log.exception("Search failed for %r", keywords)
        return "The catalogue search is unavailable right now; try again in a moment."
    if not hits:
        return f"No products found for {keywords!r}. Try other words."
    lines = [f"Results for {keywords!r} ({len(hits)}):"]
    for _, values in hits:
        row = dict(zip(_embedding_features, values, strict=False))
        lines.append(f"- {row.get('asin')} | {row.get('title')} | {row.get('price_text')}")
    return "\n".join(lines)


@tool
def product_details(asin: str) -> str:
    """Everything the shop knows about one product: description, features, ratings, options.

    This is the item page and its Description, Features and Reviews tabs in one read.
    The options (colour, size, ...) listed here are the only ones `buy_now` accepts.

    Args:
        asin: The product's ASIN, as returned by `search_products`.
    """
    asin = (asin or "").strip().upper()
    product = _lookup(PRODUCTS_FG, {"asin": asin}) if asin else None
    if not product:
        return f"No product with ASIN {asin!r} in the catalogue. Search first and use an ASIN from the results."
    bullets = _as_list(product.get("bullet_points"))
    options = _as_dict(product.get("options"))
    attributes = _as_dict(product.get("attributes"))
    description = str(product.get("description") or "").strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS] + " […]"
    rating = product.get("rating")
    reviews = int(product.get("review_count") or 0)
    rating_text = (
        f"{rating:.1f} / 5 from {reviews} review(s)"
        if isinstance(rating, (int, float)) and not pd.isna(rating) and reviews
        else "no reviews yet"
    )
    parts = [
        f"{product.get('title')}",
        f"ASIN: {asin}",
        f"Brand: {product.get('brand') or 'unknown'}",
        f"Price: {product.get('price_text')}",
        f"Rating: {rating_text}",
        f"Category: {product.get('product_category') or product.get('category') or 'unknown'}",
        "",
        "Description:",
        description or "(none)",
        "",
        "Features:",
        *([f"- {b}" for b in bullets] or ["(none listed)"]),
        "",
        "Options (choose from these when buying):",
        *([f"- {name}: {', '.join(values)}" for name, values in options.items()] or ["(no options: one version only)"]),
    ]
    if attributes:
        parts += ["", "Details:", *[f"- {k.strip()}: {str(v).strip()}" for k, v in attributes.items()]]
    return "\n".join(parts)


@tool
def buy_now(asin: str, options: str = "", shopper_confirmed: bool = False) -> str:
    """Place the order for one product, in the chosen options.

    Set `shopper_confirmed` only when the shopper has *told you to buy it* — "buy it",
    "order it", "yes" to your direct question about placing the order. Liking it,
    wanting it, or asking about it is NOT that. Leave the flag false when unsure: the
    order is then not placed and you are told to ask.

    This RECORDS the order. It does not take payment: there is no card, no charge and
    no delivery in this chat. Say so when you confirm, so nobody believes they have paid.

    Args:
        asin: The product's ASIN.
        options: The chosen options as "name: value" pairs separated by commas, e.g.
            "color: black, size: medium". Must be options the product offers.
        shopper_confirmed: True only when the shopper explicitly asked to buy.
    """
    asin = (asin or "").strip().upper()
    product = _lookup(PRODUCTS_FG, {"asin": asin}) if asin else None
    if not product:
        return f"No product with ASIN {asin!r}; nothing was ordered."
    offered = _as_dict(product.get("options"))
    chosen: dict[str, str] = {}
    for pair in (options or "").split(","):
        if ":" in pair:
            name, value = pair.split(":", 1)
            chosen[name.strip().lower()] = value.strip()
    problems = []
    for name, values in offered.items():
        picked = chosen.get(name.lower())
        if not picked:
            problems.append(f"{name} (one of: {', '.join(values)})")
        elif picked.lower() not in {v.lower() for v in values}:
            problems.append(f"{name}: {picked!r} is not offered (one of: {', '.join(values)})")
    if problems:
        return (
            "NOT ORDERED. The product needs these options chosen first: "
            + "; ".join(problems)
            + ". Ask the shopper, then call buy_now again with them."
        )
    if not shopper_confirmed:
        return (
            "NOT ORDERED. The shopper has not asked to buy this yet. Ask them whether "
            f"to place the order for {product.get('title')!r}"
            + (f" in {options}" if options else "")
            + ", and call buy_now again with shopper_confirmed=True if they say yes."
        )
    order = {
        "order_id": uuid.uuid4().hex,
        "asin": asin,
        "title": str(product.get("title") or ""),
        "options": json.dumps(chosen),
        "price_text": str(product.get("price_text") or ""),
        "customer": _customer(),
        "conversation_id": getattr(current_context.get(None), "conversation_id", "") or "",
        "ordered_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    if in_evaluation():
        log.info("eval mode: not recording order for %s", asin)
    else:
        try:
            _fs.get_feature_group(ORDERS_FG, version=FG_VERSION).insert(
                pd.DataFrame([order]),
                storage="online",
                write_options={"wait_for_job": False},
            )
        except Exception:  # noqa: BLE001
            log.exception("Could not record the order for %s", asin)
            return "The order could not be recorded because of a problem on our side. Nothing was charged. Ask the shopper whether to try again."
    return (
        f"ORDER RECORDED: {product.get('title')} ({asin})"
        + (f" in {', '.join(f'{k} {v}' for k, v in chosen.items())}" if chosen else "")
        + f", {product.get('price_text')}. Order id {order['order_id'][:8]}. "
        "Nothing has been charged and no payment was taken; tell the shopper so."
    )


TOOLS = [search_products, product_details, buy_now]
