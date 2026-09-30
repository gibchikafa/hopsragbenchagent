"""
Feature pipeline for the personalized-shopping agent: the WebShop catalogue
into the feature store.

Google's ADK recipe runs a copy of Princeton's WebShop, a simulated shop the
agent browses by clicking through search results, item pages and their
Description / Features / Reviews tabs. Here there is no website to navigate:
the catalogue lives in the feature store, and the agent reads it with keyed
lookups and vector search. That is what the recipe's search engine and item
pages were, minus the pages.

Three feature groups:

| Feature group                | Key        | Answers                                      |
|------------------------------|------------|----------------------------------------------|
| `webshop_product_embeddings` | *(vector)* | "which products match what the shopper said?" |
| `webshop_products`           | `asin`     | "everything the item page showed about this one" |
| `webshop_orders`             | `order_id` | "what was ordered, by whom, in what options" |

The catalogue is WebShop's 1,000-product subset (the recipe's
`items_shuffle_1000.json`). The recipe fetches it from Google Drive, where the
file no longer resolves; this reads the same file from a Hugging Face mirror.
Set WEBSHOP_ITEMS to a local copy to skip the download.

Run once before starting the agent:

    python feature_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

import hopsworks
import pandas as pd
from hopsworks_agents.protocol.embeddings import (
    load_sentence_transformer,
    register_sentence_transformer,
)
from hsfs.embedding import EmbeddingIndex, SimilarityFunctionType
from hsfs.feature import Feature


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FG_VERSION = 1
PRODUCTS_FG = "webshop_products"
EMBEDDINGS_FG = "webshop_product_embeddings"
ORDERS_FG = "webshop_orders"
BATCH_SIZE = 128
#: what gets embedded per product, as the recipe's indexer built its documents
MAX_EMBED_CHARS = 2000

ITEMS_URL = "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/items_shuffle_1000.json"
ITEMS_PATH = os.environ.get("WEBSHOP_ITEMS", "items_shuffle_1000.json")


def ensure_items(path: str = ITEMS_PATH) -> str:
    if os.path.exists(path):
        log.info("Using existing catalogue at %s", path)
        return path
    log.info("Downloading the WebShop catalogue → %s", path)
    urllib.request.urlretrieve(ITEMS_URL, path)
    return path


_AMOUNT = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def _amounts(text: str) -> list[float]:
    return [float(a.replace(",", "")) for a in _AMOUNT.findall(text or "")]


def _price(text: str) -> float | None:
    """The lowest dollar amount in the listing's price text, or None when it has none."""
    amounts = _amounts(text)
    return min(amounts) if amounts else None


def _price_text(text: str) -> str:
    """The listing's price as a person would write it: WebShop runs a range together ("$19.99$27.99")."""
    amounts = _amounts(text)
    if not amounts:
        return "price not listed"
    if len(amounts) == 1:
        return f"${amounts[0]:,.2f}"
    return f"${min(amounts):,.2f} to ${max(amounts):,.2f}"


def _options(raw) -> dict[str, list[str]]:
    """WebShop's `customization_options`: name → a list of choices, each a dict with a `value`.

    An empty string when the product has none. Choices marked unavailable are left out,
    so the agent never offers one the shop cannot sell.
    """
    if not isinstance(raw, dict):
        return {}
    options: dict[str, list[str]] = {}
    for name, choices in raw.items():
        if not isinstance(choices, list):
            continue
        values = []
        for choice in choices:
            if isinstance(choice, dict):
                if choice.get("is_available") is False:
                    continue
                value = str(choice.get("value") or "").strip()
            else:
                value = str(choice).strip()
            if value and value not in values:
                values.append(value)
        if values:
            options[str(name).strip()] = values
    return options


def _text_feature(name: str) -> Feature:
    """A long text column, declared TEXT so it does not count against the online row limit.

    hsfs sizes string columns from the widest value it sees; a description of a
    few thousand characters becomes a varchar that pushes the row past 30000
    bytes and the online feature group is refused. TEXT is stored out of row.
    """
    return Feature(name, type="string", online_type="text")


def read_catalogue(path: str) -> pd.DataFrame:
    """One row per product, in the columns the recipe's item page showed."""
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    rows = []
    for item in items:
        bullets = item.get("small_description") or []
        if not isinstance(bullets, list):
            bullets = [str(bullets)]
        bullets = [str(b).strip() for b in bullets if str(b).strip()]
        rating = item.get("average_rating")
        reviews = item.get("total_reviews")
        images = item.get("images") or []
        rows.append(
            {
                "asin": item["asin"],
                "title": (item.get("name") or "").strip(),
                "brand": re.sub(r"^Brand:\s*", "", item.get("brand") or "").strip(),
                "price": _price(item.get("pricing") or ""),
                "price_text": _price_text(item.get("pricing") or ""),
                "rating": float(rating) if isinstance(rating, (int, float)) else None,
                "review_count": int(reviews) if isinstance(reviews, (int, float)) else 0,
                "category": item.get("category") or "",
                "product_category": item.get("product_category") or "",
                "description": (item.get("full_description") or "").strip(),
                "bullet_points": json.dumps(bullets),
                "options": json.dumps(_options(item.get("customization_options"))),
                "attributes": json.dumps(item.get("product_information") or {}),
                "image_url": images[0] if images else "",
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["asin"]).reset_index(drop=True)
    log.info("  %d products", len(frame))
    return frame


def embedding_text(row: pd.Series) -> str:
    """Title, first bullet, description: the recipe's search document, lowercased."""
    bullets = json.loads(row["bullet_points"] or "[]")
    parts = [row["title"], bullets[0] if bullets else "", row["description"]]
    return " ".join(p for p in parts if p).lower()[:MAX_EMBED_CHARS]


def main() -> None:
    frame = read_catalogue(ensure_items())
    now = pd.Timestamp.utcnow().tz_localize(None)
    frame["indexed_at"] = now

    project = hopsworks.login()
    fs = project.get_feature_store()

    # the embedding model goes to the model registry once, here where there is internet;
    # the agent loads it from there and never downloads it in the pod
    register_sentence_transformer(EMBEDDING_MODEL, project=project)
    log.info("Embedding %d products (%s) …", len(frame), EMBEDDING_MODEL)
    model = load_sentence_transformer(EMBEDDING_MODEL, project=project)
    vectors = model.encode(
        [embedding_text(row) for _, row in frame.iterrows()],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = frame[["asin", "title", "price_text", "category", "indexed_at"]].copy()
    embeddings["embedding"] = vectors.tolist()

    index = EmbeddingIndex()
    index.add_embedding(
        name="embedding",
        dimension=EMBEDDING_DIM,
        similarity_function_type=SimilarityFunctionType.COSINE,
    )
    embeddings_fg = fs.get_or_create_feature_group(
        name=EMBEDDINGS_FG,
        version=FG_VERSION,
        description=f"WebShop products for search ({EMBEDDING_MODEL}, {EMBEDDING_DIM}d, cosine)",
        primary_key=["asin"],
        event_time="indexed_at",
        online_enabled=True,
        embedding_index=index,
    )
    embeddings_fg.insert(embeddings, write_options={"wait_for_job": True})

    products_fg = fs.get_or_create_feature_group(
        name=PRODUCTS_FG,
        version=FG_VERSION,
        description="WebShop products: what the item page showed, one row per ASIN",
        primary_key=["asin"],
        event_time="indexed_at",
        online_enabled=True,
        features=[
            Feature("asin", type="string"),
            Feature("title", type="string"),
            Feature("brand", type="string"),
            Feature("price", type="double"),
            Feature("price_text", type="string"),
            Feature("rating", type="double"),
            Feature("review_count", type="bigint"),
            Feature("category", type="string"),
            Feature("product_category", type="string"),
            _text_feature("description"),
            _text_feature("bullet_points"),
            _text_feature("options"),
            _text_feature("attributes"),
            Feature("image_url", type="string"),
            Feature("indexed_at", type="timestamp"),
        ],
    )
    products_fg.insert(frame, write_options={"wait_for_job": True})

    # Written to by the agent, one row per order. A feature group has no schema
    # until something is written and the agent can only append online, so one
    # sentinel row goes in here, where an offline write is available.
    orders_fg = fs.get_or_create_feature_group(
        name=ORDERS_FG,
        version=FG_VERSION,
        description="Append-only order ledger; an order is recorded when a row exists, never charged",
        primary_key=["order_id"],
        event_time="ordered_at",
        online_enabled=True,
        features=[
            Feature("order_id", type="string"),
            Feature("asin", type="string"),
            Feature("title", type="string"),
            _text_feature("options"),
            Feature("price_text", type="string"),
            Feature("customer", type="string"),
            Feature("conversation_id", type="string"),
            Feature("ordered_at", type="timestamp"),
        ],
    )
    orders_fg.insert(
        pd.DataFrame(
            [
                {
                    "order_id": "__seed__",
                    "asin": "",
                    "title": "",
                    "options": "{}",
                    "price_text": "",
                    "customer": "__seed__",
                    "conversation_id": "",
                    "ordered_at": now,
                }
            ]
        ),
        write_options={"wait_for_job": True},
    )

    # Online point lookups go through a feature view, not the feature group.
    for name, fg in ((PRODUCTS_FG, products_fg), (ORDERS_FG, orders_fg)):
        fs.get_or_create_feature_view(name=name, version=FG_VERSION, query=fg.select_all())
        log.info("Feature view ready: %s", name)
    log.info("Done. The agent can now search and read the catalogue.")


if __name__ == "__main__":
    main()
