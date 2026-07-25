"""
Feature pipeline for the Chinook customer-support agent.

Builds one embedding feature group holding every artist, album and track name in
the Chinook database, so the agent can disambiguate what a customer typed
("prince") against what the database actually stores ("Prince") before it goes
anywhere near SQL.

This replaces the three in-process vector stores the original notebook built at
import time. That is fine in a notebook and wrong in a deployment: it re-embeds
the whole catalogue on every pod start, holds three copies in memory per
replica, and cannot be shared, inspected, or rebuilt independently of the agent.
Indexing the catalogue is a pipeline concern — run it when the catalogue
changes, not when a pod happens to boot.

Run once before starting the agent:

    python feature_pipeline.py
"""

import logging
import os
import sqlite3
import urllib.request

import hopsworks
import pandas as pd
from hsfs.embedding import EmbeddingIndex, SimilarityFunctionType
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FG_NAME = "chinook_catalog_embeddings"
FG_VERSION = 1
BATCH_SIZE = 256

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
CHINOOK_DB = os.environ.get("CHINOOK_DB_PATH", "chinook.db")

# One row per catalogue entry. `entity_kind` is what lets a single feature group
# serve all three lookups — see the note on filtering in the agent.
CATALOGUE_QUERIES = {
    "track": "SELECT DISTINCT Name FROM Track WHERE Name IS NOT NULL",
    "artist": "SELECT DISTINCT Name FROM Artist WHERE Name IS NOT NULL",
    "album": "SELECT DISTINCT Title FROM Album WHERE Title IS NOT NULL",
}


def ensure_chinook_db(path: str = CHINOOK_DB) -> str:
    """Download the Chinook sample database if it isn't already on disk."""
    if os.path.exists(path):
        log.info("Using existing Chinook DB at %s", path)
        return path
    log.info("Downloading Chinook DB → %s", path)
    urllib.request.urlretrieve(CHINOOK_URL, path)
    return path


def read_catalogue(db_path: str) -> pd.DataFrame:
    """Every distinct artist, album and track name, one row each."""
    rows = []
    conn = sqlite3.connect(db_path)
    try:
        for kind, query in CATALOGUE_QUERIES.items():
            names = [r[0] for r in conn.execute(query).fetchall() if r[0]]
            log.info("  %-7s %d names", kind, len(names))
            rows.extend({"entity_kind": kind, "name": name} for name in names)
    finally:
        conn.close()
    frame = pd.DataFrame(rows)
    # entity_kind + name is the natural key, but a feature group primary key
    # must be a single column, so carry a composite id instead
    frame["entry_id"] = frame["entity_kind"] + ":" + frame["name"]
    return frame.drop_duplicates(subset=["entry_id"]).reset_index(drop=True)


def main() -> None:
    db_path = ensure_chinook_db()
    frame = read_catalogue(db_path)
    log.info("Embedding %d catalogue entries …", len(frame))

    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(
        frame["name"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    frame["embedding"] = vectors.tolist()
    # the feature group needs an event time; the catalogue has no natural one
    frame["indexed_at"] = pd.Timestamp.utcnow().tz_localize(None)

    project = hopsworks.login()
    fs = project.get_feature_store()

    index = EmbeddingIndex()
    index.add_embedding(
        name="embedding",
        dimension=EMBEDDING_DIM,
        similarity_function_type=SimilarityFunctionType.COSINE,
    )
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description=f"Chinook catalogue names for fuzzy lookup ({EMBEDDING_MODEL}, {EMBEDDING_DIM}d, cosine)",
        primary_key=["entry_id"],
        event_time="indexed_at",
        online_enabled=True,
        embedding_index=index,
    )

    log.info("Inserting %d rows into %s v%d …", len(frame), FG_NAME, FG_VERSION)
    fg.insert(frame, write_options={"wait_for_job": True})
    log.info("Done. The agent can now resolve catalogue names.")


if __name__ == "__main__":
    main()
