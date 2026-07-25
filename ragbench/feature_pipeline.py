"""
Feature pipeline: load vectara/open_ragbench corpus, generate embeddings,
and store in a Hopsworks feature group with vector similarity search.

Uses huggingface_hub to download individual corpus JSON files directly,
which avoids schema-detection failures caused by mixed text/image/table columns.
"""

import hashlib
import json
import logging
import os

import hopsworks
import pandas as pd
from hsfs.embedding import EmbeddingIndex, SimilarityFunctionType
from huggingface_hub import hf_hub_download, list_repo_files
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FG_NAME = "ragbench_embeddings"
FG_VERSION = 1
MAX_SECTION_LEN = 1500  # chars per chunk
MAX_SECTIONS_PER_PAPER = 3  # abstract + first 2 sections to cap total chunks
BATCH_SIZE = 256
DATASET_REPO = "vectara/open_ragbench"


def make_chunk_id(doc_id: str, idx: int) -> str:
    raw = f"{doc_id}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def iter_corpus_files():
    """Yield (filename, local_path) for every corpus/*.json in the HF repo."""
    log.info("Listing corpus files in %s …", DATASET_REPO)
    all_files = list(list_repo_files(DATASET_REPO, repo_type="dataset"))
    corpus_files = [f for f in all_files if f.startswith("pdf/arxiv/corpus/") and f.endswith(".json")]
    log.info("Found %d corpus files.", len(corpus_files))
    for hf_path in corpus_files:
        local = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=hf_path,
            repo_type="dataset",
        )
        yield hf_path, local


def extract_rows(doc_id: str, paper: dict) -> list[dict]:
    """Extract text-only chunks from a paper dict."""
    title = paper.get("title", "") or ""
    rows = []

    sections_added = 0
    for i, section in enumerate(paper.get("sections", [])):
        if sections_added >= MAX_SECTIONS_PER_PAPER:
            break
        # sections can contain text, table refs, or image refs — we want text only
        raw = section.get("text") if isinstance(section, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()[:MAX_SECTION_LEN]
        if len(text) < 50:
            continue
        rows.append(
            {
                "chunk_id": make_chunk_id(doc_id, i),
                "doc_id": doc_id[:64],
                "title": title[:256],
                "section_text": text,
            }
        )
        sections_added += 1

    # Fallback: use abstract if no sections extracted
    if not rows:
        abstract = paper.get("abstract", "") or ""
        if abstract.strip():
            rows.append(
                {
                    "chunk_id": make_chunk_id(doc_id, 0),
                    "doc_id": doc_id[:64],
                    "title": title[:256],
                    "section_text": abstract[:MAX_SECTION_LEN],
                }
            )
    return rows


def main():
    # ── 1. collect text chunks ────────────────────────────────────────────────
    all_rows: list[dict] = []
    for hf_path, local_path in iter_corpus_files():
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                paper = json.load(f)
            doc_id = str(
                paper.get("paperId") or paper.get("_id") or paper.get("id") or hf_path
            )
            all_rows.extend(extract_rows(doc_id, paper))
        except Exception as exc:
            log.warning("Skipping %s: %s", hf_path, exc)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["chunk_id"])
    log.info("Collected %d unique chunks.", len(df))

    if df.empty:
        raise RuntimeError("No text chunks extracted — check corpus structure.")

    # ── 2. generate embeddings ─────────────────────────────────────────────────
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    texts = df["section_text"].tolist()
    log.info("Embedding %d chunks (batch_size=%d) …", len(texts), BATCH_SIZE)
    vecs = embed_model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )
    df["embedding"] = vecs.tolist()

    # ── 3. create / get feature group ─────────────────────────────────────────
    project = hopsworks.login()
    fs = project.get_feature_store()

    emb_index = EmbeddingIndex()
    emb_index.add_embedding(
        name="embedding",
        dimension=EMBEDDING_DIM,
        similarity_function_type=SimilarityFunctionType.COSINE,
    )

    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="RAGBench corpus embeddings (all-MiniLM-L6-v2, 384d, cosine)",
        primary_key=["chunk_id"],
        online_enabled=True,
        embedding_index=emb_index,
    )

    # ── 4. insert ──────────────────────────────────────────────────────────────
    log.info("Inserting %d rows …", len(df))
    fg.insert(df, write_options={"wait_for_job": True})
    log.info("Done. Vector index ready.")


if __name__ == "__main__":
    main()
