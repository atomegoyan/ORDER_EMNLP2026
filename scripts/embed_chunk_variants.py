"""
Phase 2 — Embed chunk variants into ChromaDB collections.

For each chunking strategy directory (data/corpus_1887_{strategy}/), reads all
chunk files, embeds them using Cohere embed-v4.0, and stores them in a new
ChromaDB collection inside data/embeddings_1887_cohere/.

Usage:
    python scripts/embed_chunk_variants.py              # embed all strategies
    python scripts/embed_chunk_variants.py S2_atomic    # embed one strategy
"""

import os
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.config import COHERE_API_KEY, EMBEDDINGS_1887_COHERE

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = EMBEDDINGS_1887_COHERE

STRATEGY_IDS = [
    "baseline_10k",
    "S2_atomic",
    "S3_window3",
    "S4_president",
    "S8_vote",
    "S9_topic",
    "S10_amendment",
    "S11_hybrid",
    "S9_topic_noVotes",
    "S11_hybrid_noVotes",
    "fixed_500",
    "fixed_1000",
    "fixed_2000",
]

BATCH_SIZE = 96  # Cohere embed-v4.0 supports up to 96 texts per call
MAX_CHUNK_CHARS = 30000  # Truncate very long chunks to avoid API timeouts


# ── Main ──────────────────────────────────────────────────────────────────

def embed_strategy(strategy_id: str):
    """Embed all chunks for one strategy into a ChromaDB collection."""

    chunk_dir = DATA_DIR / f"corpus_1887_{strategy_id}"
    if not chunk_dir.exists():
        print(f"  SKIP: {chunk_dir} does not exist")
        return

    collection_name = f"debattre_1887_{strategy_id}_cohere"

    # Collect all chunk files
    chunk_files = sorted(chunk_dir.rglob("*.txt"))
    print(f"\n  {strategy_id}: {len(chunk_files)} chunk files -> collection '{collection_name}'")

    # Init ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    cohere_ef = embedding_functions.CohereEmbeddingFunction(
        api_key=COHERE_API_KEY, model_name="embed-v4.0"
    )

    # Skip if collection already exists and is non-empty
    try:
        existing = client.get_collection(collection_name)
        if existing.count() > 0:
            print(f"    SKIP: collection '{collection_name}' already has {existing.count()} documents")
            return
        # Collection exists but is empty — delete and recreate
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=cohere_ef,
        metadata={
            "description": f"Les Débats 1887 — chunking strategy {strategy_id}",
            "model": "Cohere embed-v4.0",
            "strategy": strategy_id,
        },
    )

    # Read all chunks
    ids_all, texts_all, metas_all = [], [], []
    for fp in chunk_files:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Truncate very long texts to avoid Cohere API timeouts
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS]
        chunk_id = fp.name  # e.g. "1887-02-04_008_002.txt"
        seance_date = chunk_id.split("_")[0]

        ids_all.append(chunk_id)
        texts_all.append(text)
        metas_all.append({
            "seance_date": seance_date,
            "section_file": fp.parent.name + "/" + fp.name,
            "chunk_length": str(len(text)),
        })

    print(f"    {len(ids_all)} non-empty chunks to embed")

    # Embed in batches (respects Cohere rate limits)
    for i in tqdm(range(0, len(ids_all), BATCH_SIZE), desc=f"    Embedding {strategy_id}"):
        batch_ids = ids_all[i : i + BATCH_SIZE]
        batch_texts = texts_all[i : i + BATCH_SIZE]
        batch_metas = metas_all[i : i + BATCH_SIZE]

        retries = 0
        while retries < 5:
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metas,
                )
                break
            except Exception as e:
                retries += 1
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str or "timeout" in err_str:
                    wait = 15 * retries
                    print(f"\n    Retryable error, waiting {wait}s (attempt {retries}/5): {type(e).__name__}")
                    time.sleep(wait)
                else:
                    print(f"\n    Error: {e}")
                    raise

    final_count = collection.count()
    print(f"    OK Collection '{collection_name}' has {final_count} documents")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else STRATEGY_IDS

    for sid in targets:
        if sid not in STRATEGY_IDS:
            print(f"Unknown strategy: {sid}. Valid: {STRATEGY_IDS}")
            continue
        embed_strategy(sid)

    # Summary
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    print("\n" + "=" * 60)
    print("All collections in ChromaDB:")
    for col in client.list_collections():
        print(f"  {col.name:50s}  count={col.count()}")


if __name__ == "__main__":
    main()
