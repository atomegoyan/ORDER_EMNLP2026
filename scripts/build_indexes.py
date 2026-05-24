"""Build Cohere / OpenRouter embeddings and BM25 search indexes for the 1887 corpora.

Handles three corpora:
  - Les Débats parlementaires 1887  (reads pre-chunked files — run chunk_debats.py first)
  - L'Intransigeant 1887            (section regex extraction → embed + BM25)
  - Le Gaulois 1887                 (section regex extraction → embed + BM25)

Embedding providers
-------------------
  cohere      — Cohere embed-v4.0 (default)
                Stores collections in  data/embeddings_1887_cohere/
  openrouter  — Any OpenRouter-hosted model (default: openai/text-embedding-3-small)
                Stores collections in  data/embeddings_1887_openrouter/

Usage (from project root):
    python -m scripts.build_indexes --all
    python -m scripts.build_indexes --all --provider openrouter
    python -m scripts.build_indexes --debats --provider openrouter --model openai/text-embedding-3-large
    python -m scripts.build_indexes --lintransigeant --legaulois
    python -m scripts.build_indexes --all --force

Note: for Les Débats, run ``python -m scripts.chunk_debats`` first to create
the split files in data/corpus_1887_splitted_v2/.
"""

import argparse
import os

import bm25s
import chromadb
import chromadb.utils.embedding_functions as chromadb_ef
import regex as re
from tqdm import tqdm

try:
    from scripts.config import (
        BM25_DEBATS_DIR,
        BM25_LEGAULOIS_DIR,
        BM25_LINTRANSIGEANT_DIR,
        COHERE_API_KEY,
        DATA_DIR,
        EMBEDDINGS_1887_COHERE,
        EMBEDDINGS_1887_OPENROUTER,
        OPENROUTER_API_KEY,
        OPENROUTER_EMBEDDING_MODEL,
    )
    from scripts.utils.embedding_functions import OpenRouterEmbeddingFunction
except ModuleNotFoundError:
    from config import (
        BM25_DEBATS_DIR,
        BM25_LEGAULOIS_DIR,
        BM25_LINTRANSIGEANT_DIR,
        COHERE_API_KEY,
        DATA_DIR,
        EMBEDDINGS_1887_COHERE,
        EMBEDDINGS_1887_OPENROUTER,
        OPENROUTER_API_KEY,
        OPENROUTER_EMBEDDING_MODEL,
    )
    from utils.embedding_functions import OpenRouterEmbeddingFunction


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _model_slug(model: str) -> str:
    """Return the trailing part of a model name suitable for use in collection names.

    Examples:
        "embed-v4.0"                     → "embed-v4.0"
        "openai/text-embedding-3-small"  → "text-embedding-3-small"
    """
    return model.split("/")[-1]


def _get_client_and_ef(provider: str, model: str):
    """Create a ChromaDB PersistentClient and matching EmbeddingFunction.

    Args:
        provider: ``"cohere"`` or ``"openrouter"``.
        model:    Model identifier (provider-specific).

    Returns:
        Tuple of (chromadb.PersistentClient, embedding_function, chroma_dir).
    """
    if provider == "cohere":
        ef = chromadb_ef.CohereEmbeddingFunction(api_key=COHERE_API_KEY, model_name=model)
        chroma_dir = EMBEDDINGS_1887_COHERE
    elif provider == "openrouter":
        ef = OpenRouterEmbeddingFunction(api_key=OPENROUTER_API_KEY, model=model)
        chroma_dir = EMBEDDINGS_1887_OPENROUTER
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'cohere' or 'openrouter'.")

    client = chromadb.PersistentClient(path=chroma_dir)
    return client, ef, chroma_dir


def _collection_name_debats(provider: str, model: str) -> str:
    if provider == "cohere":
        return "debattre_1887_10000_hierarchical_cohere"
    return f"debattre_1887_10000_hierarchical_{_model_slug(model)}"


def _collection_name_newspaper(base: str, provider: str, model: str) -> str:
    """e.g. base='legaulois_1887', provider='openrouter' → 'legaulois_1887_text-embedding-3-small'."""
    if provider == "cohere":
        return f"{base}_v1"
    return f"{base}_{_model_slug(model)}"


def _build_bm25_index(corpus: list[tuple[str, str]], index_dir: str) -> None:
    """Tokenise *corpus* and persist a bm25s index to *index_dir*.

    Args:
        corpus: list of (doc_id, text) pairs.
        index_dir: directory to save the index.
    """
    corpus_with_ids = [{"id": doc_id, "text": text} for doc_id, text in corpus]
    texts = [d["text"] for d in corpus_with_ids]
    tokenized = bm25s.tokenize(texts, stopwords="fr", show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    os.makedirs(index_dir, exist_ok=True)
    retriever.save(index_dir, corpus=corpus_with_ids)
    print(f"BM25 index saved → {index_dir} ({len(corpus_with_ids)} documents)")


# ============================================================================
# LES DÉBATS PARLEMENTAIRES 1887
# ============================================================================

_DEBATS_SPLIT_DIR = os.path.join(DATA_DIR, "corpus_1887_splitted_v2")


def _collect_debats_chunks() -> list[tuple[str, str]]:
    """Return list of (filename_stem, text) for all chunks under _DEBATS_SPLIT_DIR."""
    result = []
    for folder in sorted(os.listdir(_DEBATS_SPLIT_DIR)):
        folder_path = os.path.join(_DEBATS_SPLIT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, encoding="utf-8") as fh:
                result.append((fname, fh.read()))
    return result


def build_debats(provider: str = "cohere", model: str = "embed-v4.0", force: bool = False) -> None:
    """Embed and index the Les Débats 1887 corpus.

    Requires pre-chunked files in corpus_1887_splitted_v2/ — run
    ``python -m scripts.chunk_debats`` first if that directory is empty.

    Args:
        provider: ``"cohere"`` or ``"openrouter"``.
        model:    Embedding model identifier.
        force:    Delete and rebuild the existing collection / BM25 index.
    """
    print("\n" + "=" * 60)
    print(f"BUILDING: Les Débats parlementaires 1887  [{provider} / {model}]")
    print("=" * 60)

    # Validate that chunks exist
    if not os.path.isdir(_DEBATS_SPLIT_DIR) or not any(
        os.scandir(_DEBATS_SPLIT_DIR)
    ):
        raise FileNotFoundError(
            f"No chunked files found in {_DEBATS_SPLIT_DIR}.\n"
            "Run  python -m scripts.chunk_debats  first."
        )

    client, ef, chroma_dir = _get_client_and_ef(provider, model)
    collection_name = _collection_name_debats(provider, model)

    existing_names = [c.name for c in client.list_collections()]
    if force and collection_name in existing_names:
        print(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
        existing_names.remove(collection_name)

    if collection_name in existing_names:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
        print(
            f"Collection '{collection_name}' already exists "
            f"({collection.count()} docs) — skipping (use --force to redo)"
        )
    else:
        print(f"Embedding chunks into '{collection_name}'...")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={
                "description": (
                    "Embeddings of the Débats parlementaires 1887 corpus split "
                    f"hierarchically with a threshold of 10 000 chars using {provider}/{model}"
                ),
                "model": model,
                "provider": provider,
            },
        )
        chunks = _collect_debats_chunks()
        for fname, text in tqdm(chunks, desc="Embedding"):
            parts = fname.split("-")
            year = parts[0] if len(parts) > 0 else ""
            month = parts[1] if len(parts) > 1 else ""
            day = parts[2].split("_")[0] if len(parts) > 2 else ""
            collection.add(
                documents=[text],
                ids=[fname],
                metadatas=[{"base_name": fname, "year": year, "month": month, "day": day}],
            )
        print(f"Collection '{collection_name}' — {collection.count()} documents")

    # BM25 (provider-independent, built only once)
    if force or not os.path.isdir(BM25_DEBATS_DIR) or not os.listdir(BM25_DEBATS_DIR):
        print("Building BM25 index for Les Débats...")
        _build_bm25_index(_collect_debats_chunks(), BM25_DEBATS_DIR)
    else:
        print(f"BM25 index already present at {BM25_DEBATS_DIR} — skipping (use --force to redo)")

    print("\nDone: Les Débats parlementaires 1887")


_LINTRANSIGEANT_CORPUS_DIR = os.path.join(DATA_DIR, "lintransigeant_1887")

_LINTRANSIGEANT_SECTION_REGEX = re.compile(r"La Chambre\n.*")


def _load_lintransigeant_corpus() -> list[tuple[str, str]]:
    result = []
    for folder in sorted(os.listdir(_LINTRANSIGEANT_CORPUS_DIR)):
        folder_path = os.path.join(_LINTRANSIGEANT_CORPUS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        filepath = os.path.join(folder_path, files[0])
        with open(filepath, encoding="utf-8") as fh:
            text = fh.read().replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        result.append((files[0], text))
    return result


def _extract_la_chambre(corpus: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result = []
    for filename, text in corpus:
        matches = _LINTRANSIGEANT_SECTION_REGEX.findall(text)
        if matches:
            result.append((filename, matches[0]))
    print(f"Extracted {len(result)} / {len(corpus)} articles with 'La Chambre' section")
    return result


def build_lintransigeant(
    provider: str = "cohere", model: str = "embed-v4.0", force: bool = False
) -> None:
    """Parse, embed, and index L'Intransigeant 1887 corpus."""
    print("\n" + "=" * 60)
    print(f"BUILDING: L'Intransigeant 1887  [{provider} / {model}]")
    print("=" * 60)

    corpus_raw = _load_lintransigeant_corpus()
    print(f"Loaded {len(corpus_raw)} raw files from {_LINTRANSIGEANT_CORPUS_DIR}")
    corpus = _extract_la_chambre(corpus_raw)

    client, ef, _ = _get_client_and_ef(provider, model)
    collection_name = _collection_name_newspaper("lintransigeant_1887", provider, model)

    existing_names = [c.name for c in client.list_collections()]
    if force and collection_name in existing_names:
        print(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
        existing_names.remove(collection_name)

    if collection_name in existing_names:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
        print(
            f"Collection '{collection_name}' already exists "
            f"({collection.count()} docs) — skipping (use --force to redo)"
        )
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={
                "description": (
                    "Embeddings des sections 'La Chambre' de L'Intransigeant 1887. "
                    f"Provider: {provider}, model: {model}"
                ),
                "model": model,
                "provider": provider,
            },
        )
        for name, text in tqdm(corpus, desc="Embedding"):
            base_name = name.split(".")[0]
            year, month, day = base_name[0:4], base_name[4:6], base_name[6:8]
            collection.add(
                documents=[text],
                metadatas=[{"base_name": base_name, "year": year, "month": month, "day": day}],
                ids=[base_name],
            )
        print(f"Collection '{collection_name}' — {collection.count()} documents")

    if force or not os.path.isdir(BM25_LINTRANSIGEANT_DIR) or not os.listdir(BM25_LINTRANSIGEANT_DIR):
        print("Building BM25 index for L'Intransigeant...")
        _build_bm25_index([(n.split(".")[0], t) for n, t in corpus], BM25_LINTRANSIGEANT_DIR)
    else:
        print(f"BM25 index already present at {BM25_LINTRANSIGEANT_DIR} — skipping (use --force to redo)")

    print("\nDone: L'Intransigeant 1887")


# ============================================================================
# LE GAULOIS 1887
# ============================================================================

_LEGAULOIS_CORPUS_DIR = os.path.join(DATA_DIR, "legaulois_1887")

_LEGAULOIS_SECTION_REGEX = re.compile(
    r"CHAMBRE DES D[ÉE]PUT[ÉE]S.{50,}?[A-ZÉ\s\n]{4,}", re.DOTALL
)


def _load_legaulois_corpus() -> list[tuple[str, str]]:
    result = []
    for folder in sorted(os.listdir(_LEGAULOIS_CORPUS_DIR)):
        folder_path = os.path.join(_LEGAULOIS_CORPUS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        filepath = os.path.join(folder_path, files[0])
        with open(filepath, encoding="utf-8") as fh:
            text = fh.read().replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        result.append((files[0], text))
    return result


def _extract_chambre_deputes(corpus: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result = []
    for filename, text in corpus:
        matches = _LEGAULOIS_SECTION_REGEX.findall(text)
        if matches:
            result.append((filename, matches[0]))
    print(f"Extracted {len(result)} / {len(corpus)} articles with 'Chambre des Députés' section")
    return result


def build_legaulois(
    provider: str = "cohere", model: str = "embed-v4.0", force: bool = False
) -> None:
    """Parse, embed, and index Le Gaulois 1887 corpus."""
    print("\n" + "=" * 60)
    print(f"BUILDING: Le Gaulois 1887  [{provider} / {model}]")
    print("=" * 60)

    corpus_raw = _load_legaulois_corpus()
    print(f"Loaded {len(corpus_raw)} raw files from {_LEGAULOIS_CORPUS_DIR}")
    corpus = _extract_chambre_deputes(corpus_raw)

    client, ef, _ = _get_client_and_ef(provider, model)
    collection_name = _collection_name_newspaper("legaulois_1887", provider, model)

    existing_names = [c.name for c in client.list_collections()]
    if force and collection_name in existing_names:
        print(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
        existing_names.remove(collection_name)

    if collection_name in existing_names:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
        print(
            f"Collection '{collection_name}' already exists "
            f"({collection.count()} docs) — skipping (use --force to redo)"
        )
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={
                "description": (
                    "Embeddings des sections 'Chambre des Députés' du Gaulois 1887. "
                    f"Provider: {provider}, model: {model}"
                ),
                "model": model,
                "provider": provider,
            },
        )
        for name, text in tqdm(corpus, desc="Embedding"):
            base_name = name.split(".")[0]
            year, month, day = base_name[0:4], base_name[4:6], base_name[6:8]
            collection.add(
                documents=[text],
                metadatas=[{"base_name": base_name, "year": year, "month": month, "day": day}],
                ids=[base_name],
            )
        print(f"Collection '{collection_name}' — {collection.count()} documents")

    if force or not os.path.isdir(BM25_LEGAULOIS_DIR) or not os.listdir(BM25_LEGAULOIS_DIR):
        print("Building BM25 index for Le Gaulois...")
        _build_bm25_index([(n.split(".")[0], t) for n, t in corpus], BM25_LEGAULOIS_DIR)
    else:
        print(f"BM25 index already present at {BM25_LEGAULOIS_DIR} — skipping (use --force to redo)")

    print("\nDone: Le Gaulois 1887")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build embedding + BM25 search indexes for the 1887 newspaper corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.build_indexes --all
  python -m scripts.build_indexes --all --provider openrouter
  python -m scripts.build_indexes --debats --provider openrouter --model openai/text-embedding-3-large
  python -m scripts.build_indexes --lintransigeant --legaulois
  python -m scripts.build_indexes --all --force

Note: for --debats, run  python -m scripts.chunk_debats  first.
        """,
    )
    parser.add_argument("--all", dest="all_corpora", action="store_true", help="Build all three corpora")
    parser.add_argument("--debats", action="store_true", help="Build Les Débats parlementaires 1887")
    parser.add_argument("--lintransigeant", action="store_true", help="Build L'Intransigeant 1887")
    parser.add_argument("--legaulois", action="store_true", help="Build Le Gaulois 1887")
    parser.add_argument(
        "--provider",
        choices=["cohere", "openrouter"],
        default="cohere",
        help="Embedding provider (default: cohere)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Embedding model. Defaults: cohere→embed-v4.0, "
            "openrouter→openai/text-embedding-3-small"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild existing collections/indexes (re-calls the embedding API)",
    )
    args = parser.parse_args()

    if not any([args.all_corpora, args.debats, args.lintransigeant, args.legaulois]):
        parser.print_help()
        return

    # Resolve default model per provider
    if args.model is None:
        args.model = (
            "embed-v4.0" if args.provider == "cohere" else OPENROUTER_EMBEDDING_MODEL
        )

    if args.all_corpora or args.debats:
        build_debats(provider=args.provider, model=args.model, force=args.force)

    if args.all_corpora or args.lintransigeant:
        build_lintransigeant(provider=args.provider, model=args.model, force=args.force)

    if args.all_corpora or args.legaulois:
        build_legaulois(provider=args.provider, model=args.model, force=args.force)

    print("\nAll requested indexes are ready.")


if __name__ == "__main__":
    main()
