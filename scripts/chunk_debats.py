"""Chunk Les Débats parlementaires 1887 corpus into hierarchical text chunks.

Reads each raw .txt file from data/corpus_1887/ and splits it into two levels:
  Level 1 — section boundaries detected by uppercase-heavy header regex
  Level 2 — speech / paragraph chunks of ≤ 10 000 characters

Chunks are written to data/corpus_1887_splitted_v2/{date}/{date}_{sec}_{chunk}.txt.

Run this script ONCE before running build_indexes.py --debats.

Usage (from project root):
    python -m scripts.chunk_debats           # skip files already chunked
    python -m scripts.chunk_debats --force   # re-chunk everything
"""

import argparse
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

try:
    from scripts.config import DATA_DIR
except ModuleNotFoundError:
    from config import DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = os.path.join(DATA_DIR, "corpus_1887")
SPLIT_DIR = os.path.join(DATA_DIR, "corpus_1887_splitted_v2")

# ---------------------------------------------------------------------------
# Text splitters — module-level to avoid re-instantiating per file
# ---------------------------------------------------------------------------

_SPLITTER_L1 = RecursiveCharacterTextSplitter(
    separators=[
        r'\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n',
        r"[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*",
    ],
    chunk_size=1,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=True,
)

_SPLITTER_L2 = RecursiveCharacterTextSplitter(
    separators=[r"\nM\.", r"\n{2,}", r"\n"],
    chunk_size=10_000,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=True,
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def chunk_file(filename: str) -> int:
    """Split one raw corpus file and write chunks to SPLIT_DIR.

    Args:
        filename: basename of a .txt file inside CORPUS_DIR.

    Returns:
        Number of chunk files created.
    """
    base = filename.split(".")[0]
    out_dir = os.path.join(SPLIT_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(CORPUS_DIR, filename), encoding="utf-8") as fh:
        text = fh.read()

    first_level = _SPLITTER_L1.create_documents([text])
    chunk_count = 0
    for i, section_doc in enumerate(first_level):
        for j, chunk_doc in enumerate(_SPLITTER_L2.create_documents([section_doc.page_content])):
            out_path = os.path.join(out_dir, f"{base}_{i:03}_{j:03}.txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(chunk_doc.page_content)
            chunk_count += 1
    return chunk_count


def chunk_corpus(force: bool = False) -> None:
    """Chunk all raw corpus files.

    Args:
        force: If True, re-chunk even when output directories already exist.
    """
    os.makedirs(SPLIT_DIR, exist_ok=True)

    raw_files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt"))
    if not raw_files:
        print(f"No .txt files found in {CORPUS_DIR}")
        return

    to_process, skipped = [], 0
    for f in raw_files:
        base = f.split(".")[0]
        out_dir = os.path.join(SPLIT_DIR, base)
        if not force and os.path.isdir(out_dir) and os.listdir(out_dir):
            skipped += 1
        else:
            to_process.append(f)

    if skipped:
        print(f"Skipping {skipped} already-chunked file(s) — use --force to redo")

    if not to_process:
        print("All files already chunked. Nothing to do.")
        return

    print(f"Chunking {len(to_process)} file(s)...")
    total_chunks = 0
    for f in tqdm(to_process, desc="Chunking"):
        total_chunks += chunk_file(f)

    print(f"Done — {len(to_process)} files → {total_chunks} chunks saved to {SPLIT_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk Les Débats parlementaires 1887 corpus into hierarchical text chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-chunk files even when output chunks already exist.",
    )
    args = parser.parse_args()
    chunk_corpus(force=args.force)


if __name__ == "__main__":
    main()
