"""
Phase 1 — Generate chunk variants for Les Débats corpus.

Reads header-split sections from  data/corpus_1887_splitted_section/
and re-chunks each section with 8 strategies (4 domain-specific + 3 fixed-size + baseline).
Saves chunks to separate directories under data/, plus a chunk_metadata.json per strategy.

Usage:
    python scripts/generate_all_chunk_variants.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.segmentation_strategies.strategies import (
    speaker_turn_atomic,
    speaker_turn_windowed,
    president_mediated,
    chapter_vote_split,
    strip_vote_lists,
    topic_boundary,
    amendment_cycle,
    hybrid_topic_window,
)

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SECTION_DIR = DATA_DIR / "corpus_1887_splitted_section"

# ── Strategy definitions ───────────────────────────────────────────────────

def _make_fixed_splitter(chunk_size: int, overlap: int):
    """Return a RecursiveCharacterTextSplitter with French debate separators."""
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*",
            r"\nM\.",
            r"\n{2,}",
            r"\n",
        ],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=True,
    )

    def split_fn(text: str):
        docs = splitter.create_documents([text])
        return [d.page_content for d in docs]

    return split_fn


STRATEGIES = {
    "baseline_10k": _make_fixed_splitter(10000, 200),
    "S2_atomic": speaker_turn_atomic,
    "S3_window3": lambda s: speaker_turn_windowed(s, n_turns=3, overlap=1),
    "S4_president": president_mediated,
    "S8_vote": chapter_vote_split,
    "S9_topic": topic_boundary,
    "S10_amendment": amendment_cycle,
    "S11_hybrid": hybrid_topic_window,
    "S9_topic_noVotes": lambda s: topic_boundary(strip_vote_lists(s)),
    "S11_hybrid_noVotes": lambda s: hybrid_topic_window(strip_vote_lists(s)),
    "fixed_500": _make_fixed_splitter(500, 50),
    "fixed_1000": _make_fixed_splitter(1000, 100),
    "fixed_2000": _make_fixed_splitter(2000, 200),
}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not SECTION_DIR.exists():
        print(f"ERROR: Section directory not found: {SECTION_DIR}")
        sys.exit(1)

    # Collect all section files: data/corpus_1887_splitted_section/{date}/{date}_{idx}.txt
    section_files = sorted(SECTION_DIR.rglob("*.txt"))
    print(f"Found {len(section_files)} section files in {SECTION_DIR}")

    for strategy_name, strategy_fn in STRATEGIES.items():
        out_dir = DATA_DIR / f"corpus_1887_{strategy_name}"
        out_dir.mkdir(exist_ok=True)

        metadata = {}  # chunk_id → info
        total_chunks = 0
        empty_sections = 0

        for section_file in section_files:
            text = section_file.read_text(encoding="utf-8")
            if not text.strip():
                empty_sections += 1
                continue

            # Extract seance_date and section_idx from filename
            # e.g. "1887-02-04_008.txt" → date="1887-02-04", section_idx="008"
            stem = section_file.stem  # "1887-02-04_008"
            parts = stem.split("_")
            seance_date = parts[0]  # "1887-02-04"
            section_idx = "_".join(parts[1:]) if len(parts) > 1 else "000"

            # Create per-date subdirectory
            date_dir = out_dir / seance_date
            date_dir.mkdir(exist_ok=True)

            # Apply strategy
            chunks = strategy_fn(text)
            if not chunks:
                chunks = [text.strip()]

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{seance_date}_{section_idx}_{chunk_idx:03}.txt"
                chunk_path = date_dir / chunk_id
                chunk_path.write_text(chunk_text, encoding="utf-8")

                metadata[chunk_id] = {
                    "seance_date": seance_date,
                    "section_file": section_file.name,
                    "section_idx": section_idx,
                    "chunk_idx": chunk_idx,
                    "chunk_length": len(chunk_text),
                }
                total_chunks += 1

        # Save metadata
        meta_path = out_dir / "chunk_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=1)

        print(
            f"  {strategy_name:20s}  →  {total_chunks:6d} chunks  "
            f"({empty_sections} empty sections skipped)  "
            f"→ {out_dir}"
        )


if __name__ == "__main__":
    main()
