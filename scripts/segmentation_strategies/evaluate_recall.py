"""
Evaluate segmentation-strategy recall for doc_2 retrieval.

Loads all questions from the cluster JSON files, segments each doc_2_section
using every registered strategy, embeds question + segments via Cohere, and
computes recall@k.  Outputs a per-question CSV and an aggregate CSV, plus a
timestamped log file.

Usage
-----
    cd <repo-root>
    python scripts/segmentation_strategies/evaluate_recall.py

Environment
-----------
    COHERE_API_KEY          – required
    COHERE_EMBEDDING_MODEL  – optional, default embed-v4.0
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
print(REPO_ROOT)
SCRIPTS_DIR = REPO_ROOT / "scripts"
CLUSTER_DIR = REPO_ROOT / "data" / "cluster_questions"
OUTPUT_DIR = SCRIPT_DIR / "results"

# Ensure importability
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from segmentation_strategies.strategies import (
    STRATEGIES,
    anchor_expansion,
)
from segmentation_strategies.evaluate import (
    coverage_score,
    load_records,
    retrieval_rank,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K_VALUES = (1, 3, 5, 10)
ANCHOR_CONTEXT_TURNS = (0, 1, 2)
COHERE_BATCH_SIZE = 96          # Cohere embed API batch limit
COHERE_RATE_LIMIT_PAUSE = 1.0   # seconds between batches


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"evaluate_recall_{ts}.log"

    logger = logging.getLogger("eval_recall")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
def make_embed_fn() -> Callable[[List[str]], List[List[float]]]:
    import cohere
    from scripts.config import COHERE_API_KEY

    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY not found. Check your .env file.")
    model = os.environ.get("COHERE_EMBEDDING_MODEL", "embed-v4.0")
    co = cohere.Client(COHERE_API_KEY)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), COHERE_BATCH_SIZE):
            batch = texts[i : i + COHERE_BATCH_SIZE]
            resp = co.embed(texts=batch, model=model, input_type="search_document")
            all_vecs.extend(resp.embeddings)
            if i + COHERE_BATCH_SIZE < len(texts):
                time.sleep(COHERE_RATE_LIMIT_PAUSE)
        return all_vecs

    return embed_fn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_records(cluster_dir: Path) -> List[Dict]:
    """Load and merge all cluster JSON files."""
    records: List[Dict] = []
    pattern = str(cluster_dir / "clusters_*_for_llm.json")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        recs = load_records(fpath)
        # Tag each record with the cluster file it came from
        for r in recs:
            r["_source_file"] = fname
        records.extend(recs)
    return records


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------
def evaluate_record(
    record: Dict,
    strategy_name: str,
    segments: List[str],
    embed_fn: Callable,
) -> Dict:
    """Evaluate a single record for one strategy, return row dict."""
    question = record.get("question", "")
    doc_2 = record["doc_2"]
    covered = coverage_score(segments, doc_2)

    rank: Optional[int] = None
    if covered and question:
        rank = retrieval_rank(question, segments, doc_2, embed_fn)

    row = {
        "question": question,
        "strategy": strategy_name,
        "cluster": record.get("_source_file", ""),
        "n_segments": len(segments),
        "coverage": covered,
        "rank": rank if rank is not None else -1,
    }
    for k in K_VALUES:
        row[f"recall@{k}"] = int(rank is not None and rank < k) if covered else 0

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger = setup_logging(OUTPUT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Load data ---
    logger.info("Loading cluster files from %s", CLUSTER_DIR)
    all_records = load_all_records(CLUSTER_DIR)
    valid = [r for r in all_records if r.get("doc_2") and r.get("doc_2_section")]
    logger.info("Total questions: %d   valid (doc_2 + section): %d", len(all_records), len(valid))

    # --- Embedding function ---
    logger.info("Initialising Cohere embedding function")
    embed_fn = make_embed_fn()

    # --- Build strategy dict (including S5 anchor variants) ---
    strategies: Dict[str, Callable] = dict(STRATEGIES)
    for ctx in ANCHOR_CONTEXT_TURNS:
        name = f"S5_anchor_ctx{ctx}"
        _ctx = ctx  # capture
        strategies[name] = lambda section, _c=_ctx: None  # placeholder; handled below

    strategy_names = list(strategies.keys())
    logger.info("Strategies to evaluate: %s", strategy_names)

    # --- Per-record evaluation ---
    per_question_rows: List[Dict] = []
    total = len(valid) * len(strategy_names)
    done = 0

    for i, record in enumerate(valid):
        section = record["doc_2_section"]
        doc_2 = record["doc_2"]
        question = record.get("question", "")
        cluster = record.get("_source_file", "")

        for sname in strategy_names:
            # Segment
            if sname.startswith("S5_anchor_ctx"):
                ctx = int(sname[-1])
                segments = anchor_expansion(section, doc_2, context_turns=ctx)
            else:
                segments = STRATEGIES[sname](section)

            # Evaluate
            row = evaluate_record(record, sname, segments, embed_fn)
            per_question_rows.append(row)

            done += 1
            if done % 100 == 0:
                logger.info("Progress: %d / %d  (%.0f%%)", done, total, 100 * done / total)

    # --- Write per-question CSV ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_q_path = OUTPUT_DIR / f"recall_per_question_{ts}.csv"
    fieldnames = ["question", "strategy", "cluster", "n_segments", "coverage", "rank"] + [
        f"recall@{k}" for k in K_VALUES
    ]
    with open(per_q_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_question_rows)
    logger.info("Per-question CSV written: %s  (%d rows)", per_q_path, len(per_question_rows))

    # --- Aggregate CSV ---
    import pandas as pd

    df = pd.DataFrame(per_question_rows)
    recall_cols = [f"recall@{k}" for k in K_VALUES]
    agg = (
        df.groupby("strategy")
        .agg(
            n_questions=("question", "count"),
            coverage_rate=("coverage", "mean"),
            mean_rank=("rank", lambda s: s[s >= 0].mean() if (s >= 0).any() else -1),
            avg_segments=("n_segments", "mean"),
            **{col: (col, "mean") for col in recall_cols},
        )
        .reset_index()
    )
    agg_path = OUTPUT_DIR / f"recall_aggregate_{ts}.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("Aggregate CSV written: %s", agg_path)

    # --- Print summary ---
    logger.info("\n=== AGGREGATE RESULTS ===")
    for _, row in agg.iterrows():
        parts = [f"{row['strategy']:30s}"]
        parts.append(f"cov={row['coverage_rate']:.2%}")
        parts.append(f"rank={row['mean_rank']:.1f}")
        for col in recall_cols:
            parts.append(f"{col}={row[col]:.2%}")
        logger.info("  ".join(parts))

    logger.info("Done.")


if __name__ == "__main__":
    main()
