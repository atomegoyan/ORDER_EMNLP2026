"""
Evaluation utilities for segmentation strategies.

Metrics
-------
coverage_score          – Is doc_2 contained in ≥1 segment?
size_distribution       – Descriptive stats on segment lengths
segment_count           – How many segments does each strategy produce?
retrieval_simulation    – Embed question + segments, compute recall@k
compare_strategies      – Run all deterministic strategies on a dataset
                          and return a summary DataFrame
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Coverage
# ---------------------------------------------------------------------------

def _contains(segment: str, source_chunk: str, fuzzy: bool = True) -> bool:
    """Check whether `source_chunk` is (approximately) found in `segment`."""
    if source_chunk in segment:
        return True
    if not fuzzy:
        return False
    # Fuzzy: check first 150 chars of source_chunk
    anchor = source_chunk[:150].strip()
    return anchor in segment


def coverage_score(
    segments: List[str],
    source_chunk: str,
    fuzzy: bool = True,
) -> bool:
    """
    Return True if any segment contains `source_chunk`.

    Parameters
    ----------
    segments : List[str]
        Output of a segmentation strategy.
    source_chunk : str
        The ``doc_2`` evidence text that must be locatable.
    fuzzy : bool
        If True, also accept a partial prefix match (first 150 chars).
    """
    return any(_contains(seg, source_chunk, fuzzy) for seg in segments)


def coverage_rate(
    results: List[Dict],
    strategy_name: str,
    source_key: str = "doc_2",
    segments_key: str = "segments",
    fuzzy: bool = True,
) -> float:
    """
    Coverage rate across a list of result dicts.

    Each dict must have ``segments_key`` (list of str from strategy) and
    ``source_key`` (str, the evidence chunk to locate).

    Returns fraction of examples where coverage is True.
    """
    hits = sum(
        coverage_score(r[segments_key], r[source_key], fuzzy)
        for r in results
    )
    return hits / len(results) if results else 0.0


# ---------------------------------------------------------------------------
# 2. Size distribution
# ---------------------------------------------------------------------------

def size_distribution(segments: List[str]) -> Dict[str, float]:
    """
    Compute descriptive statistics of segment character lengths.

    Returns a dict with keys: count, mean, median, std, min, max, p25, p75.
    """
    lengths = [len(s) for s in segments if s]
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "std": 0,
                "min": 0, "max": 0, "p25": 0, "p75": 0}
    arr = np.array(lengths, dtype=float)
    return {
        "count": len(lengths),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def aggregate_size_stats(
    all_segments: List[List[str]],
) -> Dict[str, float]:
    """
    Aggregate `size_distribution` over many segmented sections.
    Pools all segment lengths together.
    """
    flat = [seg for segs in all_segments for seg in segs if seg]
    return size_distribution(flat)


# ---------------------------------------------------------------------------
# 3. Retrieval simulation
# ---------------------------------------------------------------------------

def cosine_similarities(
    query_vec: np.ndarray,
    segment_vecs: np.ndarray,
) -> np.ndarray:
    """Return cosine similarity between query and each segment vector."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(segment_vecs, axis=1, keepdims=True) + 1e-9
    normed = segment_vecs / norms
    return normed @ q


def retrieval_rank(
    question: str,
    segments: List[str],
    source_chunk: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    fuzzy: bool = True,
) -> Optional[int]:
    """
    Embed the `question` and all `segments`, then return the rank (0-indexed)
    of the first segment that contains `source_chunk`.

    Returns None if `source_chunk` is not found in any segment.
    """
    if not segments:
        return None

    gold_indices = [
        i for i, seg in enumerate(segments)
        if _contains(seg, source_chunk, fuzzy)
    ]
    if not gold_indices:
        return None

    texts = [question] + segments
    vecs = np.array(embed_fn(texts), dtype=np.float32)
    query_vec = vecs[0]
    seg_vecs = vecs[1:]

    sims = cosine_similarities(query_vec, seg_vecs)
    ranking = np.argsort(-sims).tolist()  # descending

    # Return the best (lowest) rank among gold segments
    ranks = [ranking.index(gi) for gi in gold_indices if gi in ranking]
    return min(ranks) if ranks else None


def recall_at_k(
    question: str,
    segments: List[str],
    source_chunk: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    k: int = 3,
    fuzzy: bool = True,
) -> bool:
    """Return True if the gold segment is ranked in top-k."""
    rank = retrieval_rank(question, segments, source_chunk, embed_fn, fuzzy)
    return rank is not None and rank < k


# ---------------------------------------------------------------------------
# 4. Per-strategy comparison across a dataset
# ---------------------------------------------------------------------------

def compare_strategies(
    records: List[Dict],
    strategies: Dict[str, Callable[[str], List[str]]],
    section_key: str = "doc_2_section",
    source_key: str = "doc_2",
    question_key: str = "question",
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    k_values: Tuple[int, ...] = (1, 3, 5),
    fuzzy: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Apply each strategy to every record and compute all metrics.

    Parameters
    ----------
    records : List[Dict]
        Items from ``clusters_9_for_llm.json`` (must have section_key,
        source_key, and optionally question_key).
    strategies : Dict[str, callable]
        Map of strategy name → segmentation function
        (fn(section: str) -> List[str]).
    embed_fn : callable, optional
        If provided, retrieval simulation metrics are also computed.
    k_values : tuple
        K values for recall@k.
    verbose : bool
        Print progress.

    Returns
    -------
    Dict mapping strategy_name → metrics dict.
    """
    # Filter records that have a usable section
    valid = [
        r for r in records
        if r.get(section_key) and r.get(source_key)
    ]

    if verbose:
        print(f"Evaluating {len(strategies)} strategies on {len(valid)} records.")

    results: Dict[str, Dict] = {}

    for name, fn in strategies.items():
        if verbose:
            print(f"  → {name} ...", end=" ", flush=True)

        all_segments: List[List[str]] = []
        covered = 0
        rank_list: List[Optional[int]] = []

        for r in valid:
            section = r[section_key]
            doc_2 = r[source_key]
            question = r.get(question_key, "")

            try:
                segs = fn(section)
            except Exception as exc:
                if verbose:
                    print(f"\n    Warning: strategy {name} raised {exc} on record; skipping.")
                segs = [section]

            all_segments.append(segs)

            if coverage_score(segs, doc_2, fuzzy):
                covered += 1

            if embed_fn and question:
                rank = retrieval_rank(question, segs, doc_2, embed_fn, fuzzy)
                rank_list.append(rank)

        cov_rate = covered / len(valid) if valid else 0.0
        size_stats = aggregate_size_stats(all_segments)
        avg_n_segs = statistics.mean(len(s) for s in all_segments) if all_segments else 0

        metrics: Dict = {
            "coverage_rate": cov_rate,
            "avg_n_segments": avg_n_segs,
            **{f"size_{k}": v for k, v in size_stats.items()},
        }

        if embed_fn and rank_list:
            for k in k_values:
                recall = sum(1 for r in rank_list if r is not None and r < k) / len(rank_list)
                metrics[f"recall@{k}"] = recall
            valid_ranks = [r for r in rank_list if r is not None]
            metrics["mean_rank"] = float(np.mean(valid_ranks)) if valid_ranks else float("inf")
            metrics["not_found_rate"] = sum(1 for r in rank_list if r is None) / len(rank_list)

        results[name] = metrics

        if verbose:
            print(f"coverage={cov_rate:.2%}, avg_segs={avg_n_segs:.1f}, median_chars={size_stats['median']:.0f}")

    return results


# ---------------------------------------------------------------------------
# 5. Load helpers
# ---------------------------------------------------------------------------

def load_records(json_path: str) -> List[Dict]:
    """
    Load the cluster questions JSON file and flatten all questions into a
    list of flat dicts (one per question), preserving all source fields.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    # Support both a list of clusters and a list of questions
    for item in data:
        if "questions" in item:
            for q in item["questions"]:
                records.append(q)
        else:
            records.append(item)
    return records


def load_sections_deduplicated(records: List[Dict], section_key: str = "doc_2_section") -> List[str]:
    """Return unique non-null sections from records."""
    seen = set()
    sections = []
    for r in records:
        sec = r.get(section_key)
        if sec and sec not in seen:
            seen.add(sec)
            sections.append(sec)
    return sections


# ---------------------------------------------------------------------------
# 6. Pretty-print summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    col_width = max(len(k) for k in results) + 2
    all_metrics = sorted({m for v in results.values() for m in v})

    header = f"{'Strategy':<{col_width}}" + "".join(f"{m:>14}" for m in all_metrics)
    print(header)
    print("─" * len(header))

    for strategy, metrics in sorted(results.items()):
        row = f"{strategy:<{col_width}}"
        for m in all_metrics:
            val = metrics.get(m)
            if val is None:
                row += f"{'—':>14}"
            elif isinstance(val, float):
                row += f"{val:>14.3f}"
            else:
                row += f"{val:>14}"
        print(row)
