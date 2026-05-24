"""
run_C4_joint_multiseed.py

Multi-seed experiment for Joint (Chunking × Metadata) Task-Conditioned Optimization.

Uses the pre-built JSONL cache from scripts/c4_joint/c4_1_build_cache.py.
For each seed: split → cluster → score every (chunking, metadata) pair per cluster
→ pick best pair per cluster per routing → evaluate test set (baseline vs TC).

Protocol matches C4_task_conditioned_joint.ipynb:
  - Split via np.random.default_rng(seed).permutation()
  - UMAP(10D, cosine, random_state=seed) + HDBSCAN(min_cluster_size=5, min_samples=5)
  - Score: Coverage@k per cluster × pair × routing on train set
  - Test: assign via nearest centroid (noise cluster -1 treated as real cluster) in UMAP space
  - Evaluate: Coverage@{3,5,10} and Recall@{3,5,10} on test set, baseline vs TC

Outputs:
    data/multiseed_results/C4_joint_runs_raw.csv
    data/multiseed_results/C4_joint_per_question.csv
"""

import gc
import os
import datetime
import json
import sys
from collections import defaultdict

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.metrics.pairwise import euclidean_distances

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.c4_joint.c4_common import (
    BASELINE_PAIR,
    CLUSTER_DIM,
    JOINT_PAIRS,
    K_VALUES,
    ROUTING_CONFIGS,
    SCORE_K_PER_ROUTING,
    load_gold_collections,
    load_joint,
    load_questions_df,
)
from scripts.evaluation_utils import compute_coverage_row_list
from scripts.retrieval_utils import compute_recall_metrics_dataframe

# ── Config ────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 2021, 7, 9999]
TRAIN_RATIO = 0.5

RESULTS_DIR = os.path.join(BASE_DIR, "data", "multiseed_results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_questions(questions_df, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(questions_df))
    n_train = int(len(questions_df) * TRAIN_RATIO)
    train_df = questions_df.iloc[idx[:n_train]].reset_index(drop=True)
    test_df  = questions_df.iloc[idx[n_train:]].reset_index(drop=True)
    return train_df, test_df


def cluster_questions(train_df, seed):
    """UMAP(CLUSTER_DIM) + HDBSCAN on training embeddings. Returns (reducer, labels, centroids)."""
    X_train = np.array(train_df["embeddings"].tolist())
    reducer = umap.UMAP(n_components=CLUSTER_DIM, metric="cosine", random_state=seed)
    X_train_umap = reducer.fit_transform(X_train)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric="euclidean")
    labels = clusterer.fit_predict(X_train_umap)
    centroids = {cid: X_train_umap[labels == cid].mean(axis=0) for cid in set(labels)}
    return reducer, labels, centroids


def assign_clusters(X_test_umap, centroids):
    """Assign each test point to its nearest centroid (noise cluster -1 included)."""
    valid_cids = sorted(centroids.keys())
    C = np.array([centroids[c] for c in valid_cids])
    d = euclidean_distances(X_test_umap, C)
    return np.array([valid_cids[i] for i in d.argmin(axis=1)])


def _n_results_for(routing_key, k):
    """UMS-based routings request 2*k documents to distribute across collections."""
    return 2 * k if routing_key in ("ums", "qre_ums") else k


def _load_with_sources(pair, uid_to_sources):
    df = load_joint(*pair)
    if df is None:
        return None
    if "source_1" not in df.columns:
        df = df.merge(uid_to_sources, on="unique_id", how="left")
    return df


# ── Single fold ───────────────────────────────────────────────────────────────

def run_fold(questions_df, seed, gold_collections, uid_to_sources):
    """Run one seed: split → cluster → score pairs → evaluate.

    Returns:
        results_rows : list of dicts, one per condition (routing × {baseline, tc}),
                       with keys condition, coverage@k, recall@k for all k.
        per_q_df     : DataFrame with one row per (unique_id, condition),
                       with columns coverage@k, recall@k for all k.
    """
    # 1. Split
    train_df, test_df = split_questions(questions_df, seed)
    train_ids   = set(train_df["unique_id"])
    all_test_uids = set(test_df["unique_id"])

    # 2. Cluster training questions
    reducer, labels, centroids = cluster_questions(train_df, seed)
    cluster_ids = sorted(centroids.keys())
    cluster_train_qids = {
        cid: set(train_df.loc[labels == cid, "unique_id"])
        for cid in cluster_ids
    }
    n_clusters = len([c for c in cluster_ids if c != -1])
    print(f"  {n_clusters} clusters (+noise), train={len(train_df)}, test={len(test_df)}")

    # 3. Score all (chunking, metadata) pairs per cluster × routing on train set
    print(f"  Scoring {len(JOINT_PAIRS)} pairs × {len(cluster_ids)} clusters × 4 routings…")
    score_rows = []
    for pair_idx, pair in enumerate(JOINT_PAIRS):
        chunking_id, metadata_id = pair
        df = _load_with_sources(pair, uid_to_sources)
        if df is None:
            for rk in ROUTING_CONFIGS:
                for cid in cluster_ids:
                    score_rows.append({
                        "routing": rk, "cluster": int(cid),
                        "chunking": chunking_id, "metadata": metadata_id,
                        "score": float("nan"),
                    })
            continue

        df_train = df[df["unique_id"].isin(train_ids)].copy()
        strat_uids = set(df_train["unique_id"])
        del df; gc.collect()

        for rk, rcfg in ROUTING_CONFIGS.items():
            k_score = SCORE_K_PER_ROUTING[rk]
            routed = rcfg["fn"](df_train.copy(), n_results=k_score, **rcfg["kwargs"])
            for cid, qids in cluster_train_qids.items():
                valid_qids = qids & strat_uids
                subset = routed[routed["unique_id"].isin(valid_qids)]
                if len(subset) == 0:
                    score = float("nan")
                else:
                    cov = compute_coverage_row_list(
                        subset, k=k_score, collections_dict=gold_collections,
                        docs_col=rcfg["docs_col"], sources_col=rcfg["sources_col"],
                    )
                    score = float(cov[2].mean())
                score_rows.append({
                    "routing": rk, "cluster": int(cid),
                    "chunking": chunking_id, "metadata": metadata_id,
                    "score": score,
                })
            del routed
        del df_train; gc.collect()

        if (pair_idx + 1) % 50 == 0:
            print(f"    … {pair_idx + 1}/{len(JOINT_PAIRS)} pairs scored")

    scores_df = pd.DataFrame(score_rows)

    # 4. Pick best pair per (routing, cluster)
    cluster_to_pair = {}
    for rk in ROUTING_CONFIGS:
        sub = scores_df[scores_df["routing"] == rk]
        cmap = {}
        for cid, grp in sub.groupby("cluster"):
            valid = grp.dropna(subset=["score"])
            if len(valid) == 0:
                cmap[int(cid)] = BASELINE_PAIR
            else:
                top = valid.loc[valid["score"].idxmax()]
                cmap[int(cid)] = (top["chunking"], top["metadata"])
        cluster_to_pair[rk] = cmap
    del scores_df, score_rows; gc.collect()

    # 5. Assign test questions to nearest centroid
    X_test = np.array(test_df["embeddings"].tolist())
    X_test_umap = reducer.transform(X_test)
    test_clusters = assign_clusters(X_test_umap, centroids)
    test_assign = test_df[["unique_id"]].copy()
    test_assign["cluster"] = test_clusters
    del X_test, X_test_umap, train_df; gc.collect()

    # 6. Evaluate baseline and TC for each routing
    # Accumulate into {condition_key: {metric: value}} and per-question rows
    metric_acc = {}   # condition_key -> {metric_col: value}
    per_q_acc = {}    # condition_key -> {uid -> {metric_col: value}}

    for rk, rcfg in ROUTING_CONFIGS.items():
        base_key = f"baseline_{rk}"
        tc_key   = f"tc_{rk}"
        metric_acc[base_key] = {"condition": base_key}
        metric_acc[tc_key]   = {"condition": tc_key}
        per_q_acc[base_key]  = defaultdict(dict)
        per_q_acc[tc_key]    = defaultdict(dict)

        # Map test UIDs to their TC-assigned pair
        cmap = cluster_to_pair[rk]
        pair_to_uids = defaultdict(set)
        for _, r in test_assign.iterrows():
            pair = cmap.get(int(r["cluster"]), BASELINE_PAIR)
            pair_to_uids[pair].add(r["unique_id"])

        # Load baseline once; load each TC pair once — iterate over k inside
        base_sub = _load_with_sources(BASELINE_PAIR, uid_to_sources)
        if base_sub is not None:
            base_sub = base_sub[base_sub["unique_id"].isin(all_test_uids)].copy()

        # Pre-load TC pairs for this routing (each pair loaded once, applied at all k)
        tc_pair_dfs = {}
        for pair, uids in pair_to_uids.items():
            df = _load_with_sources(pair, uid_to_sources)
            if df is not None:
                tc_pair_dfs[pair] = (df[df["unique_id"].isin(uids)].copy(), uids)
            del df; gc.collect()

        for k in K_VALUES:
            n_res = _n_results_for(rk, k)

            # --- Baseline ---
            if base_sub is not None and len(base_sub) > 0:
                base_routed = rcfg["fn"](base_sub.copy(), n_results=n_res, **rcfg["kwargs"])
                cov_b = compute_coverage_row_list(
                    base_routed, k=k, collections_dict=gold_collections,
                    docs_col=rcfg["docs_col"], sources_col=rcfg["sources_col"],
                )
                rec_b = compute_recall_metrics_dataframe(base_routed, k_values=[k])
                metric_acc[base_key][f"coverage@{k}"] = float(cov_b[2].mean())
                metric_acc[base_key][f"recall@{k}"]   = float(np.mean(rec_b[f"recall@{k}"]))
                for uid, cv, rv in zip(
                    base_routed["unique_id"].values,
                    cov_b[2].values,
                    rec_b[f"recall@{k}"],
                ):
                    per_q_acc[base_key][uid][f"coverage@{k}"] = float(cv)
                    per_q_acc[base_key][uid][f"recall@{k}"]   = float(rv)
                del base_routed

            # --- Task-conditioned ---
            tc_chunks = []
            for pair, (df_sub, _) in tc_pair_dfs.items():
                routed = rcfg["fn"](df_sub.copy(), n_results=n_res, **rcfg["kwargs"])
                if len(routed) > 0:
                    tc_chunks.append(routed)

            if tc_chunks:
                tc_routed = pd.concat(tc_chunks, ignore_index=True)
                cov_t = compute_coverage_row_list(
                    tc_routed, k=k, collections_dict=gold_collections,
                    docs_col=rcfg["docs_col"], sources_col=rcfg["sources_col"],
                )
                rec_t = compute_recall_metrics_dataframe(tc_routed, k_values=[k])
                metric_acc[tc_key][f"coverage@{k}"] = float(cov_t[2].mean())
                metric_acc[tc_key][f"recall@{k}"]   = float(np.mean(rec_t[f"recall@{k}"]))
                for uid, cv, rv in zip(
                    tc_routed["unique_id"].values,
                    cov_t[2].values,
                    rec_t[f"recall@{k}"],
                ):
                    per_q_acc[tc_key][uid][f"coverage@{k}"] = float(cv)
                    per_q_acc[tc_key][uid][f"recall@{k}"]   = float(rv)
                del tc_routed
            del tc_chunks

        del base_sub, tc_pair_dfs; gc.collect()

    # Convert metric_acc → results_rows
    results_rows = list(metric_acc.values())

    # Convert per_q_acc → wide DataFrame (one row per uid × condition)
    per_q_rows = []
    for cond_key, uid_dict in per_q_acc.items():
        for uid, metrics in uid_dict.items():
            per_q_rows.append({"unique_id": uid, "condition": cond_key, **metrics})
    per_q_df = pd.DataFrame(per_q_rows) if per_q_rows else pd.DataFrame()

    return results_rows, per_q_df


# ── Aggregation & display ─────────────────────────────────────────────────────

def aggregate_and_display(all_runs):
    """Print Table 1 (absolute values) and Table 2 (Δ TC − Baseline)."""
    flat_rows = []
    for run_id, run_results in enumerate(all_runs):
        for row in run_results:
            flat_rows.append({"run": run_id, **row})
    df = pd.DataFrame(flat_rows)

    metrics = [f"{m}@{k}" for m in ["coverage", "recall"] for k in K_VALUES]

    # Table 1: absolute values
    print("\n" + "=" * 90)
    print("TABLE 1: Mean ± Std across seeds (Coverage@k, Recall@k) — C4 Joint")
    print("=" * 90)
    agg = df.groupby("condition")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]

    header = f"{'Condition':<35}" + "".join(f"  {m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))
    conditions = df["condition"].unique()
    for prefix in ("baseline_", "tc_"):
        for cond in sorted(c for c in conditions if c.startswith(prefix)):
            if cond not in agg.index:
                continue
            row_str = f"{cond:<35}"
            for m in metrics:
                mean = agg.loc[cond, f"{m}_mean"]
                std  = agg.loc[cond, f"{m}_std"]
                row_str += f"  {mean:.3f}±{std:.3f}"
            print(row_str)

    # Table 2: Δ TC − Baseline
    print("\n" + "=" * 90)
    print("TABLE 2: Δ (Task-Conditioned Joint − Global Baseline) per routing — C4 Joint")
    print("=" * 90)
    header2 = f"{'Routing':<20}" + "".join(f"  {m:>14}" for m in metrics)
    print(header2)
    print("-" * len(header2))
    for rk, rcfg in ROUTING_CONFIGS.items():
        base_key = f"baseline_{rk}"
        tc_key   = f"tc_{rk}"
        if base_key not in agg.index or tc_key not in agg.index:
            continue
        row_str = f"{rcfg['label']:<20}"
        for m in metrics:
            delta_mean = agg.loc[tc_key, f"{m}_mean"] - agg.loc[base_key, f"{m}_mean"]
            delta_std  = np.sqrt(
                agg.loc[tc_key, f"{m}_std"] ** 2 + agg.loc[base_key, f"{m}_std"] ** 2
            )
            row_str += f"  {delta_mean:+.3f}±{delta_std:.3f}"
        print(row_str)
    print()


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(all_runs, all_per_q, meta=None):
    """Save per-run aggregate and per-question results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    flat_rows = []
    for run_id, run_results in enumerate(all_runs):
        for row in run_results:
            flat_rows.append({"run": run_id, **row})
    df = pd.DataFrame(flat_rows)
    path = os.path.join(RESULTS_DIR, "C4_joint_runs_raw.csv")
    df.to_csv(path, index=False)
    print(f"Saved raw run results → {path}")

    if all_per_q:
        pq_frames = [pq.assign(run=run_id) for run_id, pq in enumerate(all_per_q) if len(pq) > 0]
        if pq_frames:
            pq_all = pd.concat(pq_frames, ignore_index=True)
            pq_path = os.path.join(RESULTS_DIR, "C4_joint_per_question.csv")
            pq_all.to_csv(pq_path, index=False)
            print(f"Saved per-question results → {pq_path}")

    if meta is not None:
        meta["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta_path = os.path.join(RESULTS_DIR, "C4_experiment_config.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved experiment config  → {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load questions and gold collections once (shared across all seeds)
    print("Loading questions from ChromaDB…")
    questions_df = load_questions_df()
    print(f"  {len(questions_df)} questions loaded")

    uid_to_sources = questions_df[["unique_id", "source_1", "source_2"]].copy()

    print("Loading gold collections…")
    gold_collections = load_gold_collections()
    for label, col in gold_collections.items():
        print(f"  {label}: {col.count()} docs cached")

    all_runs   = []
    all_per_q  = []
    total      = len(SEEDS)

    for run_num, seed in enumerate(SEEDS, 1):
        print(f"\n[{run_num}/{total}] Seed={seed}")
        fold_results, per_q = run_fold(
            questions_df, seed, gold_collections, uid_to_sources
        )
        all_runs.append(fold_results)
        all_per_q.append(per_q)
        print(f"  Done ({len(fold_results)} conditions evaluated)")

    aggregate_and_display(all_runs)
    n_total = len(questions_df)
    meta = {
        "script": "run_C4_joint_multiseed.py",
        "seeds": SEEDS,
        "train_ratio": TRAIN_RATIO,
        "n_total": n_total,
        "n_train": int(n_total * TRAIN_RATIO),
        "n_test": n_total - int(n_total * TRAIN_RATIO),
        "k_values": list(K_VALUES),
        "score_k_per_routing": SCORE_K_PER_ROUTING,
        "routing_configs": list(ROUTING_CONFIGS.keys()),
        "n_joint_pairs": len(JOINT_PAIRS),
        "baseline_pair": list(BASELINE_PAIR),
        "umap_params": {"n_components": CLUSTER_DIM, "metric": "cosine"},
        "hdbscan_params": {"min_cluster_size": 5, "min_samples": 5},
    }
    save_results(all_runs, all_per_q, meta=meta)


if __name__ == "__main__":
    main()
