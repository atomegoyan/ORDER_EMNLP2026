"""
run_C2_multiseed_kfold.py

Multi-seed 75/25 split experiment for Task-Conditioned Chunking Strategy selection.

Protocol (matching C2_task_conditioned_inference.ipynb):
  - 3 seeds, each with a 75/25 train/test split → 3 experiments
  - Split uses np.random.default_rng(seed).permutation() to match the notebook exactly
  - Train: UMAP(10D, cosine) + HDBSCAN(min_cluster_size=5, min_samples=5)
  - Score: Coverage@k per cluster × strategy × routing on train set
  - Test: assign via nearest centroid in UMAP space
  - Evaluate: Coverage@{3,5,10} and Recall@{3,5,10} on test set
  - Output: Table 1 (mean ± std absolute), Table 2 (Δ TC − Baseline per routing)
"""

import datetime
import json
import os
import sys

# Force UTF-8 stdout/stderr so Unicode chars (Δ, ±) survive Windows cp1252 consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.metrics.pairwise import euclidean_distances

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.config import COHERE_API_KEY, EMBEDDINGS_1887_COHERE
from scripts.evaluation_utils import compute_coverage_row_list
from scripts.retrieval_utils import (
    apply_forced_equal_from_all_collections,
    apply_forced_multicollection,
    apply_no_rerouting,
    apply_query_rerouting,
    compute_recall_metrics_dataframe,
)
from scripts.utils.chroma_utils import load_chroma_collection

# ── Config ────────────────────────────────────────────────────────────────────

# DO 10 seeds
#SEEDS = [42, 123, 2021, 7, 9999, 2022, 31415, 271828, 0, 5555] 
# # 5 seeds for 5-fold-like robustness check (no actual folds)
# SEEDS = [42, 123, 2021, 7, 9999, 2022, 31415, 271828, 0, 5555]
SEEDS = [42, 123, 2021, 7, 9999]
TRAIN_RATIO = 0.5
K_VALUES = [3, 5, 10]
CLUSTER_DIM = 10

CACHE_DIR = os.path.join(BASE_DIR, "data", "RETRIEVER_results_chunking_comparison")
BASELINE = "10000_hierarchical"

STRATEGY_IDS = [
    "10000_hierarchical",
    "baseline_10k",
    "S2_atomic",
    "S3_window3",
    "S4_president",
    "S8_vote",
    "fixed_500",
    "fixed_1000",
    "fixed_2000",
    "S9_topic",
    "S9_topic_noVotes",
    "S10_amendment",
    "S11_hybrid",
    "S11_hybrid_noVotes",
]

CLASS_TO_COLLECTIONS = {
    "Le Gaulois + L'Intransigeant": ["Le Gaulois", "L'Intransigeant"],
    "L'Intransigeant + Les Débats": ["Le Gaulois", "L'Intransigeant", "Les Débats"],
    "Le Gaulois + Les Débats": ["Le Gaulois", "L'Intransigeant", "Les Débats"],
}

ROUTING_CONFIGS = {
    "no_rerouting": {
        "fn": apply_no_rerouting,
        "kwargs": {},
        "docs_col": "documents",
        "sources_col": "sources",
        "label": "No Rerouting",
    },
    "qre": {
        "fn": apply_query_rerouting,
        "kwargs": {"class_to_collections": CLASS_TO_COLLECTIONS},
        "docs_col": "documents_rerouted",
        "sources_col": "sources_rerouted",
        "label": "Query Rerouting (QRe)",
    },
    "ums": {
        "fn": apply_forced_equal_from_all_collections,
        "kwargs": {},
        "docs_col": "documents_forced_equal",
        "sources_col": "sources_forced_equal",
        "label": "UMS (Forced Equal)",
    },
    "qre_ums": {
        "fn": apply_forced_multicollection,
        "kwargs": {"class_to_collections": CLASS_TO_COLLECTIONS},
        "docs_col": "documents_forced",
        "sources_col": "sources_forced",
        "label": "QRe + UMS",
    },
}

# k used for training-phase strategy selection (larger k for multi-collection configs)
SCORE_K_PER_ROUTING = {"no_rerouting": 3, "qre": 3, "ums": 6, "qre_ums": 6}

GOLD_COLLECTION_CONFIGS = [
    {"name": "legaulois_1887_v1", "label": "Le Gaulois"},
    {"name": "lintransigeant_1887_v1", "label": "L'Intransigeant"},
    {"name": "debattre_1887_10000_hierarchical_cohere", "label": "Les Débats"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeCollection:
    """In-memory drop-in replacement for a ChromaDB collection (gold doc lookup)."""
    def __init__(self, id_to_text: dict):
        self._cache = id_to_text

    def get(self, ids):
        return {"documents": [self._cache.get(ids[0], "")]}


def load_jsonl_as_df(strategy_id: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"universal_{strategy_id}.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    for col in ["gold_ids", "retrieved_ids", "distances", "documents", "sources"]:
        if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(json.loads)
    return df


def load_condition_raw(df: pd.DataFrame, strategy_col: str, strategy_dfs: dict) -> pd.DataFrame:
    chunks = []
    for sid in df[strategy_col].unique():
        strat_df = strategy_dfs[sid]
        qids = df[df[strategy_col] == sid]["unique_id"].tolist()
        subset = strat_df[strat_df["unique_id"].isin(qids)]
        chunks.append(subset)
    return pd.concat(chunks, ignore_index=True)


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup():
    """Load questions, embeddings, strategy DFs, and gold collections once."""
    print("Loading questions from ChromaDB...")
    collection_questions = load_chroma_collection(
        EMBEDDINGS_1887_COHERE, api_key=COHERE_API_KEY, collection_name="questions_MH_v1"
    )
    result = collection_questions.get(include=["metadatas", "documents", "embeddings"])
    questions_df = pd.DataFrame(
        {"question": result["documents"], "embeddings": list(result["embeddings"]), "metadata": result["metadatas"]}
    )
    questions_df = pd.concat(
        [questions_df.drop("metadata", axis=1), questions_df["metadata"].apply(pd.Series)], axis=1
    )
    questions_df = questions_df.loc[:, ~questions_df.columns.duplicated()]
    questions_df = questions_df[questions_df["question"] != "AUCUNE QUESTION"].reset_index(drop=True)
    print(f"  {len(questions_df)} questions loaded")

    print("Loading strategy DataFrames from cache...")
    strategy_dfs = {}
    for sid in STRATEGY_IDS:
        strategy_dfs[sid] = load_jsonl_as_df(sid)
    print(f"  {len(strategy_dfs)} strategy DFs loaded")

    print("Loading gold collections into memory...")
    gold_collections = {}
    for cfg in GOLD_COLLECTION_CONFIGS:
        col = load_chroma_collection(EMBEDDINGS_1887_COHERE, api_key=COHERE_API_KEY, collection_name=cfg["name"])
        raw = col.get(include=["documents"])
        id_to_text = {doc_id: doc for doc_id, doc in zip(raw["ids"], raw["documents"])}
        gold_collections[cfg["label"]] = FakeCollection(id_to_text)
        print(f"  {cfg['label']}: {len(id_to_text)} docs cached")

    return questions_df, strategy_dfs, gold_collections


# ── Single fold experiment ────────────────────────────────────────────────────

def run_fold(train_df, test_df, strategy_dfs, gold_collections, seed):
    """Run one fold: cluster train → score → assign test → evaluate. Returns list of result dicts."""
    # Embeddings
    X_train = np.stack(train_df["embeddings"].values)
    X_test = np.stack(test_df["embeddings"].values)

    # UMAP + HDBSCAN on train
    reducer = umap.UMAP(n_components=CLUSTER_DIM, metric="cosine", random_state=seed)
    X_train_umap = reducer.fit_transform(X_train)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric="euclidean")
    train_labels = clusterer.fit_predict(X_train_umap)

    train_df = train_df.copy()
    train_df["cluster"] = train_labels

    # Centroids (including noise cluster -1)
    cluster_ids = sorted(set(train_labels))
    centroids = {}
    for cid in cluster_ids:
        mask = train_labels == cid
        centroids[cid] = X_train_umap[mask].mean(axis=0)

    # cluster → set of training unique_ids
    cluster_train_qids = {
        cid: set(train_df[train_df["cluster"] == cid]["unique_id"].tolist())
        for cid in cluster_ids
    }

    # Score each strategy × cluster × routing on train
    cluster_scores_by_routing = {
        rk: {cid: {} for cid in cluster_ids} for rk in ROUTING_CONFIGS
    }

    for sid in STRATEGY_IDS:
        strat_df = strategy_dfs[sid]
        strat_uids = set(strat_df["unique_id"].tolist())

        for rk, rcfg in ROUTING_CONFIGS.items():
            rk_score_k = SCORE_K_PER_ROUTING[rk]
            routed_full = rcfg["fn"](strat_df.copy(), n_results=rk_score_k, **rcfg["kwargs"])

            for cid, qids in cluster_train_qids.items():
                valid_qids = qids & strat_uids
                if not valid_qids:
                    continue
                subset = routed_full[routed_full["unique_id"].isin(valid_qids)]
                if len(subset) == 0:
                    continue
                cov = compute_coverage_row_list(
                    subset, k=rk_score_k, collections_dict=gold_collections,
                    docs_col=rcfg["docs_col"], sources_col=rcfg["sources_col"],
                )
                cluster_scores_by_routing[rk][cid][sid] = float(cov[2].mean())

    # Best strategy per cluster per routing
    cluster_to_strategy_by_routing = {}
    for rk in ROUTING_CONFIGS:
        cmap = {}
        for cid in cluster_ids:
            valid = cluster_scores_by_routing[rk][cid]
            cmap[cid] = max(valid, key=valid.get) if valid else BASELINE
        cluster_to_strategy_by_routing[rk] = cmap

    # Assign test questions to nearest centroid
    X_test_umap = reducer.transform(X_test)
    cluster_ids_ordered = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[cid] for cid in cluster_ids_ordered])
    dists = euclidean_distances(X_test_umap, centroid_matrix)
    nearest_idx = dists.argmin(axis=1)
    assigned_clusters = [cluster_ids_ordered[i] for i in nearest_idx]

    test_df = test_df.copy()
    test_df["baseline_strategy"] = BASELINE
    for rk in ROUTING_CONFIGS:
        test_df[f"assigned_strategy_{rk}"] = [
            cluster_to_strategy_by_routing[rk].get(cid, BASELINE) for cid in assigned_clusters
        ]

    # Build CONDITION_CONFIGS
    raw_baseline = load_condition_raw(test_df, "baseline_strategy", strategy_dfs)
    raw_tc = {}
    for rk in ROUTING_CONFIGS:
        raw_tc[rk] = load_condition_raw(test_df, f"assigned_strategy_{rk}", strategy_dfs)

    condition_configs = {}
    for rk, rcfg in ROUTING_CONFIGS.items():
        condition_configs[f"baseline_{rk}"] = (
            raw_baseline.copy(), rcfg["fn"], rcfg["kwargs"],
            rcfg["docs_col"], rcfg["sources_col"], f"Global Baseline + {rcfg['label']}",
        )
        condition_configs[f"tc_{rk}"] = (
            raw_tc[rk].copy(), rcfg["fn"], rcfg["kwargs"],
            rcfg["docs_col"], rcfg["sources_col"], f"Task-Conditioned + {rcfg['label']}",
        )

    # Evaluation loop — collect aggregate row AND per-question scores
    results_rows = []
    per_q_rows = []
    for cond_key, (raw_df, routing_fn, routing_kwargs, docs_col, sources_col, display_label) in condition_configs.items():
        rk = cond_key.split("_", 1)[1]  # strip "baseline_" or "tc_" prefix
        row = {"condition": cond_key, "label": display_label}
        for k in K_VALUES:
            n_res = 2 * k if rk in ("ums", "qre_ums") else k
            df_k = routing_fn(raw_df.copy(), n_results=n_res, **routing_kwargs)
            cov = compute_coverage_row_list(
                df_k, k=k, collections_dict=gold_collections,
                docs_col=docs_col, sources_col=sources_col,
            )
            row[f"coverage@{k}"] = float(cov[2].mean())
            recall = compute_recall_metrics_dataframe(df_k, k_values=[k])
            row[f"recall@{k}"] = float(np.mean(recall[f"recall@{k}"]))

            # Per-question scores for significance testing
            for uid, cov_val, rec_val in zip(
                df_k["unique_id"].values, cov[2].values, recall[f"recall@{k}"]
            ):
                per_q_rows.append({
                    "unique_id": uid,
                    "condition": cond_key,
                    f"coverage@{k}": float(cov_val),
                    f"recall@{k}": float(rec_val),
                })
        results_rows.append(row)

    # Collapse per_q_rows: one row per (unique_id, condition) with all k values
    per_q_df = pd.DataFrame(per_q_rows)
    per_q_wide = per_q_df.groupby(["unique_id", "condition"]).first().reset_index()

    return results_rows, per_q_wide


# ── Aggregation & display ─────────────────────────────────────────────────────

def save_results(all_runs, all_per_q, out_dir=None, meta=None):
    """Save per-run aggregate and per-question results to CSV.

    Writes:
      - C2_runs_raw.csv       : one row per (run, condition), aggregate metric values
      - C2_per_question.csv   : one row per (run, unique_id, condition), per-question scores
      - C2_experiment_config.json : experiment hyper-parameters and split sizes
    """
    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "data", "multiseed_results")
    os.makedirs(out_dir, exist_ok=True)

    flat_rows = []
    for run_id, run_results in enumerate(all_runs):
        for row in run_results:
            flat_rows.append({"run": run_id, **row})
    df = pd.DataFrame(flat_rows)
    path = os.path.join(out_dir, "C2_runs_raw.csv")
    df.to_csv(path, index=False)
    print(f"Saved raw run results → {path}")

    pq_frames = []
    for run_id, pq_df in enumerate(all_per_q):
        pq_frames.append(pq_df.assign(run=run_id))
    pq_all = pd.concat(pq_frames, ignore_index=True)
    pq_path = os.path.join(out_dir, "C2_per_question.csv")
    pq_all.to_csv(pq_path, index=False)
    print(f"Saved per-question results → {pq_path}")

    if meta is not None:
        meta["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta_path = os.path.join(out_dir, "C2_experiment_config.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved experiment config  → {meta_path}")

    return df


def aggregate_and_display(all_runs):
    """Compute mean ± std across runs and print results tables."""
    # Flatten all runs into a single DataFrame
    flat_rows = []
    for run_id, run_results in enumerate(all_runs):
        for row in run_results:
            flat_rows.append({"run": run_id, **row})
    df = pd.DataFrame(flat_rows)

    metrics = [f"{m}@{k}" for m in ["coverage", "recall"] for k in K_VALUES]

    # Table 1: absolute values (mean ± std)
    print("\n" + "=" * 90)
    print("TABLE 1: Mean ± Std across folds/seeds (Coverage@k, Recall@k)")
    print("=" * 90)

    agg = df.groupby("condition")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]

    header = f"{'Condition':<35}" + "".join(f"  {m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))

    # Print baseline rows first, then TC rows
    conditions = df["condition"].unique()
    baseline_conds = sorted([c for c in conditions if c.startswith("baseline_")])
    tc_conds = sorted([c for c in conditions if c.startswith("tc_")])

    for cond in baseline_conds + tc_conds:
        if cond not in agg.index:
            continue
        row_str = f"{cond:<35}"
        for m in metrics:
            mean = agg.loc[cond, f"{m}_mean"]
            std = agg.loc[cond, f"{m}_std"]
            row_str += f"  {mean:.3f}±{std:.3f}"
        print(row_str)

    # Table 2: Δ TC − Baseline per routing
    print("\n" + "=" * 90)
    print("TABLE 2: Δ (Task-Conditioned − Global Baseline) per routing")
    print("=" * 90)

    header2 = f"{'Routing':<20}" + "".join(f"  {m:>14}" for m in metrics)
    print(header2)
    print("-" * len(header2))

    for rk, rcfg in ROUTING_CONFIGS.items():
        base_key = f"baseline_{rk}"
        tc_key = f"tc_{rk}"
        if base_key not in agg.index or tc_key not in agg.index:
            continue
        row_str = f"{rcfg['label']:<20}"
        for m in metrics:
            delta_mean = agg.loc[tc_key, f"{m}_mean"] - agg.loc[base_key, f"{m}_mean"]
            # Propagate std: std(A-B) ≈ sqrt(std_A^2 + std_B^2) (conservative)
            delta_std = np.sqrt(agg.loc[tc_key, f"{m}_std"] ** 2 + agg.loc[base_key, f"{m}_std"] ** 2)
            row_str += f"  {delta_mean:+.3f}±{delta_std:.3f}"
        print(row_str)

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    questions_df, strategy_dfs, gold_collections = setup()

    all_questions = questions_df.reset_index(drop=True)
    all_runs = []
    all_per_q = []
    total = len(SEEDS)

    for run_num, seed in enumerate(SEEDS, 1):
        rng = np.random.default_rng(seed)
        shuffled_idx = rng.permutation(len(all_questions))
        n_train = int(len(all_questions) * TRAIN_RATIO)
        train_df = all_questions.iloc[shuffled_idx[:n_train]].reset_index(drop=True)
        test_df  = all_questions.iloc[shuffled_idx[n_train:]].reset_index(drop=True)

        print(f"\n[{run_num}/{total}] Seed={seed} "
              f"(train={len(train_df)}, test={len(test_df)})")

        fold_results, per_q = run_fold(train_df, test_df, strategy_dfs, gold_collections, seed)
        all_runs.append(fold_results)
        all_per_q.append(per_q)
        print(f"  Done ({len(fold_results)} conditions evaluated)")

    n_total = len(all_questions)
    meta = {
        "script": "run_C2_multiseed_kfold.py",
        "seeds": SEEDS,
        "train_ratio": TRAIN_RATIO,
        "n_total": n_total,
        "n_train": int(n_total * TRAIN_RATIO),
        "n_test": n_total - int(n_total * TRAIN_RATIO),
        "k_values": K_VALUES,
        "score_k_per_routing": SCORE_K_PER_ROUTING,
        "routing_configs": list(ROUTING_CONFIGS.keys()),
        "strategy_ids": STRATEGY_IDS,
        "n_strategies": len(STRATEGY_IDS),
        "baseline_strategy": BASELINE,
        "umap_params": {"n_components": CLUSTER_DIM, "metric": "cosine"},
        "hdbscan_params": {"min_cluster_size": 5, "min_samples": 5},
    }
    save_results(all_runs, all_per_q, meta=meta)
    aggregate_and_display(all_runs)


if __name__ == "__main__":
    main()
