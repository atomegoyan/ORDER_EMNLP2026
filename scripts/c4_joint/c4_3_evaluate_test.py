"""C4 step 3 — evaluate test set under 8 conditions (4 routings × {baseline, TC}).

Memory strategy:
- Reuse the reducer/centroids/cluster_to_pair from step 2 (no re-clustering).
- For each routing, group test questions by the pair selected for their cluster
  and load each needed joint JSONL exactly once.
- Coverage & recall are computed per (routing × condition × k); per-question
  retrieved lists are not retained after each pair is consumed.

Outputs:
    data/JOINT_results/_artifacts/results_table.csv   long table (routing, condition, k, metric, value)
    data/JOINT_results/_artifacts/wins_summary.csv    pair frequencies per routing
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from scripts.c4_joint.c4_common import (
    ARTIFACTS_DIR, BASELINE_PAIR, K_VALUES,
    ROUTING_CONFIGS, SCORE_K_PER_ROUTING,
    load_gold_collections, load_joint,
    load_questions_df, train_test_split_ids,
)
from scripts.evaluation_utils import compute_coverage_row_list
from scripts.retrieval_utils import compute_recall_metrics_dataframe

REDUCER_PATH   = os.path.join(ARTIFACTS_DIR, 'reducer.joblib')
CENTROIDS_PATH = os.path.join(ARTIFACTS_DIR, 'centroids.joblib')
PAIR_MAP_PATH  = os.path.join(ARTIFACTS_DIR, 'cluster_to_pair.json')
RESULTS_PATH   = os.path.join(ARTIFACTS_DIR, 'results_table.csv')
WINS_PATH      = os.path.join(ARTIFACTS_DIR, 'wins_summary.csv')

# n_results to request from each routing for evaluation at a given k.
# - For routings that retrieve from a single collection (no_rerouting, qre),
#   we request exactly k documents.
# - For UMS-based routings that split across collections, we follow the
#   "k_0 = 2*k" convention used elsewhere in the paper (2 docs per source
#   when 2 collections are active, ~k docs per source split into n_collections).
def _n_results_for(routing_key: str, k: int) -> int:
    if routing_key in ('ums', 'qre_ums'):
        return 2 * k
    return k


def _assign_clusters(X_test_umap, centroids):
    """Return array of cluster ids (one per test question), -1 excluded."""
    cluster_ids = [c for c in centroids if c != -1]
    if not cluster_ids:
        cluster_ids = list(centroids)
    C = np.array([centroids[c] for c in cluster_ids])
    d = euclidean_distances(X_test_umap, C)
    return np.array([cluster_ids[i] for i in d.argmin(axis=1)])


def _load_joint_with_sources(pair, uid_to_sources):
    """Load joint(pair) and ensure source_1/source_2 columns are present."""
    df = load_joint(*pair)
    if df is None:
        return None
    if 'source_1' not in df.columns:
        df = df.merge(uid_to_sources, on='unique_id', how='left')
    return df


def _apply_routing(df, uids, routing_key, n_results):
    """Apply routing fn at given n_results and filter to uids."""
    rcfg = ROUTING_CONFIGS[routing_key]
    routed = rcfg['fn'](df, n_results=n_results, **rcfg['kwargs'])
    return routed[routed['unique_id'].isin(uids)].copy()


def _evaluate_at_k(routed_df, routing_key, gold_collections, k):
    """Compute Coverage@k and Recall@k on a routed_df produced with n_results
    sized for this specific k. Returns {'Coverage@k': v, 'Recall@k': v}."""
    rcfg = ROUTING_CONFIGS[routing_key]
    recalls = compute_recall_metrics_dataframe(routed_df, k_values=[k])
    cov = compute_coverage_row_list(
        routed_df, k=k, collections_dict=gold_collections,
        docs_col=rcfg['docs_col'], sources_col=rcfg['sources_col'],
    )
    return {
        f'Recall@{k}':   float(np.mean(recalls[f'recall@{k}'])),
        f'Coverage@{k}': float(cov[2].mean()),
    }


def main():
    # ---- Load artifacts ----------------------------------------------------
    reducer    = joblib.load(REDUCER_PATH)
    centroids  = joblib.load(CENTROIDS_PATH)
    with open(PAIR_MAP_PATH, encoding='utf-8') as f:
        cluster_to_pair = {rk: {int(c): tuple(p) for c, p in m.items()}
                           for rk, m in json.load(f).items()}

    # ---- Test split --------------------------------------------------------
    print('[c4_3] loading questions...')
    questions_df = load_questions_df()
    _, test_ids = train_test_split_ids(questions_df)
    test_df = questions_df[questions_df['unique_id'].isin(test_ids)].reset_index(drop=True)
    uid_to_sources = questions_df[['unique_id', 'source_1', 'source_2']]
    print(f'[c4_3] test n={len(test_df)}')

    # ---- Project to UMAP & assign clusters --------------------------------
    X_test = np.array(test_df['embeddings'].tolist())
    X_test_umap = reducer.transform(X_test)
    test_clusters = _assign_clusters(X_test_umap, centroids)
    test_df = test_df[['unique_id']].assign(cluster=test_clusters)
    del questions_df, X_test, X_test_umap
    import gc; gc.collect()

    # ---- Gold collections --------------------------------------------------
    gold_collections = load_gold_collections()

    # ---- Per-routing: group test UIDs by selected pair --------------------
    rows = []
    wins_rows = []
    all_test_uids = set(test_df['unique_id'])
    for rk in ROUTING_CONFIGS:
        cmap = cluster_to_pair[rk]

        # task-conditioned: pair -> set of uids
        pair_to_uids = defaultdict(set)
        for _, r in test_df.iterrows():
            pair = cmap.get(int(r['cluster']), BASELINE_PAIR)
            pair_to_uids[pair].add(r['unique_id'])
        for pair, uids in pair_to_uids.items():
            wins_rows.append({
                'routing': rk, 'pair': f'{pair[0]}__{pair[1]}',
                'chunking': pair[0], 'metadata': pair[1],
                'n_test_questions': len(uids),
            })

        base_metrics: dict = {}
        tc_metrics: dict = {}

        # Apply routing once per k (UMS depth depends on k); cache loaded joints.
        for k in K_VALUES:
            n_res = _n_results_for(rk, k)

            # ----- baseline condition -----
            base_df = _load_joint_with_sources(BASELINE_PAIR, uid_to_sources)
            if base_df is not None:
                base_routed = _apply_routing(base_df, all_test_uids, rk, n_res)
                if len(base_routed) > 0:
                    base_metrics.update(_evaluate_at_k(base_routed, rk, gold_collections, k))
                del base_routed
            del base_df; gc.collect()

            # ----- task-conditioned condition -----
            tc_chunks = []
            for pair, uids in pair_to_uids.items():
                df = _load_joint_with_sources(pair, uid_to_sources)
                if df is None:
                    continue
                routed = _apply_routing(df, uids, rk, n_res)
                if len(routed) > 0:
                    tc_chunks.append(routed)
                del df; gc.collect()
            if tc_chunks:
                tc_routed = pd.concat(tc_chunks, ignore_index=True)
                tc_metrics.update(_evaluate_at_k(tc_routed, rk, gold_collections, k))
                del tc_routed
            del tc_chunks; gc.collect()

        for cond, metrics in (('baseline', base_metrics), ('task_conditioned', tc_metrics)):
            for metric, val in metrics.items():
                rows.append({
                    'routing': rk, 'condition': cond,
                    'metric': metric, 'value': val,
                })
        print(f'[c4_3] {rk}: baseline={base_metrics}  tc={tc_metrics}')

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f'[c4_3] results → {RESULTS_PATH}')

    wins_df = pd.DataFrame(wins_rows)
    wins_df.to_csv(WINS_PATH, index=False)
    print(f'[c4_3] wins summary → {WINS_PATH}')

    # ---- Pretty pivot ------------------------------------------------------
    print('\n[c4_3] Coverage / Recall (rows = routing × condition, cols = metric)')
    pivot = results_df.pivot_table(
        index=['routing', 'condition'], columns='metric', values='value',
    )
    print(pivot.round(4).to_string())


if __name__ == '__main__':
    main()
