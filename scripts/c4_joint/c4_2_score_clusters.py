"""C4 step 2 — score every (chunking, metadata) pair per cluster, per routing.

Memory strategy:
- Fit UMAP/HDBSCAN once, persist (reducer, centroids, train labels) to disk.
- For each pair: load JSONL → for each routing apply once on the full df → slice
  per cluster → compute Coverage → drop the dataframe before next pair.
- Per-pair scores are checkpointed to a single Parquet file so the run can
  resume from arbitrary interruptions.

Outputs (under `data/JOINT_results/_artifacts/`):
    reducer.joblib                    UMAP model (used at test time)
    centroids.joblib                  dict: cluster_id -> centroid (CLUSTER_DIM-d)
    train_labels.joblib               (train_unique_ids, labels) arrays
    cluster_scores.parquet            long table: routing,cluster,chunking,metadata,score
    cluster_to_pair.json              {routing: {cluster_id: [chunking, metadata]}}

Usage:
    python -m scripts.c4_joint.c4_2_score_clusters
    python -m scripts.c4_joint.c4_2_score_clusters --rescore        # ignore parquet
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.c4_joint.c4_common import (
    ARTIFACTS_DIR, BASELINE_PAIR, JOINT_PAIRS,
    ROUTING_CONFIGS, SCORE_K_PER_ROUTING,
    cluster_train, load_gold_collections, load_joint,
    load_questions_df, train_test_split_ids,
)
from scripts.evaluation_utils import compute_coverage_row_list

REDUCER_PATH    = os.path.join(ARTIFACTS_DIR, 'reducer.joblib')
CENTROIDS_PATH  = os.path.join(ARTIFACTS_DIR, 'centroids.joblib')
LABELS_PATH     = os.path.join(ARTIFACTS_DIR, 'train_labels.joblib')
SCORES_PATH     = os.path.join(ARTIFACTS_DIR, 'cluster_scores.parquet')
PAIR_MAP_PATH   = os.path.join(ARTIFACTS_DIR, 'cluster_to_pair.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rescore', action='store_true',
                        help='Ignore existing parquet checkpoint and recompute all scores.')
    args = parser.parse_args()

    # ---- Questions + split + clustering -------------------------------------
    print('[c4_2] loading questions...')
    questions_df = load_questions_df()
    train_ids, test_ids = train_test_split_ids(questions_df)
    train_df = questions_df[questions_df['unique_id'].isin(train_ids)].reset_index(drop=True)
    print(f'[c4_2] train={len(train_df)}  test={len(test_ids)}')

    # Drop heavy embedding storage once we no longer need questions_df.
    uid_to_sources = questions_df[['unique_id', 'source_1', 'source_2']].copy()
    del questions_df
    gc.collect()

    print('[c4_2] clustering training questions (UMAP + HDBSCAN)...')
    reducer, X_train_umap, labels, centroids = cluster_train(train_df)
    n_clusters = len(set(labels) - {-1})
    print(f'[c4_2] {n_clusters} clusters (+ noise), {len(centroids)} centroids')

    cluster_train_qids = {
        cid: set(train_df.loc[labels == cid, 'unique_id'].tolist())
        for cid in centroids
    }
    train_unique_ids = train_df['unique_id'].to_numpy()
    del train_df, X_train_umap
    gc.collect()

    # Persist clustering artifacts so step 3 doesn't have to recompute.
    joblib.dump(reducer, REDUCER_PATH)
    joblib.dump(centroids, CENTROIDS_PATH)
    joblib.dump({'unique_ids': train_unique_ids, 'labels': labels}, LABELS_PATH)
    print(f'[c4_2] artifacts saved → {ARTIFACTS_DIR}')

    # ---- Load gold collections (needed by coverage) -------------------------
    print('[c4_2] loading gold collections...')
    gold_collections = load_gold_collections()

    # ---- Stream-score each pair --------------------------------------------
    already_done = set()
    rows_existing = []
    if (not args.rescore) and os.path.exists(SCORES_PATH):
        prev = pd.read_parquet(SCORES_PATH)
        rows_existing = prev.to_dict('records')
        already_done = {(r['routing'], r['chunking'], r['metadata']) for r in rows_existing}
        # We need every (routing, chunking, metadata) triple — only skip pair if
        # ALL routings already scored.
        per_pair_routings = {}
        for r in rows_existing:
            per_pair_routings.setdefault((r['chunking'], r['metadata']), set()).add(r['routing'])
        fully_done = {p for p, rks in per_pair_routings.items()
                      if rks >= set(ROUTING_CONFIGS)}
        print(f'[c4_2] checkpoint: {len(fully_done)} pairs fully scored, '
              f'{len(JOINT_PAIRS) - len(fully_done)} remaining.')
    else:
        fully_done = set()

    score_rows = list(rows_existing)
    flush_every = 20
    n_new = 0

    for pair in tqdm(JOINT_PAIRS, desc='[c4_2] pairs'):
        if pair in fully_done:
            continue
        chunking_id, metadata_id = pair
        df = load_joint(chunking_id, metadata_id)
        if df is None:
            # Missing pair → NaN for every (routing, cluster).
            for rk in ROUTING_CONFIGS:
                for cid in cluster_train_qids:
                    score_rows.append({
                        'routing': rk, 'cluster': int(cid),
                        'chunking': chunking_id, 'metadata': metadata_id,
                        'score': float('nan'),
                    })
            continue

        # Merge source_1/source_2 once (compute_coverage_row_list needs them).
        if 'source_1' not in df.columns:
            df = df.merge(uid_to_sources, on='unique_id', how='left')
        strat_uids = set(df['unique_id'].tolist())

        for rk, rcfg in ROUTING_CONFIGS.items():
            if (rk, chunking_id, metadata_id) in already_done:
                continue
            k = SCORE_K_PER_ROUTING[rk]
            routed_full = rcfg['fn'](df, n_results=k, **rcfg['kwargs'])

            for cid, qids in cluster_train_qids.items():
                valid_qids = qids & strat_uids
                subset = routed_full[routed_full['unique_id'].isin(valid_qids)]
                if len(subset) == 0:
                    score = float('nan')
                else:
                    cov = compute_coverage_row_list(
                        subset, k=k, collections_dict=gold_collections,
                        docs_col=rcfg['docs_col'], sources_col=rcfg['sources_col'],
                    )
                    score = float(cov[2].mean())
                score_rows.append({
                    'routing': rk, 'cluster': int(cid),
                    'chunking': chunking_id, 'metadata': metadata_id,
                    'score': score,
                })

            del routed_full

        del df
        gc.collect()
        n_new += 1
        if n_new % flush_every == 0:
            pd.DataFrame(score_rows).to_parquet(SCORES_PATH, index=False)

    # Final flush
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_parquet(SCORES_PATH, index=False)
    print(f'[c4_2] scored rows: {len(scores_df)}  → {SCORES_PATH}')

    # ---- Pick best pair per (routing, cluster) -----------------------------
    cluster_to_pair = {}
    for rk in ROUTING_CONFIGS:
        sub = scores_df[scores_df['routing'] == rk]
        cmap = {}
        for cid, grp in sub.groupby('cluster'):
            valid = grp.dropna(subset=['score'])
            if len(valid) == 0:
                cmap[int(cid)] = list(BASELINE_PAIR)
            else:
                top = valid.loc[valid['score'].idxmax()]
                cmap[int(cid)] = [top['chunking'], top['metadata']]
        cluster_to_pair[rk] = cmap

    with open(PAIR_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(cluster_to_pair, f, indent=2)
    print(f'[c4_2] cluster→pair map saved → {PAIR_MAP_PATH}')

    # Quick summary
    print('\n[c4_2] Summary (no_rerouting):')
    nr = cluster_to_pair['no_rerouting']
    n_same = sum(tuple(v) == BASELINE_PAIR for v in nr.values())
    print(f'  {n_same}/{len(nr)} clusters pick the baseline pair '
          f'({BASELINE_PAIR[0]}, {BASELINE_PAIR[1]})')


if __name__ == '__main__':
    main()
