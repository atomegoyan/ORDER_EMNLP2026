"""C4 step 1 — build joint (chunking, metadata) retrieval JSONLs.

Memory strategy:
- Iterate one chunking at a time.
- Load the chunking's universal JSONL ONCE; reuse as `base_df` for every metadata
  strategy under that chunking, then drop before moving to the next chunking.
- Stream-write each built joint JSONL to disk and del immediately.
- Skip pairs whose result is already on disk (joint cache, chunking cache for
  baseline metadata, or C3 cache for hierarchical chunking).

Usage:
    python -m scripts.c4_joint.c4_1_build_cache              # all 196 pairs
    python -m scripts.c4_joint.c4_1_build_cache --chunkings S10_amendment fixed_500
    python -m scripts.c4_joint.c4_1_build_cache --metadatas exclude_noisy
    python -m scripts.c4_joint.c4_1_build_cache --rebuild    # ignore existing joint cache
"""
from __future__ import annotations

import argparse
import gc
import os

import pandas as pd
from tqdm import tqdm

from scripts.c4_joint.c4_common import (
    BASELINE_CHUNKING, BASELINE_METADATA,
    CHUNKING_IDS, METADATA_IDS, FILTER_STRATEGIES,
    N_PER_COLLECTION,
    joint_cache_path, load_chunking_universal, load_c3_metadata, save_joint,
    load_questions_df,
)


def build_joint_df(chunking_id, metadata_id, where_filter, needs_metas, rerank_fn,
                   base_df, debat_collection, embedding_by_uid):
    """Re-query chunking-specific Les Débats collection with metadata filter / rerank.
    Newspaper rows are copied as-is from base_df.

    Returns the built dataframe (caller is responsible for writing + freeing).
    """
    include_fields = ['documents', 'distances']
    if needs_metas or rerank_fn is not None:
        include_fields.append('metadatas')

    out_rows = []
    for _, row in tqdm(base_df.iterrows(), total=len(base_df),
                       desc=f'{chunking_id}·{metadata_id}', leave=False):
        np_ids, np_dists, np_docs, np_srcs, np_metas = [], [], [], [], []
        for i, src in enumerate(row['sources']):
            if src != 'Les Débats':
                np_ids.append(row['retrieved_ids'][i])
                np_dists.append(row['distances'][i])
                np_docs.append(row['documents'][i])
                np_srcs.append(src)
                np_metas.append(None)

        uid = row['unique_id']
        emb = embedding_by_uid.get(uid)

        db_ids, db_dists, db_docs, db_metas, db_srcs = [], [], [], [], []
        if emb is not None:
            q = {'query_embeddings': [list(emb)], 'n_results': N_PER_COLLECTION,
                 'include': include_fields}
            if where_filter is not None:
                q['where'] = where_filter
            try:
                res = debat_collection.query(**q)
                db_ids   = res['ids'][0]
                db_dists = res['distances'][0]
                db_docs  = res['documents'][0]
                db_metas = res['metadatas'][0] if 'metadatas' in res else [None] * len(db_ids)
                db_srcs  = ['Les Débats'] * len(db_ids)
            except Exception:
                pass

        all_ids   = np_ids   + db_ids
        all_dists = np_dists + db_dists
        all_docs  = np_docs  + db_docs
        all_srcs  = np_srcs  + db_srcs
        all_metas = np_metas + db_metas

        if rerank_fn is not None and len(all_ids) > 0:
            r_ids, r_dists, _, r_srcs = rerank_fn(all_ids, all_dists, all_metas, all_srcs)
            id_to_doc = dict(zip(all_ids, all_docs))
            r_docs = [id_to_doc.get(rid, '') for rid in r_ids]
        else:
            combined = sorted(zip(all_ids, all_dists, all_docs, all_srcs), key=lambda x: x[1])
            r_ids   = [x[0] for x in combined]
            r_dists = [x[1] for x in combined]
            r_docs  = [x[2] for x in combined]
            r_srcs  = [x[3] for x in combined]

        out_rows.append({
            'unique_id':       uid,
            'gold_ids':        row['gold_ids'],
            'predicted_class': row.get('predicted_class', ''),
            'retrieved_ids':   r_ids,
            'distances':       r_dists,
            'documents':       r_docs,
            'sources':         r_srcs,
        })

    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunkings', nargs='+', default=None,
                        help='Subset of chunking IDs to build (default: all 14).')
    parser.add_argument('--metadatas', nargs='+', default=None,
                        help='Subset of metadata IDs to build (default: all 14).')
    parser.add_argument('--rebuild', action='store_true',
                        help='Re-build even if joint cache already exists.')
    args = parser.parse_args()

    chunkings = args.chunkings or CHUNKING_IDS
    metadatas = args.metadatas or METADATA_IDS

    # Lazy imports — only needed when we actually re-query Chroma.
    questions_df = load_questions_df()
    embedding_by_uid = dict(zip(questions_df['unique_id'],
                                 questions_df['embeddings'].tolist()))
    print(f'[c4_1] questions loaded: n={len(questions_df)}')

    # Cache the Chroma collection per chunking (loaded lazily, then dropped).
    from scripts.config import COHERE_API_KEY, EMBEDDINGS_1887_COHERE
    from scripts.utils.chroma_utils import load_chroma_collection

    n_built = n_skipped = n_failed = 0
    for chunking_id in chunkings:
        base_df = load_chunking_universal(chunking_id)
        if base_df is None:
            print(f'[c4_1] [WARN] no chunking cache for {chunking_id} — skipping.')
            n_failed += len(metadatas)
            continue

        debat_collection = None
        for metadata_id in metadatas:
            target_path = joint_cache_path(chunking_id, metadata_id)
            if (not args.rebuild) and os.path.exists(target_path):
                n_skipped += 1
                continue

            # No need to write joint cache for the two "free" cases — downstream
            # scripts read those directly via load_joint().
            if metadata_id == BASELINE_METADATA:
                n_skipped += 1
                continue
            if chunking_id == BASELINE_CHUNKING and (not args.rebuild) \
                    and load_c3_metadata(metadata_id) is not None:
                n_skipped += 1
                continue

            if debat_collection is None:
                col_name = f'debattre_1887_{chunking_id}_cohere'
                try:
                    debat_collection = load_chroma_collection(
                        EMBEDDINGS_1887_COHERE, api_key=COHERE_API_KEY,
                        collection_name=col_name,
                    )
                except Exception as e:
                    print(f'[c4_1] [WARN] cannot load {col_name}: {e}')
                    debat_collection = False  # sentinel: "tried and failed"

            if debat_collection is False:
                n_failed += 1
                continue

            where_filter, needs_metas, rerank_fn = FILTER_STRATEGIES[metadata_id]
            print(f'[c4_1] building {chunking_id} × {metadata_id} ...')
            df_built = build_joint_df(
                chunking_id, metadata_id, where_filter, needs_metas, rerank_fn,
                base_df, debat_collection, embedding_by_uid,
            )
            save_joint(df_built, chunking_id, metadata_id)
            del df_built
            gc.collect()
            n_built += 1

        del base_df, debat_collection
        gc.collect()

    print(f'\n[c4_1] done. built={n_built}  skipped={n_skipped}  failed={n_failed}')


if __name__ == '__main__':
    main()
