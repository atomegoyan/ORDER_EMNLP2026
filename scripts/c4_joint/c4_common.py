"""Shared configuration + I/O helpers for the C4 joint task-conditioned pipeline.

Memory-conscious design: callers should consume joint dataframes one pair at a time
and free them between iterations rather than holding all 196 in memory.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ---- Make the repo importable when this module is loaded as a script -----------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.retrieval_utils import (  # noqa: E402
    apply_no_rerouting,
    apply_query_rerouting,
    apply_forced_multicollection,
    apply_forced_equal_from_all_collections,
    rerank_type_boost,
    rerank_speaker_boost,
    rerank_combined,
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = _REPO_ROOT
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHUNKING_CACHE_DIR = os.path.join(DATA_DIR, 'RETRIEVER_results_chunking_comparison')
META_CACHE_DIR     = os.path.join(DATA_DIR, 'METADATA_results_NER')
JOINT_CACHE_DIR    = os.path.join(DATA_DIR, 'JOINT_results')
ARTIFACTS_DIR      = os.path.join(DATA_DIR, 'JOINT_results', '_artifacts')
os.makedirs(JOINT_CACHE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ============================================================
# Experiment configuration
# ============================================================
SEED = 42
TRAIN_RATIO = 0.75
K_VALUES = [3, 5, 10]
CLUSTER_DIM = 10
N_PER_COLLECTION = 15

BASELINE_CHUNKING = '10000_hierarchical'
BASELINE_METADATA = 'baseline'
BASELINE_PAIR = (BASELINE_CHUNKING, BASELINE_METADATA)

CHUNKING_IDS = [
    '10000_hierarchical', 'baseline_10k',
    'S2_atomic', 'S3_window3', 'S4_president', 'S8_vote',
    'fixed_500', 'fixed_1000', 'fixed_2000',
    'S9_topic', 'S9_topic_noVotes',
    'S10_amendment',
    'S11_hybrid', 'S11_hybrid_noVotes',
]

NOISY_DOC_TYPES = [
    'vote_list', 'absence_list', 'fragment', 'chamber_header',
    'scrutin', 'excuse', 'conge', 'compte_rendu_header', 'tirage_sort',
]
DEBATE_TYPES = ['debate', 'legal_text', 'law_presentation', 'law_adoption',
                'petition', 'sommaire']

FILTER_STRATEGIES = {
    'baseline':                  (None, False, None),
    'exclude_noisy':             ({'doc_type': {'$nin': NOISY_DOC_TYPES}}, False, None),
    'debates_only':              ({'doc_type': 'debate'}, False, None),
    'debates_and_legal':         ({'doc_type': {'$in': DEBATE_TYPES}}, False, None),
    'min_speakers_2':            ({'speaker_count': {'$gte': 2}}, False, None),
    'min_length_500':            ({'doc_length': {'$gte': 500}}, False, None),
    'exclude_noisy_min_length':  (
        {'$and': [{'doc_type': {'$nin': NOISY_DOC_TYPES}}, {'doc_length': {'$gte': 200}}]},
        False, None,
    ),
    'rerank_type_boost':         (None, True, rerank_type_boost),
    'rerank_speaker_boost':      (None, True, rerank_speaker_boost),
    'rerank_combined':           (None, True, rerank_combined),
    'exclude_noisy__rerank_speaker': (
        {'doc_type': {'$nin': NOISY_DOC_TYPES}}, True, rerank_speaker_boost,
    ),
    'debates_legal__rerank_speaker': (
        {'doc_type': {'$in': DEBATE_TYPES}}, True, rerank_speaker_boost,
    ),
    'debates_legal__rerank_combined': (
        {'doc_type': {'$in': DEBATE_TYPES}}, True, rerank_combined,
    ),
    'triple_filter__rerank_combined': (
        {'$and': [
            {'doc_type': {'$in': DEBATE_TYPES}},
            {'speaker_count': {'$gte': 2}},
            {'doc_length': {'$gte': 200}},
        ]},
        True, rerank_combined,
    ),
}
METADATA_IDS = list(FILTER_STRATEGIES.keys())
JOINT_PAIRS = [(c, m) for c in CHUNKING_IDS for m in METADATA_IDS]

CLASS_TO_COLLECTIONS = {
    "Le Gaulois + L'Intransigeant": ['Le Gaulois', "L'Intransigeant"],
    "L'Intransigeant + Les Débats": ['Le Gaulois', "L'Intransigeant", 'Les Débats'],
    "Le Gaulois + Les Débats":      ['Le Gaulois', "L'Intransigeant", 'Les Débats'],
}

ROUTING_CONFIGS = {
    'no_rerouting': {
        'fn': apply_no_rerouting, 'kwargs': {},
        'docs_col': 'documents', 'sources_col': 'sources',
        'label': 'No Rerouting',
    },
    'qre': {
        'fn': apply_query_rerouting, 'kwargs': {'class_to_collections': CLASS_TO_COLLECTIONS},
        'docs_col': 'documents_rerouted', 'sources_col': 'sources_rerouted',
        'label': 'Query Rerouting (QRe)',
    },
    'ums': {
        'fn': apply_forced_equal_from_all_collections, 'kwargs': {},
        'docs_col': 'documents_forced_equal', 'sources_col': 'sources_forced_equal',
        'label': 'UMS (Forced Equal)',
    },
    'qre_ums': {
        'fn': apply_forced_multicollection, 'kwargs': {'class_to_collections': CLASS_TO_COLLECTIONS},
        'docs_col': 'documents_forced', 'sources_col': 'sources_forced',
        'label': 'QRe + UMS',
    },
}
SCORE_K_PER_ROUTING = {'no_rerouting': 3, 'qre': 3, 'ums': 6, 'qre_ums': 6}

GOLD_COLLECTION_CONFIGS = [
    {'name': 'legaulois_1887_v1',                       'label': 'Le Gaulois'},
    {'name': 'lintransigeant_1887_v1',                  'label': "L'Intransigeant"},
    {'name': 'debattre_1887_10000_hierarchical_cohere', 'label': 'Les Débats'},
]

# ============================================================
# JSONL helpers
# ============================================================
_LIST_COLS = ['gold_ids', 'retrieved_ids', 'distances', 'documents', 'sources']


def _parse_lists(df: pd.DataFrame) -> pd.DataFrame:
    for col in _LIST_COLS:
        if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(json.loads)
    return df


def _serialize(val):
    return val if isinstance(val, list) else json.loads(val)


def joint_cache_path(chunking_id: str, metadata_id: str) -> str:
    return os.path.join(JOINT_CACHE_DIR, f'joint_{chunking_id}__{metadata_id}.jsonl')


def load_jsonl_as_df(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return _parse_lists(pd.DataFrame(rows))


def load_chunking_universal(chunking_id: str) -> Optional[pd.DataFrame]:
    return load_jsonl_as_df(os.path.join(CHUNKING_CACHE_DIR, f'universal_{chunking_id}.jsonl'))


def load_c3_metadata(metadata_id: str) -> Optional[pd.DataFrame]:
    return load_jsonl_as_df(os.path.join(META_CACHE_DIR, f'meta_{metadata_id}.jsonl'))


def load_joint(chunking_id: str, metadata_id: str) -> Optional[pd.DataFrame]:
    """Resolve a joint pair df from the lightest available source.

    Priority:
      1. Joint cache on disk
      2. (*, baseline) → reuse chunking universal JSONL
      3. (10000_hierarchical, *) → reuse C3 metadata JSONL
      Returns None if nothing can be resolved.
    """
    df = load_jsonl_as_df(joint_cache_path(chunking_id, metadata_id))
    if df is not None:
        return df
    if metadata_id == BASELINE_METADATA:
        return load_chunking_universal(chunking_id)
    if chunking_id == BASELINE_CHUNKING:
        return load_c3_metadata(metadata_id)
    return None


def save_joint(df: pd.DataFrame, chunking_id: str, metadata_id: str) -> None:
    path = joint_cache_path(chunking_id, metadata_id)
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {
                'unique_id':       row['unique_id'],
                'gold_ids':        _serialize(row['gold_ids']),
                'predicted_class': row.get('predicted_class', ''),
                'retrieved_ids':   _serialize(row['retrieved_ids']),
                'distances':       _serialize(row['distances']),
                'documents':       _serialize(row['documents']),
                'sources':         _serialize(row['sources']),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ============================================================
# Questions loader + split + clustering (single source of truth)
# ============================================================
def load_questions_df() -> pd.DataFrame:
    """Load the questions collection from local Chroma. No API calls."""
    from scripts.config import COHERE_API_KEY, EMBEDDINGS_1887_COHERE
    from scripts.utils.chroma_utils import load_chroma_collection

    col = load_chroma_collection(
        EMBEDDINGS_1887_COHERE, api_key=COHERE_API_KEY,
        collection_name='questions_MH_v1',
    )
    result = col.get(include=['metadatas', 'documents', 'embeddings'])
    df = pd.DataFrame({
        'question':   result['documents'],
        'embeddings': list(result['embeddings']),
        'metadata':   result['metadatas'],
    })
    df = pd.concat([df.drop('metadata', axis=1), df['metadata'].apply(pd.Series)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df['question'] != 'AUCUNE QUESTION'].reset_index(drop=True)
    if 'gold_ids' in df.columns and isinstance(df['gold_ids'].iloc[0], str):
        df['gold_ids'] = df['gold_ids'].apply(json.loads)
    return df


def train_test_split_ids(questions_df: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    shuffled_idx = rng.permutation(len(questions_df))
    n_train = int(len(questions_df) * TRAIN_RATIO)
    train_ids = set(questions_df.iloc[shuffled_idx[:n_train]]['unique_id'].tolist())
    test_ids  = set(questions_df.iloc[shuffled_idx[n_train:]]['unique_id'].tolist())
    return train_ids, test_ids


def cluster_train(train_df: pd.DataFrame):
    """Fit UMAP(CLUSTER_DIM) + HDBSCAN on training embeddings.

    Returns (reducer, X_train_umap, labels, centroids).
    """
    import hdbscan
    import umap

    X_train = np.array(train_df['embeddings'].tolist())
    reducer = umap.UMAP(n_components=CLUSTER_DIM, metric='cosine', random_state=SEED)
    X_train_umap = reducer.fit_transform(X_train)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric='euclidean')
    labels = clusterer.fit_predict(X_train_umap)
    centroids = {}
    for cid in set(labels):
        centroids[cid] = X_train_umap[labels == cid].mean(axis=0)
    return reducer, X_train_umap, labels, centroids


# ============================================================
# Gold collections
# ============================================================
def load_gold_collections():
    from scripts.config import COHERE_API_KEY, EMBEDDINGS_1887_COHERE
    from scripts.utils.chroma_utils import load_chroma_collection

    out = {}
    for cfg in GOLD_COLLECTION_CONFIGS:
        out[cfg['label']] = load_chroma_collection(
            EMBEDDINGS_1887_COHERE, api_key=COHERE_API_KEY,
            collection_name=cfg['name'],
        )
    return out
