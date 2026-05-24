# Standard library imports
import os
import sys
import json
import time
import logging
import random

# Third-party imports
import pandas as pd
import numpy
import regex as re
from tqdm import tqdm
import chromadb

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Add parent directory to path to import local scripts folder
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Local imports
from scripts.config import COHERE_API_KEY, DATA_DIR, IMAGE_DIR, HIPPORAG_DIR, EMBEDDINGS_1887_COHERE, QUESTIONS_FILE, HIPPORAG_DIR_MH
from scripts.utils.chroma_utils import query_collection, list_collections, load_chroma_collection
from scripts.utils.json_reading import read_jsonl_basic

# Set environment variable for HippoRAG (uses OpenAI client internally)
os.environ['OPENAI_API_KEY'] = COHERE_API_KEY


#

# ── Load document collections ─────────────────────────────────────────────────────

COLLECTION_CONFIGS = [
    {"name": "legaulois_1887_v1",                       "label": "Le Gaulois"},
    {"name": "lintransigeant_1887_v1",                  "label": "L'Intransigeant"},
    {"name": "debattre_1887_10000_hierarchical_cohere", "label": "Les Débats"},
]

# Questions files configuration - modify this section to add/remove question types
QUESTION_DATASETS = [
    {
        'name': 'Generic_LeGaulois_Debats',
        'file': 'MH_Generic_legaulois_v1.jsonl',
        'column_mapping': {
            'id_1': 'journal_id',
            'id_2': 'debat_id',
            'doc_1': 'journal_doc',
            'doc_2': 'debat_doc',
            'metadata_1': 'journal_metadata',
            'metadata_2': 'debat_metadata'
        },
        'source1_name': 'Le Gaulois',
        'source2_name': 'Les Débats',
        'question_type': 'MH - Generic'
    },
    {
        'name': 'Generic_LIntransigeant_Debats',
        'file': 'generated_questions_lintransigeant_all.jsonl',
        'column_mapping': {
            'id_1': 'journal_id',
            'id_2': 'debat_id',
            'doc_1': 'journal_doc',
            'doc_2': 'debat_doc',
            'metadata_1': 'journal_metadata',
            'metadata_2': 'debat_metadata'
        },
        'source1_name': "L'Intransigeant",
        'source2_name': "Les Débats",
        'question_type': 'MH - Generic'
    },
    {
        'name': 'BridgeEntity_LIntransigeant_Debats',
        'file': 'MH_BridgeEntity_lintransigeant_v2.jsonl',
        'column_mapping': {
            'id_1': 'journal_id',
            'id_2': 'debat_id',
            'doc_1': 'journal_doc',
            'doc_2': 'debat_doc',
            'metadata_1': 'journal_metadata',
            'metadata_2': 'debat_metadata'
        },
        'source1_name': "L'Intransigeant",
        'source2_name': 'Les Débats',
        'question_type': 'MH - BridgeEntity'
    },
    {
        'name': 'Comparative_LeGaulois_LIntransigeant',
        'file': 'MH_Comparative_legaulois_lintransigeant_v3.jsonl',
        'column_mapping': {
            'id_1': 'legaulois_id',
            'id_2': 'lintransigeant_id',
            'doc_1': 'legaulois_doc',
            'doc_2': 'lintransigeant_doc',
            'metadata_1': 'legaulois_metadata',
            'metadata_2': 'lintransigeant_metadata'
        },
        'source1_name': 'Le Gaulois',
        'source2_name': "L'Intransigeant",
        'question_type': 'MH - Comparative'
    },                                                                                        
    {
        'name': 'Generic_LeGaulois_LIntransigeant',
        'file': 'MH_Generic_legaulois_lintransigeant_v3.jsonl',
        'column_mapping': {
            'id_1': 'legaulois_id',
            'id_2': 'lintransigeant_id',
            'doc_1': 'legaulois_doc',
            'doc_2': 'lintransigeant_doc',
            'metadata_1': 'legaulois_metadata',
            'metadata_2': 'lintransigeant_metadata'
        },
        'source1_name': 'Le Gaulois',
        'source2_name': "L'Intransigeant",
        'question_type': 'MH - Generic'
    }
]


def main():
    print("Step 1: Loading document collections from ChromaDB...")
    collections_dict = {}
    for cfg in COLLECTION_CONFIGS:
        col = load_chroma_collection(
            EMBEDDINGS_1887_COHERE, api_key=COHERE_API_KEY, collection_name=cfg["name"]
        )
        collections_dict[cfg["label"]] = col
        print(f"  Loaded: {cfg['label']}")
    docs_legaulois = collections_dict["Le Gaulois"].get()["documents"]
    ids_legaulois = collections_dict["Le Gaulois"].get()["ids"]
    ids_legaulois = ["Le Gaulois_" + str(id) for id in ids_legaulois]

    docs_lintransigeant = collections_dict["L'Intransigeant"].get()["documents"]
    ids_lintransigeant = collections_dict["L'Intransigeant"].get()["ids"]
    ids_lintransigeant = ["L'Intransigeant_" + str(id) for id in ids_lintransigeant]

    docs_debats = collections_dict["Les Débats"].get()["documents"]
    ids_debats = collections_dict["Les Débats"].get()["ids"]
    ids_debats = ["Les Débats_" + str(id) for id in ids_debats]

    all_docs = docs_legaulois + docs_lintransigeant + docs_debats
    all_ids = ids_legaulois + ids_lintransigeant + ids_debats

    # Combine the lists directly
    
    print(f"Total documents before deduplication: {len(all_docs)}")

    # Deduplicate by tying them together in a dictionary. 
    # Duplicate IDs will just overwrite each other safely.
    unique_docs_dict = dict(zip(all_ids, all_docs))
    
    # Extract them back into parallel lists
    doc_ids = list(unique_docs_dict.keys())
    docs = list(unique_docs_dict.values())
    
    print(f"{len(docs)} unique documents to index.")
    
    config = BaseConfig(
        save_dir=HIPPORAG_DIR_MH,
        llm_name="command-a-03-2025",
        llm_base_url="https://api.cohere.ai/compatibility/v1",
        embedding_model_name="embed-v4.0",
        embedding_base_url="https://api.cohere.ai/compatibility/v1",
        seed=None,
    )
    rag = HippoRAG(global_config=config)

    print("Indexing documents...")
    rag.index(docs=docs, doc_ids=doc_ids)
    print(f"Indexing complete. Output saved to {HIPPORAG_DIR_MH}")


if __name__ == "__main__":
    main()
