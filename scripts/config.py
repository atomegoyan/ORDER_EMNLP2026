"""
Configuration constants for the Debattre project.
All paths and constants are defined here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API KEYS
# ============================================================================

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not COHERE_API_KEY:
    import warnings
    warnings.warn("COHERE_API_KEY not set — Cohere embedding functions will not work.", stacklevel=2)

# ============================================================================
# BASE DIRECTORIES
# ============================================================================

# Allow base directory to be overridden via environment variable
PROJECT_BASE_DIR = os.getenv("PROJECT_BASE_DIR") or os.getcwd()
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(PROJECT_BASE_DIR, 'data')

# ============================================================================
# EMBEDDINGS PATHS
# ============================================================================

EMBEDDINGS_1887_COHERE = os.path.join(DATA_DIR, "embeddings_1887_cohere")
EMBEDDINGS_1887_OPENROUTER = os.path.join(DATA_DIR, "embeddings_1887_openrouter")
EMBEDDINGS_CS1 = os.path.join(DATA_DIR, "embeddings_cs1")
EMBEDDINGS_SECTIONS = os.path.join(DATA_DIR, "embeddings_sections")
EMBEDDINGS_PP = os.path.join(DATA_DIR, "embeddings_pp")
EMBEDDINGS_CS4 = os.path.join(DATA_DIR, "embeddings_cs4")

# ============================================================================
# COLLECTION NAMES
# ============================================================================

COLLECTION_DEBATTRE_1887_SECTION = "debattre_1887__sectioncohere"

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIR = os.path.join(DATA_DIR, "RAG_results")
QUESTIONS_DIR = os.path.join(DATA_DIR, "questions_singlehop")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
HIPPORAG_DIR = os.path.join(DATA_DIR, "hipprag_output")
HIPPORAG_DIR_MH = os.path.join(DATA_DIR, "hipprag_mh_output")

# ============================================================================
# COMMON FILE PATHS
# ============================================================================

QUESTIONS_FILE = os.path.join(QUESTIONS_DIR, "questions_generated_progressive.jsonl")
RETRIEVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "retrieval_results.jsonl")

# ============================================================================
# EMBEDDING MODELS
# ============================================================================

COHERE_EMBEDDING_MODEL = "embed-v4.0"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
SENTENCE_TRANSFORMERS_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# MULTI-HOP EVALUATION SHARED CONSTANTS
# ============================================================================

# Newspaper collection configs (shared across all evaluation notebooks)
NEWSPAPER_CONFIGS = [
    {"name": "legaulois_1887_v1", "label": "Le Gaulois"},
    {"name": "lintransigeant_1887_v1", "label": "L'Intransigeant"},
]

# Default Les Débats collection
DEBATS_COLLECTION_NAME = "debattre_1887_10000_hierarchical_cohere"

# ============================================================================
# BM25 INDEX PATHS
# ============================================================================

BM25_DEBATS_DIR = os.path.join(DATA_DIR, "bm25_1887")
BM25_LINTRANSIGEANT_DIR = os.path.join(DATA_DIR, "bm25_lintransigeant_1887")
BM25_LEGAULOIS_DIR = os.path.join(DATA_DIR, "bm25_legaulois_1887")

BM25_COLLECTIONS = {
    "Les Débats": BM25_DEBATS_DIR,
    "L'Intransigeant": BM25_LINTRANSIGEANT_DIR,
    "Le Gaulois": BM25_LEGAULOIS_DIR,
}

# Multi-hop questions collection
QUESTIONS_COLLECTION_NAME = "questions_MH_v1"

# Default K values for retrieval evaluation
DEFAULT_K_VALUES = [3, 5, 10]

# Default class-to-collections mapping for query rerouting
DEFAULT_CLASS_TO_COLLECTIONS = {
    "Le Gaulois + L'Intransigeant": ["Le Gaulois", "L'Intransigeant"],
    "L'Intransigeant + Les Débats": ["Le Gaulois", "L'Intransigeant", "Les Débats"],
    "Le Gaulois + Les Débats": ["Le Gaulois", "L'Intransigeant", "Les Débats"],
}
