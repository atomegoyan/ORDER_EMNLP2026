"""
B1_hipporag_indexing.py

Index documents into HippoRAG using Cohere models.
Reads questions from the single-hop JSONL file, extracts the associated
documents, and indexes them into HippoRAG.

Usage:
    python -m scripts.B1_hipporag_indexing
"""

import os
import sys
import json

import pandas as pd
from tqdm import tqdm
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Ensure project root is on the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.config import COHERE_API_KEY, HIPPORAG_DIR, QUESTIONS_FILE
from scripts.utils.json_reading import read_jsonl_basic

# HippoRAG uses the OpenAI client internally; point it at Cohere
os.environ["OPENAI_API_KEY"] = COHERE_API_KEY


def prepare_questions_for_retrieval(df_questions: pd.DataFrame) -> pd.DataFrame:
    """Parse raw question rows into a clean DataFrame for retrieval."""
    prepared_data = []
    for idx, row in df_questions.iterrows():
        try:
            questions_data = json.loads(row["questions_v4_simplified"])
            question_text = questions_data.get("question", "")
            if question_text == "AUCUNE QUESTION" or not question_text.strip():
                continue
            prepared_data.append({
                "id": row["id"],
                "question": question_text,
                "gold_ids": [row["id"]],
                "document": row["document"],
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping row {idx}: {e}")
    return pd.DataFrame(prepared_data)


def main():
    print(f"Loading questions from {QUESTIONS_FILE}")
    df_questions = prepare_questions_for_retrieval(
        pd.DataFrame(read_jsonl_basic(QUESTIONS_FILE))
    )
    print(f"{len(df_questions)} documents to index.")

    docs = df_questions["document"].tolist()
    doc_ids = df_questions["id"].tolist()

    config = BaseConfig(
        save_dir=HIPPORAG_DIR,
        llm_name="command-a-03-2025",
        llm_base_url="https://api.cohere.ai/compatibility/v1",
        embedding_model_name="embed-v4.0",
        embedding_base_url="https://api.cohere.ai/compatibility/v1",
        seed=None,
    )
    rag = HippoRAG(global_config=config)

    print("Indexing documents...")
    rag.index(docs=docs, doc_ids=doc_ids)
    print(f"Indexing complete. Output saved to {HIPPORAG_DIR}")


if __name__ == "__main__":
    main()