"""
Utility functions for working with ChromaDB collections.
"""

import os
from typing import Optional, Dict, Any, List
import chromadb
from chromadb.utils import embedding_functions

from scripts.config import (
    COHERE_EMBEDDING_MODEL,
    SENTENCE_TRANSFORMERS_EMBEDDING_MODEL,
)


def get_embedding_function(
    embedding_type: str = "cohere",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Get the appropriate embedding function based on type.

    Args:
        embedding_type: "cohere" or "sentence_transformers"
        api_key: API key (required for Cohere)
        model_name: Embedding model name (uses defaults if not provided)

    Returns:
        ChromaDB embedding function
    """
    if embedding_type == "cohere":
        if api_key is None:
            raise ValueError("api_key is required for Cohere embedding function")
        model = model_name or COHERE_EMBEDDING_MODEL
        return embedding_functions.CohereEmbeddingFunction(
            api_key=api_key, model_name=model
        )
    elif embedding_type == "sentence_transformers":
        model = model_name or SENTENCE_TRANSFORMERS_EMBEDDING_MODEL
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model
        )
    else:
        raise ValueError(f"Unsupported embedding_type: {embedding_type}. "
                        f"Choose from: 'cohere', 'sentence_transformers'")


def load_chroma_collection(
    collection_path: str,
    collection_name: str,
    embedding_type: str = "cohere",
    api_key: Optional[str] = None,
    embedding_model: Optional[str] = None
):
    """
    Load an existing ChromaDB collection.

    Args:
        collection_path: Path to the ChromaDB persistent storage directory
        collection_name: Name of the collection to load
        embedding_type: "cohere" or "sentence_transformers"
        api_key: API key (required for Cohere)
        embedding_model: Embedding model name (uses defaults if not provided)

    Returns:
        ChromaDB collection object

    Raises:
        ValueError: If collection path doesn't exist or collection can't be loaded
    """
    if not os.path.exists(collection_path):
        raise ValueError(f"Collection path does not exist: {collection_path}")

    client = chromadb.PersistentClient(path=collection_path)
    embedding_fn = get_embedding_function(
        embedding_type=embedding_type, api_key=api_key, model_name=embedding_model
    )

    try:
        return client.get_collection(
            name=collection_name, embedding_function=embedding_fn
        )
    except Exception as e:
        raise ValueError(f"Failed to load collection '{collection_name}': {str(e)}")


def list_collections(collection_path: str) -> List[str]:
    """
    List all collection names in a ChromaDB path.

    Args:
        collection_path: Path to the ChromaDB persistent storage directory

    Returns:
        List of collection names
    """
    client = chromadb.PersistentClient(path=collection_path)
    return [col.name for col in client.list_collections()]


def query_collection(
    collection,
    query_texts: List[str],
    n_results: int = 10,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Query a ChromaDB collection with text queries.

    Args:
        collection: ChromaDB collection object
        query_texts: List of query texts
        n_results: Number of results to return per query
        where: Optional metadata filter
        where_document: Optional document content filter
        include: Fields to include (defaults to documents, metadatas, distances)

    Returns:
        Query results dictionary
    """
    if include is None:
        include = ["documents", "metadatas", "distances"]

    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        where_document=where_document,
        include=include
    )
