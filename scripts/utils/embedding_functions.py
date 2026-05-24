"""Custom ChromaDB-compatible embedding functions.

Currently provided:
  - OpenRouterEmbeddingFunction: calls the OpenRouter /embeddings API endpoint,
    supporting any model available there (e.g. openai/text-embedding-3-small).
"""

import requests


class OpenRouterEmbeddingFunction:
    """Embedding function for ChromaDB backed by the OpenRouter embeddings API.

    Implements the ChromaDB EmbeddingFunction duck-type protocol:
    ``__call__(self, input: list[str]) -> list[list[float]]``

    Args:
        api_key: OpenRouter API key (OPENROUTER_API_KEY in config.py).
        model:   OpenRouter model name, e.g. ``"openai/text-embedding-3-small"``.

    Example::

        from scripts.utils.embedding_functions import OpenRouterEmbeddingFunction
        ef = OpenRouterEmbeddingFunction(api_key=OPENROUTER_API_KEY)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection("my_col", embedding_function=ef)
    """

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small") -> None:
        self.api_key = api_key
        self.model = model

    def name(self) -> str:
        """Required by ChromaDB ≥ 1.5 to identify the embedding function."""
        return f"openrouter/{self.model}"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a list of texts.

        Args:
            input: List of strings to embed.

        Returns:
            List of embedding vectors (one per input string).

        Raises:
            requests.HTTPError: If the API returns a non-2xx status.
        """
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": list(input)},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """ChromaDB ≥ 0.6 calls this method at query time.

        Delegates to ``__call__`` since OpenRouter uses the same endpoint for
        both document and query embeddings.
        """
        return self(input)
