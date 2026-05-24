"""Embedding model wrappers compatible with the SentenceTransformer interface
expected by EmbeddingStore and LinearRAG.

Supported backends
------------------
CohereEmbeddingModel  — Cohere embed-v4.0 (or any embed-* model)
                        Uses input_type="search_document" for indexing (is_query=False)
                        and input_type="search_query"    for retrieval (is_query=True).
"""

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class CohereEmbeddingModel:
    """Cohere-backed embedding model with the same .encode() signature as
    sentence_transformers.SentenceTransformer.

    Args:
        api_key:    Cohere API key.
        model:      Model name, e.g. ``"embed-v4.0"``.
        batch_size: Maximum number of texts per API call (Cohere cap = 96).
    """

    _MAX_BATCH = 96  # Hard Cohere limit per embed request

    def __init__(self, api_key: str, model: str = "embed-v4.0", batch_size: int = 96):
        import cohere  # lazy import so the file can be imported without cohere installed

        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model
        self.default_batch_size = min(batch_size, self._MAX_BATCH)

    # ------------------------------------------------------------------
    # Public interface (mirrors SentenceTransformer)
    # ------------------------------------------------------------------

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int | None = None,
        is_query: bool = False,
    ) -> np.ndarray:
        """Encode *texts* and return an (N, D) float32 ndarray.

        Args:
            texts:               A single string or a list of strings.
            normalize_embeddings: Ignored – Cohere returns unit-norm embeddings by default.
            show_progress_bar:   Ignored (kept for API parity).
            batch_size:          Override default batch size.
            is_query:            ``True`` → ``input_type="search_query"`` (used at retrieval time).
                                 ``False`` → ``input_type="search_document"`` (used at index time).
        """
        single_string = isinstance(texts, str)
        if single_string:
            texts = [texts]

        texts = list(texts)
        if len(texts) == 0:
            return np.array([])

        effective_batch = min(batch_size or self.default_batch_size, self._MAX_BATCH)
        input_type = "search_query" if is_query else "search_document"

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), effective_batch):
            batch = texts[i : i + effective_batch]
            success = False
            for attempt in range(5):
                try:
                    response = self.client.embed(
                        texts=batch,
                        model=self.model,
                        input_type=input_type,
                        embedding_types=["float"],
                    )
                    all_embeddings.extend(response.embeddings.float)
                    success = True
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "Cohere embed error (attempt %d/5): %s — retrying in %ds",
                        attempt + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
            if not success:
                raise RuntimeError(
                    f"Cohere embed failed after 5 retries for batch starting at index {i}"
                )

        result = np.array(all_embeddings, dtype=np.float32)
        # Mirror sentence_transformers: a bare string input → 1-D array (dim,)
        # A list input → always 2-D array (n, dim)
        return result[0] if single_string else result
