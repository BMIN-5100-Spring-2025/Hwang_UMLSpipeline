from __future__ import annotations

"""Sentence-level embedding utilities for hybrid document representations."""

import logging
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise ImportError("sentence-transformers is required for SentenceEmbedder. Please `pip install sentence-transformers`."
                      ) from e

MODEL_ALIASES = {
    "sapbert": "pritamdeka/SapBERT-from-PubMedBERT-fulltext",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


class SentenceEmbedder:
    """Thin wrapper around a SentenceTransformer model that always returns a numpy vector."""

    def __init__(self, model_key: str = "sapbert", device: Optional[str] = None):
        model_name = MODEL_ALIASES.get(model_key, model_key)
        logging.info(f"Loading SentenceTransformer model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:  # type: ignore[override]
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec 