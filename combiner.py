from __future__ import annotations

"""Utilities for combining CUI2Vec and SBERT embeddings."""

from typing import Optional, Sequence
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


class EmbeddingCombiner:
    """Combine two embeddings with different strategies."""

    def __init__(self, strategy: str = "concat", output_dim: int = 256):
        self.strategy = strategy
        if strategy == "linear":
            if torch is None:
                raise ImportError("PyTorch required for linear fusion strategy")
            self._proj = nn.Linear(0, 0)  # placeholder; will be set on first call
        elif strategy == "concat":
            self._proj = None
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

    def _init_linear(self, dim_in: int, output_dim: int = 256):
        assert torch is not None
        self._proj = torch.nn.Linear(dim_in, output_dim)
        logging.info(f"Initialized linear projection: {dim_in} -> {output_dim}")

    def combine(self, cui_vec: Optional[np.ndarray], sbert_vec: np.ndarray) -> np.ndarray:
        if cui_vec is None:
            combined = sbert_vec
        elif self.strategy == "concat":
            combined = np.concatenate([cui_vec, sbert_vec])
        else:  # linear
            if self._proj is None or self._proj.in_features == 0:
                self._init_linear(cui_vec.shape[-1] + sbert_vec.shape[-1])
            vec = np.concatenate([cui_vec, sbert_vec])
            with torch.no_grad():
                combined = self._proj(torch.from_numpy(vec).float()).numpy()
        return combined 