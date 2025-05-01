"""High-level document embedding pipeline that fuses CUI2Vec and sentence-transformer vectors."""

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import logging

from sentence_embedder import SentenceEmbedder
from combiner import EmbeddingCombiner
from embeddings import ConceptEmbedder


class DocumentEmbedder:
    """Generate unified high‑dim document vectors."""

    def __init__(
        self,
        concept_embedder: ConceptEmbedder,
        sbert_model: str = "sapbert",
        fusion_strategy: str = "concat",
        idf_weights: Dict[str, float] | None = None,
        sif_a: float = 1e-3,
    ) -> None:
        self.concept_embedder = concept_embedder
        self.sentence_embedder = SentenceEmbedder(sbert_model)
        self.combiner = EmbeddingCombiner(strategy=fusion_strategy)
        # IDF weights for SIF pooling
        self.idf = idf_weights or {}
        self.sif_a = sif_a

    def _aggregate_cui_vectors(self, cui_to_vec: Dict[str, np.ndarray]) -> np.ndarray | None:
        """Return IDF/SIF-weighted average of concept vectors.

        Args:
            cui_to_vec: mapping from CUI to its embedding.

        Returns:
            A numpy array representing the pooled document vector, or None if no vectors.
        """
        if not cui_to_vec:
            expected_dim = self.concept_embedder.model.vector_size
            logging.debug("No CUI vectors found for pooling, returning zero vector.")
            return np.zeros(expected_dim, dtype=np.float32) # Use float32 for consistency

        vectors = []
        weights = []
        for cui, vec in cui_to_vec.items():
            if self.idf:
                # Smooth Inverse Frequency style weight.  If idf absent for CUI, default 0.
                idf_val = self.idf.get(cui, 0.0)
                w = self.sif_a / (self.sif_a + idf_val)
            else:
                w = 1.0  # simple mean when no idf provided
            vectors.append(vec * w)
            weights.append(w)

        stacked = np.stack(vectors)
        pooled = np.sum(stacked, axis=0) / (np.sum(weights) + 1e-9)
        return pooled

    def embed_document(self, umls_entities: List[Dict[str, Any]], text: str) -> np.ndarray:
        # 1. CUI embeddings
        cui_to_vec = self.concept_embedder.embed_document(umls_entities)
        pooled_cui_vec = self._aggregate_cui_vectors(cui_to_vec)

        # 2. Sentence embedding for full document (simple for now)
        sbert_vec = self.sentence_embedder.encode(text)

        # 3. Combine
        unified_vec = self.combiner.combine(pooled_cui_vec, sbert_vec)
        return unified_vec

    # ------------- stats -------------
    def coverage_stats(self):
        return dict(self.concept_embedder.stats)

    # -------------------- IDF helpers --------------------
    def set_idf_weights(self, idf: Dict[str, float]):
        """Update/attach the IDF dictionary used for SIF pooling."""
        self.idf = idf 