"""High-level document embedding pipeline that fuses CUI2Vec and sentence-transformer vectors."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("sentence-transformers is required. Please `pip install sentence-transformers`.") from e

from combiner import EmbeddingCombiner
from embeddings import ConceptEmbedder

MODEL_ALIASES = {
    "sapbert": "pritamdeka/SapBERT-from-PubMedBERT-fulltext",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

class DocumentEmbedder:
    """Generate unified high‑dim document vectors."""

    def __init__(
        self,
        concept_embedder: ConceptEmbedder,
        sbert_model: str = "sapbert",
        fusion_strategy: str = "concat",
        idf_weights: Dict[str, float] | None = None,
        sif_a: float = 1e-3,
        device: Optional[str] = None
    ) -> None:
        self.concept_embedder = concept_embedder
        model_name = MODEL_ALIASES.get(sbert_model, sbert_model)
        logging.info(f"Loading SentenceTransformer model: {model_name}")
        self._sbert_model = SentenceTransformer(model_name, device=device)
        self.combiner = EmbeddingCombiner(strategy=fusion_strategy)
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
            return np.zeros(expected_dim, dtype=np.float32)

        vectors = []
        weights = []
        for cui, vec in cui_to_vec.items():
            if self.idf:
                idf_val = self.idf.get(cui, 0.0)
                w = self.sif_a / (self.sif_a + idf_val)
            else:
                w = 1.0
            vectors.append(vec * w)
            weights.append(w)

        stacked = np.stack(vectors)
        pooled = np.sum(stacked, axis=0) / (np.sum(weights) + 1e-9)
        return pooled

    def embed_document(self, umls_entities: List[Dict[str, Any]], text: str) -> np.ndarray:
        # 1. CUI embeddings
        cui_to_vec = self.concept_embedder.embed_document(umls_entities)
        pooled_cui_vec = self._aggregate_cui_vectors(cui_to_vec)

        # 2. Sentence embedding for full document
        sbert_vec = self._sbert_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        # 3. Combine
        unified_vec = self.combiner.combine(pooled_cui_vec, sbert_vec)
        return unified_vec

    def coverage_stats(self):
        return dict(self.concept_embedder.stats)

    def set_idf_weights(self, idf: Dict[str, float]):
        # Make SIF parameter accessible for plotly hover
        self.idf = idf 