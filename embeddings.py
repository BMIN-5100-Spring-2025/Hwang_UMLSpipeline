from typing import Dict, List, Optional
import numpy as np
from gensim.models import KeyedVectors
import logging

class ConceptEmbedder:
    """Handles loading and mapping of CUI embeddings."""
    
    def __init__(self, embedding_path: str, embedding_dim: int = 500) -> None:
        """
        Initialize the concept embedder.
        
        Args:
            embedding_path: Path to the word2vec format embeddings file
            embedding_dim: Dimension of the embeddings (default 500 for Cui2Vec)
        """
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self._load_embeddings(embedding_path)
        
    def _load_embeddings(self, path: str) -> None:
        """Load pre-trained embeddings from file."""
        try:
            logging.info(f"Loading embeddings from {path}")
            self.embeddings = KeyedVectors.load_word2vec_format(path, binary=True)
            logging.info(f"Loaded {len(self.embeddings.key_to_index)} embeddings")
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            raise
            
    def get_embedding(self, cui: str) -> Optional[np.ndarray]:
        """Get embedding vector for a CUI."""
        try:
            return self.embeddings[cui]
        except KeyError:
            logging.warning(f"No embedding found for CUI: {cui}")
            return None
            
    def embed_document(self, umls_entities: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for all CUIs in a document.
        
        Args:
            umls_entities: List of UMLS entity dictionaries from NLPProcessor
            
        Returns:
            Dictionary mapping CUIs to their embedding vectors
        """
        embeddings = {}
        for entity in umls_entities:
            cui = entity['cui']
            if cui not in embeddings:
                embedding = self.get_embedding(cui)
                if embedding is not None:
                    embeddings[cui] = embedding
        return embeddings 