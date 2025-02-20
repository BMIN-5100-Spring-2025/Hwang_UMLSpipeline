from typing import Dict, List, Optional
import numpy as np
from gensim.models import KeyedVectors
import logging

class ConceptEmbedder:
    """Handles loading and mapping of CUI embeddings using gensim KeyedVectors."""
    
    def __init__(self, embeddings_filepath: str):
        """
        Initialize the embedder by loading embeddings using gensim's KeyedVectors.
        
        The embeddings file should be in word2vec text format.
        """
        self.embeddings_filepath = embeddings_filepath
        self.model = KeyedVectors.load_word2vec_format(embeddings_filepath, binary=False)
    
    def get_embedding(self, cui: str):
        """
        Retrieve the embedding for a given CUI.
        
        If an exact match is not found, try to return the most similar embedding as a fallback
        using gensim's most_similar method.
        
        Args:
            cui: The unique concept identifier.
            
        Returns:
            A numpy array for the embedding, or None if no similar embedding could be determined.
        """
        if cui in self.model:
            return self.model[cui]
        else:
            try:
                # This will look for the most similar key. If found,
                # choose the top similar cui's embedding as a fallback.
                similar = self.model.most_similar(positive=[cui])
                if similar:
                    similar_cui, similarity = similar[0]
                    return self.model[similar_cui]
            except Exception as err:
                print(f"Warning: No embedding found for CUI '{cui}'; fallback failed: {err}")
            return None

    def embed_document(self, umls_entities: list) -> dict:
        """
        Create embeddings for all CUIs found in a document.
        
        Args:
            umls_entities: List of UMLS entity dictionaries from NLPProcessor.
            
        Returns:
            Dictionary mapping each CUI to its embedding vector.
        """
        embeddings = {}
        for entity in umls_entities:
            cui = entity['cui']
            if cui not in embeddings:
                embedding = self.get_embedding(cui)
                if embedding is not None:
                    embeddings[cui] = embedding
        return embeddings

    @property
    def embeddings(self) -> dict:
        """
        Return a dictionary mapping each CUI in the model to its embedding vector.
        """
        return {word: self.model[word] for word in self.model.key_to_index} 