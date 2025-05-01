from typing import Dict, List, Optional
import numpy as np
from gensim.models import KeyedVectors
import logging
from pathlib import Path
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class ConceptEmbedder:
    """Handles loading and mapping of CUI embeddings using gensim KeyedVectors."""
    
    def __init__(
        self,
        embeddings_filepath: str,
        fallback_strategy: str = "text2vec",
        sbert_model: str = None,
        mrrel_path: Optional[str] = None,
    ):
        """
        Initialize the embedder by loading embeddings using gensim's KeyedVectors.
        
        Args:
            embeddings_filepath: path to CUI2Vec vectors (word2vec text format)
            fallback_strategy: 'text2vec' or 'graph'
            sbert_model: which sentence transformer model to use for fallback mapping (default: all-MiniLM-L12-v2)
            mrrel_path: path to UMLS MRREL file
        """
        self.embeddings_filepath = embeddings_filepath
        self.model = KeyedVectors.load_word2vec_format(embeddings_filepath, binary=False)

        self.fallback_strategy = fallback_strategy
        self.sentence_model_name = {
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            None: "sentence-transformers/all-MiniLM-L12-v2"
        }.get(sbert_model, sbert_model or "sentence-transformers/all-MiniLM-L12-v2")

        if fallback_strategy == "text2vec":
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for text2vec fallback")
            logging.info(f"Loading SentenceTransformer model: {self.sentence_model_name}")
            self.sbert = SentenceTransformer(self.sentence_model_name)
            # Precompute CUI2Vec embeddings normalized for efficient similarity search
            self._cui_mat = np.stack([self.model[cui] for cui in self.model.index_to_key])
            self._cui_norm = self._cui_mat / np.linalg.norm(self._cui_mat, axis=1, keepdims=True)
        else:
            self.sbert = None

        # Load graph if requested and path supplied
        if fallback_strategy == "graph":
            if mrrel_path is None:
                logging.warning("Graph fallback selected but no MRREL path provided; fallback will return None")
                self._umls_graph = None
            else:
                from graph_utils import load_umls_graph
                self._umls_graph = load_umls_graph(Path(mrrel_path))

        proj_path = Path(embeddings_filepath).with_suffix(".sbert_proj.npy")
        if proj_path.exists():
            self._proj = np.load(proj_path)
            if self._proj.shape != (384, self.model.vector_size):
                logging.warning("Projection matrix shape mismatch; ignoring.")
                self._proj = None
            else:
                logging.info("Loaded SBERT→CUI2Vec projection matrix: %s", proj_path)
        else:
            self._proj = None

        self.stats = defaultdict(int)
        self._fallback_cache: Dict[str, Optional[np.ndarray]] = {}
    
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
        # 1. Check cache first
        if cui in self._fallback_cache:
            cached_result = self._fallback_cache[cui]
            if cached_result is not None:
                self.stats['cache_hit'] += 1
            return cached_result

        # 2. Check exact match in model
        if cui in self.model:
            self.stats['exact'] += 1
            return self.model[cui]

        # 3. Attempt fallbacks
        vec = None
        if self.fallback_strategy == "text2vec" and self.sbert is not None:
            vec = self._text_to_vec_fallback(cui)
            if vec is not None:
                self.stats['text2vec'] += 1
        elif self.fallback_strategy == "graph":
            vec = self._graph_fallback(cui)
            if vec is not None:
                self.stats['graph'] += 1

        # 4. Cache the result and return
        self._fallback_cache[cui] = vec
        if vec is None:
            self.stats['missing'] += 1
        return vec

    def _text_to_vec_fallback(self, cui: str):
        """Approximate embedding by encoding the term text and mapping to closest CUI2Vec."""
        try:
            query_vec = self.sbert.encode(cui, convert_to_numpy=True, show_progress_bar=False)
            if self._proj is not None:
                query_vec = query_vec @ self._proj
            elif query_vec.shape[0] != self._cui_mat.shape[1]:
                logging.debug("No projection; SBERT dim %s mismatch. skipping", query_vec.shape[0])
                return None

            query_vec = query_vec / np.linalg.norm(query_vec)
            # Cosine similarity search
            sims = np.dot(self._cui_norm, query_vec)
            best_idx = int(np.argmax(sims))
            return self._cui_mat[best_idx]
        except Exception as err:
            logging.warning(f"Text2Vec fallback failed for {cui}: {err}")
            return None

    #  Graph fallback 
    def _graph_fallback(self, cui: str, max_hops: int = 2):
        """Traverse UMLS graph outward until neighbor with embedding found, and average them."""
        if self._umls_graph is None:
            return None

        if cui not in self._umls_graph:
            return None

        frontier = {cui}
        visited = {cui}
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            collected: List[np.ndarray] = []
            for node in frontier:
                for nb in self._umls_graph.get(node, []):
                    if nb in self.model:
                        collected.append(self.model[nb])
                    if nb not in visited:
                        next_frontier.add(nb)
            if collected:
                return np.mean(np.stack(collected), axis=0)
            frontier = next_frontier
            visited.update(frontier)
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