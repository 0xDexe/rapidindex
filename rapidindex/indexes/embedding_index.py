# rapidindex/indexes/embedding_index.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from loguru import logger


class EmbeddingIndex:
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}  # section_id -> embedding
        logger.info("EmbeddingIndex initialized")
    
    def add_section(self, section_id: str, text: str):
        """Generate and store embedding for section."""
        if len(text) > 50:
            embedding = self.model.encode(text, show_progress_bar=False)
            self.embeddings[section_id] = embedding
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Re-rank candidates using semantic similarity."""
        if not candidates:
            return []
        
        # Encode query
        query_embedding = self.model.encode(query, show_progress_bar=False)
        
        # Calculate similarities
        similarities = []
        for section_id in candidates:
            if section_id in self.embeddings:
                section_embedding = self.embeddings[section_id]
                # Cosine similarity
                similarity = np.dot(query_embedding, section_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(section_embedding)
                )
                similarities.append((section_id, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def should_rerank(self, bm25_scores: List[float], threshold: float = 0.1) -> bool:
        """Decide if reranking is needed based on score variance."""
        if len(bm25_scores) < 2:
            return False
        
        score_variance = np.var(bm25_scores)
        return bool(score_variance < threshold)


__all__ = ['EmbeddingIndex']