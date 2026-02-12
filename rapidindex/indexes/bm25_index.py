# rapidindex/indexes/bm25_index.py

from rank_bm25 import BM25Okapi
import nltk
from typing import List, Tuple, Dict
from loguru import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class BM25Index:
    """BM25-based keyword search index."""
    
    def __init__(self):
        self.index = None
        self.documents = {}  # doc_id -> Document
        self.section_map = {}  # section_id -> (doc_id, section)
        self.tokenized_corpus = []
        logger.info("BM25Index initialized")
    
    def add_document(self, document):
        """Add document to BM25 index."""
        from ..core.document import Document
        
        self.documents[document.id] = document
        
        # Tokenize each section
        for section in document.sections:
            # Store section mapping
            self.section_map[section.id] = (document.id, section)
            
            # Tokenize section content
            tokens = self._tokenize(section.title + " " + section.content)
            self.tokenized_corpus.append(tokens)
        
        # Rebuild BM25 index
        self._rebuild_index()
        logger.info(f"Document added to BM25: {document.id} ({len(document.sections)} sections)")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant sections."""
        if not self.index:
            logger.warning("BM25 index is empty")
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Map back to section IDs
        results = []
        section_ids = list(self.section_map.keys())
        
        for idx in top_indices:
            if idx < len(section_ids):
                section_id = section_ids[idx]
                score = scores[idx]
                results.append((section_id, score))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        tokens = nltk.word_tokenize(text.lower())
        return [t for t in tokens if len(t) > 2 and t.isalnum()]
    
    def _rebuild_index(self):
        """Rebuild BM25 index from corpus."""
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
            logger.debug(f"BM25 index rebuilt: {len(self.tokenized_corpus)} sections")


__all__ = ['BM25Index']