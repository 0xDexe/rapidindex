# rapidindex/core/retriever.py
"""
Retrieval engine 

This module implements the core retrieval logic combining:
- BM25 keyword search
- Optional embedding-based reranking
- LLM reasoning for final selection
- Multi-layer caching
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from loguru import logger

from pydantic import BaseModel, Field

from .document import Document, DocumentSection
from .exceptions import SearchError, InvalidQueryError, NoResultsError
from ..indexes.bm25_index import BM25Index
from ..indexes.embedding_index import EmbeddingIndex
from ..reasoning.llm_client import LLMClient
from ..cache.cache_manager import CacheManager


class RetrievalMode(str, Enum):
    """Retrieval strategy modes."""
    KEYWORD = "keyword"           # BM25 only
    SEMANTIC = "semantic"         # Embeddings only
    HYBRID = "hybrid"             # BM25 + Embeddings
    REASONING = "reasoning"       # Full pipeline with LLM


class RetrievalConfig(BaseModel):
    """Configuration for retrieval operations."""
    
    mode: RetrievalMode = Field(default=RetrievalMode.REASONING)
    bm25_top_k: int = Field(default=20, ge=1, le=100)
    embedding_top_k: int = Field(default=10, ge=1, le=50)
    final_top_k: int = Field(default=5, ge=1, le=20)
    
    # Reranking settings
    use_reranking: bool = Field(default=True)
    rerank_threshold: float = Field(default=0.1, ge=0, le=1)
    
    # LLM settings
    use_llm_reasoning: bool = Field(default=True)
    max_reasoning_tokens: int = Field(default=1000)
    max_answer_tokens: int = Field(default=2000)
    
    # Caching
    enable_cache: bool = Field(default=True)
    cache_ttl: int = Field(default=3600)  # 1 hour
    
    # Query processing
    enable_query_expansion: bool = Field(default=False)
    min_query_length: int = Field(default=3)
    
    class Config:
        use_enum_values = True


@dataclass
class RetrievalMetrics:
    """Metrics collected during retrieval."""
    
    query: str
    total_time_ms: float
    bm25_time_ms: float
    embedding_time_ms: float
    llm_time_ms: float
    
    bm25_candidates: int
    embedding_candidates: int
    final_results: int
    
    cache_hit: bool
    mode: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'total_time_ms': round(self.total_time_ms, 2),
            'bm25_time_ms': round(self.bm25_time_ms, 2),
            'embedding_time_ms': round(self.embedding_time_ms, 2),
            'llm_time_ms': round(self.llm_time_ms, 2),
            'bm25_candidates': self.bm25_candidates,
            'embedding_candidates': self.embedding_candidates,
            'final_results': self.final_results,
            'cache_hit': self.cache_hit,
            'mode': self.mode
        }


class SearchResult(BaseModel):
    """Search result with answer and metadata."""
    
    query: str
    answer: str
    sections: List[Dict[str, Any]]
    reasoning: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    
    # Metadata
    retrieval_mode: str
    total_time_ms: float
    cache_hit: bool
    metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True


class Retriever:
    """
    Main retrieval engine for RapidIndex.
    
    Coordinates BM25 search, embedding reranking, and LLM reasoning
    to retrieve the most relevant document sections for a query.
    
    Architecture:
        Query → Cache Check → BM25 → [Optional Reranking] → [Optional LLM] → Result
    
    """
    
    def __init__(
        self,
        bm25_index: BM25Index,
        embedding_index: Optional[EmbeddingIndex] = None,
        llm_client: Optional[LLMClient] = None,
        cache_manager: Optional[CacheManager] = None,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize retriever.
        
        Args:
            bm25_index: BM25 keyword search index
            embedding_index: Optional embedding index for semantic search
            llm_client: Optional LLM client for reasoning
            cache_manager: Optional cache manager
            config: Retrieval configuration
        """
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.llm_client = llm_client
        self.cache_manager = cache_manager
        self.config = config or RetrievalConfig()
        
        # Metrics
        self._total_searches = 0
        self._cache_hits = 0
        
        logger.info(
            "Retriever initialized",
            mode=self.config.mode,
            has_embeddings=embedding_index is not None,
            has_llm=llm_client is not None,
            has_cache=cache_manager is not None
        )
    
    async def search(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None
    ) -> SearchResult:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            config: Optional config override
            
        Returns:
            SearchResult with answer and metadata
            
        Raises:
            InvalidQueryError: If query is invalid
            SearchError: If search fails
        """
        config = config or self.config
        start_time = time.time()
        
        # Validate query
        self._validate_query(query)
        
        # Track metrics
        self._total_searches += 1
        
        logger.info(
            "Starting search",
            query=query,
            mode=config.mode,
            search_count=self._total_searches
        )
        
        try:
            # Check cache
            cache_result = await self._check_cache(query, config)
            if cache_result:
                self._cache_hits += 1
                logger.info("Cache hit", query=query)
                return cache_result
            
            # Execute retrieval based on mode
            if config.mode == RetrievalMode.KEYWORD:
                result = await self._keyword_search(query, config)
            elif config.mode == RetrievalMode.SEMANTIC:
                result = await self._semantic_search(query, config)
            elif config.mode == RetrievalMode.HYBRID:
                result = await self._hybrid_search(query, config)
            else:  # REASONING
                result = await self._reasoning_search(query, config)
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            result.total_time_ms = total_time
            
            # Cache result
            if config.enable_cache and self.cache_manager:
                await self._cache_result(query, result, config)
            
            logger.success(
                "Search completed",
                query=query,
                time_ms=round(total_time, 2),
                num_sections=len(result.sections),
                confidence=result.confidence
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Search failed",
                query=query,
                error=str(e),
                error_type=type(e).__name__
            )
            raise SearchError(f"Search failed: {str(e)}") from e
    
    async def _keyword_search(
        self,
        query: str,
        config: RetrievalConfig
    ) -> SearchResult:
        """BM25 keyword search only."""
        
        bm25_start = time.time()
        
        # BM25 search
        candidates = self.bm25_index.search(
            query,
            top_k=config.final_top_k
        )
        
        bm25_time = (time.time() - bm25_start) * 1000
        
        if not candidates:
            raise NoResultsError(f"No results found for query: {query}")
        
        # Get section details
        sections = self._get_section_details(candidates)
        
        # Create metrics
        metrics = RetrievalMetrics(
            query=query,
            total_time_ms=bm25_time,
            bm25_time_ms=bm25_time,
            embedding_time_ms=0,
            llm_time_ms=0,
            bm25_candidates=len(candidates),
            embedding_candidates=0,
            final_results=len(sections),
            cache_hit=False,
            mode=RetrievalMode.KEYWORD
        )
        
        return SearchResult(
            query=query,
            answer=self._generate_simple_answer(query, sections),
            sections=sections,
            confidence=0.6,  # Lower confidence without LLM
            retrieval_mode=RetrievalMode.KEYWORD,
            total_time_ms=bm25_time,
            cache_hit=False,
            metrics=metrics.to_dict()
        )
    
    async def _semantic_search(
        self,
        query: str,
        config: RetrievalConfig
    ) -> SearchResult:
        """Embedding-based semantic search."""
        
        if not self.embedding_index:
            raise SearchError("Embedding index not available")
        
        embed_start = time.time()
        
        # Get all section IDs (or use a subset)
        all_section_ids = list(self.bm25_index.section_map.keys())
        
        # Rerank using embeddings
        candidates = self.embedding_index.rerank(
            query,
            all_section_ids,
            top_k=config.final_top_k
        )
        
        embed_time = (time.time() - embed_start) * 1000
        
        if not candidates:
            raise NoResultsError(f"No results found for query: {query}")
        
        # Get section details
        sections = self._get_section_details(candidates)
        
        # Create metrics
        metrics = RetrievalMetrics(
            query=query,
            total_time_ms=embed_time,
            bm25_time_ms=0,
            embedding_time_ms=embed_time,
            llm_time_ms=0,
            bm25_candidates=0,
            embedding_candidates=len(candidates),
            final_results=len(sections),
            cache_hit=False,
            mode=RetrievalMode.SEMANTIC
        )
        
        return SearchResult(
            query=query,
            answer=self._generate_simple_answer(query, sections),
            sections=sections,
            confidence=0.7,  # Medium confidence
            retrieval_mode=RetrievalMode.SEMANTIC,
            total_time_ms=embed_time,
            cache_hit=False,
            metrics=metrics.to_dict()
        )
    
    async def _hybrid_search(
        self,
        query: str,
        config: RetrievalConfig
    ) -> SearchResult:
        """Hybrid search combining BM25 and embeddings."""
        
        # BM25 first pass
        bm25_start = time.time()
        bm25_candidates = self.bm25_index.search(
            query,
            top_k=config.bm25_top_k
        )
        bm25_time = (time.time() - bm25_start) * 1000
        
        if not bm25_candidates:
            raise NoResultsError(f"No results found for query: {query}")
        
        logger.info(
            "BM25 search completed",
            candidates=len(bm25_candidates),
            time_ms=round(bm25_time, 2)
        )
        
        # Optional embedding reranking
        embed_time = 0
        final_candidates = bm25_candidates
        
        if (config.use_reranking and 
            self.embedding_index and 
            self._should_rerank(bm25_candidates, config)):
            
            embed_start = time.time()
            
            # Extract section IDs
            section_ids = [cand[0] for cand in bm25_candidates]
            
            # Rerank with embeddings
            reranked = self.embedding_index.rerank(
                query,
                section_ids,
                top_k=config.embedding_top_k
            )
            
            final_candidates = reranked
            embed_time = (time.time() - embed_start) * 1000
            
            logger.info(
                "Embedding reranking completed",
                candidates=len(reranked),
                time_ms=round(embed_time, 2)
            )
        
        # Take top K
        final_candidates = final_candidates[:config.final_top_k]
        
        # Get section details
        sections = self._get_section_details(final_candidates)
        
        # Create metrics
        metrics = RetrievalMetrics(
            query=query,
            total_time_ms=bm25_time + embed_time,
            bm25_time_ms=bm25_time,
            embedding_time_ms=embed_time,
            llm_time_ms=0,
            bm25_candidates=len(bm25_candidates),
            embedding_candidates=len(final_candidates),
            final_results=len(sections),
            cache_hit=False,
            mode=RetrievalMode.HYBRID
        )
        
        return SearchResult(
            query=query,
            answer=self._generate_simple_answer(query, sections),
            sections=sections,
            confidence=0.75,  # Good confidence with hybrid
            retrieval_mode=RetrievalMode.HYBRID,
            total_time_ms=bm25_time + embed_time,
            cache_hit=False,
            metrics=metrics.to_dict()
        )
    
    async def _reasoning_search(
        self,
        query: str,
        config: RetrievalConfig
    ) -> SearchResult:
        """Full reasoning-based search with LLM."""
        
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to hybrid")
            return await self._hybrid_search(query, config)
        
        # First get candidates using hybrid search
        bm25_start = time.time()
        bm25_candidates = self.bm25_index.search(
            query,
            top_k=config.bm25_top_k
        )
        bm25_time = (time.time() - bm25_start) * 1000
        
        if not bm25_candidates:
            raise NoResultsError(f"No results found for query: {query}")
        
        # Optional reranking
        embed_time = 0
        candidates = bm25_candidates
        
        if (config.use_reranking and 
            self.embedding_index and
            self._should_rerank(bm25_candidates, config)):
            
            embed_start = time.time()
            section_ids = [cand[0] for cand in bm25_candidates]
            candidates = self.embedding_index.rerank(
                query,
                section_ids,
                top_k=config.embedding_top_k
            )
            embed_time = (time.time() - embed_start) * 1000
        
        # LLM reasoning
        llm_start = time.time()
        
        # Prepare section data for LLM
        section_data = self._prepare_sections_for_llm(candidates)
        
        # Get LLM reasoning
        reasoning_result = await self.llm_client.reason_over_sections(
            query,
            section_data,
            max_tokens=config.max_reasoning_tokens
        )
        
        # Get selected sections
        selected_sections = self._get_selected_sections(
            reasoning_result['selected_sections']
        )
        
        # Generate final answer
        answer = await self.llm_client.generate_answer(
            query,
            selected_sections,
            max_tokens=config.max_answer_tokens
        )
        
        llm_time = (time.time() - llm_start) * 1000
        
        logger.info(
            "LLM reasoning completed",
            selected=len(selected_sections),
            time_ms=round(llm_time, 2),
            confidence=reasoning_result['confidence']
        )
        
        # Create metrics
        metrics = RetrievalMetrics(
            query=query,
            total_time_ms=bm25_time + embed_time + llm_time,
            bm25_time_ms=bm25_time,
            embedding_time_ms=embed_time,
            llm_time_ms=llm_time,
            bm25_candidates=len(bm25_candidates),
            embedding_candidates=len(candidates),
            final_results=len(selected_sections),
            cache_hit=False,
            mode=RetrievalMode.REASONING
        )
        
        return SearchResult(
            query=query,
            answer=answer,
            sections=selected_sections,
            reasoning=reasoning_result['reasoning'],
            confidence=reasoning_result['confidence'],
            retrieval_mode=RetrievalMode.REASONING,
            total_time_ms=bm25_time + embed_time + llm_time,
            cache_hit=False,
            metrics=metrics.to_dict()
        )
    
    def _validate_query(self, query: str) -> None:
        """Validate search query."""
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")
        
        if len(query.strip()) < self.config.min_query_length:
            raise InvalidQueryError(
                f"Query too short (minimum {self.config.min_query_length} characters)"
            )
    
    async def _check_cache(
        self,
        query: str,
        config: RetrievalConfig
    ) -> Optional[SearchResult]:
        """Check if query result is cached."""
        if not config.enable_cache or not self.cache_manager:
            return None
        
        cache_key = self._make_cache_key(query, config)
        
        try:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                # Mark as cache hit
                cached['cache_hit'] = True
                return SearchResult(**cached)
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    async def _cache_result(
        self,
        query: str,
        result: SearchResult,
        config: RetrievalConfig
    ) -> None:
        """Cache search result."""
        cache_key = self._make_cache_key(query, config)
        
        try:
            await self.cache_manager.set(
                cache_key,
                result.dict(),
                ttl=config.cache_ttl
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def _make_cache_key(self, query: str, config: RetrievalConfig) -> str:
        """Generate cache key from query and config."""
        import hashlib
        
        # Normalize query
        normalized = query.lower().strip()
        
        # Include relevant config in key
        config_str = f"{config.mode}:{config.final_top_k}:{config.use_reranking}"
        
        # Hash
        key_str = f"{normalized}:{config_str}"
        return f"search:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def _should_rerank(
        self,
        candidates: List[Tuple[str, float]],
        config: RetrievalConfig
    ) -> bool:
        """Determine if reranking is needed based on score variance."""
        if len(candidates) < 2:
            return False
        
        # Extract scores
        scores = [score for _, score in candidates]
        
        # Calculate variance
        import statistics
        try:
            variance = statistics.variance(scores)
            return variance < config.rerank_threshold
        except statistics.StatisticsError:
            return False
    
    def _get_section_details(
        self,
        candidates: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Get full section details from candidate IDs."""
        sections = []
        
        for section_id, score in candidates:
            doc_id, section = self.bm25_index.section_map.get(
                section_id,
                (None, None)
            )
            
            if section:
                sections.append({
                    'id': section.id,
                    'title': section.title,
                    'content': section.content,
                    'pages': section.page_numbers,
                    'keywords': section.keywords,
                    'score': float(score),
                    'document_id': doc_id
                })
        
        return sections
    
    def _prepare_sections_for_llm(
        self,
        candidates: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Prepare section data for LLM reasoning."""
        sections = []
        
        for section_id, score in candidates:
            doc_id, section = self.bm25_index.section_map.get(
                section_id,
                (None, None)
            )
            
            if section:
                sections.append({
                    'id': section.id,
                    'title': section.title,
                    'pages': section.page_numbers,
                    'preview': section.content[:300],  # First 300 chars
                    'content': section.content,  # Full content
                    'score': float(score)
                })
        
        return sections
    
    def _get_selected_sections(
        self,
        section_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get full details for selected sections."""
        sections = []
        
        for section_id in section_ids:
            doc_id, section = self.bm25_index.section_map.get(
                section_id,
                (None, None)
            )
            
            if section:
                sections.append({
                    'id': section.id,
                    'title': section.title,
                    'content': section.content,
                    'pages': section.page_numbers,
                    'keywords': section.keywords,
                    'document_id': doc_id
                })
        
        return sections
    
    def _generate_simple_answer(
        self,
        query: str,
        sections: List[Dict[str, Any]]
    ) -> str:
        """Generate simple answer without LLM."""
        if not sections:
            return "No relevant sections found."
        
        # Simple concatenation with section info
        answer_parts = []
        
        for i, section in enumerate(sections[:3], 1):
            answer_parts.append(
                f"From '{section['title']}' (Pages {section['pages']}): "
                f"{section['content'][:200]}..."
            )
        
        return "\n\n".join(answer_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        cache_hit_rate = (
            self._cache_hits / self._total_searches 
            if self._total_searches > 0 
            else 0
        )
        
        return {
            'total_searches': self._total_searches,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': round(cache_hit_rate, 3),
            'mode': self.config.mode,
            'has_embeddings': self.embedding_index is not None,
            'has_llm': self.llm_client is not None,
            'has_cache': self.cache_manager is not None
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_searches = 0
        self._cache_hits = 0
        logger.info("Retrieval statistics reset")


# Export
__all__ = [
    'Retriever',
    'RetrievalMode',
    'RetrievalConfig',
    'SearchResult',
    'RetrievalMetrics'
]