# rapidindex/core/retriever.py
"""
Retrieval engine

Combines BM25 keyword search, optional embedding reranking,
LLM reasoning, and multi-layer caching.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
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
    KEYWORD  = "keyword"
    SEMANTIC = "semantic"
    HYBRID   = "hybrid"
    REASONING = "reasoning"


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = Field(default=RetrievalMode.REASONING)
    bm25_top_k: int = Field(default=20, ge=1, le=100)
    embedding_top_k: int = Field(default=10, ge=1, le=50)
    final_top_k: int = Field(default=5, ge=1, le=20)

    use_reranking: bool = Field(default=True)
    # FIX: threshold is now applied to *normalised* scores so the 0–1
    # range is meaningful.  0.05 means "rerank when the top normalised
    # variance is tighter than 5 %".
    rerank_threshold: float = Field(default=0.05, ge=0, le=1)

    use_llm_reasoning: bool = Field(default=True)
    max_reasoning_tokens: int = Field(default=1000)
    max_answer_tokens: int = Field(default=2000)

    enable_cache: bool = Field(default=True)
    cache_ttl: int = Field(default=3600)

    enable_query_expansion: bool = Field(default=False)
    min_query_length: int = Field(default=3)

    class Config:
        use_enum_values = True


@dataclass
class RetrievalMetrics:
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
        return {
            "query": self.query,
            "total_time_ms": round(self.total_time_ms, 2),
            "bm25_time_ms": round(self.bm25_time_ms, 2),
            "embedding_time_ms": round(self.embedding_time_ms, 2),
            "llm_time_ms": round(self.llm_time_ms, 2),
            "bm25_candidates": self.bm25_candidates,
            "embedding_candidates": self.embedding_candidates,
            "final_results": self.final_results,
            "cache_hit": self.cache_hit,
            "mode": self.mode,
        }


class SearchResult(BaseModel):
    query: str
    answer: str
    sections: List[Dict[str, Any]]
    reasoning: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    retrieval_mode: str
    total_time_ms: float
    cache_hit: bool
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class Retriever:
    """Main retrieval engine for RapidIndex."""

    def __init__(
        self,
        bm25_index: BM25Index,
        embedding_index: Optional[EmbeddingIndex] = None,
        llm_client: Optional[LLMClient] = None,
        cache_manager: Optional[CacheManager] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.llm_client = llm_client
        self.cache_manager = cache_manager
        self.config = config or RetrievalConfig()

        self._total_searches = 0
        self._cache_hits = 0

        logger.info(
            "Retriever initialized",
            mode=self.config.mode,
            has_embeddings=embedding_index is not None,
            has_llm=llm_client is not None,
            has_cache=cache_manager is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> SearchResult:
        config = config or self.config
        start_time = time.time()

        self._validate_query(query)
        self._total_searches += 1

        logger.info("Starting search", query=query, mode=config.mode)

        try:
            cache_result = await self._check_cache(query, config)
            if cache_result:
                self._cache_hits += 1
                logger.info("Cache hit", query=query)
                return cache_result

            if config.mode == RetrievalMode.KEYWORD:
                result = await self._keyword_search(query, config)
            elif config.mode == RetrievalMode.SEMANTIC:
                result = await self._semantic_search(query, config)
            elif config.mode == RetrievalMode.HYBRID:
                result = await self._hybrid_search(query, config)
            else:
                result = await self._reasoning_search(query, config)

            result.total_time_ms = (time.time() - start_time) * 1000

            if config.enable_cache and self.cache_manager:
                await self._cache_result(query, result, config)

            logger.success(
                "Search completed",
                query=query,
                time_ms=round(result.total_time_ms, 2),
                num_sections=len(result.sections),
                confidence=result.confidence,
            )
            return result

        except Exception as exc:
            logger.error("Search failed", query=query, error=str(exc))
            raise SearchError(f"Search failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    async def _keyword_search(
        self, query: str, config: RetrievalConfig
    ) -> SearchResult:
        t0 = time.time()
        candidates = self.bm25_index.search(query, top_k=config.final_top_k)
        bm25_ms = (time.time() - t0) * 1000

        if not candidates:
            raise NoResultsError(f"No results found for: {query}")

        sections = self._get_section_details(candidates)
        metrics = RetrievalMetrics(
            query=query, total_time_ms=bm25_ms, bm25_time_ms=bm25_ms,
            embedding_time_ms=0, llm_time_ms=0,
            bm25_candidates=len(candidates), embedding_candidates=0,
            final_results=len(sections), cache_hit=False, mode=RetrievalMode.KEYWORD,
        )
        return SearchResult(
            query=query, answer=self._generate_simple_answer(query, sections),
            sections=sections, confidence=0.6, retrieval_mode=RetrievalMode.KEYWORD,
            total_time_ms=bm25_ms, cache_hit=False, metrics=metrics.to_dict(),
        )

    async def _semantic_search(
        self, query: str, config: RetrievalConfig
    ) -> SearchResult:
        if not self.embedding_index:
            raise SearchError("Embedding index not available")

        t0 = time.time()
        all_ids = list(self.bm25_index.section_map.keys())
        candidates = self.embedding_index.rerank(query, all_ids, top_k=config.final_top_k)
        embed_ms = (time.time() - t0) * 1000

        if not candidates:
            raise NoResultsError(f"No results found for: {query}")

        sections = self._get_section_details(candidates)
        metrics = RetrievalMetrics(
            query=query, total_time_ms=embed_ms, bm25_time_ms=0,
            embedding_time_ms=embed_ms, llm_time_ms=0,
            bm25_candidates=0, embedding_candidates=len(candidates),
            final_results=len(sections), cache_hit=False, mode=RetrievalMode.SEMANTIC,
        )
        return SearchResult(
            query=query, answer=self._generate_simple_answer(query, sections),
            sections=sections, confidence=0.7, retrieval_mode=RetrievalMode.SEMANTIC,
            total_time_ms=embed_ms, cache_hit=False, metrics=metrics.to_dict(),
        )

    async def _hybrid_search(
        self, query: str, config: RetrievalConfig
    ) -> SearchResult:
        t0 = time.time()
        bm25_candidates = self.bm25_index.search(query, top_k=config.bm25_top_k)
        bm25_ms = (time.time() - t0) * 1000

        if not bm25_candidates:
            raise NoResultsError(f"No results found for: {query}")

        embed_ms = 0.0
        final_candidates = bm25_candidates

        if (
            config.use_reranking
            and self.embedding_index
            and self._should_rerank(bm25_candidates, config)
        ):
            t1 = time.time()
            section_ids = [cid for cid, _ in bm25_candidates]
            final_candidates = self.embedding_index.rerank(
                query, section_ids, top_k=config.embedding_top_k
            )
            embed_ms = (time.time() - t1) * 1000

        final_candidates = final_candidates[: config.final_top_k]
        sections = self._get_section_details(final_candidates)
        metrics = RetrievalMetrics(
            query=query, total_time_ms=bm25_ms + embed_ms,
            bm25_time_ms=bm25_ms, embedding_time_ms=embed_ms, llm_time_ms=0,
            bm25_candidates=len(bm25_candidates),
            embedding_candidates=len(final_candidates),
            final_results=len(sections), cache_hit=False, mode=RetrievalMode.HYBRID,
        )
        return SearchResult(
            query=query, answer=self._generate_simple_answer(query, sections),
            sections=sections, confidence=0.75, retrieval_mode=RetrievalMode.HYBRID,
            total_time_ms=bm25_ms + embed_ms, cache_hit=False, metrics=metrics.to_dict(),
        )

    async def _reasoning_search(
        self, query: str, config: RetrievalConfig
    ) -> SearchResult:
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to hybrid")
            return await self._hybrid_search(query, config)

        t0 = time.time()
        bm25_candidates = self.bm25_index.search(query, top_k=config.bm25_top_k)
        bm25_ms = (time.time() - t0) * 1000

        if not bm25_candidates:
            raise NoResultsError(f"No results found for: {query}")

        embed_ms = 0.0
        candidates = bm25_candidates

        if (
            config.use_reranking
            and self.embedding_index
            and self._should_rerank(bm25_candidates, config)
        ):
            t1 = time.time()
            section_ids = [cid for cid, _ in bm25_candidates]
            candidates = self.embedding_index.rerank(
                query, section_ids, top_k=config.embedding_top_k
            )
            embed_ms = (time.time() - t1) * 1000

        t2 = time.time()
        section_data = self._prepare_sections_for_llm(candidates)
        reasoning_result = await self.llm_client.reason_over_sections(
            query, section_data, max_tokens=config.max_reasoning_tokens
        )
        selected_sections = self._get_selected_sections(
            reasoning_result["selected_sections"]
        )
        answer = await self.llm_client.generate_answer(
            query, selected_sections, max_tokens=config.max_answer_tokens
        )
        llm_ms = (time.time() - t2) * 1000

        metrics = RetrievalMetrics(
            query=query, total_time_ms=bm25_ms + embed_ms + llm_ms,
            bm25_time_ms=bm25_ms, embedding_time_ms=embed_ms, llm_time_ms=llm_ms,
            bm25_candidates=len(bm25_candidates), embedding_candidates=len(candidates),
            final_results=len(selected_sections), cache_hit=False,
            mode=RetrievalMode.REASONING,
        )
        return SearchResult(
            query=query, answer=answer, sections=selected_sections,
            reasoning=reasoning_result["reasoning"],
            confidence=reasoning_result["confidence"],
            retrieval_mode=RetrievalMode.REASONING,
            total_time_ms=bm25_ms + embed_ms + llm_ms,
            cache_hit=False, metrics=metrics.to_dict(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_query(self, query: str) -> None:
        if not query or not query.strip():
            raise InvalidQueryError("Query cannot be empty")
        if len(query.strip()) < self.config.min_query_length:
            raise InvalidQueryError(
                f"Query too short (minimum {self.config.min_query_length} characters)"
            )

    def _should_rerank(
        self,
        candidates: List[Tuple[str, float]],
        config: RetrievalConfig,
    ) -> bool:
        """Return True when BM25 scores are too close to distinguish reliably.

        FIX: BM25 scores are unbounded (easily 0–20+), so a fixed variance
        threshold of 0.1 was always triggered regardless of actual spread.
        We normalise to [0, 1] first so the threshold is meaningful.
        """
        if len(candidates) < 2:
            return False

        scores = [score for _, score in candidates]
        max_score = max(scores)

        if max_score == 0:
            return True  # all zero → can't distinguish → do rerank

        normalised = [s / max_score for s in scores]

        try:
            var = statistics.variance(normalised)
            # Low variance = scores bunched together = reranking will help
            return var < config.rerank_threshold
        except statistics.StatisticsError:
            return False

    def _get_section_details(
        self, candidates: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        sections = []
        for section_id, score in candidates:
            doc_id, section = self.bm25_index.section_map.get(
                section_id, (None, None)
            )
            if section:
                sections.append({
                    "id": section.id,
                    "title": section.title,
                    "content": section.content,
                    "pages": section.page_numbers,
                    "keywords": section.keywords,
                    "score": float(score),
                    "document_id": doc_id,
                })
        return sections

    def _prepare_sections_for_llm(
        self, candidates: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        sections = []
        for section_id, score in candidates:
            doc_id, section = self.bm25_index.section_map.get(
                section_id, (None, None)
            )
            if section:
                sections.append({
                    "id": section.id,
                    "title": section.title,
                    "pages": section.page_numbers,
                    "preview": section.content[:300],
                    "content": section.content,
                    "score": float(score),
                })
        return sections

    def _get_selected_sections(
        self, section_ids: List[str]
    ) -> List[Dict[str, Any]]:
        sections = []
        for section_id in section_ids:
            doc_id, section = self.bm25_index.section_map.get(
                section_id, (None, None)
            )
            if section:
                sections.append({
                    "id": section.id,
                    "title": section.title,
                    "content": section.content,
                    "pages": section.page_numbers,
                    "keywords": section.keywords,
                    "document_id": doc_id,
                })
        return sections

    def _generate_simple_answer(
        self, query: str, sections: List[Dict[str, Any]]
    ) -> str:
        if not sections:
            return "No relevant sections found."
        parts = [
            f"From '{s['title']}' (Pages {s['pages']}): {s['content'][:200]}..."
            for s in sections[:3]
        ]
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _check_cache(
        self, query: str, config: RetrievalConfig
    ) -> Optional[SearchResult]:
        if not config.enable_cache or not self.cache_manager:
            return None
        try:
            cached = await self.cache_manager.get(self._make_cache_key(query, config))
            if cached:
                cached["cache_hit"] = True
                return SearchResult(**cached)
        except Exception as exc:
            logger.warning(f"Cache check failed: {exc}")
        return None

    async def _cache_result(
        self, query: str, result: SearchResult, config: RetrievalConfig
    ) -> None:
        try:
            await self.cache_manager.set(
                self._make_cache_key(query, config),
                result.dict(),
                ttl=config.cache_ttl,
            )
        except Exception as exc:
            logger.warning(f"Cache set failed: {exc}")

    def _make_cache_key(self, query: str, config: RetrievalConfig) -> str:
        import hashlib
        key = f"{query.lower().strip()}:{config.mode}:{config.final_top_k}:{config.use_reranking}"
        return f"search:{hashlib.md5(key.encode()).hexdigest()}"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        hit_rate = (
            self._cache_hits / self._total_searches if self._total_searches else 0
        )
        return {
            "total_searches": self._total_searches,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(hit_rate, 3),
            "mode": self.config.mode,
            "has_embeddings": self.embedding_index is not None,
            "has_llm": self.llm_client is not None,
            "has_cache": self.cache_manager is not None,
        }

    def reset_stats(self) -> None:
        self._total_searches = 0
        self._cache_hits = 0
        logger.info("Retrieval statistics reset")


__all__ = [
    "Retriever",
    "RetrievalMode",
    "RetrievalConfig",
    "SearchResult",
    "RetrievalMetrics",
]
