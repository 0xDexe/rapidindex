# benchmarks/run_benchmark.py
"""Run benchmarks on RapidIndex."""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from loguru import logger

from rapidindex import RapidIndexer, Retriever, RetrievalMode
from rapidindex.indexes.bm25_index import BM25Index
from rapidindex.indexes.embedding_index import EmbeddingIndex
from rapidindex.reasoning.llm_client import LLMClient
from rapidindex.cache.cache_manager import CacheManager
from rapidindex.storage.sqllite import SQLiteStorage
from rapidindex.utils.config import config
from rapidindex.core.retriever import RetrievalConfig

from .datasets.financebench_loader import FinanceBenchDataset
from .evaluate import exact_match, contains_match, number_match, calculate_accuracy


async def run_benchmark(
    mode: RetrievalMode = RetrievalMode.REASONING,
    sample_size: int = 50,
    output_file: str = "benchmark_results.json",
    llm_provider: Optional[str] = None,
):
    """Run benchmark."""

    provider = llm_provider or config.llm_provider
    # FIX: RetrievalMode is a str-enum.  In CPython < 3.11 str-enums happen
    # to serialise with json.dump, but in 3.11+ the behaviour changed and
    # the enum is no longer automatically unwrapped.  Always use .value so
    # the output is a plain string regardless of Python version.
    mode_value: str = mode.value if isinstance(mode, RetrievalMode) else str(mode)

    logger.info(f"Starting benchmark")
    logger.info(f"Mode: {mode_value}")
    logger.info(f"LLM Provider: {provider}")
    logger.info(f"Sample size: {sample_size}")

    # Initialise components
    storage = SQLiteStorage(config.database_url)
    bm25_index = BM25Index()
    embedding_index = EmbeddingIndex()

    if provider == "ollama":
        llm_client = LLMClient(
            provider="ollama",
            model=config.ollama_model,
            ollama_url=config.ollama_url,
        )
        logger.info(f"Using Ollama: {config.ollama_model} (FREE)")
    elif provider == "anthropic":
        llm_client = LLMClient(
            provider="anthropic",
            api_key=config.anthropic_api_key,
            model=config.anthropic_model,
        )
        logger.info(f"Using Anthropic: {config.anthropic_model} (PAID)")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    cache_manager = CacheManager()
    indexer = RapidIndexer(
        storage=storage,
        bm25_index=bm25_index,
        embedding_index=embedding_index,
    )

    retrieval_config = RetrievalConfig(mode=mode)
    retriever = Retriever(
        bm25_index=bm25_index,
        embedding_index=embedding_index,
        llm_client=llm_client,
        cache_manager=cache_manager,
        config=retrieval_config,
    )

    # Load dataset
    dataset = FinanceBenchDataset()
    questions, documents = dataset.load()

    # Index documents
    logger.info(f"Indexing {len(documents)} documents...")
    for doc_path in documents:
        try:
            indexer.index_document(str(doc_path))
        except Exception as exc:
            logger.error(f"Failed to index {doc_path}: {exc}")

    # Run queries
    logger.info(f"Running {sample_size} queries...")
    results = []

    for i, question_data in enumerate(questions[:sample_size]):
        query = question_data["question"]
        ground_truth = question_data.get("answer", "")

        logger.info(f"Query {i + 1}/{sample_size}: {query}")

        start_time = time.time()

        try:
            result = await retriever.search(query)
            latency_ms = (time.time() - start_time) * 1000
            answer = result.answer

            result_data = {
                "query": query,
                "answer": answer,
                "ground_truth": ground_truth,
                "exact_match": exact_match(answer, ground_truth),
                "contains_match": contains_match(answer, ground_truth),
                "number_match": number_match(answer, ground_truth),
                "confidence": result.confidence,
                "latency_ms": latency_ms,
                "num_sections": len(result.sections),
                # FIX: use plain string so json.dump never hits a
                # non-serialisable enum value.
                "mode": mode_value,
                "provider": provider,
            }
            results.append(result_data)

            logger.info(
                f"Latency: {latency_ms:.2f}ms, "
                f"Confidence: {result.confidence}, "
                f"Match: {result_data['contains_match']}"
            )

        except Exception as exc:
            logger.error(f"Query failed: {exc}")
            results.append({
                "query": query,
                "error": str(exc),
                "latency_ms": 0,
                "confidence": 0,
                "mode": mode_value,
                "provider": provider,
            })

    metrics = calculate_accuracy(results)
    metrics["provider"] = provider  # type: ignore

    logger.success("Benchmark complete!")
    logger.info(f"Provider: {provider}")
    logger.info(f"Accuracy: {metrics['contains_accuracy']:.1%}")
    logger.info(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")

    output = {
        # FIX: plain string, not enum object
        "mode": mode_value,
        "provider": provider,
        "sample_size": sample_size,
        "metrics": metrics,
        "results": results,
    }

    with open(output_file, "w") as fh:
        json.dump(output, fh, indent=2)

    logger.info(f"Results saved to: {output_file}")
    return metrics


if __name__ == "__main__":
    asyncio.run(
        run_benchmark(mode=RetrievalMode.REASONING, sample_size=50)
    )
