#!/usr/bin/env python3
# benchmarking/run_benchmark.py
"""
Run RapidIndex against FinanceBench.

Quick start
-----------
# Smoke test — 5 questions, local Ollama
python -m benchmarking.run_benchmark --sample-size 5 --provider ollama

# Full open-source set — 150 questions, Anthropic
python -m benchmarking.run_benchmark --sample-size 150 --provider anthropic

# Keyword-only mode (no LLM, measures pure retrieval)
python -m benchmarking.run_benchmark --sample-size 20 --mode keyword

# Single company deep-dive
python -m benchmarking.run_benchmark --company 3M --provider ollama
"""

import argparse
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
    company_filter: Optional[str] = None,
    data_dir: str = "./benchmarking/datasets/financebench",
) -> dict:

    provider = llm_provider or config.llm_provider
    mode_value: str = mode.value if isinstance(mode, RetrievalMode) else str(mode)

    logger.info(f"Mode: {mode_value}  |  Provider: {provider}  |  Sample: {sample_size}")

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    storage = SQLiteStorage(config.database_url)
    bm25_index = BM25Index()
    embedding_index = EmbeddingIndex()

    if provider == "ollama":
        llm_client = LLMClient(
            provider="ollama",
            model=config.ollama_model,
            ollama_url=config.ollama_url,
        )
    elif provider == "anthropic":
        llm_client = LLMClient(
            provider="anthropic",
            api_key=config.anthropic_api_key,
            model=config.anthropic_model,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    cache_manager = CacheManager()
    indexer = RapidIndexer(
        storage=storage,
        bm25_index=bm25_index,
        embedding_index=embedding_index,
    )
    retriever = Retriever(
        bm25_index=bm25_index,
        embedding_index=embedding_index,
        llm_client=llm_client,
        cache_manager=cache_manager,
        config=RetrievalConfig(mode=mode),
    )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset = FinanceBenchDataset(data_dir=data_dir)
    questions, documents = dataset.load()

    if company_filter:
        questions = [
            q for q in questions
            if q.get("company", "").upper() == company_filter.upper()
        ]
        logger.info(f"Filtered to company={company_filter}: {len(questions)} questions")

    if not questions:
        logger.error("No questions loaded — run: python -m benchmarking.setup_financebench")
        return {}

    if not documents:
        logger.error(
            "No PDFs found — run: "
            "python -m benchmarking.setup_financebench --download-pdfs"
        )
        return {}

    # ------------------------------------------------------------------
    # Index documents (warm-reload skips already-indexed ones)
    # ------------------------------------------------------------------
    logger.info(f"Indexing {len(documents)} PDFs...")
    for doc_path in documents:
        try:
            indexer.index_document(str(doc_path))
        except Exception as exc:
            logger.error(f"Failed to index {doc_path.name}: {exc}")

    # ------------------------------------------------------------------
    # Run queries
    # ------------------------------------------------------------------
    questions_to_run = questions[:sample_size]
    logger.info(f"Running {len(questions_to_run)} queries...")
    results = []

    for i, q in enumerate(questions_to_run):
        query        = q["question"]
        ground_truth = q.get("answer", "")
        doc_name     = q.get("doc_name", "")
        q_type       = q.get("question_type", "unknown")

        pdf_present = (Path(data_dir) / "pdfs" / f"{doc_name}.pdf").exists()
        if not pdf_present:
            logger.warning(f"PDF not on disk, answer may be wrong: {doc_name}")

        logger.info(f"[{i+1}/{len(questions_to_run)}] ({q_type}) {query[:80]}...")

        t0 = time.time()
        try:
            result = await retriever.search(query)
            latency_ms = (time.time() - t0) * 1000
            answer = result.answer

            row = {
                "financebench_id": q.get("financebench_id"),
                "company":         q.get("company"),
                "doc_name":        doc_name,
                "question_type":   q_type,
                "question":        query,
                "answer":          answer,
                "ground_truth":    ground_truth,
                "exact_match":     exact_match(answer, ground_truth),
                "contains_match":  contains_match(answer, ground_truth),
                "number_match":    number_match(answer, ground_truth),
                "confidence":      result.confidence,
                "latency_ms":      latency_ms,
                "num_sections":    len(result.sections),
                "mode":            mode_value,
                "provider":        provider,
            }
            results.append(row)

            mark = "✓" if row["contains_match"] else "✗"
            logger.info(
                f"  {mark}  {latency_ms:.0f}ms  "
                f"conf={result.confidence:.2f}  "
                f"{answer[:60]}..."
            )

        except Exception as exc:
            logger.error(f"  Query failed: {exc}")
            results.append({
                "financebench_id": q.get("financebench_id"),
                "company":         q.get("company"),
                "doc_name":        doc_name,
                "question_type":   q_type,
                "question":        query,
                "ground_truth":    ground_truth,
                "error":           str(exc),
                "latency_ms":      0.0,
                "confidence":      0.0,
                "mode":            mode_value,
                "provider":        provider,
            })

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    overall = calculate_accuracy(results)
    overall["provider"] = provider
    overall["mode"] = mode_value

    by_type: dict = {}
    for q_type in FinanceBenchDataset.QUESTION_TYPES:
        subset = [r for r in results if r.get("question_type") == q_type]
        if subset:
            by_type[q_type] = calculate_accuracy(subset)

    _print_report(overall, by_type, mode_value, provider)

    output = {
        "mode":        mode_value,
        "provider":    provider,
        "sample_size": sample_size,
        "metrics":     overall,
        "by_type":     by_type,
        "results":     results,
    }
    with open(output_file, "w") as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"Results saved → {output_file}")

    return overall


def _print_report(overall: dict, by_type: dict, mode: str, provider: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RapidIndex  ·  FinanceBench Results")
    print(f"  Mode: {mode}   Provider: {provider}")
    print(sep)
    print(f"  Total questions : {overall.get('total', 0)}")
    print(f"  Contains match  : {overall.get('contains_accuracy', 0):.1%}")
    print(f"  Number match    : {overall.get('number_accuracy', 0):.1%}")
    print(f"  Avg latency     : {overall.get('avg_latency_ms', 0):.0f} ms")
    print(f"  Avg confidence  : {overall.get('avg_confidence', 0):.2f}")

    if by_type:
        print(f"\n  By question type:")
        for q_type, m in by_type.items():
            label = FinanceBenchDataset.QUESTION_TYPES.get(q_type, q_type)
            print(
                f"    {q_type:<22}  n={m['total']:<4}  "
                f"contains={m['contains_accuracy']:.1%}  "
                f"number={m['number_accuracy']:.1%}"
            )
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RapidIndex on FinanceBench")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in RetrievalMode],
        default=RetrievalMode.REASONING.value,
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--provider", choices=["anthropic", "ollama"], default=None)
    parser.add_argument("--company", default=None, help="Filter to one company, e.g. 3M")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument(
        "--data-dir",
        default="./benchmarking/datasets/financebench",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            mode=RetrievalMode(args.mode),
            sample_size=args.sample_size,
            output_file=args.output,
            llm_provider=args.provider,
            company_filter=args.company,
            data_dir=args.data_dir,
        )
    )


if __name__ == "__main__":
    main()
