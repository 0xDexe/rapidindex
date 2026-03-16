# rapidindex/core/indexer.py

from typing import Optional, List
from pathlib import Path

import numpy as np
from loguru import logger

from .document import Document
from ..parsers.registry import registry
from ..indexes.bm25_index import BM25Index
from ..indexes.embedding_index import EmbeddingIndex
from ..storage.sqllite import SQLiteStorage
from ..utils.config import config


class RapidIndexer:
    """Main document indexer with index persistence and warm-reload."""

    def __init__(
        self,
        storage: Optional[SQLiteStorage] = None,
        bm25_index: Optional[BM25Index] = None,
        embedding_index: Optional[EmbeddingIndex] = None,
    ):
        self.storage = storage or SQLiteStorage(config.database_url)
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_index = embedding_index or EmbeddingIndex(config.embedding_model)

        # Rebuild in-memory indexes from durable storage on every startup.
        # Previously the indexes started empty on every restart, making the
        # system unusable without a full re-index after every process restart.
        self._warm_reload()

        logger.info("RapidIndexer initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(self, file_path: str) -> str:
        """Parse, store, and index a single document."""
        logger.info(f"Indexing document: {file_path}")

        parser = registry.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for: {file_path}")

        document = parser.parse(file_path)

        # Persist to SQLite first so sections have stable IDs.
        self.storage.save_document(document)

        # Add to BM25.
        self.bm25_index.add_document(document)

        # Compute embeddings, write them back to storage so warm-reload
        # can restore them without recomputing.
        for section in document.sections:
            self.embedding_index.add_section(
                section.id,
                f"{section.title}\n{section.content}",
            )
            embedding = self.embedding_index.embeddings.get(section.id)
            if embedding is not None:
                self.storage.update_section_embedding(
                    section.id, embedding.tolist()
                )

        # Snapshot the BM25 index so the next process restart is fast.
        self.bm25_index.save(str(self._bm25_cache_path))

        logger.success(f"Document indexed: {document.id}")
        return document.id

    def index_directory(self, directory: str) -> List[str]:
        """Index all supported documents in a directory."""
        dir_path = Path(directory)
        doc_ids = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and registry.supports(str(file_path)):
                try:
                    doc_id = self._index_document_no_snapshot(str(file_path))
                    doc_ids.append(doc_id)
                except Exception as exc:
                    logger.error(f"Failed to index {file_path}: {exc}")

        # Single BM25 snapshot after bulk indexing is cheaper than one per file.
        if doc_ids:
            self.bm25_index.save(str(self._bm25_cache_path))
            logger.info(f"BM25 index snapshotted after bulk indexing {len(doc_ids)} docs")

        logger.info(f"Indexed {len(doc_ids)} documents from {directory}")
        return doc_ids

    # ------------------------------------------------------------------
    # Warm-reload
    # ------------------------------------------------------------------

    def _warm_reload(self) -> None:
        """Restore in-memory indexes from durable storage.

        Strategy:
        1. Try to load the BM25 pickle (fast — avoids re-tokenising).
        2. Walk every stored document to load/recompute embeddings into
           EmbeddingIndex.embeddings.
        3. If the BM25 pickle was stale or missing, rebuild it from
           storage and write a fresh snapshot.
        """
        documents = self.storage.list_documents()
        if not documents:
            logger.info("Storage is empty — nothing to reload")
            return

        logger.info(f"Warm reload: {len(documents)} documents in storage")

        bm25_loaded = self.bm25_index.load(str(self._bm25_cache_path))

        embeddings_loaded = 0
        embeddings_recomputed = 0

        for doc_summary in documents:
            # get_document returns sections with stored embeddings included.
            doc = self.storage.get_document(doc_summary.id)
            if not doc:
                logger.warning(f"Could not load document body: {doc_summary.id}")
                continue

            # Rebuild BM25 from stored text if the pickle was absent/corrupt.
            if not bm25_loaded:
                self.bm25_index.add_document(doc)

            # Restore or recompute embeddings.
            for section in doc.sections:
                if section.embedding:
                    # Stored as JSON list → convert to numpy for EmbeddingIndex.
                    self.embedding_index.embeddings[section.id] = np.array(
                        section.embedding
                    )
                    embeddings_loaded += 1
                else:
                    # Embedding missing — compute and persist it now.
                    self.embedding_index.add_section(
                        section.id,
                        f"{section.title}\n{section.content}",
                    )
                    emb = self.embedding_index.embeddings.get(section.id)
                    if emb is not None:
                        self.storage.update_section_embedding(
                            section.id, emb.tolist()
                        )
                    embeddings_recomputed += 1

        if not bm25_loaded:
            self.bm25_index.save(str(self._bm25_cache_path))
            logger.info("BM25 index rebuilt from storage and cached")

        logger.info(
            f"Warm reload complete — "
            f"{embeddings_loaded} embeddings restored, "
            f"{embeddings_recomputed} recomputed"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_document_no_snapshot(self, file_path: str) -> str:
        """Like index_document but skips the per-file BM25 snapshot.

        Used by index_directory so we can write a single snapshot at the end.
        """
        parser = registry.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for: {file_path}")

        document = parser.parse(file_path)
        self.storage.save_document(document)
        self.bm25_index.add_document(document)

        for section in document.sections:
            self.embedding_index.add_section(
                section.id, f"{section.title}\n{section.content}"
            )
            embedding = self.embedding_index.embeddings.get(section.id)
            if embedding is not None:
                self.storage.update_section_embedding(
                    section.id, embedding.tolist()
                )

        logger.success(f"Document indexed (no snapshot): {document.id}")
        return document.id

    @property
    def _bm25_cache_path(self) -> Path:
        """Canonical path for the BM25 pickle snapshot."""
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "bm25_index.pkl"


__all__ = ["RapidIndexer"]
