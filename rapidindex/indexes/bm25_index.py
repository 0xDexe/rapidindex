# rapidindex/indexes/bm25_index.py

import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import nltk
from typing import List, Tuple, Dict, Optional
from loguru import logger

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class BM25Index:
    """BM25-based keyword search index."""

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.documents: Dict = {}           # doc_id -> Document
        self.section_map: Dict = {}         # section_id -> (doc_id, section)

        # FIX: explicit ordered list that mirrors tokenized_corpus position-for-position.
        # Previously we used list(self.section_map.keys())[idx] which breaks
        # the moment any dict ordering assumption is violated (e.g. after a
        # pickle round-trip on Python < 3.7, or after selective deletion).
        self.section_ids: List[str] = []    # index i -> section_id
        self.tokenized_corpus: List[List[str]] = []

        logger.info("BM25Index initialized")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_document(self, document) -> None:
        """Add document to BM25 index (idempotent per section)."""
        self.documents[document.id] = document

        added = 0
        for section in document.sections:
            if section.id in self.section_map:
                # Already indexed — skip to keep corpus/section_ids in sync.
                continue

            self.section_map[section.id] = (document.id, section)
            tokens = self._tokenize(section.title + " " + section.content)
            self.section_ids.append(section.id)
            self.tokenized_corpus.append(tokens)
            added += 1

        if added:
            self._rebuild_index()

        logger.info(
            f"Document added to BM25: {document.id} "
            f"({added} new sections, {len(document.sections) - added} already indexed)"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return (section_id, score) pairs sorted by relevance."""
        if not self.index:
            logger.warning("BM25 index is empty")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.section_ids):
                score = float(scores[idx])
                if score > 0:                   # skip zero-score noise
                    results.append((self.section_ids[idx], score))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> bool:
        """Pickle the index to disk so it survives restarts."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(
                    {
                        "section_ids": self.section_ids,
                        "tokenized_corpus": self.tokenized_corpus,
                        "section_map": self.section_map,
                        "documents": self.documents,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"BM25 index saved: {path} ({len(self.section_ids)} sections)")
            return True
        except Exception as exc:
            logger.error(f"Failed to save BM25 index: {exc}")
            return False

    def load(self, path: str) -> bool:
        """Load a previously saved index from disk."""
        if not Path(path).exists():
            return False
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)

            self.section_ids = data["section_ids"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.section_map = data["section_map"]
            self.documents = data["documents"]
            self._rebuild_index()

            logger.info(
                f"BM25 index loaded: {path} ({len(self.section_ids)} sections)"
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load BM25 index: {exc}")
            # Reset to clean state so callers can rebuild safely
            self.__init__()
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        tokens = nltk.word_tokenize(text.lower())
        return [t for t in tokens if len(t) > 2 and t.isalnum()]

    def _rebuild_index(self) -> None:
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
            logger.debug(f"BM25 index rebuilt: {len(self.tokenized_corpus)} sections")


__all__ = ["BM25Index"]
