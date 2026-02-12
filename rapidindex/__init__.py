# rapidindex/__init__.py
"""RapidIndex - Production-grade reasoning-based RAG."""

from .core.indexer import RapidIndexer
from .core.retriever import Retriever, RetrievalMode, SearchResult
from .core.document import Document, DocumentSection
from .utils.config import config

__version__ = "0.1.0"

__all__ = [
    'RapidIndexer',
    'Retriever',
    'RetrievalMode',
    'SearchResult',
    'Document',
    'DocumentSection',
    'config'
]