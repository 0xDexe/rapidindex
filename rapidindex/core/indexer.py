# rapidindex/core/indexer.py

from typing import Optional, List
from pathlib import Path
from loguru import logger

from .document import Document
from ..parsers.registry import registry
from ..indexes.bm25_index import BM25Index
from ..indexes.embedding_index import EmbeddingIndex
from ..storage.sqllite import SQLiteStorage
from ..utils.config import config


class RapidIndexer:
    """Main document indexer."""
    
    def __init__(
        self,
        storage: Optional[SQLiteStorage] = None,
        bm25_index: Optional[BM25Index] = None,
        embedding_index: Optional[EmbeddingIndex] = None
    ):
        
        self.storage = storage or SQLiteStorage(config.database_url)
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_index = embedding_index or EmbeddingIndex(config.embedding_model)
        
        logger.info("RapidIndexer initialized")
    
    def index_document(self, file_path: str) -> str:
        """Index a document."""
        logger.info(f"Indexing document: {file_path}")
        
        # Get parser
        parser = registry.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for: {file_path}")
        
        # Parse document
        document = parser.parse(file_path)
        
        # Save to storage
        self.storage.save_document(document)
        
        # Add to BM25 index
        self.bm25_index.add_document(document)
        
        # Add to embedding index
        for section in document.sections:
            self.embedding_index.add_section(
                section.id,
                f"{section.title}\n{section.content}"
            )
        
        logger.success(f"Document indexed: {document.id}")
        return document.id
    
    def index_directory(self, directory: str) -> List[str]:
        """Index all documents in directory."""
        dir_path = Path(directory)
        doc_ids = []
        
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and registry.supports(str(file_path)):
                try:
                    doc_id = self.index_document(str(file_path))
                    doc_ids.append(doc_id)
                except Exception as e:
                    logger.error(f"Failed to index {file_path}: {e}")
        
        logger.info(f"Indexed {len(doc_ids)} documents from {directory}")
        return doc_ids


__all__ = ['RapidIndexer']