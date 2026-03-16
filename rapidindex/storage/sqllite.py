# rapidindex/storage/sqlite.py
"""SQLite storage backend."""

from typing import List, Optional
from sqlalchemy.orm import Session
from loguru import logger

from .models import DocumentModel, SectionModel, init_db, get_session
from ..core.document import Document, DocumentSection


class SQLiteStorage:
    """SQLite storage backend."""

    def __init__(self, database_url: str = "sqlite:///./rapidindex.db"):
        self.database_url = database_url
        self.engine = init_db(database_url)
        logger.info(f"SQLite storage initialized: {database_url}")

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def save_document(self, document: Document) -> bool:
        session = get_session(self.engine)
        try:
            doc_model = DocumentModel(
                id=document.id,
                title=document.title,
                file_path=document.file_path,
                document_type=document.document_type,
                metadata=document.metadata,
                indexed_at=document.indexed_at,
            )
            for section in document.sections:
                doc_model.sections.append(
                    SectionModel(
                        id=section.id,
                        document_id=document.id,
                        title=section.title,
                        content=section.content,
                        page_numbers=section.page_numbers,
                        level=section.level,
                        keywords=section.keywords,
                        embedding=section.embedding,
                    )
                )
            session.add(doc_model)
            session.commit()
            logger.info(f"Document saved: {document.id}")
            return True
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save document: {exc}")
            return False
        finally:
            session.close()

    def get_document(self, doc_id: str) -> Optional[Document]:
        session = get_session(self.engine)
        try:
            doc_model = session.query(DocumentModel).filter_by(id=doc_id).first()
            if not doc_model:
                return None
            return self._model_to_document(doc_model)
        finally:
            session.close()

    def list_documents(self) -> List[Document]:
        session = get_session(self.engine)
        try:
            docs = session.query(DocumentModel).all()
            result = []
            for d in docs:
                metadata = self._parse_metadata(d.metadata)
                result.append(
                    Document(
                        id=d.id,
                        title=d.title,
                        file_path=d.file_path,
                        document_type=d.document_type,
                        sections=[],   # lightweight listing — no section bodies
                        metadata=metadata,
                        indexed_at=d.indexed_at,
                    )
                )
            return result
        finally:
            session.close()

    def delete_document(self, doc_id: str) -> bool:
        session = get_session(self.engine)
        try:
            doc = session.query(DocumentModel).filter_by(id=doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                logger.info(f"Document deleted: {doc_id}")
                return True
            return False
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to delete document: {exc}")
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Embeddings persistence
    # ------------------------------------------------------------------

    def update_section_embedding(
        self, section_id: str, embedding: List[float]
    ) -> bool:
        """Persist a computed embedding vector for a section.

        Called by the indexer after EmbeddingIndex.add_section() so that
        warm-reload can restore embeddings without recomputing them.
        """
        session = get_session(self.engine)
        try:
            section = session.query(SectionModel).filter_by(id=section_id).first()
            if section:
                section.embedding = embedding
                session.commit()
                return True
            logger.warning(f"Section not found for embedding update: {section_id}")
            return False
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to update section embedding {section_id}: {exc}")
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _model_to_document(self, doc_model: DocumentModel) -> Document:
        sections = [
            DocumentSection(
                id=s.id,
                title=s.title,
                content=s.content,
                page_numbers=s.page_numbers or [],
                level=s.level,
                keywords=s.keywords or [],
                embedding=s.embedding,   # may be None or List[float]
            )
            for s in doc_model.sections
        ]
        return Document(
            id=doc_model.id,
            title=doc_model.title,
            file_path=doc_model.file_path,
            document_type=doc_model.document_type,
            sections=sections,
            metadata=self._parse_metadata(doc_model.metadata),
            indexed_at=doc_model.indexed_at,
        )

    @staticmethod
    def _parse_metadata(raw) -> dict:
        if not raw:
            return {}
        if isinstance(raw, str):
            import json
            return json.loads(raw)
        return raw


__all__ = ["SQLiteStorage"]
