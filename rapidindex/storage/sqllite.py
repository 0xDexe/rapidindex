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
        """Initialize SQLite storage."""
        self.database_url = database_url
        self.engine = init_db(database_url)
        logger.info(f"SQLite storage initialized: {database_url}")
    
    def save_document(self, document: Document) -> bool:
        """Save document to database."""
        session = get_session(self.engine)
        
        try:
            # Create document model
            doc_model = DocumentModel(
                id=document.id,
                title=document.title,
                file_path=document.file_path,
                document_type=document.document_type,
                metadata=document.metadata,
                indexed_at=document.indexed_at
            )
            
            # Create section models
            for section in document.sections:
                section_model = SectionModel(
                    id=section.id,
                    document_id=document.id,
                    title=section.title,
                    content=section.content,
                    page_numbers=section.page_numbers,
                    level=section.level,
                    keywords=section.keywords,
                    embedding=section.embedding
                )
                doc_model.sections.append(section_model)
            
            session.add(doc_model)
            session.commit()
            logger.info(f"Document saved: {document.id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save document: {e}")
            return False
        finally:
            session.close()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        session = get_session(self.engine)
        
        try:
            doc_model = session.query(DocumentModel).filter_by(id=doc_id).first()
            
            if not doc_model:
                return None
            
            # Convert to Document
            sections = [
                DocumentSection(
                    id=s.id,
                    title=s.title,
                    content=s.content,
                    page_numbers=s.page_numbers or [],
                    level=s.level,
                    keywords=s.keywords or [],
                    embedding=s.embedding
                )
                for s in doc_model.sections
            ]
            
            metadata = doc_model.metadata if doc_model.metadata else {}
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            
            return Document(
                id=doc_model.id,
                title=doc_model.title,
                file_path=doc_model.file_path,
                document_type=doc_model.document_type,
                sections=sections,
                metadata=metadata,
                indexed_at=doc_model.indexed_at
            )
            
        finally:
            session.close()
    
    def list_documents(self) -> List[Document]:
        """List all documents."""
        session = get_session(self.engine)
        
        try:
            docs = session.query(DocumentModel).all()
            result = []
            for d in docs:
                metadata = d.metadata if d.metadata else {}
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)
                result.append(Document(
                    id=d.id,
                    title=d.title,
                    file_path=d.file_path,
                    document_type=d.document_type,
                    sections=[],  # Don't load sections for listing
                    metadata=metadata,
                    indexed_at=d.indexed_at
                ))
            return result
        finally:
            session.close()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document."""
        session = get_session(self.engine)
        
        try:
            doc = session.query(DocumentModel).filter_by(id=doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                logger.info(f"Document deleted: {doc_id}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete document: {e}")
            return False
        finally:
            session.close()


__all__ = ['SQLiteStorage']