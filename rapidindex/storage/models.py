# rapidindex/storage/models.py
"""SQLAlchemy models for SQLite."""

from sqlalchemy import Column, String, Integer, Text, JSON, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


class DocumentModel(Base):
    """Document table."""
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    metadata = Column(JSON)
    indexed_at = Column(DateTime, default=datetime.now)
    
    # Relationship
    sections = relationship("SectionModel", back_populates="document", cascade="all, delete-orphan")


class SectionModel(Base):
    """Section table."""
    __tablename__ = 'sections'
    
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    page_numbers = Column(JSON)
    level = Column(Integer, default=1)
    keywords = Column(JSON)
    embedding = Column(JSON)  # Store as JSON array
    
    # Relationship
    document = relationship("DocumentModel", back_populates="sections")


def init_db(database_url: str):
    """Initialize database."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get database session."""
    Session = sessionmaker(bind=engine)
    return Session()


__all__ = ['DocumentModel', 'SectionModel', 'init_db', 'get_session', 'Base']