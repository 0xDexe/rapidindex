# rapidindex/storage/models.py
"""SQLAlchemy models for SQLite."""

from sqlalchemy import Column, String, Integer, Text, JSON, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


class DocumentModel(Base):
    """Document table."""
    __tablename__ = 'documents'

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title         = Column(String, nullable=False)
    file_path     = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    # FIX: 'metadata' is reserved by SQLAlchemy's DeclarativeBase.
    # Renamed to 'doc_metadata' in Python; the DB column stays 'doc_metadata'.
    doc_metadata  = Column(JSON, name='doc_metadata')
    indexed_at    = Column(DateTime, default=datetime.now)

    sections = relationship(
        "SectionModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class SectionModel(Base):
    """Section table."""
    __tablename__ = 'sections'

    id            = Column(String, primary_key=True)
    document_id   = Column(String, ForeignKey('documents.id'), nullable=False)
    title         = Column(String, nullable=False)
    content       = Column(Text, nullable=False)
    page_numbers  = Column(JSON)
    level         = Column(Integer, default=1)
    keywords      = Column(JSON)
    embedding     = Column(JSON)   # stored as a JSON array of floats

    document = relationship("DocumentModel", back_populates="sections")


def init_db(database_url: str):
    """Initialize database and create tables."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get a new database session."""
    Session = sessionmaker(bind=engine)
    return Session()


__all__ = ['DocumentModel', 'SectionModel', 'init_db', 'get_session', 'Base']
