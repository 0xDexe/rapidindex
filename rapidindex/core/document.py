# rapidindex/core/document.py

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):    
    id: str
    title: str
    content: str
    page_numbers: List[int] = Field(default_factory=list)
    level: int = 1
    parent_id: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    """Main document model."""
    
    id: str
    title: str
    file_path: str
    document_type: str
    sections: List[DocumentSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    indexed_at: datetime = Field(default_factory=datetime.now)
    
    def get_all_text(self) -> str:
        """Get full document text."""
        return "\n\n".join([s.content for s in self.sections])
    
    def get_section_by_id(self, section_id: str) -> Optional[DocumentSection]:
        """Retrieve specific section."""
        return next((s for s in self.sections if s.id == section_id), None)
    
    class Config:
        arbitrary_types_allowed = True


__all__ = ['Document', 'DocumentSection']