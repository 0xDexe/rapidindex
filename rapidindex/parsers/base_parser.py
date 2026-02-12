# rapidindex/parsers/base_parser.py
"""
Base parser module for document parsing.

This module provides the abstract base class for all document parsers,
defining the interface and common functionality for parsing different
document formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import hashlib
import mimetypes
from datetime import datetime

from pydantic import BaseModel, Field, validator
from loguru import logger

from ..core.document import Document, DocumentSection
from ..core.exceptions import (
    ParseError,
    UnsupportedFormatError,
    FileNotFoundError as RapidIndexFileNotFoundError,
    FileSizeExceededError
)


class ParserConfig(BaseModel):
    """Configuration for document parsers."""
    
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    extract_metadata: bool = Field(default=True)
    extract_images: bool = Field(default=False)
    min_section_length: int = Field(default=50, ge=0)
    max_section_length: int = Field(default=10000, ge=100)
    preserve_formatting: bool = Field(default=False)
    language: Optional[str] = Field(default=None)
    
    class Config:
        frozen = True  # Immutable config


class ParserMetrics(BaseModel):
    """Metrics collected during parsing."""
    
    parsing_time_seconds: float
    file_size_bytes: int
    num_pages: int = 0
    num_sections: int = 0
    num_images: int = 0
    num_tables: int = 0
    word_count: int = 0
    character_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return self.dict()


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.
    
    This class defines the interface that all parsers must implement
    and provides common functionality for document parsing operations.
    
    Attributes:
        supported_formats: List of file extensions this parser supports
        config: Parser configuration settings
        
    Example:
        >>> class MyParser(BaseParser):
        ...     supported_formats = ['.txt']
        ...     def _parse_implementation(self, file_path: Path) -> Document:
        ...         # Implementation here
        ...         pass
        >>> parser = MyParser()
        >>> document = parser.parse('document.txt')
    """
    
    # Subclasses must define supported formats
    supported_formats: List[str] = []
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize the parser.
        
        Args:
            config: Optional parser configuration. If None, uses defaults.
        """
        self.config = config or ParserConfig()
        self._metrics: Optional[ParserMetrics] = None
        
        logger.info(
            f"Initialized {self.__class__.__name__}",
            supported_formats=self.supported_formats,
            config=self.config.dict()
        )
    
    def parse(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Parse a document file into a structured Document object.
        
        This is the main entry point for parsing. It handles validation,
        error handling, and metrics collection, then delegates to the
        subclass-specific _parse_implementation method.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom document ID. If None, auto-generated.
            metadata: Optional additional metadata to attach to document
            
        Returns:
            Document: Parsed document with sections and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If file format not supported
            FileSizeExceededError: If file exceeds size limit
            ParseError: If parsing fails
            
        Example:
            >>> parser = PDFParser()
            >>> doc = parser.parse('report.pdf')
            >>> print(f"Parsed {len(doc.sections)} sections")
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # Validate file
            self._validate_file(file_path)
            
            logger.info(
                "Starting document parsing",
                file_path=str(file_path),
                parser=self.__class__.__name__
            )
            
            # Delegate to subclass implementation
            document = self._parse_implementation(file_path)
            
            # Post-process document
            document = self._post_process(
                document,
                file_path,
                document_id,
                metadata
            )
            
            # Calculate metrics
            end_time = datetime.now()
            self._metrics = ParserMetrics(
                parsing_time_seconds=(end_time - start_time).total_seconds(),
                file_size_bytes=file_path.stat().st_size,
                num_sections=len(document.sections),
                word_count=self._count_words(document),
                character_count=self._count_characters(document)
            )
            
            logger.success(
                "Document parsed successfully",
                document_id=document.id,
                num_sections=len(document.sections),
                parsing_time=self._metrics.parsing_time_seconds
            )
            
            return document
            
        except Exception as e:
            logger.error(
                "Parsing failed",
                file_path=str(file_path),
                error=str(e),
                error_type=type(e).__name__
            )
            
            if isinstance(e, (
                RapidIndexFileNotFoundError,
                UnsupportedFormatError,
                FileSizeExceededError,
                ParseError
            )):
                raise
            else:
                raise ParseError(
                    f"Unexpected error parsing {file_path}: {str(e)}"
                ) from e
    
    @abstractmethod
    def _parse_implementation(self, file_path: Path) -> Document:
        """
        Parse the document file (implemented by subclasses).
        
        This method must be implemented by all parser subclasses to handle
        the specific parsing logic for their document format.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Document: Parsed document structure
            
        Raises:
            ParseError: If parsing fails
        """
        pass
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate the file before parsing.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If format not supported
            FileSizeExceededError: If file too large
        """
        # Check file exists
        if not file_path.exists():
            raise RapidIndexFileNotFoundError(
                f"File not found: {file_path}"
            )
        
        # Check file format
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise UnsupportedFormatError(
                f"Unsupported format '{file_extension}'. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise FileSizeExceededError(
                f"File size ({file_size_mb:.2f}MB) exceeds maximum "
                f"allowed size ({self.config.max_file_size_mb}MB)"
            )
        
        logger.debug(
            "File validation passed",
            file_path=str(file_path),
            size_mb=f"{file_size_mb:.2f}"
        )
    
    def _post_process(
        self,
        document: Document,
        file_path: Path,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Post-process the parsed document.
        
        Args:
            document: Parsed document
            file_path: Original file path
            document_id: Optional custom document ID
            metadata: Optional additional metadata
            
        Returns:
            Document: Post-processed document
        """
        # Set or override document ID
        if document_id:
            document.id = document_id
        elif not document.id:
            document.id = self._generate_document_id(file_path)
        
        # Merge metadata
        if metadata:
            document.metadata.update(metadata)
        
        # Add parser metadata
        document.metadata.update({
            'parser': self.__class__.__name__,
            'parsed_at': datetime.now().isoformat(),
            'file_path': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower()
        })
        
        # Filter sections by length
        if self.config.min_section_length > 0:
            original_count = len(document.sections)
            document.sections = [
                s for s in document.sections
                if len(s.content) >= self.config.min_section_length
            ]
            filtered_count = original_count - len(document.sections)
            if filtered_count > 0:
                logger.debug(
                    f"Filtered {filtered_count} sections below minimum length",
                    min_length=self.config.min_section_length
                )
        
        # Validate sections
        for i, section in enumerate(document.sections):
            if not section.id:
                section.id = f"sec_{i+1:04d}"
        
        return document
    
    def _generate_document_id(self, file_path: Path) -> str:
        """
        Generate a unique document ID from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Unique document ID (12-character hash)
        """
        # Use file path and modification time for uniqueness
        unique_string = f"{file_path.absolute()}_{file_path.stat().st_mtime}"
        hash_object = hashlib.sha256(unique_string.encode())
        return hash_object.hexdigest()[:12]
    
    def _extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """
        Extract keywords from text (simple implementation).
        
        Override this method in subclasses for more sophisticated
        keyword extraction using NLP libraries.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List[str]: Extracted keywords
        """
        import re
        
        # Simple word extraction
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_counts.most_common(max_keywords)]
    
    def _count_words(self, document: Document) -> int:
        """
        Count total words in document.
        
        Args:
            document: Document to count words in
            
        Returns:
            int: Total word count
        """
        total = 0
        for section in document.sections:
            total += len(section.content.split())
        return total
    
    def _count_characters(self, document: Document) -> int:
        """
        Count total characters in document.
        
        Args:
            document: Document to count characters in
            
        Returns:
            int: Total character count
        """
        total = 0
        for section in document.sections:
            total += len(section.content)
        return total
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Override this in subclasses to use libraries like langdetect.
        
        Args:
            text: Text to detect language for
            
        Returns:
            str: ISO language code (e.g., 'en', 'es')
        """
        # Default implementation - subclasses should implement proper language detection
        return 'en'
    
    def get_metrics(self) -> Optional[ParserMetrics]:
        """
        Get metrics from the last parse operation.
        
        Returns:
            ParserMetrics: Metrics from last parse, or None if no parse yet
        """
        return self._metrics
    
    def supports_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this parser supports the given file format.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if format is supported
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List[str]: List of supported extensions
        """
        return cls.supported_formats
    
    def __repr__(self) -> str:
        """String representation of parser."""
        return (
            f"{self.__class__.__name__}("
            f"formats={self.supported_formats})"
        )


class TextBasedParser(BaseParser):
    """
    Base class for text-based document parsers.
    
    Provides common functionality for parsers that work with
    text-based formats (PDF, DOCX, HTML, etc.).
    """
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of paragraphs
        """
        import re
        
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter empty paragraphs
        paragraphs = [
            self._clean_text(p)
            for p in paragraphs
            if p.strip()
        ]
        
        return paragraphs
    
    def _is_heading(self, text: str, next_text: Optional[str] = None) -> bool:
        """
        Heuristic to detect if text is a heading.
        
        Args:
            text: Text to check
            next_text: Following text (for context)
            
        Returns:
            bool: True if likely a heading
        """
        import re
        
        if not text or len(text) > 200:
            return False
        
        # Check for numbered headings (1. Title, 1.1 Subtitle)
        if re.match(r'^[\d\.]+\s+[A-Z]', text):
            return True
        
        # Check for all caps (but not too long)
        if text.isupper() and len(text.split()) >= 2 and len(text) < 100:
            return True
        
        # Check for title case without ending punctuation
        if (text.istitle() and 
            not text.endswith(('.', ',', ';', ':')) and
            len(text.split()) >= 2):
            return True
        
        # Check if short and followed by longer text
        if (next_text and 
            len(text) < 100 and 
            len(next_text) > len(text) * 2):
            return True
        
        return False


# Export public API
__all__ = [
    'BaseParser',
    'TextBasedParser',
    'ParserConfig',
    'ParserMetrics',
]