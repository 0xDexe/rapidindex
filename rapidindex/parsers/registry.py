# rapidindex/parsers/registry.py
"""Parser registry for managing document parsers."""

from typing import Dict, Optional, Type
from pathlib import Path
from loguru import logger

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .text_parser import TxtParser


class ParserRegistry:
    """Registry for document parsers."""
    
    def __init__(self):
        self._parsers: Dict[str, Type[BaseParser]] = {}
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """Register built-in parsers."""
        self.register(PDFParser)
        logger.info("Default parsers registered")
    
    def register(self, parser_class: Type[BaseParser]):
        """Register a parser."""
        for fmt in parser_class.supported_formats:
            self._parsers[fmt] = parser_class
            logger.debug(f"Registered parser for {fmt}: {parser_class.__name__}")
    
    def get_parser(self, file_path: str, config=None) -> Optional[BaseParser]:
        """Get appropriate parser for file."""
        ext = Path(file_path).suffix.lower()
        
        parser_class = self._parsers.get(ext)
        if parser_class:
            return parser_class(config=config)
        
        logger.warning(f"No parser found for extension: {ext}")
        return None
    
    def supports(self, file_path: str) -> bool:
        """Check if file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self._parsers


# Global registry instance
registry = ParserRegistry()


__all__ = ['ParserRegistry', 'registry']