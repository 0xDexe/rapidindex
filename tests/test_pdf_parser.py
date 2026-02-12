# tests/unit/test_pdf_parser.py
"""
Unit tests for PDF parser.
"""

import pytest
from pathlib import Path
from rapidindex.parsers.pdf_parser import PDFParser, PDFParserConfig
from rapidindex.core.exceptions import FileNotFoundError, UnsupportedFormatError


class TestPDFParser:
    """Test suite for PDF parser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return PDFParser()
    
    @pytest.fixture
    def parser_with_ocr(self):
        """Create parser with OCR enabled."""
        config = PDFParserConfig(
            use_ocr=True,
            extract_tables=True,
            extract_images=True
        )
        return PDFParser(config)
    
    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser.supported_formats == ['.pdf']
        assert isinstance(parser.config, PDFParserConfig)
    
    def test_file_not_found(self, parser):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            parser.parse('nonexistent.pdf')
    
    def test_unsupported_format(self, parser):
        """Test error handling for wrong format."""
        with pytest.raises(UnsupportedFormatError):
            parser.parse('document.docx')
    
    def test_parse_simple_pdf(self, parser, tmp_path):
        """Test parsing a simple text-based PDF."""
        # This would use a fixture PDF file
        # pdf_file = tmp_path / "test.pdf"
        # document = parser.parse(pdf_file)
        # assert len(document.sections) > 0
        # assert document.document_type == 'pdf'
        pass
    
    def test_metadata_extraction(self, parser):
        """Test PDF metadata extraction."""
        # Test with real PDF
        pass
    
    def test_table_extraction(self, parser_with_ocr):
        """Test table extraction from PDF."""
        # Test with PDF containing tables
        pass
    
    def test_ocr_fallback(self, parser_with_ocr):
        """Test OCR on scanned PDF."""
        # Test with scanned PDF
        pass
    
    def test_metrics_collection(self, parser):
        """Test that metrics are collected."""
        # Parse document
        # metrics = parser.get_metrics()
        # assert metrics.parsing_time_seconds > 0
        pass