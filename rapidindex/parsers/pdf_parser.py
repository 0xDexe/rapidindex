# rapidindex/parsers/pdf_parser.py
"""
PDF document parser 
- Text extraction from text-based PDFs
- OCR for scanned PDFs
- Table extraction
- Image extraction
- Metadata extraction
- Hierarchical structure detection
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import re
from io import BytesIO

# PDF libraries
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams

# OCR support (optional)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    convert_from_path = None
    pytesseract = None
    OCR_AVAILABLE = False

# Image handling
try:
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False

from loguru import logger
from pydantic import BaseModel, Field

from .base_parser import TextBasedParser, ParserConfig
from ..core.document import Document, DocumentSection
from ..core.exceptions import ParseError, CorruptedFileError


class PDFParserConfig(ParserConfig):
    
    use_ocr: bool = Field(default=False, description="Enable OCR for scanned PDFs")
    ocr_language: str = Field(default='eng', description="OCR language (e.g., 'eng', 'fra')")
    extract_tables: bool = Field(default=True, description="Extract tables from PDF")
    extract_images: bool = Field(default=False, description="Extract images from PDF")
    merge_hyphenated_words: bool = Field(default=True, description="Merge hyphenated words")
    detect_headers: bool = Field(default=True, description="Auto-detect section headers")
    min_text_per_page: int = Field(default=50, description="Min chars to consider page as text-based")
    dpi: int = Field(default=300, ge=100, le=600, description="DPI for OCR")
    
    # Layout analysis parameters
    line_margin: float = Field(default=0.5, description="pdfminer LAParams line_margin")
    word_margin: float = Field(default=0.1, description="pdfminer LAParams word_margin")
    char_margin: float = Field(default=2.0, description="pdfminer LAParams char_margin")


class PDFMetadata(BaseModel):
    """PDF document metadata."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    keywords: List[str] = Field(default_factory=list)
    num_pages: int = 0
    is_encrypted: bool = False
    is_scanned: bool = False
    pdf_version: Optional[str] = None


class PDFParser(TextBasedParser):
    supported_formats = ['.pdf']
    
    def __init__(self, config: Optional[PDFParserConfig] = None):
        """
        Initialize PDF parser.
        
        Args:
            config: Parser configuration
        """
        self.config = config or PDFParserConfig()
        super().__init__(self.config)
        
        # Check OCR availability
        if self.config.use_ocr and not OCR_AVAILABLE:
            logger.warning(
                "OCR requested but dependencies not installed. "
                "Install with: pip install pdf2image pytesseract"
            )
            self.config.use_ocr = False
    
    def _parse_implementation(self, file_path: Path) -> Document:
        """
        Parse PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document: Parsed document with sections
            
        Raises:
            ParseError: If parsing fails
            CorruptedFileError: If PDF is corrupted
        """
        try:
            # Extract metadata first
            metadata = self._extract_metadata(file_path)
            
            logger.info(
                "PDF metadata extracted",
                num_pages=metadata.num_pages,
                is_scanned=metadata.is_scanned,
                encrypted=metadata.is_encrypted
            )
            
            # Choose extraction method based on PDF type
            if metadata.is_scanned and self.config.use_ocr:
                sections = self._extract_with_ocr(file_path, metadata.num_pages)
            else:
                sections = self._extract_with_text_extraction(file_path, metadata.num_pages)
            
            # Extract tables if enabled
            if self.config.extract_tables:
                tables = self._extract_tables(file_path)
                if tables:
                    logger.info(f"Extracted {len(tables)} tables from PDF")
                    metadata.num_pages  # Add to metadata
            
            # Create document
            document = Document(
                id="",  # Will be set in post_process
                title=metadata.title or self._extract_title_from_content(sections),
                file_path=str(file_path),
                document_type='pdf',
                sections=sections,
                metadata={
                    **metadata.dict(exclude_none=True),
                    'extraction_method': 'ocr' if metadata.is_scanned else 'text'
                }
            )
            
            logger.success(
                "PDF parsed successfully",
                num_sections=len(sections),
                total_text_length=sum(len(s.content) for s in sections)
            )
            
            return document
            
        except Exception as e:
            if "PdfReadError" in type(e).__name__ or "corrupted" in str(e).lower():
                raise CorruptedFileError(
                    f"PDF file is corrupted or invalid: {str(e)}"
                ) from e
            raise ParseError(
                f"Failed to parse PDF: {str(e)}"
            ) from e
    
    def _extract_metadata(self, file_path: Path) -> PDFMetadata:
        """
        Extract PDF metadata.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            PDFMetadata: Extracted metadata
        """
        metadata = PDFMetadata()
        
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                # Basic info
                metadata.num_pages = len(reader.pages)
                metadata.is_encrypted = reader.is_encrypted
                
                # PDF version
                if hasattr(reader, 'pdf_header'):
                    metadata.pdf_version = reader.pdf_header
                
                # Document info
                if reader.metadata:
                    info = reader.metadata
                    metadata.title = self._clean_metadata_string(info.get('/Title'))
                    metadata.author = self._clean_metadata_string(info.get('/Author'))
                    metadata.subject = self._clean_metadata_string(info.get('/Subject'))
                    metadata.creator = self._clean_metadata_string(info.get('/Creator'))
                    metadata.producer = self._clean_metadata_string(info.get('/Producer'))
                    
                    # Parse dates
                    if '/CreationDate' in info:
                        metadata.creation_date = self._parse_pdf_date(str(info['/CreationDate']))
                    if '/ModDate' in info:
                        metadata.modification_date = self._parse_pdf_date(str(info['/ModDate']))
                    
                    # Keywords
                    if '/Keywords' in info:
                        keywords_str = self._clean_metadata_string(info['/Keywords'])
                        if keywords_str:
                            metadata.keywords = [
                                k.strip() for k in keywords_str.split(',')
                            ]
                
                # Check if scanned (low text content on first few pages)
                metadata.is_scanned = self._is_scanned_pdf(reader)
                
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
        
        return metadata
    
    def _is_scanned_pdf(self, reader: PyPDF2.PdfReader) -> bool:
        """
        Heuristic to detect if PDF is scanned.
        
        Args:
            reader: PDF reader object
            
        Returns:
            bool: True if likely scanned
        """
        # Check first 3 pages or all pages if fewer
        pages_to_check = min(3, len(reader.pages))
        
        for i in range(pages_to_check):
            try:
                text = reader.pages[i].extract_text()
                # If any page has substantial text, consider it text-based
                if text and len(text.strip()) > self.config.min_text_per_page:
                    return False
            except Exception:
                continue
        
        return True
    
    def _extract_with_text_extraction(
        self,
        file_path: Path,
        num_pages: int
    ) -> List[DocumentSection]:
        """
        Extract text from text-based PDF using multiple methods.
        
        Args:
            file_path: Path to PDF
            num_pages: Number of pages
            
        Returns:
            List[DocumentSection]: Extracted sections
        """
        sections = []
        
        # Try pdfplumber first (best for layout)
        try:
            sections = self._extract_with_pdfplumber(file_path)
            if sections:
                logger.info("Successfully extracted text with pdfplumber")
                return sections
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            sections = self._extract_with_pypdf2(file_path)
            if sections:
                logger.info("Successfully extracted text with PyPDF2")
                return sections
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Final fallback to pdfminer
        try:
            sections = self._extract_with_pdfminer(file_path)
            if sections:
                logger.info("Successfully extracted text with pdfminer")
                return sections
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {e}")
        
        # If all fail, raise error
        if not sections:
            raise ParseError("All text extraction methods failed")
        
        return sections
    
    def _extract_with_pdfplumber(self, file_path: Path) -> List[DocumentSection]:
        """Extract text using pdfplumber (best for layout preservation)."""
        sections = []
        
        with pdfplumber.open(file_path) as pdf:
            current_section = None
            section_counter = 0
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text with layout
                    text = page.extract_text()
                    
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    # Clean and split into paragraphs
                    paragraphs = self._split_into_paragraphs(text)
                    
                    for i, para in enumerate(paragraphs):
                        if not para.strip():
                            continue
                        
                        # Check if it's a heading
                        next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
                        
                        if self.config.detect_headers and self._is_heading(para, next_para):
                            # Save previous section
                            if current_section and current_section.content.strip():
                                sections.append(current_section)
                            
                            # Start new section
                            section_counter += 1
                            current_section = DocumentSection(
                                id=f"sec_{section_counter:04d}",
                                title=self._clean_text(para),
                                content="",
                                page_numbers=[page_num],
                                level=self._detect_heading_level(para),
                                keywords=[]
                            )
                        else:
                            # Add to current section
                            if current_section is None:
                                # Create first section
                                section_counter += 1
                                current_section = DocumentSection(
                                    id=f"sec_{section_counter:04d}",
                                    title="Introduction",
                                    content="",
                                    page_numbers=[page_num],
                                    level=1,
                                    keywords=[]
                                )
                            
                            current_section.content += para + "\n\n"
                            
                            # Track page numbers
                            if page_num not in current_section.page_numbers:
                                current_section.page_numbers.append(page_num)
                
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue
            
            # Add final section
            if current_section and current_section.content.strip():
                sections.append(current_section)
        
        # Post-process sections
        for section in sections:
            section.content = self._clean_text(section.content)
            if self.config.merge_hyphenated_words:
                section.content = self._merge_hyphenated_words(section.content)
            section.keywords = self._extract_keywords(section.content)
        
        return sections
    
    def _extract_with_pypdf2(self, file_path: Path) -> List[DocumentSection]:
        """Extract text using PyPDF2 (fallback method)."""
        sections = []
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            all_text = []
            page_map = {}  # Track which text came from which page
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        all_text.append(text)
                        page_map[len(all_text) - 1] = page_num
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
        
        # Join all text and split into sections
        full_text = "\n\n".join(all_text)
        paragraphs = self._split_into_paragraphs(full_text)
        
        current_section = None
        section_counter = 0
        
        for i, para in enumerate(paragraphs):
            next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
            
            if self.config.detect_headers and self._is_heading(para, next_para):
                if current_section:
                    sections.append(current_section)
                
                section_counter += 1
                current_section = DocumentSection(
                    id=f"sec_{section_counter:04d}",
                    title=self._clean_text(para),
                    content="",
                    page_numbers=[1],  # Page tracking is less accurate with PyPDF2
                    level=self._detect_heading_level(para),
                    keywords=[]
                )
            else:
                if current_section is None:
                    section_counter += 1
                    current_section = DocumentSection(
                        id=f"sec_{section_counter:04d}",
                        title="Content",
                        content="",
                        page_numbers=[1],
                        level=1,
                        keywords=[]
                    )
                
                current_section.content += para + "\n\n"
        
        if current_section:
            sections.append(current_section)
        
        # Post-process
        for section in sections:
            section.content = self._clean_text(section.content)
            if self.config.merge_hyphenated_words:
                section.content = self._merge_hyphenated_words(section.content)
            section.keywords = self._extract_keywords(section.content)
        
        return sections
    
    def _extract_with_pdfminer(self, file_path: Path) -> List[DocumentSection]:
        """Extract text using pdfminer (most robust fallback)."""
        
        # Configure layout analysis
        laparams = LAParams(
            line_margin=self.config.line_margin,
            word_margin=self.config.word_margin,
            char_margin=self.config.char_margin
        )
        
        # Extract all text
        text = pdfminer_extract_text(
            str(file_path),
            laparams=laparams
        )
        
        if not text or len(text.strip()) < 100:
            return []
        
        # Process into sections
        paragraphs = self._split_into_paragraphs(text)
        sections = []
        current_section = None
        section_counter = 0
        
        for i, para in enumerate(paragraphs):
            next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
            
            if self.config.detect_headers and self._is_heading(para, next_para):
                if current_section:
                    sections.append(current_section)
                
                section_counter += 1
                current_section = DocumentSection(
                    id=f"sec_{section_counter:04d}",
                    title=self._clean_text(para),
                    content="",
                    page_numbers=[1],
                    level=self._detect_heading_level(para),
                    keywords=[]
                )
            else:
                if current_section is None:
                    section_counter += 1
                    current_section = DocumentSection(
                        id=f"sec_{section_counter:04d}",
                        title="Content",
                        content="",
                        page_numbers=[1],
                        level=1,
                        keywords=[]
                    )
                
                current_section.content += para + "\n\n"
        
        if current_section:
            sections.append(current_section)
        
        # Post-process
        for section in sections:
            section.content = self._clean_text(section.content)
            if self.config.merge_hyphenated_words:
                section.content = self._merge_hyphenated_words(section.content)
            section.keywords = self._extract_keywords(section.content)
        
        return sections
    
    def _extract_with_ocr(
        self,
        file_path: Path,
        num_pages: int
    ) -> List[DocumentSection]:
        """
        Extract text from scanned PDF using OCR.
        
        Args:
            file_path: Path to PDF
            num_pages: Number of pages
            
        Returns:
            List[DocumentSection]: Extracted sections
        """
        if not OCR_AVAILABLE or pytesseract is None:
            raise ParseError(
                "OCR dependencies not installed. "
                "Install with: pip install pdf2image pytesseract"
            )
        
        logger.info(f"Starting OCR extraction for {num_pages} pages")
        
        sections = []
        
        try:
            # Ensure convert_from_path is available
            if convert_from_path is None:
                raise ParseError(
                    "OCR dependencies not installed. "
                    "Install with: pip install pdf2image pytesseract"
                )
            # Convert PDF to images
            images = convert_from_path(
                str(file_path),
                dpi=self.config.dpi,
                fmt='jpeg'
            )
            
            logger.info(f"Converted {len(images)} pages to images")
            
            for page_num, image in enumerate(images, 1):
                try:
                    # Perform OCR
                    text = pytesseract.image_to_string(
                        image,
                        lang=self.config.ocr_language
                    )
                    
                    if text and len(text.strip()) > 10:
                        # Create section per page for scanned docs
                        section = DocumentSection(
                            id=f"sec_{page_num:04d}",
                            title=f"Page {page_num}",
                            content=self._clean_text(text),
                            page_numbers=[page_num],
                            level=1,
                            keywords=self._extract_keywords(text)
                        )
                        sections.append(section)
                    
                    logger.debug(f"OCR completed for page {page_num}")
                    
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
                    continue
            
            logger.success(f"OCR extraction complete: {len(sections)} pages processed")
            
        except Exception as e:
            raise ParseError(f"OCR extraction failed: {str(e)}") from e
        
        return sections
    
    def _extract_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            List[Dict]: Extracted tables with metadata
        """
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_tables = page.extract_tables()
                        
                        for table_num, table in enumerate(page_tables, 1):
                            if table and len(table) > 1:  # At least header + 1 row
                                tables.append({
                                    'page': page_num,
                                    'table_number': table_num,
                                    'data': table,
                                    'rows': len(table),
                                    'columns': len(table[0]) if table else 0
                                })
                    except Exception as e:
                        logger.warning(f"Error extracting tables from page {page_num}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    def _detect_heading_level(self, text: str) -> int:
        """
        Detect heading level from text.
        
        Args:
            text: Heading text
            
        Returns:
            int: Heading level (1-6)
        """
        # Check for numbered headings (1., 1.1., 1.1.1.)
        match = re.match(r'^([\d\.]+)\s+', text)
        if match:
            level = len(match.group(1).split('.'))
            return min(level, 6)
        
        # Check font size indicators (if preserved)
        if text.isupper() and len(text) < 50:
            return 1
        
        if text.istitle() and len(text) < 100:
            return 2
        
        return 1
    
    def _merge_hyphenated_words(self, text: str) -> str:
        """
        Merge hyphenated.
        
        Args:
            text: Text to process
            
        Returns:
            str: Text with merged words
        """
        # Match word- at end of line followed by word continuation
        pattern = r'(\w+)-\s*\n\s*(\w+)'
        return re.sub(pattern, r'\1\2', text)
    
    def _extract_title_from_content(self, sections: List[DocumentSection]) -> str:
        """
        Extract title from first section or heading.
        
        Args:
            sections: Document sections
            
        Returns:
            str: Extracted title
        """
        if not sections:
            return "Untitled"
        
        # Try first section title
        first_title = sections[0].title
        if first_title and first_title != "Content" and first_title != "Introduction":
            return first_title
        
        # Try first substantial text
        for section in sections[:3]:
            if section.content and len(section.content) > 20:
                # Get first sentence
                first_sentence = section.content.split('.')[0].strip()
                if len(first_sentence) < 200:
                    return first_sentence
        
        return "Untitled"
    
    def _clean_metadata_string(self, value: Any) -> Optional[str]:
        """Clean metadata string value."""
        if not value:
            return None
        
        text = str(value).strip()
        
        # Remove PDF encoding artifacts
        text = text.replace('\x00', '')
        
        return text if text else None
    
    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse PDF date format (D:YYYYMMDDHHmmSSOHH'mm').
        
        Args:
            date_str: PDF date string
            
        Returns:
            datetime: Parsed date or None
        """
        try:
            # Remove D: prefix
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Extract basic date parts (YYYYMMDDHHMMSS)
            if len(date_str) >= 14:
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(date_str[8:10])
                minute = int(date_str[10:12])
                second = int(date_str[12:14])
                
                return datetime(year, month, day, hour, minute, second)
        except Exception:
            pass
        
        return None


# Export
__all__ = ['PDFParser', 'PDFParserConfig', 'PDFMetadata']