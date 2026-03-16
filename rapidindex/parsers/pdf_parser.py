# rapidindex/parsers/pdf_parser.py
"""
PDF document parser
- Text extraction from text-based PDFs
- OCR for scanned PDFs
- Table extraction → searchable DocumentSections
- Image extraction
- Metadata extraction
- Hierarchical structure detection
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import re
from io import BytesIO

import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    convert_from_path = None
    pytesseract = None
    OCR_AVAILABLE = False

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
    use_ocr: bool = Field(default=False)
    ocr_language: str = Field(default="eng")
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=False)
    merge_hyphenated_words: bool = Field(default=True)
    detect_headers: bool = Field(default=True)
    min_text_per_page: int = Field(default=50)
    dpi: int = Field(default=300, ge=100, le=600)
    line_margin: float = Field(default=0.5)
    word_margin: float = Field(default=0.1)
    char_margin: float = Field(default=2.0)


class PDFMetadata(BaseModel):
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
    supported_formats = [".pdf"]

    def __init__(self, config: Optional[PDFParserConfig] = None):
        self.config = config or PDFParserConfig()
        super().__init__(self.config)

        if self.config.use_ocr and not OCR_AVAILABLE:
            logger.warning(
                "OCR requested but dependencies not installed. "
                "Install with: pip install pdf2image pytesseract"
            )
            self.config.use_ocr = False

    def _parse_implementation(self, file_path: Path) -> Document:
        try:
            metadata = self._extract_metadata(file_path)

            logger.info(
                "PDF metadata extracted",
                num_pages=metadata.num_pages,
                is_scanned=metadata.is_scanned,
                encrypted=metadata.is_encrypted,
            )

            if metadata.is_scanned and self.config.use_ocr:
                sections = self._extract_with_ocr(file_path, metadata.num_pages)
            else:
                sections = self._extract_with_text_extraction(file_path, metadata.num_pages)

            # FIX: tables were extracted but then discarded via a bare
            # attribute expression (`metadata.num_pages`).  We now convert
            # each table into a DocumentSection so its content is indexed
            # and searchable, and record the count in metadata.
            extra_metadata: Dict[str, Any] = {}
            if self.config.extract_tables:
                tables = self._extract_tables(file_path)
                if tables:
                    logger.info(f"Extracted {len(tables)} tables from PDF")
                    extra_metadata["table_count"] = len(tables)
                    for table in tables:
                        table_section = self._table_to_section(table)
                        if table_section:
                            sections.append(table_section)

            document = Document(
                id="",
                title=metadata.title or self._extract_title_from_content(sections),
                file_path=str(file_path),
                document_type="pdf",
                sections=sections,
                metadata={
                    **metadata.dict(exclude_none=True),
                    "extraction_method": "ocr" if metadata.is_scanned else "text",
                    **extra_metadata,
                },
            )

            logger.success(
                "PDF parsed successfully",
                num_sections=len(sections),
                total_text_length=sum(len(s.content) for s in sections),
            )

            return document

        except Exception as exc:
            if "PdfReadError" in type(exc).__name__ or "corrupted" in str(exc).lower():
                raise CorruptedFileError(
                    f"PDF file is corrupted or invalid: {exc}"
                ) from exc
            raise ParseError(f"Failed to parse PDF: {exc}") from exc

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _table_to_section(self, table: Dict[str, Any]) -> Optional[DocumentSection]:
        """Convert a raw table dict to a DocumentSection with pipe-formatted text."""
        text = self._table_to_text(table.get("data", []))
        if not text or len(text) < 10:
            return None

        sec_id = f"tbl_p{table['page']:04d}_t{table['table_number']:02d}"
        title = f"Table {table['table_number']} (Page {table['page']})"

        return DocumentSection(
            id=sec_id,
            title=title,
            content=text,
            page_numbers=[table["page"]],
            level=2,
            keywords=self._extract_keywords(text),
        )

    def _table_to_text(self, table_data: List) -> str:
        """Render a 2-D table (list of lists) as pipe-delimited text."""
        if not table_data:
            return ""
        rows = []
        for row in table_data:
            if row:
                rows.append(" | ".join(str(cell or "").strip() for cell in row))
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(self, file_path: Path) -> PDFMetadata:
        metadata = PDFMetadata()

        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                metadata.num_pages = len(reader.pages)
                metadata.is_encrypted = reader.is_encrypted

                if hasattr(reader, "pdf_header"):
                    metadata.pdf_version = reader.pdf_header

                if reader.metadata:
                    info = reader.metadata
                    metadata.title = self._clean_metadata_string(info.get("/Title"))
                    metadata.author = self._clean_metadata_string(info.get("/Author"))
                    metadata.subject = self._clean_metadata_string(info.get("/Subject"))
                    metadata.creator = self._clean_metadata_string(info.get("/Creator"))
                    metadata.producer = self._clean_metadata_string(info.get("/Producer"))

                    if "/CreationDate" in info:
                        metadata.creation_date = self._parse_pdf_date(str(info["/CreationDate"]))
                    if "/ModDate" in info:
                        metadata.modification_date = self._parse_pdf_date(str(info["/ModDate"]))

                    if "/Keywords" in info:
                        kw = self._clean_metadata_string(info["/Keywords"])
                        if kw:
                            metadata.keywords = [k.strip() for k in kw.split(",")]

                metadata.is_scanned = self._is_scanned_pdf(reader)

        except Exception as exc:
            logger.warning(f"Error extracting PDF metadata: {exc}")

        return metadata

    def _is_scanned_pdf(self, reader: PyPDF2.PdfReader) -> bool:
        pages_to_check = min(3, len(reader.pages))
        for i in range(pages_to_check):
            try:
                text = reader.pages[i].extract_text()
                if text and len(text.strip()) > self.config.min_text_per_page:
                    return False
            except Exception:
                continue
        return True

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_with_text_extraction(
        self, file_path: Path, num_pages: int
    ) -> List[DocumentSection]:
        for method, extractor in [
            ("pdfplumber", self._extract_with_pdfplumber),
            ("PyPDF2", self._extract_with_pypdf2),
            ("pdfminer", self._extract_with_pdfminer),
        ]:
            try:
                sections = extractor(file_path)
                if sections:
                    logger.info(f"Text extracted with {method}")
                    return sections
            except Exception as exc:
                logger.warning(f"{method} extraction failed: {exc}")

        raise ParseError("All text extraction methods failed")

    def _extract_with_pdfplumber(self, file_path: Path) -> List[DocumentSection]:
        sections = []

        with pdfplumber.open(file_path) as pdf:
            current_section = None
            section_counter = 0

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if not text or len(text.strip()) < 10:
                        continue

                    paragraphs = self._split_into_paragraphs(text)

                    for i, para in enumerate(paragraphs):
                        if not para.strip():
                            continue

                        next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None

                        if self.config.detect_headers and self._is_heading(para, next_para):
                            if current_section and current_section.content.strip():
                                sections.append(current_section)
                            section_counter += 1
                            current_section = DocumentSection(
                                id=f"sec_{section_counter:04d}",
                                title=self._clean_text(para),
                                content="",
                                page_numbers=[page_num],
                                level=self._detect_heading_level(para),
                                keywords=[],
                            )
                        else:
                            if current_section is None:
                                section_counter += 1
                                current_section = DocumentSection(
                                    id=f"sec_{section_counter:04d}",
                                    title="Introduction",
                                    content="",
                                    page_numbers=[page_num],
                                    level=1,
                                    keywords=[],
                                )
                            current_section.content += para + "\n\n"
                            if page_num not in current_section.page_numbers:
                                current_section.page_numbers.append(page_num)

                except Exception as exc:
                    logger.warning(f"Error extracting page {page_num}: {exc}")
                    continue

            if current_section and current_section.content.strip():
                sections.append(current_section)

        for section in sections:
            section.content = self._clean_text(section.content)
            if self.config.merge_hyphenated_words:
                section.content = self._merge_hyphenated_words(section.content)
            section.keywords = self._extract_keywords(section.content)

        return sections

    def _extract_with_pypdf2(self, file_path: Path) -> List[DocumentSection]:
        sections = []

        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            all_text = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        all_text.append(text)
                except Exception as exc:
                    logger.warning(f"Error extracting page {page_num}: {exc}")

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
                    page_numbers=[1],
                    level=self._detect_heading_level(para),
                    keywords=[],
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
                        keywords=[],
                    )
                current_section.content += para + "\n\n"

        if current_section:
            sections.append(current_section)

        for section in sections:
            section.content = self._clean_text(section.content)
            if self.config.merge_hyphenated_words:
                section.content = self._merge_hyphenated_words(section.content)
            section.keywords = self._extract_keywords(section.content)

        return sections

    def _extract_with_pdfminer(self, file_path: Path) -> List[DocumentSection]:
        laparams = LAParams(
            line_margin=self.config.line_margin,
            word_margin=self.config.word_margin,
            char_margin=self.config.char_margin,
        )

        text = pdfminer_extract_text(str(file_path), laparams=laparams)
        if not text or len(text.strip()) < 100:
            return []

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
                    keywords=[],
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
                        keywords=[],
                    )
                current_section.content += para + "\n\n"

        if current_section:
            sections.append(current_section)

        for section in sections:
            section.content = self._clean_text(section.content)
            if self.config.merge_hyphenated_words:
                section.content = self._merge_hyphenated_words(section.content)
            section.keywords = self._extract_keywords(section.content)

        return sections

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _extract_with_ocr(
        self, file_path: Path, num_pages: int
    ) -> List[DocumentSection]:
        if not OCR_AVAILABLE or pytesseract is None or convert_from_path is None:
            raise ParseError(
                "OCR dependencies not installed. "
                "Install with: pip install pdf2image pytesseract"
            )

        logger.info(f"Starting OCR extraction for {num_pages} pages")
        sections = []

        try:
            images = convert_from_path(str(file_path), dpi=self.config.dpi, fmt="jpeg")
            logger.info(f"Converted {len(images)} pages to images")

            for page_num, image in enumerate(images, 1):
                try:
                    text = pytesseract.image_to_string(
                        image, lang=self.config.ocr_language
                    )
                    if text and len(text.strip()) > 10:
                        sections.append(
                            DocumentSection(
                                id=f"sec_{page_num:04d}",
                                title=f"Page {page_num}",
                                content=self._clean_text(text),
                                page_numbers=[page_num],
                                level=1,
                                keywords=self._extract_keywords(text),
                            )
                        )
                    logger.debug(f"OCR completed for page {page_num}")
                except Exception as exc:
                    logger.warning(f"OCR failed for page {page_num}: {exc}")

            logger.success(f"OCR extraction complete: {len(sections)} pages")

        except Exception as exc:
            raise ParseError(f"OCR extraction failed: {exc}") from exc

        return sections

    # ------------------------------------------------------------------
    # Table extraction (raw data only — conversion in _table_to_section)
    # ------------------------------------------------------------------

    def _extract_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        for table_num, table in enumerate(page.extract_tables(), 1):
                            if table and len(table) > 1:
                                tables.append(
                                    {
                                        "page": page_num,
                                        "table_number": table_num,
                                        "data": table,
                                        "rows": len(table),
                                        "columns": len(table[0]) if table else 0,
                                    }
                                )
                    except Exception as exc:
                        logger.warning(f"Error extracting tables from page {page_num}: {exc}")
        except Exception as exc:
            logger.warning(f"Table extraction failed: {exc}")
        return tables

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _detect_heading_level(self, text: str) -> int:
        match = re.match(r"^([\d\.]+)\s+", text)
        if match:
            return min(len(match.group(1).split(".")), 6)
        if text.isupper() and len(text) < 50:
            return 1
        if text.istitle() and len(text) < 100:
            return 2
        return 1

    def _merge_hyphenated_words(self, text: str) -> str:
        return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    def _extract_title_from_content(self, sections: List[DocumentSection]) -> str:
        if not sections:
            return "Untitled"
        first_title = sections[0].title
        if first_title and first_title not in ("Content", "Introduction"):
            return first_title
        for section in sections[:3]:
            if section.content and len(section.content) > 20:
                first_sentence = section.content.split(".")[0].strip()
                if len(first_sentence) < 200:
                    return first_sentence
        return "Untitled"

    def _clean_metadata_string(self, value: Any) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip().replace("\x00", "")
        return text if text else None

    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        try:
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            if len(date_str) >= 14:
                return datetime(
                    int(date_str[0:4]),
                    int(date_str[4:6]),
                    int(date_str[6:8]),
                    int(date_str[8:10]),
                    int(date_str[10:12]),
                    int(date_str[12:14]),
                )
        except Exception:
            pass
        return None


__all__ = ["PDFParser", "PDFParserConfig", "PDFMetadata"]
