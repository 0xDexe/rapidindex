# rapidindex/parsers/txt_parser.py


from pathlib import Path
from typing import List

from .base_parser import TextBasedParser
from ..core.document import Document, DocumentSection


class TxtParser(TextBasedParser):
    """
    Parser for plain text files.
    
    Example:
        >>> parser = TxtParser()
        >>> doc = parser.parse('document.txt')
    """
    
    supported_formats = ['.txt', '.text']
    
    def _parse_implementation(self, file_path: Path) -> Document:
        """Parse plain text file."""
        
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(content)
        
        # Create sections
        sections: List[DocumentSection] = []
        current_heading = "Content"
        current_content = []
        section_counter = 0
        
        for i, para in enumerate(paragraphs):
            # Check if paragraph is a heading
            next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else None
            
            if self._is_heading(para, next_para):
                # Save previous section
                if current_content:
                    section_counter += 1
                    sections.append(DocumentSection(
                        id=f"sec_{section_counter:04d}",
                        title=current_heading,
                        content="\n\n".join(current_content),
                        page_numbers=[1],  # Text files don't have pages
                        level=1,
                        keywords=self._extract_keywords("\n".join(current_content))
                    ))
                
                # Start new section
                current_heading = para
                current_content = []
            else:
                # Add to current section
                current_content.append(para)
        
        # Add final section
        if current_content:
            section_counter += 1
            sections.append(DocumentSection(
                id=f"sec_{section_counter:04d}",
                title=current_heading,
                content="\n\n".join(current_content),
                page_numbers=[1],
                level=1,
                keywords=self._extract_keywords("\n".join(current_content))
            ))
        
        # Create document
        return Document(
            id="",  # Will be set in post_process
            title=self._extract_title(file_path),
            file_path=str(file_path),
            document_type='txt',
            sections=sections,
            metadata={
                'encoding': 'utf-8',
                'num_paragraphs': len(paragraphs)
            }
        )
    
    def _extract_title(self, file_path: Path) -> str:
        """Extract title from filename."""
        return file_path.stem.replace('_', ' ').replace('-', ' ').title()