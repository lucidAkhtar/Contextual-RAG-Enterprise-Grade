"""
Document processor for PDF ingestion and preprocessing.
Handles PDF extraction, text cleaning, and metadata management.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from llama_index.core.schema import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentProcessor:
    """
    Processes PDF documents and extracts structured content.
    Implements Single Responsibility Principle.
    """
    
    def __init__(self):
        self.supported_formats = [".pdf"]
        logger.info("Initialized DocumentProcessor")
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load and parse PDF document.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of Document objects with page-level metadata
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            documents = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Extract tables if present
                tables = self._extract_tables(page)
                
                # Clean text
                cleaned_text = self._clean_text(text)
                
                if cleaned_text.strip():
                    metadata = {
                        "page": page_num + 1,
                        "source": str(path),
                        "total_pages": len(doc),
                        "has_tables": len(tables) > 0,
                        "tables_count": len(tables)
                    }
                    
                    # Add table information to text if present
                    if tables:
                        table_text = self._format_tables(tables)
                        cleaned_text += f"\n\n{table_text}"
                    
                    documents.append(
                        Document(
                            text=cleaned_text,
                            metadata=metadata
                        )
                    )
            
            doc.close()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text from PDF
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers (heuristic: very short lines at top/bottom)
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if len(line.strip()) > 10 or any(char.isalnum() for char in line)
        ]
        
        text = '\n'.join(cleaned_lines)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def _extract_tables(self, page: Any) -> List[List[List[str]]]:
        """
        Extract tables from PDF page.
        
        Args:
            page: PyMuPDF page object
        
        Returns:
            List of tables (each table is a list of rows)
        """
        try:
            tables = page.find_tables()
            extracted_tables = []
            
            for table in tables:
                table_data = table.extract()
                if table_data:
                    extracted_tables.append(table_data)
            
            return extracted_tables
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
            return []
    
    def _format_tables(self, tables: List[List[List[str]]]) -> str:
        """
        Format extracted tables as text.
        
        Args:
            tables: List of table data
        
        Returns:
            Formatted table text
        """
        formatted_text = []
        
        for idx, table in enumerate(tables, 1):
            formatted_text.append(f"[Table {idx}]")
            for row in table:
                # Filter out None and empty cells
                cleaned_row = [str(cell) for cell in row if cell]
                if cleaned_row:
                    formatted_text.append(" | ".join(cleaned_row))
            formatted_text.append("")  # Empty line between tables
        
        return "\n".join(formatted_text)
    
    def extract_text_span(
        self,
        pdf_path: str,
        page: int,
        start_char: int,
        end_char: int
    ) -> str:
        """
        Extract specific text span from PDF.
        Useful for ground truth extraction.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
            start_char: Start character position
            end_char: End character position
        
        Returns:
            Extracted text span
        """
        try:
            doc = fitz.open(pdf_path)
            page_obj = doc[page - 1]  # Convert to 0-indexed
            text = page_obj.get_text()
            doc.close()
            
            return text[start_char:end_char]
        except Exception as e:
            logger.error(f"Error extracting text span: {e}")
            return ""
    
    def get_page_text(self, pdf_path: str, page: int) -> str:
        """
        Get full text of a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
        
        Returns:
            Page text
        """
        try:
            doc = fitz.open(pdf_path)
            page_obj = doc[page - 1]
            text = self._clean_text(page_obj.get_text())
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error getting page text: {e}")
            return ""
