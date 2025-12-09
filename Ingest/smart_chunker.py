"""
Smart Chunker - Intelligent file processing based on file type
- Excel/XLSX: Row-wise chunking (each row = one chunk)
- PDF: Paragraph-wise chunking
- JSON: Block-wise chunking (each object = one chunk)

Sub-chunking applied when content exceeds max_chunk_size
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 300))

logger.info(f"SmartChunker loaded with CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP} from .env")

@dataclass
class Chunk:
    """Represents a single chunk of content"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    source_file: str = ""
    file_type: str = ""
    primary_chunk_index: int = 0  # Main chunk index (row/paragraph/block)
    sub_chunk_index: int = 0  # Sub-chunk index if content was split
    total_sub_chunks: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id,
            'source_file': self.source_file,
            'file_type': self.file_type,
            'primary_chunk_index': self.primary_chunk_index,
            'sub_chunk_index': self.sub_chunk_index,
            'total_sub_chunks': self.total_sub_chunks
        }


class SubChunker:
    """Handles sub-chunking when content exceeds max size"""
    
    def __init__(self, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def needs_sub_chunking(self, text: str) -> bool:
        """Check if text needs to be sub-chunked"""
        return len(text) > self.max_chunk_size
    
    def sub_chunk(self, text: str) -> List[str]:
        """Split text into smaller chunks with overlap"""
        if not self.needs_sub_chunking(text):
            return [text]
        
        chunks = []
        
        # Try to split on sentence boundaries first
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence exceeds max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + sentence
                else:
                    # Single sentence is too long, force split
                    chunks.extend(self._force_split(sentence))
                    current_chunk = ""
            else:
                current_chunk += sentence
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on common sentence endings
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s + ' ' for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of previous chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good break point
        overlap_region = text[-self.chunk_overlap:]
        space_idx = overlap_region.find(' ')
        
        if space_idx != -1:
            return overlap_region[space_idx + 1:]
        return overlap_region
    

    def _force_split(self, text: str) -> List[str]:
        """
        FIXED: Force split text ensuring word boundaries
        Never cuts words mid-character
        """
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk_words = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.max_chunk_size and current_chunk_words:
                # Save current chunk
                chunks.append(' '.join(current_chunk_words))
                
                # Calculate overlap in words
                overlap_chars = 0
                overlap_words = []
                for w in reversed(current_chunk_words):
                    if overlap_chars + len(w) + 1 > self.chunk_overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_chars += len(w) + 1
                
                # Start new chunk with overlap
                current_chunk_words = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk_words)
            else:
                current_chunk_words.append(word)
                current_length += word_length
        
        # Add remaining words
        if current_chunk_words:
            chunks.append(' '.join(current_chunk_words))
        
        return chunks


class BaseChunker(ABC):
    """Abstract base class for file-specific chunkers"""
    
    def __init__(self, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.sub_chunker = SubChunker(max_chunk_size, chunk_overlap)
    
    @abstractmethod
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Process file and return list of chunks"""
        pass
    
    @abstractmethod
    def get_file_type(self) -> str:
        """Return the file type this chunker handles"""
        pass
    
    def _apply_sub_chunking(
        self, 
        content: str, 
        base_metadata: Dict[str, Any],
        source_file: str,
        primary_index: int
    ) -> List[Chunk]:
        """Apply sub-chunking if needed and return list of Chunk objects"""
        
        sub_chunks = self.sub_chunker.sub_chunk(content)
        chunks = []
        
        for sub_idx, sub_content in enumerate(sub_chunks):
            chunk = Chunk(
                content=sub_content,
                metadata=base_metadata.copy(),
                chunk_id=f"{Path(source_file).stem}_{primary_index}_{sub_idx}",
                source_file=os.path.basename(source_file),
                file_type=self.get_file_type(),
                primary_chunk_index=primary_index,
                sub_chunk_index=sub_idx,
                total_sub_chunks=len(sub_chunks)
            )
            chunks.append(chunk)
        
        return chunks


class ExcelChunker(BaseChunker):
    """
    Excel/XLSX chunker - processes row by row
    Each row represents a single product/article/record
    """
    
    def __init__(self, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(max_chunk_size, chunk_overlap)
        self._pandas = None
    
    def _get_pandas(self):
        """Lazy load pandas"""
        if self._pandas is None:
            import pandas as pd
            self._pandas = pd
        return self._pandas
    
    def get_file_type(self) -> str:
        return "excel"
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Process Excel file row by row"""
        pd = self._get_pandas()
        
        logger.info(f"Processing Excel file: {file_path}")
        
        # Determine file extension
        ext = Path(file_path).suffix.lower()
        
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif ext in ['.xlsx', '.xls', '.xlsm']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported Excel format: {ext}")
        except UnicodeDecodeError:
            # Try with different encoding
            df = pd.read_csv(file_path, encoding='latin-1')
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        all_chunks = []
        columns = list(df.columns)
        

        for row_idx, row in df.iterrows():
            # Create content from all columns
            row_content = self._row_to_text(row, columns)
            
            if not row_content or len(row_content.strip()) < 10:
                continue
            
            # Extract header (MODIFIED - no truncation)
            header_value = ""
            for col in ['name', 'Name', 'title', 'Title', 'header', 'Header', 'product_name', 'Header_1']:
                if col in row and pd.notna(row[col]):
                    header_value = str(row[col])[:100]  # Full header, no truncation
                    break
            if not header_value and len(columns) > 0:
                first_val = row.get(columns[0])
                if pd.notna(first_val):
                    header_value = str(first_val)[:100]  # Full header, no truncation
            
            metadata = {
                'row_index': int(row_idx),
                'columns': columns,
                'row_data': {col: str(val) if pd.notna(val) else None 
                            for col, val in row.items()},
                'header': header_value,
                'content_type': 'product'
            }
            
            # Apply sub-chunking
            chunks = self._apply_sub_chunking(
                content=row_content,
                base_metadata=metadata,
                source_file=file_path,
                primary_index=int(row_idx)
            )
            
            # ======== ADD THIS NEW CODE ========
            # Filter tiny chunks and add headers
            filtered_chunks = []
            for chunk in chunks:
                # Skip tiny chunks
                if len(chunk.content.strip()) < 100:
                    continue
                
                # PREPEND HEADER to text
                if header_value:
                    chunk.content = f"{header_value}\n\n{chunk.content}"
                
                filtered_chunks.append(chunk)
            
            all_chunks.extend(filtered_chunks)
        
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(df)} rows")
        return all_chunks
    
    def _row_to_text(self, row, columns: List[str]) -> str:
        """Convert a row to searchable text"""
        pd = self._get_pandas()
        
        parts = []
        for col in columns:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                # Include column name for context
                parts.append(f"{col}: {str(value).strip()}")
        
        return "\n".join(parts)


class PDFChunker(BaseChunker):
    """
    PDF chunker - processes paragraph by paragraph
    Each paragraph represents related content
    """
    
    def __init__(self, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(max_chunk_size, chunk_overlap)
    
    def get_file_type(self) -> str:
        return "pdf"
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Process PDF file paragraph by paragraph"""
        logger.info(f"Processing PDF file: {file_path}")
        
        # Extract text from PDF
        text = self._extract_pdf_text(file_path)
        
        if not text:
            logger.warning(f"No text extracted from PDF: {file_path}")
            return []
        
        # Split into paragraphs
        paragraphs = self._extract_paragraphs(text)
        
        logger.info(f"Extracted {len(paragraphs)} paragraphs")
        
        all_chunks = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 20:  # Skip very short paragraphs
                continue
            
            # Create metadata
            # Extract potential header (first sentence or capitalized text)
            first_sentence = paragraph.split('.')[0][:100] if '.' in paragraph else paragraph[:100]
            
            metadata = {
                'paragraph_index': para_idx,
                'total_paragraphs': len(paragraphs),
                'paragraph_length': len(paragraph),
                'header': first_sentence.strip(),
                'content_type': 'documentation',  # Default for PDF
                'page_number': -1,  # Will be updated if page tracking is implemented
                'total_pages': 0,
                'section_title': ''
            }
            
            # Apply sub-chunking if needed
            chunks = self._apply_sub_chunking(
                content=paragraph,
                base_metadata=metadata,
                source_file=file_path,
                primary_index=para_idx
            )
            
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(paragraphs)} paragraphs")
        return all_chunks
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz)"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.warning("PyMuPDF not installed, trying pdfplumber")
            return self._extract_with_pdfplumber(file_path)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Fallback PDF extraction using pdfplumber"""
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.error("Neither PyMuPDF nor pdfplumber installed")
            return ""
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        # Split on double newlines or multiple newlines
        raw_paragraphs = re.split(r'\n\s*\n+', text)
        
        paragraphs = []
        for para in raw_paragraphs:
            # Clean up the paragraph
            cleaned = ' '.join(para.split())
            if cleaned and len(cleaned) >= 20:
                paragraphs.append(cleaned)
        
        return paragraphs


class JSONChunker(BaseChunker):
    """
    JSON chunker - processes block by block (object by object)
    Each JSON object/block represents a complete entity
    """
    
    def __init__(self, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(max_chunk_size, chunk_overlap)
        
        # Fields to prioritize for content
        self.content_fields = [
            'Header_1', 'Description_1', 'Description_2', 'Description_3',
            'Description_4', 'Description_6', 'WholeContent_1', 'WholeContent_2',
            'content', 'text', 'body', 'description', 'title', 'name',
            'summary', 'abstract'
        ]
    
    def get_file_type(self) -> str:
        return "json"
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """Process JSON file block by block"""
        logger.info(f"Processing JSON file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading JSON file: {e}")
            return []
        
        # Handle different JSON structures
        if isinstance(data, list):
            blocks = data
        elif isinstance(data, dict):
            # If dict, treat it as single block or look for array fields
            array_fields = [k for k, v in data.items() if isinstance(v, list)]
            if array_fields:
                # Use the first array field found
                blocks = data[array_fields[0]]
                logger.info(f"Using array field '{array_fields[0]}' with {len(blocks)} items")
            else:
                blocks = [data]
        else:
            logger.warning(f"Unexpected JSON structure: {type(data)}")
            return []
        
        logger.info(f"Found {len(blocks)} blocks to process")
        
        all_chunks = []
        
        
        for block_idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            
            # Convert block to text
            block_content = self._block_to_text(block)
            
            if not block_content or len(block_content.strip()) < 20:
                continue
            
            # Create metadata
            metadata = self._extract_block_metadata(block, block_idx, len(blocks))
            
            # Apply sub-chunking
            chunks = self._apply_sub_chunking(
                content=block_content,
                base_metadata=metadata,
                source_file=file_path,
                primary_index=block_idx
            )
            
            # ======== ADD THIS NEW CODE ========
            # Filter tiny chunks and add headers
            header = metadata.get('header', '')
            filtered_chunks = []
            
            for chunk in chunks:
                # Skip tiny chunks
                if len(chunk.content.strip()) < 100:
                    continue
                
                # PREPEND HEADER to text
                if header:
                    chunk.content = f"{header}\n\n{chunk.content}"
                
                filtered_chunks.append(chunk)
            
            all_chunks.extend(filtered_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(blocks)} blocks")
        return all_chunks
    
    
    def _block_to_text(self, block: Dict[str, Any]) -> str:
        """Convert a JSON block to searchable text - FIXED"""
        parts = []
        
        # First, add content from priority fields (NO FIELD NAMES!)
        for field in self.content_fields:
            if field in block and block[field]:
                value = str(block[field]).strip()
                if value and len(value) > 5:
                    # FIXED: Just append value, NO "field: value" format
                    parts.append(value)
        
        # Then add other fields (NO FIELD NAMES!)
        for key, value in block.items():
            if key in self.content_fields:
                continue  # Already added
            
            if value and not isinstance(value, (dict, list)):
                str_value = str(value).strip()
                if str_value and len(str_value) > 3:
                    # FIXED: Just append value, NO "key: value" format
                    parts.append(str_value)
        
        # Use double newline as separator for better readability
        return "\n\n".join(parts)
    
    def _extract_block_metadata(
        self, 
        block: Dict[str, Any], 
        block_idx: int,
        total_blocks: int
    ) -> Dict[str, Any]:
        """Extract metadata from a JSON block"""
        metadata = {
            'block_index': block_idx,
            'total_blocks': total_blocks,
            'content_type': 'article',  # Default for JSON
            'header': '',
            'primary_field': '',
            'original_keys': list(block.keys())
        }
        
        # Extract header/title if available
        for field in ['Header_1', 'title', 'name', 'header', 'Title', 'Name']:
            if field in block and block[field]:
                metadata['header'] = str(block[field])[:100]
                metadata['primary_field'] = field
                break
        
        # Extract ID if available
        for field in ['id', 'ID', '_id', 'record_id']:
            if field in block:
                metadata['record_id'] = block[field]
                break
        
        # Store field names that had content
        metadata['fields_with_content'] = [
            k for k, v in block.items() 
            if v and not isinstance(v, (dict, list))
        ]
        
        return metadata


class SmartChunker:
    """
    Main chunker class that automatically selects the right chunking strategy
    based on file type
    """
    
    def __init__(self, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize chunkers
        self.chunkers = {
            'excel': ExcelChunker(max_chunk_size, chunk_overlap),
            'pdf': PDFChunker(max_chunk_size, chunk_overlap),
            'json': JSONChunker(max_chunk_size, chunk_overlap)
        }
        
        # File extension to chunker mapping
        self.extension_map = {
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.xlsm': 'excel',
            '.csv': 'excel',
            '.pdf': 'pdf',
            '.json': 'json'
        }
    
    def get_chunker_type(self, file_path: str) -> Optional[str]:
        """Determine which chunker to use for a file"""
        ext = Path(file_path).suffix.lower()
        return self.extension_map.get(ext)
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Automatically chunk a file using the appropriate strategy
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of Chunk objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        chunker_type = self.get_chunker_type(file_path)
        
        if chunker_type is None:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")
        
        logger.info(f"Using {chunker_type} chunker for {file_path}")
        
        chunker = self.chunkers[chunker_type]
        return chunker.chunk_file(file_path)
    
    def chunk_directory(
        self, 
        directory: str, 
        recursive: bool = True
    ) -> Dict[str, List[Chunk]]:
        """
        Process all supported files in a directory
        
        Args:
            directory: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary mapping file paths to their chunks
        """
        results = {}
        
        if recursive:
            pattern = Path(directory).rglob('*')
        else:
            pattern = Path(directory).glob('*')
        
        for file_path in pattern:
            if not file_path.is_file():
                continue
            
            if self.get_chunker_type(str(file_path)) is None:
                continue
            
            try:
                chunks = self.chunk_file(str(file_path))
                results[str(file_path)] = chunks
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        return results
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        return list(self.extension_map.keys())


def test_smart_chunker():
    """Test the smart chunker with sample data"""
    print("Testing Smart Chunker")
    print("=" * 60)
    
    chunker = SmartChunker(max_chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    print(f"\nSupported extensions: {chunker.get_supported_extensions()}")
    
    # Test sub-chunker
    sub_chunker = SubChunker(max_chunk_size=100, chunk_overlap=20)
    
    long_text = "This is a very long text that needs to be split into smaller chunks. " * 10
    sub_chunks = sub_chunker.sub_chunk(long_text)
    
    print(f"\nSub-chunking test:")
    print(f"  Original length: {len(long_text)}")
    print(f"  Number of sub-chunks: {len(sub_chunks)}")
    for i, chunk in enumerate(sub_chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")
    
    print("\nSmart Chunker ready for use!")
    return True


if __name__ == "__main__":
    test_smart_chunker()