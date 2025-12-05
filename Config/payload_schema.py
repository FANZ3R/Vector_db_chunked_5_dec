"""
Payload Schema for chatbot_embeddings Collection
Defines the structure for all vectors stored in Qdrant

Collection: chatbot_embeddings
Purpose: Store embeddings from multiple file types for RAG chatbot
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import uuid


class FileType(str, Enum):
    """Supported file types"""
    EXCEL = "excel"      # .xlsx, .xls, .xlsm, .csv
    PDF = "pdf"          # .pdf
    JSON = "json"        # .json
    TEXT = "text"        # .txt, .md (future)
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    """Type of content"""
    PRODUCT = "product"
    ARTICLE = "article"
    REPORT = "report"
    DOCUMENTATION = "documentation"
    PROCEDURE = "procedure"
    GENERAL = "general"


@dataclass
class BasePayload:
    """
    Base payload structure for all embeddings in chatbot_embeddings collection
    
    This is the COMMON structure that ALL vectors will have regardless of file type
    """
    
    # ============ REQUIRED FIELDS (no defaults) ============
    text: str                        # The actual text content (searchable)
    
    # ============ IDENTIFICATION ============
    chunk_id: str = ""               # Unique identifier for this chunk
    collection_name: str = "chatbot_embeddings"  # Collection identifier
    
    # ============ CONTENT ============
    content_length: int = 0          # Length of text in characters
    
    # ============ SOURCE TRACKING ============
    source_file: str = ""            # Original filename (e.g., "products.xlsx")
    source_path: str = ""            # Full path if needed
    file_type: str = "unknown"       # excel, pdf, json, text
    content_type: str = "general"    # product, article, report, etc.
    
    # ============ CHUNKING INFO ============
    primary_chunk_index: int = 0     # Main chunk number (row/paragraph/block)
    sub_chunk_index: int = 0         # Sub-chunk within primary (0 if not sub-chunked)
    total_sub_chunks: int = 1        # Total sub-chunks for this primary chunk
    
    # ============ METADATA ============
    header: str = ""                 # Title/header for display
    summary: str = ""                # Brief summary if available
    
    # ============ TIMESTAMPS ============
    created_at: str = ""             # When this embedding was created
    updated_at: str = ""             # When this embedding was last updated
    
    # ============ VERSIONING ============
    version: str = "1.0"             # Schema version for future migrations
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())
        self.content_length = len(self.text) if self.text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExcelPayload(BasePayload):
    """
    Extended payload for Excel/CSV data
    Adds row-specific metadata
    """
    
    # ============ EXCEL-SPECIFIC ============
    row_index: int = -1              # Original row number in spreadsheet
    sheet_name: str = ""             # Sheet name (for multi-sheet Excel)
    columns: List[str] = field(default_factory=list)  # Column names
    row_data: Dict[str, Any] = field(default_factory=dict)  # Original row as dict
    
    # ============ DATA QUALITY ============
    null_columns: List[str] = field(default_factory=list)  # Columns with null values
    
    def __post_init__(self):
        super().__post_init__()
        self.file_type = FileType.EXCEL.value


@dataclass
class PDFPayload(BasePayload):
    """
    Extended payload for PDF documents
    Adds paragraph/page-specific metadata
    """
    
    # ============ PDF-SPECIFIC ============
    page_number: int = -1            # Page number in PDF
    paragraph_index: int = -1        # Paragraph number on page
    total_paragraphs: int = 0        # Total paragraphs in document
    total_pages: int = 0             # Total pages in PDF
    
    # ============ DOCUMENT STRUCTURE ============
    section_title: str = ""          # Section/chapter title if detected
    is_header: bool = False          # Is this a header/title
    is_footer: bool = False          # Is this a footer
    
    def __post_init__(self):
        super().__post_init__()
        self.file_type = FileType.PDF.value


@dataclass
class JSONPayload(BasePayload):
    """
    Extended payload for JSON data
    Adds block-specific metadata
    """
    
    # ============ JSON-SPECIFIC ============
    block_index: int = -1            # Index of JSON object in array
    total_blocks: int = 0            # Total objects in JSON array
    json_path: str = ""              # JSON path to this block (e.g., "data[0]")
    
    # ============ FIELD INFO ============
    fields_with_content: List[str] = field(default_factory=list)  # Fields that had data
    primary_field: str = ""          # Main content field used
    
    # ============ ORIGINAL STRUCTURE ============
    original_keys: List[str] = field(default_factory=list)  # All keys in original object
    
    def __post_init__(self):
        super().__post_init__()
        self.file_type = FileType.JSON.value


# ============ PAYLOAD SCHEMA DOCUMENTATION ============

PAYLOAD_SCHEMA = {
    "collection_name": "chatbot_embeddings",
    "description": "Unified collection for RAG chatbot embeddings from multiple file types",
    "version": "1.0",
    
    "common_fields": {
        "chunk_id": "string - Unique identifier (UUID)",
        "collection_name": "string - Always 'chatbot_embeddings'",
        "text": "string - The embedded text content",
        "content_length": "integer - Length of text",
        "source_file": "string - Original filename",
        "source_path": "string - Full file path",
        "file_type": "string - excel|pdf|json|text|unknown",
        "content_type": "string - product|article|report|documentation|procedure|general",
        "primary_chunk_index": "integer - Main chunk number",
        "sub_chunk_index": "integer - Sub-chunk number (0 if not sub-chunked)",
        "total_sub_chunks": "integer - Total sub-chunks for primary",
        "header": "string - Title/header for display",
        "summary": "string - Brief summary",
        "created_at": "string - ISO timestamp",
        "updated_at": "string - ISO timestamp",
        "version": "string - Schema version"
    },
    
    "excel_specific_fields": {
        "row_index": "integer - Row number in spreadsheet",
        "sheet_name": "string - Excel sheet name",
        "columns": "array[string] - Column names",
        "row_data": "object - Original row as key-value pairs",
        "null_columns": "array[string] - Columns with null values"
    },
    
    "pdf_specific_fields": {
        "page_number": "integer - Page number",
        "paragraph_index": "integer - Paragraph number",
        "total_paragraphs": "integer - Total paragraphs",
        "total_pages": "integer - Total pages",
        "section_title": "string - Section/chapter title",
        "is_header": "boolean - Is header text",
        "is_footer": "boolean - Is footer text"
    },
    
    "json_specific_fields": {
        "block_index": "integer - JSON object index",
        "total_blocks": "integer - Total JSON objects",
        "json_path": "string - Path to block",
        "fields_with_content": "array[string] - Fields with data",
        "primary_field": "string - Main content field",
        "original_keys": "array[string] - All keys in object"
    },
    
    "indexing_recommendations": {
        "keyword_indices": [
            "file_type",
            "content_type", 
            "source_file"
        ],
        "integer_indices": [
            "primary_chunk_index",
            "row_index",
            "page_number",
            "block_index"
        ],
        "full_text_indices": [
            "text",
            "header"
        ]
    }
}


def create_excel_payload(
    text: str,
    source_file: str,
    row_index: int,
    columns: List[str],
    row_data: Dict[str, Any],
    header: str = "",
    content_type: str = "general",
    primary_chunk_index: int = None,
    sub_chunk_index: int = 0,
    total_sub_chunks: int = 1,
    sheet_name: str = ""
) -> Dict[str, Any]:
    """Factory function to create Excel payload"""
    
    payload = ExcelPayload(
        chunk_id=f"{source_file}_{row_index}_{sub_chunk_index}",
        text=text,
        source_file=source_file,
        file_type=FileType.EXCEL.value,
        content_type=content_type,
        primary_chunk_index=primary_chunk_index if primary_chunk_index is not None else row_index,
        sub_chunk_index=sub_chunk_index,
        total_sub_chunks=total_sub_chunks,
        header=header,
        row_index=row_index,
        sheet_name=sheet_name,
        columns=columns,
        row_data=row_data,
        null_columns=[k for k, v in row_data.items() if v is None or v == ""]
    )
    
    return payload.to_dict()


def create_pdf_payload(
    text: str,
    source_file: str,
    page_number: int,
    paragraph_index: int,
    total_paragraphs: int = 0,
    total_pages: int = 0,
    header: str = "",
    section_title: str = "",
    content_type: str = "documentation",
    primary_chunk_index: int = None,
    sub_chunk_index: int = 0,
    total_sub_chunks: int = 1
) -> Dict[str, Any]:
    """Factory function to create PDF payload"""
    
    payload = PDFPayload(
        chunk_id=f"{source_file}_{paragraph_index}_{sub_chunk_index}",
        text=text,
        source_file=source_file,
        file_type=FileType.PDF.value,
        content_type=content_type,
        primary_chunk_index=primary_chunk_index if primary_chunk_index is not None else paragraph_index,
        sub_chunk_index=sub_chunk_index,
        total_sub_chunks=total_sub_chunks,
        header=header,
        page_number=page_number,
        paragraph_index=paragraph_index,
        total_paragraphs=total_paragraphs,
        total_pages=total_pages,
        section_title=section_title
    )
    
    return payload.to_dict()


def create_json_payload(
    text: str,
    source_file: str,
    block_index: int,
    total_blocks: int = 0,
    fields_with_content: List[str] = None,
    original_keys: List[str] = None,
    header: str = "",
    primary_field: str = "",
    content_type: str = "article",
    primary_chunk_index: int = None,
    sub_chunk_index: int = 0,
    total_sub_chunks: int = 1
) -> Dict[str, Any]:
    """Factory function to create JSON payload"""
    
    payload = JSONPayload(
        chunk_id=f"{source_file}_{block_index}_{sub_chunk_index}",
        text=text,
        source_file=source_file,
        file_type=FileType.JSON.value,
        content_type=content_type,
        primary_chunk_index=primary_chunk_index if primary_chunk_index is not None else block_index,
        sub_chunk_index=sub_chunk_index,
        total_sub_chunks=total_sub_chunks,
        header=header,
        block_index=block_index,
        total_blocks=total_blocks,
        fields_with_content=fields_with_content or [],
        primary_field=primary_field,
        original_keys=original_keys or []
    )
    
    return payload.to_dict()


def get_payload_schema() -> Dict[str, Any]:
    """Return the payload schema documentation"""
    return PAYLOAD_SCHEMA


# ============ EXAMPLE PAYLOADS ============

EXAMPLE_EXCEL_PAYLOAD = {
    "chunk_id": "products_5_0",
    "collection_name": "chatbot_embeddings",
    "text": "product_id: 101\nname: Premium Widget\nprice: 299.99\ncategory: Electronics\ndescription: High-quality widget for enterprise use",
    "content_length": 124,
    "source_file": "products.xlsx",
    "source_path": "/data/products.xlsx",
    "file_type": "excel",
    "content_type": "product",
    "primary_chunk_index": 5,
    "sub_chunk_index": 0,
    "total_sub_chunks": 1,
    "header": "Premium Widget",
    "summary": "",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "version": "1.0",
    "row_index": 5,
    "sheet_name": "Products",
    "columns": ["product_id", "name", "price", "category", "description"],
    "row_data": {
        "product_id": 101,
        "name": "Premium Widget",
        "price": 299.99,
        "category": "Electronics",
        "description": "High-quality widget for enterprise use"
    },
    "null_columns": []
}

EXAMPLE_PDF_PAYLOAD = {
    "chunk_id": "annual_report_12_0",
    "collection_name": "chatbot_embeddings",
    "text": "The quarterly results demonstrate significant growth in the Asia-Pacific region. Revenue increased by 23% compared to the previous quarter, driven primarily by new product launches and expanded distribution channels.",
    "content_length": 234,
    "source_file": "annual_report.pdf",
    "source_path": "/data/reports/annual_report.pdf",
    "file_type": "pdf",
    "content_type": "report",
    "primary_chunk_index": 12,
    "sub_chunk_index": 0,
    "total_sub_chunks": 1,
    "header": "Q3 Financial Results",
    "summary": "",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "version": "1.0",
    "page_number": 5,
    "paragraph_index": 12,
    "total_paragraphs": 45,
    "total_pages": 20,
    "section_title": "Financial Performance",
    "is_header": False,
    "is_footer": False
}

EXAMPLE_JSON_PAYLOAD = {
    "chunk_id": "articles_3_0",
    "collection_name": "chatbot_embeddings",
    "text": "Header_1: Market Analysis 2024\nDescription_1: Comprehensive overview of market trends and competitive landscape\nWholeContent_1: The market experienced significant shifts in Q3 2024...",
    "content_length": 312,
    "source_file": "articles.json",
    "source_path": "/data/articles.json",
    "file_type": "json",
    "content_type": "article",
    "primary_chunk_index": 3,
    "sub_chunk_index": 0,
    "total_sub_chunks": 1,
    "header": "Market Analysis 2024",
    "summary": "",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "version": "1.0",
    "block_index": 3,
    "total_blocks": 150,
    "json_path": "data[3]",
    "fields_with_content": ["Header_1", "Description_1", "WholeContent_1"],
    "primary_field": "WholeContent_1",
    "original_keys": ["Header_1", "Description_1", "Description_2", "WholeContent_1", "id"]
}


if __name__ == "__main__":
    print("Payload Schema for chatbot_embeddings Collection")
    print("=" * 60)
    
    import json
    print("\nSchema Documentation:")
    print(json.dumps(PAYLOAD_SCHEMA, indent=2))
    
    print("\n" + "=" * 60)
    print("Example Excel Payload:")
    print(json.dumps(EXAMPLE_EXCEL_PAYLOAD, indent=2))
    
    print("\n" + "=" * 60)
    print("Example PDF Payload:")
    print(json.dumps(EXAMPLE_PDF_PAYLOAD, indent=2))
    
    print("\n" + "=" * 60)
    print("Example JSON Payload:")
    print(json.dumps(EXAMPLE_JSON_PAYLOAD, indent=2))