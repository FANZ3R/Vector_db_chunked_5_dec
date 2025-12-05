# Multi-File RAG System with Smart Chunking

A Retrieval-Augmented Generation (RAG) system that intelligently processes different file types with appropriate chunking strategies:

- **Excel/CSV files**: Row-wise chunking (each row = one product/record/article)
- **PDF files**: Paragraph-wise chunking (each paragraph = related content)
- **JSON files**: Block-wise chunking (each object = complete entity)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │  Excel   │    │   PDF    │    │   JSON   │                 │
│   │ (.xlsx)  │    │ (.pdf)   │    │ (.json)  │                 │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                 │
│        │               │               │                        │
│        ▼               ▼               ▼                        │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │ Row-wise │    │Paragraph │    │Block-wise│                 │
│   │ Chunking │    │ Chunking │    │ Chunking │                 │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                 │
│        │               │               │                        │
│        └───────────────┼───────────────┘                        │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │  Sub-Chunking   │ (if content > max_chunk_size) │
│              │  with Overlap   │                                │
│              └────────┬────────┘                                │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │   Embeddings    │ (sentence-transformers)       │
│              └────────┬────────┘                                │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │     Qdrant      │ (Vector Database)             │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Qdrant_rag/
├── Config/
│   ├── __init__.py
│   ├── local_embeddings.py    # Sentence transformer embeddings
│   └── qdrant_cfg.py          # Qdrant connection & utilities
├── Ingest/
│   ├── __init__.py
│   ├── smart_chunker.py       # File-type specific chunking
│   └── unified_ingestor.py    # Main ingestion pipeline
├── Retrieve/
│   ├── __init__.py
│   └── enhanced_hybrid_search.py  # Search with context expansion
├── Data/                      # Place your data files here
├── chatbot.py                 # Streamlit RAG chatbot
├── ingest_cli.py              # CLI for ingestion
├── docker-compose.yml         # Qdrant container
├── requirements.txt
├── .env
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker-compose up -d
```

### 3. Configure Environment

Edit `.env` file:
```
QDRANT_URL=http://localhost:6333
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_COLLECTION_NAME=business_dataset
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Usage

### Ingesting Data

#### Using CLI Tool

```bash
# Ingest a single file
python ingest_cli.py file Data/products.xlsx

# Ingest a directory
python ingest_cli.py directory Data/ --recursive

# Ingest with custom collection
python ingest_cli.py file Data/report.pdf --collection my_collection

# Preview chunks without ingesting
python ingest_cli.py preview Data/sample.json --limit 5

# Show collection statistics
python ingest_cli.py stats --collection business_dataset
```

#### Using Python

```python
from Ingest import UnifiedIngestor

# Initialize ingestor
ingestor = UnifiedIngestor(
    collection_name="my_collection",
    max_chunk_size=500,
    chunk_overlap=50
)

# Ingest a single file
result = ingestor.ingest_file("Data/products.xlsx")
print(f"Created {result['points_created']} vectors")

# Ingest entire directory
result = ingestor.ingest_directory("Data/", recursive=True)
print(f"Processed {result['files_processed']} files")
```

### Searching

```python
from Retrieve import EnhancedHybridSearch

# Initialize search engine
search = EnhancedHybridSearch("my_collection")

# Basic search
results = search.search(
    query="product specifications",
    limit=10,
    score_threshold=0.3
)

# Search specific file type
results = search.search_by_file_type(
    query="quarterly report",
    file_type="pdf",
    limit=5
)

# Search specific source file
results = search.search_by_source(
    query="inventory",
    source_file="products.xlsx",
    limit=5
)

# Get collection statistics
stats = search.get_collection_stats()
print(stats)
```

### Running the Chatbot

```bash
streamlit run chatbot.py
```

## Chunking Strategies Explained

### Excel/CSV (Row-wise)

Each row in a spreadsheet typically represents a complete record:
- Product entry
- Customer record
- Inventory item
- Transaction

**Example:**
```
Row 1: {"product_id": 1, "name": "Widget A", "price": 99.99, "category": "Electronics"}
→ Chunk 1: "product_id: 1\nname: Widget A\nprice: 99.99\ncategory: Electronics"
```

### PDF (Paragraph-wise)

Each paragraph in a document contains related information:
- Topic explanation
- Procedure step
- Report section

**Example:**
```
"The quarterly results show significant growth in the Asia-Pacific region. 
Revenue increased by 23% compared to the previous quarter, driven primarily 
by new product launches and expanded distribution channels."
→ One chunk (unless too long, then sub-chunked)
```

### JSON (Block-wise)

Each JSON object represents a complete entity:
- Article/product data
- Configuration entry
- Record with multiple fields

**Example:**
```json
{
  "Header_1": "Market Analysis 2024",
  "Description_1": "Comprehensive overview of market trends...",
  "WholeContent_1": "The market experienced significant shifts..."
}
→ All fields combined into one chunk
```

### Sub-Chunking

When any primary chunk exceeds `max_chunk_size`:
1. Split on sentence boundaries
2. Add overlap between sub-chunks
3. Maintain parent-child relationship in metadata

## Metadata Structure

Each vector in Qdrant includes:

```json
{
  "text": "Actual chunk content",
  "source_file": "products.xlsx",
  "file_type": "excel",
  "primary_chunk_index": 5,
  "sub_chunk_index": 0,
  "total_sub_chunks": 1,
  "chunk_id": "products_5_0",
  "content_length": 450,
  
  // File-type specific metadata
  // Excel:
  "row_index": 5,
  "columns": ["id", "name", "price"],
  "row_data": {"id": 5, "name": "Widget", "price": 99.99},
  
  // PDF:
  "paragraph_index": 12,
  "total_paragraphs": 45,
  
  // JSON:
  "block_index": 3,
  "total_blocks": 100,
  "header": "Market Analysis 2024"
}
```

## API Reference

### SmartChunker

```python
chunker = SmartChunker(max_chunk_size=500, chunk_overlap=50)

# Chunk a file (auto-detects type)
chunks = chunker.chunk_file("data.xlsx")

# Chunk entire directory
results = chunker.chunk_directory("Data/", recursive=True)

# Get supported extensions
extensions = chunker.get_supported_extensions()
# ['.xlsx', '.xls', '.xlsm', '.csv', '.pdf', '.json']
```

### UnifiedIngestor

```python
ingestor = UnifiedIngestor(
    collection_name="my_collection",
    max_chunk_size=500,
    chunk_overlap=50
)

# Ingest single file
result = ingestor.ingest_file("data.pdf")

# Ingest directory
result = ingestor.ingest_directory("Data/", recursive=True)

# Get statistics
stats = ingestor.get_collection_stats()
```

### EnhancedHybridSearch

```python
search = EnhancedHybridSearch("my_collection")

# Main search method
results = search.search(
    query="search text",
    limit=10,
    score_threshold=0.3,
    file_type="excel",      # Optional filter
    source_file="data.xlsx", # Optional filter
    expand_context=True      # Combine sub-chunks
)

# Get available filters
filters = search.get_available_filters()
# {"file_types": ["excel", "pdf", "json"], "source_files": [...]}

# Get collection stats
stats = search.get_collection_stats()
```

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# View Qdrant logs
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

### PDF Extraction Issues

Install PDF libraries:
```bash
pip install PyMuPDF
# or
pip install pdfplumber
```

### Memory Issues with Large Files

Reduce batch size in `.env`:
```
BATCH_SIZE=16
```

### Missing Dependencies

```bash
pip install openpyxl  # For Excel files
pip install PyMuPDF   # For PDF files
```

## Performance Tips

1. **Use appropriate chunk sizes**: 
   - Smaller chunks (300-500) for precise retrieval
   - Larger chunks (500-1000) for more context

2. **Optimize batch size**: 
   - Higher batch size = faster processing, more memory
   - Lower batch size = slower processing, less memory

3. **Index your collections**:
   - Qdrant automatically indexes vectors
   - Add payload indices for frequently filtered fields

4. **Use filters when possible**:
   - Filter by file_type or source_file to narrow search space

## License

MIT License