"""
Unified Data Ingestor
Processes files using smart chunking strategies and stores in Qdrant
- Excel: Row-wise
- PDF: Paragraph-wise  
- JSON: Block-wise
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Config.local_embeddings import get_local_embedding_model
from Config.qdrant_cfg import get_qdrant_client, ensure_collection_exists
from Config.payload_schema import (
    create_excel_payload, 
    create_pdf_payload, 
    create_json_payload,
    BasePayload
)
from Ingest.smart_chunker import SmartChunker, Chunk
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 300))

logger.info(f"UnifiedIngestor loaded with CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP} from .env")

class UnifiedIngestor:
    """
    Unified ingestion pipeline that processes multiple file types
    with appropriate chunking strategies
    """
    
    def __init__(
        self,
        collection_name: str = None,
        max_chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model: str = None
    ):
        """
        Initialize the unified ingestor
        
        Args:
            collection_name: Qdrant collection name
            max_chunk_size: Maximum chunk size for sub-chunking
            chunk_overlap: Overlap between sub-chunks
            embedding_model: Name of embedding model to use
        """
        # Load from environment with defaults
        self.collection_name = collection_name or os.getenv('DEFAULT_COLLECTION_NAME', 'business_dataset')
        self.max_chunk_size = max_chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
        
        # Initialize components
        logger.info("Initializing Unified Ingestor...")
        
        self.embedding_model = get_local_embedding_model(embedding_model)
        self.qdrant_client = get_qdrant_client()
        self.smart_chunker = SmartChunker(self.max_chunk_size, self.chunk_overlap)
        
        # Ensure collection exists
        embedding_dim = self.embedding_model.get_embedding_dimension()
        if not ensure_collection_exists(self.collection_name, embedding_dim):
            raise Exception(f"Failed to create/access collection: {self.collection_name}")
        
        logger.info(f"Ingestor ready:")
        logger.info(f"  Collection: {self.collection_name}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Max chunk size: {self.max_chunk_size}")
        logger.info(f"  Chunk overlap: {self.chunk_overlap}")
    
    def ingest_file(
        self, 
        file_path: str,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single file into Qdrant
        
        Args:
            file_path: Path to file to ingest
            additional_metadata: Extra metadata to add to all chunks
            
        Returns:
            Ingestion summary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        start_time = datetime.now()
        logger.info(f"Starting ingestion: {file_path}")
        
        # Step 1: Chunk the file
        logger.info("Step 1: Chunking file...")
        try:
            chunks = self.smart_chunker.chunk_file(file_path)
        except Exception as e:
            raise Exception(f"Chunking failed: {e}")
        
        if not chunks:
            return {
                'status': 'empty',
                'file': file_path,
                'message': 'No chunks extracted from file'
            }
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Add additional metadata
        if additional_metadata:
            for chunk in chunks:
                chunk.metadata.update(additional_metadata)
        
        # Step 3: Create embeddings
        logger.info("Step 2: Creating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self._create_embeddings(texts)
        
        # Step 4: Upload to Qdrant
        logger.info("Step 3: Uploading to Qdrant...")
        points_created = self._upload_chunks(chunks, embeddings)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        summary = {
            'status': 'success',
            'file': file_path,
            'file_type': chunks[0].file_type if chunks else 'unknown',
            'total_chunks': len(chunks),
            'points_created': points_created,
            'collection': self.collection_name,
            'elapsed_seconds': round(elapsed, 2)
        }
        
        logger.info(f"Ingestion complete: {points_created} points created in {elapsed:.2f}s")
        return summary
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest all supported files in a directory
        
        Args:
            directory: Path to directory
            recursive: Whether to process subdirectories
            additional_metadata: Extra metadata for all files
            
        Returns:
            Combined ingestion summary
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        start_time = datetime.now()
        logger.info(f"Starting directory ingestion: {directory}")
        
        # Find all supported files
        supported_extensions = self.smart_chunker.get_supported_extensions()
        files_to_process = []
        
        if recursive:
            for ext in supported_extensions:
                files_to_process.extend(Path(directory).rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files_to_process.extend(Path(directory).glob(f"*{ext}"))
        
        files_to_process = [str(f) for f in files_to_process]
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process each file
        results = {
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0,
            'total_points': 0,
            'file_results': [],
            'errors': []
        }
        
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                file_result = self.ingest_file(file_path, additional_metadata)
                results['file_results'].append(file_result)
                
                if file_result['status'] == 'success':
                    results['files_processed'] += 1
                    results['total_chunks'] += file_result['total_chunks']
                    results['total_points'] += file_result['points_created']
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results['files_failed'] += 1
                results['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        elapsed = (datetime.now() - start_time).total_seconds()
        results['elapsed_seconds'] = round(elapsed, 2)
        results['collection'] = self.collection_name
        
        logger.info(f"\nDirectory ingestion complete:")
        logger.info(f"  Files processed: {results['files_processed']}")
        logger.info(f"  Files failed: {results['files_failed']}")
        logger.info(f"  Total chunks: {results['total_chunks']}")
        logger.info(f"  Total points: {results['total_points']}")
        logger.info(f"  Elapsed: {elapsed:.2f}s")
        
        return results
    
    def ingest_multiple_files(
        self,
        file_paths: List[str],
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest multiple specific files
        
        Args:
            file_paths: List of file paths
            additional_metadata: Extra metadata for all files
            
        Returns:
            Combined ingestion summary
        """
        start_time = datetime.now()
        
        results = {
            'files_processed': 0,
            'files_failed': 0,
            'total_chunks': 0,
            'total_points': 0,
            'file_results': [],
            'errors': []
        }
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                file_result = self.ingest_file(file_path, additional_metadata)
                results['file_results'].append(file_result)
                
                if file_result['status'] == 'success':
                    results['files_processed'] += 1
                    results['total_chunks'] += file_result['total_chunks']
                    results['total_points'] += file_result['points_created']
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results['files_failed'] += 1
                results['errors'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        elapsed = (datetime.now() - start_time).total_seconds()
        results['elapsed_seconds'] = round(elapsed, 2)
        results['collection'] = self.collection_name
        
        return results
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.embed_documents(
                texts,
                batch_size=self.batch_size,
                show_progress=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def _upload_chunks(
        self, 
        chunks: List[Chunk], 
        embeddings: List[List[float]]
    ) -> int:
        """Upload chunks with embeddings to Qdrant using proper payload schema"""
        
        # Create points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Create payload based on file type
            payload = self._create_payload_for_chunk(chunk)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upload in batches
        upload_batch_size = 100
        total_uploaded = 0
        
        for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading"):
            batch = points[i:i + upload_batch_size]
            
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_uploaded += len(batch)
            except Exception as e:
                logger.error(f"Failed to upload batch {i}: {e}")
        
        return total_uploaded
    
    def _create_payload_for_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """Create properly structured payload based on file type"""
        
        file_type = chunk.file_type
        metadata = chunk.metadata
        
        if file_type == "excel":
            return create_excel_payload(
                text=chunk.content,
                source_file=chunk.source_file,
                row_index=metadata.get('row_index', chunk.primary_chunk_index),
                columns=metadata.get('columns', []),
                row_data=metadata.get('row_data', {}),
                header=metadata.get('header', ''),
                content_type=metadata.get('content_type', 'general'),
                primary_chunk_index=chunk.primary_chunk_index,
                sub_chunk_index=chunk.sub_chunk_index,
                total_sub_chunks=chunk.total_sub_chunks,
                sheet_name=metadata.get('sheet_name', '')
            )
        
        elif file_type == "pdf":
            return create_pdf_payload(
                text=chunk.content,
                source_file=chunk.source_file,
                page_number=metadata.get('page_number', -1),
                paragraph_index=metadata.get('paragraph_index', chunk.primary_chunk_index),
                total_paragraphs=metadata.get('total_paragraphs', 0),
                total_pages=metadata.get('total_pages', 0),
                header=metadata.get('header', ''),
                section_title=metadata.get('section_title', ''),
                content_type=metadata.get('content_type', 'documentation'),
                primary_chunk_index=chunk.primary_chunk_index,
                sub_chunk_index=chunk.sub_chunk_index,
                total_sub_chunks=chunk.total_sub_chunks
            )
        
        elif file_type == "json":
            return create_json_payload(
                text=chunk.content,
                source_file=chunk.source_file,
                block_index=metadata.get('block_index', chunk.primary_chunk_index),
                total_blocks=metadata.get('total_blocks', 0),
                fields_with_content=metadata.get('fields_with_content', []),
                original_keys=metadata.get('original_keys', []),
                header=metadata.get('header', ''),
                primary_field=metadata.get('primary_field', ''),
                content_type=metadata.get('content_type', 'article'),
                primary_chunk_index=chunk.primary_chunk_index,
                sub_chunk_index=chunk.sub_chunk_index,
                total_sub_chunks=chunk.total_sub_chunks
            )
        
        else:
            # Fallback to base payload structure
            return {
                'chunk_id': chunk.chunk_id,
                'collection_name': self.collection_name,
                'text': chunk.content,
                'content_length': len(chunk.content),
                'source_file': chunk.source_file,
                'file_type': file_type,
                'content_type': metadata.get('content_type', 'general'),
                'primary_chunk_index': chunk.primary_chunk_index,
                'sub_chunk_index': chunk.sub_chunk_index,
                'total_sub_chunks': chunk.total_sub_chunks,
                'header': metadata.get('header', ''),
                'version': '1.0',
                **metadata
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Sample points for analysis
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Analyze distribution
            file_type_counts = {}
            source_file_counts = {}
            
            for point in sample_points:
                file_type = point.payload.get('file_type', 'unknown')
                source = point.payload.get('source_file', 'unknown')
                
                file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
                source_file_counts[source] = source_file_counts.get(source, 0) + 1
            
            return {
                'collection_name': self.collection_name,
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'status': collection_info.status.value,
                'file_type_distribution': file_type_counts,
                'source_file_distribution': source_file_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


def create_test_files():
    """Create sample test files for testing"""
    import json
    
    test_dir = Path("/home/claude/Qdrant_rag/Data/test_files")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample JSON
    sample_json = [
        {
            "Header_1": "Product A - Premium Widget",
            "Description_1": "A high-quality widget designed for enterprise use. Features include durability, efficiency, and easy maintenance.",
            "WholeContent_1": "The Premium Widget (Product A) is our flagship offering in the widget category. It combines cutting-edge technology with robust construction."
        },
        {
            "Header_1": "Product B - Standard Gadget",
            "Description_1": "An affordable gadget suitable for everyday use. Offers basic functionality with reliable performance.",
            "WholeContent_1": "The Standard Gadget (Product B) provides excellent value for budget-conscious consumers. It includes all essential features."
        },
        {
            "Header_1": "Service Package - Maintenance Plus",
            "Description_1": "Comprehensive maintenance service for all our products. Includes regular checkups and priority support.",
            "WholeContent_1": "Maintenance Plus is our premium service offering that ensures your equipment runs smoothly year-round."
        }
    ]
    
    json_path = test_dir / "sample_products.json"
    with open(json_path, 'w') as f:
        json.dump(sample_json, f, indent=2)
    
    # Create sample CSV
    csv_content = """product_id,product_name,category,price,description
1,Widget Pro,Electronics,299.99,Professional grade widget with advanced features
2,Gadget Basic,Tools,49.99,Entry-level gadget for beginners
3,Device Ultra,Electronics,599.99,Ultra-premium device with all bells and whistles
4,Tool Master,Tools,149.99,Master-class tool for professionals
5,System Core,Software,999.99,Core system software for enterprise deployment
"""
    
    csv_path = test_dir / "sample_inventory.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    
    print(f"Created test files in {test_dir}")
    return str(test_dir)


def test_unified_ingestor():
    """Test the unified ingestor"""
    print("Testing Unified Ingestor")
    print("=" * 60)
    
    try:
        # Create test files
        test_dir = create_test_files()
        
        # Initialize ingestor
        ingestor = UnifiedIngestor(
            collection_name="test_unified_ingest",
            max_chunk_size=None,
            chunk_overlap=None
        )
        
        # Test single file ingestion
        print("\n--- Testing Single File Ingestion (JSON) ---")
        json_result = ingestor.ingest_file(f"{test_dir}/sample_products.json")
        print(f"JSON Result: {json_result}")
        
        print("\n--- Testing Single File Ingestion (CSV) ---")
        csv_result = ingestor.ingest_file(f"{test_dir}/sample_inventory.csv")
        print(f"CSV Result: {csv_result}")
        
        # Get collection stats
        print("\n--- Collection Statistics ---")
        stats = ingestor.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Unified Ingestor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_unified_ingestor()