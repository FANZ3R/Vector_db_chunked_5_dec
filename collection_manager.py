"""
Collection Manager for Qdrant
Manages multiple collections, payload indices, and collection operations
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ COLLECTION CONFIGURATIONS ============

COLLECTIONS = {
    "chatbot_embeddings": {
        "description": "Main collection for RAG chatbot - stores embeddings from Excel, PDF, JSON",
        "vector_size": 384,  # all-MiniLM-L6-v2
        "distance": Distance.COSINE,
        "payload_indices": {
            # Keyword indices for filtering
            "file_type": PayloadSchemaType.KEYWORD,
            "content_type": PayloadSchemaType.KEYWORD,
            "source_file": PayloadSchemaType.KEYWORD,
            "collection_name": PayloadSchemaType.KEYWORD,
            # Integer indices for range queries
            "primary_chunk_index": PayloadSchemaType.INTEGER,
            "row_index": PayloadSchemaType.INTEGER,
            "page_number": PayloadSchemaType.INTEGER,
            "block_index": PayloadSchemaType.INTEGER,
        },
        "text_indices": ["text", "header"]
    },
    "test_business_data": {
        "description": "Test collection for development and testing",
        "vector_size": 384,
        "distance": Distance.COSINE,
        "payload_indices": {
            "file_type": PayloadSchemaType.KEYWORD,
            "source_file": PayloadSchemaType.KEYWORD,
        },
        "text_indices": ["text"]
    }
}


class CollectionManager:
    """
    Manages Qdrant collections for the RAG system
    
    Features:
    - Create collections with proper configuration
    - Set up payload indices for efficient filtering
    - Manage multiple collections
    - Collection statistics and health checks
    """
    
    def __init__(self, qdrant_url: str = None):
        """
        Initialize the collection manager
        
        Args:
            qdrant_url: Qdrant server URL (defaults to env variable)
        """
        self.url = qdrant_url or os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.client = QdrantClient(url=self.url)
        
        logger.info(f"CollectionManager connected to Qdrant at {self.url}")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> bool:
        """
        Create a new collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (384 for all-MiniLM-L6-v2)
            distance: Distance metric
            recreate: If True, delete existing collection first
            
        Returns:
            True if successful
        """
        try:
            # Check if collection exists
            existing = self.list_collections()
            
            if collection_name in existing:
                if recreate:
                    logger.info(f"Deleting existing collection '{collection_name}'")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection '{collection_name}' already exists")
                    return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            logger.info(f"Created collection '{collection_name}' with {vector_size} dimensions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False
    
    def setup_collection_from_config(
        self,
        collection_name: str,
        recreate: bool = False
    ) -> bool:
        """
        Create and configure a collection from predefined config
        
        Args:
            collection_name: Name of collection (must be in COLLECTIONS)
            recreate: If True, recreate the collection
            
        Returns:
            True if successful
        """
        if collection_name not in COLLECTIONS:
            logger.error(f"Unknown collection: {collection_name}")
            logger.info(f"Available collections: {list(COLLECTIONS.keys())}")
            return False
        
        config = COLLECTIONS[collection_name]
        
        # Create the collection
        success = self.create_collection(
            collection_name=collection_name,
            vector_size=config["vector_size"],
            distance=config["distance"],
            recreate=recreate
        )
        
        if not success:
            return False
        
        # Set up payload indices
        if "payload_indices" in config:
            self.setup_payload_indices(collection_name, config["payload_indices"])
        
        # Set up text indices
        if "text_indices" in config:
            for field in config["text_indices"]:
                self.create_text_index(collection_name, field)
        
        logger.info(f"Collection '{collection_name}' fully configured")
        return True
    
    def setup_payload_indices(
        self,
        collection_name: str,
        indices: Dict[str, PayloadSchemaType]
    ) -> bool:
        """
        Set up payload indices for efficient filtering
        
        Args:
            collection_name: Collection to index
            indices: Dict mapping field names to schema types
            
        Returns:
            True if successful
        """
        try:
            for field_name, schema_type in indices.items():
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
                logger.info(f"Created index on '{field_name}' ({schema_type})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create payload indices: {e}")
            return False
    
    def create_text_index(
        self,
        collection_name: str,
        field_name: str,
        tokenizer: TokenizerType = TokenizerType.WORD,
        min_token_len: int = 2,
        max_token_len: int = 20
    ) -> bool:
        """
        Create a full-text search index on a field
        
        Args:
            collection_name: Collection to index
            field_name: Field to create text index on
            tokenizer: Tokenizer type
            min_token_len: Minimum token length
            max_token_len: Maximum token length
            
        Returns:
            True if successful
        """
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=tokenizer,
                    min_token_len=min_token_len,
                    max_token_len=max_token_len
                )
            )
            logger.info(f"Created text index on '{field_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create text index on '{field_name}': {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection"""
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "status": info.status.value,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
                "payload_schema": {
                    k: str(v) for k, v in (info.payload_schema or {}).items()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about collection contents"""
        try:
            info = self.get_collection_info(collection_name)
            
            if not info:
                return {}
            
            # Sample points for analysis
            sample_points = self.client.scroll(
                collection_name=collection_name,
                limit=200,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Analyze distributions
            file_type_dist = {}
            content_type_dist = {}
            source_file_dist = {}
            
            for point in sample_points:
                payload = point.payload or {}
                
                ft = payload.get('file_type', 'unknown')
                ct = payload.get('content_type', 'unknown')
                sf = payload.get('source_file', 'unknown')
                
                file_type_dist[ft] = file_type_dist.get(ft, 0) + 1
                content_type_dist[ct] = content_type_dist.get(ct, 0) + 1
                source_file_dist[sf] = source_file_dist.get(sf, 0) + 1
            
            return {
                **info,
                "sample_size": len(sample_points),
                "file_type_distribution": file_type_dist,
                "content_type_distribution": content_type_dist,
                "source_file_distribution": source_file_dist
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def clear_collection(self, collection_name: str) -> bool:
        """Clear all points from a collection without deleting it"""
        try:
            # Get collection config
            info = self.client.get_collection(collection_name)
            
            # Delete and recreate with same config
            self.client.delete_collection(collection_name)
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=info.config.params.vectors.size,
                    distance=info.config.params.vectors.distance
                )
            )
            
            logger.info(f"Cleared collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def delete_by_source_file(
        self,
        collection_name: str,
        source_file: str
    ) -> int:
        """
        Delete all points from a specific source file
        
        Args:
            collection_name: Collection to delete from
            source_file: Source file name to filter by
            
        Returns:
            Number of points deleted
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        try:
            # First count how many will be deleted
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file)
                        )
                    ]
                )
            )
            
            count = count_result.count
            
            if count == 0:
                logger.info(f"No points found for source_file='{source_file}'")
                return 0
            
            # Delete the points
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted {count} points from source_file='{source_file}'")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete by source file: {e}")
            return 0
    
    def delete_by_file_type(
        self,
        collection_name: str,
        file_type: str
    ) -> int:
        """Delete all points of a specific file type"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        try:
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_type",
                            match=MatchValue(value=file_type)
                        )
                    ]
                )
            )
            
            count = count_result.count
            
            if count == 0:
                return 0
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_type",
                            match=MatchValue(value=file_type)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted {count} points with file_type='{file_type}'")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete by file type: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Qdrant connection"""
        try:
            collections = self.list_collections()
            
            return {
                "status": "healthy",
                "url": self.url,
                "collections_count": len(collections),
                "collections": collections,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": self.url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def print_collection_summary(self, collection_name: str = None):
        """Print a formatted summary of collection(s)"""
        
        if collection_name:
            collections = [collection_name]
        else:
            collections = self.list_collections()
        
        print("\n" + "=" * 60)
        print("QDRANT COLLECTION SUMMARY")
        print("=" * 60)
        
        for name in collections:
            stats = self.get_collection_stats(name)
            
            if not stats:
                print(f"\n{name}: Unable to retrieve stats")
                continue
            
            print(f"\n📦 Collection: {name}")
            print("-" * 40)
            print(f"   Status: {stats.get('status', 'unknown')}")
            print(f"   Total Points: {stats.get('points_count', 0):,}")
            print(f"   Vector Size: {stats.get('vector_size', 'N/A')}")
            print(f"   Distance: {stats.get('distance', 'N/A')}")
            
            if stats.get('file_type_distribution'):
                print("\n   File Types:")
                for ft, count in stats['file_type_distribution'].items():
                    print(f"      - {ft}: {count}")
            
            if stats.get('source_file_distribution'):
                print("\n   Source Files:")
                for sf, count in list(stats['source_file_distribution'].items())[:5]:
                    print(f"      - {sf}: {count}")
                if len(stats['source_file_distribution']) > 5:
                    print(f"      ... and {len(stats['source_file_distribution']) - 5} more")
        
        print("\n" + "=" * 60)


def setup_chatbot_embeddings_collection(recreate: bool = False) -> bool:
    """
    Convenience function to set up the main chatbot_embeddings collection
    
    Args:
        recreate: If True, recreate the collection from scratch
        
    Returns:
        True if successful
    """
    manager = CollectionManager()
    return manager.setup_collection_from_config("chatbot_embeddings", recreate=recreate)


def main():
    """CLI for collection management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qdrant Collection Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    subparsers.add_parser('list', help='List all collections')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a collection')
    create_parser.add_argument('name', help='Collection name')
    create_parser.add_argument('--recreate', action='store_true', help='Recreate if exists')
    
    # Setup command (uses predefined config)
    setup_parser = subparsers.add_parser('setup', help='Setup collection from config')
    setup_parser.add_argument('name', choices=list(COLLECTIONS.keys()), help='Collection name')
    setup_parser.add_argument('--recreate', action='store_true', help='Recreate if exists')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get collection info')
    info_parser.add_argument('name', help='Collection name')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get collection statistics')
    stats_parser.add_argument('name', nargs='?', help='Collection name (all if not specified)')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a collection')
    delete_parser.add_argument('name', help='Collection name')
    delete_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all points from collection')
    clear_parser.add_argument('name', help='Collection name')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm clearing')
    
    # Health command
    subparsers.add_parser('health', help='Health check')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = CollectionManager()
    
    if args.command == 'list':
        collections = manager.list_collections()
        print(f"\nCollections ({len(collections)}):")
        for col in collections:
            print(f"  - {col}")
    
    elif args.command == 'create':
        success = manager.create_collection(args.name, recreate=args.recreate)
        print(f"{'✅ Created' if success else '❌ Failed'}: {args.name}")
    
    elif args.command == 'setup':
        success = manager.setup_collection_from_config(args.name, recreate=args.recreate)
        print(f"{'✅ Setup complete' if success else '❌ Setup failed'}: {args.name}")
    
    elif args.command == 'info':
        info = manager.get_collection_info(args.name)
        if info:
            import json
            print(json.dumps(info, indent=2))
        else:
            print(f"Collection '{args.name}' not found")
    
    elif args.command == 'stats':
        manager.print_collection_summary(args.name)
    
    elif args.command == 'delete':
        if not args.confirm:
            print(f"To delete '{args.name}', add --confirm flag")
        else:
            success = manager.delete_collection(args.name)
            print(f"{'✅ Deleted' if success else '❌ Failed'}: {args.name}")
    
    elif args.command == 'clear':
        if not args.confirm:
            print(f"To clear '{args.name}', add --confirm flag")
        else:
            success = manager.clear_collection(args.name)
            print(f"{'✅ Cleared' if success else '❌ Failed'}: {args.name}")
    
    elif args.command == 'health':
        health = manager.health_check()
        import json
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    main()