#!/usr/bin/env python3
"""
Batch ingestion script for multiple JSON files
Handles directory scanning and optional collection reset
"""

import os
import sys
from pathlib import Path
from typing import List
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Ingest.custom_json_ingest import FlexibleJSONIngestor
from Config.qdrant_cfg import get_qdrant_client

from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from project root
project_root = Path(__file__).parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 300))
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_json_files(data_dir: str = "Data") -> List[str]:
    """
    Get all JSON files from the data directory
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of JSON file paths
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files")
    return [str(f) for f in json_files]


def clear_collection(collection_name: str) -> bool:
    """
    Delete and recreate a collection (removes all data)
    
    Args:
        collection_name: Name of collection to clear
        
    Returns:
        True if successful
    """
    try:
        client = get_qdrant_client()
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)# Use docker compo
            logger.info(f"Collection '{collection_name}' deleted successfully")
        else:
            logger.info(f"Collection '{collection_name}' doesn't exist yet")
        
        return True
        
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        return False


def get_collection_stats(collection_name: str) -> dict:
    """Get statistics about existing collection"""
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            return {"exists": False}
        
        info = client.get_collection(collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "vectors_count": getattr(info, 'vectors_count', None) or getattr(info, 'indexed_vectors_count', None)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"exists": False}


def ingest_all_files(
    data_dir: str = "Data",
    collection_name: str = "business_dataset_full",
    clear_existing: bool = False,
    max_records_per_file: int = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    auto_confirm: bool = False
):
    """
    Ingest all JSON files from directory into Qdrant
    
    Args:
        data_dir: Directory containing JSON files
        collection_name: Target collection name
        clear_existing: Whether to clear existing data first
        max_records_per_file: Limit records per file (for testing)
        chunk_size: Text chunk size
        chunk_overlap: Overlap between chunks
        auto_confirm: If True, skip all confirmation prompts
    """
    
    print("=" * 80)
    print("BATCH JSON INGESTION")
    print("=" * 80)
    
    # Get all JSON files
    json_files = get_json_files(data_dir)
    
    if not json_files:
        print("No JSON files found. Exiting.")
        return
    
    print(f"\nFound {len(json_files)} JSON files:")
    for i, file in enumerate(json_files, 1):
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"  {i}. {Path(file).name} ({file_size:.2f} MB)")
    
    # Check existing collection
    existing_stats = get_collection_stats(collection_name)
    
    if existing_stats["exists"]:
        print(f"\n⚠️  Collection '{collection_name}' already exists!")
        print(f"   Current points: {existing_stats['points_count']:,}")
        print(f"   Current vectors: {existing_stats['vectors_count']:,}")
        
        if clear_existing:
            print("\n🗑️  Clear existing data: YES (specified by clear_existing=True)")
        else:
            if not auto_confirm:
                print("\n⚠️  WARNING: Running without clearing will ADD to existing data")
                print("   This may create DUPLICATES if files were already processed!")
                
                response = input("\nDo you want to CLEAR existing data first? (yes/no): ").strip().lower()
                clear_existing = response in ['yes', 'y']
            else:
                print("\n⚠️  AUTO MODE: Keeping existing data (clear_existing=False)")
    else:
        print(f"\n✨ Collection '{collection_name}' will be created")
    
    # Clear collection if requested
    if clear_existing:
        print("\n🗑️  Clearing existing collection...")
        if not clear_collection(collection_name):
            print("Failed to clear collection. Exiting.")
            return
        print("✅ Collection cleared successfully")
    
    # Confirm before proceeding
    print(f"\n📋 Processing Configuration:")
    print(f"   - Collection: {collection_name}")
    print(f"   - Files to process: {len(json_files)}")
    print(f"   - Chunk size: {chunk_size}")
    print(f"   - Chunk overlap: {chunk_overlap}")
    if max_records_per_file:
        print(f"   - Max records per file: {max_records_per_file} (TESTING MODE)")
    
    if not auto_confirm:
        response = input("\n▶️  Proceed with ingestion? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return
    else:
        print("\n▶️  AUTO MODE: Proceeding automatically...")
    
    # Initialize ingestor
    print("\n" + "=" * 80)
    print("STARTING INGESTION")
    print("=" * 80)
    
    ingestor = FlexibleJSONIngestor()
    
    # Process each file
    all_summaries = []
    total_records = 0
    total_chunks = 0
    total_points = 0
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing File {i}/{len(json_files)}: {Path(json_file).name}")
        print(f"{'=' * 80}")
        
        try:
            summary = ingestor.process_json_file(
                json_file_path=json_file,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_records=max_records_per_file
            )
            
            all_summaries.append(summary)
            total_records += summary['total_records']
            total_chunks += summary['total_chunks']
            total_points += summary['points_created']
            
            print(f"\n✅ File {i} completed:")
            print(f"   - Records processed: {summary['total_records']:,}")
            print(f"   - Chunks created: {summary['total_chunks']:,}")
            print(f"   - Points uploaded: {summary['points_created']:,}")
            
        except Exception as e:
            print(f"\n❌ Error processing {json_file}: {e}")
            logger.error(f"Failed to process {json_file}: {e}")
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Final Statistics:")
    print(f"   - Files processed: {len(all_summaries)}/{len(json_files)}")
    print(f"   - Total records: {total_records:,}")
    print(f"   - Total chunks: {total_chunks:,}")
    print(f"   - Total points uploaded: {total_points:,}")
    print(f"   - Collection: {collection_name}")
    
    # Verify final collection state
    final_stats = get_collection_stats(collection_name)
    if final_stats["exists"]:
        print(f"\n✅ Collection verification:")
        print(f"   - Points in collection: {final_stats['points_count']:,}")
        if final_stats.get('vectors_count') is not None:
            print(f"   - Vectors in collection: {final_stats['vectors_count']:,}")
    
    # Save summary to file
    summary_file = f"ingestion_summary_{collection_name}.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump({
                'collection_name': collection_name,
                'total_files': len(json_files),
                'processed_files': len(all_summaries),
                'total_records': total_records,
                'total_chunks': total_chunks,
                'total_points': total_points,
                'file_summaries': all_summaries
            }, f, indent=2)
        print(f"\n📄 Detailed summary saved to: {summary_file}")
    except Exception as e:
        logger.warning(f"Could not save summary file: {e}")


def main():
    """Main entry point"""
    
    print("\n🚀 Batch JSON Ingestion Tool")
    print("=" * 80)
    
    # Check for auto mode (no prompts)
    auto_mode = '--auto' in sys.argv or '--yes' in sys.argv
    
    if auto_mode:
        print("🤖 AUTO MODE: Running with automatic confirmation")
    
    # Configuration
    DATA_DIR = "Data"
    COLLECTION_NAME = "business_dataset_full"
    CLEAR_EXISTING = True  # Set to True to always clear before ingestion
    
    # For testing with limited data, set max_records_per_file
    # For full production, set to None
    MAX_RECORDS_PER_FILE = None  # None = process all records
    
    # Chunk settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Run ingestion
    ingest_all_files(
        data_dir=DATA_DIR,
        collection_name=COLLECTION_NAME,
        clear_existing=CLEAR_EXISTING,
        max_records_per_file=MAX_RECORDS_PER_FILE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        auto_confirm=auto_mode
    )


if __name__ == "__main__":
    try:
        # Show help if requested
        if '--help' in sys.argv or '-h' in sys.argv:
            print("Usage: python batch_ingest_all.py [OPTIONS]")
            print("\nOptions:")
            print("  --auto, --yes    Run without confirmation prompts (automatic mode)")
            print("  -h, --help       Show this help message")
            print("\nExamples:")
            print("  python batch_ingest_all.py              # Interactive mode (asks for confirmation)")
            print("  python batch_ingest_all.py --auto       # Automatic mode (no prompts)")
            print("\nConfiguration:")
            print("  Edit the script to change:")
            print("  - CLEAR_EXISTING: Clear old data before ingestion (default: True)")
            print("  - DATA_DIR: Directory with JSON files (default: 'Data')")
            print("  - COLLECTION_NAME: Target collection (default: 'business_dataset_full')")
            sys.exit(0)
        
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error during ingestion")
        sys.exit(1)