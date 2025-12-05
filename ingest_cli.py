#!/usr/bin/env python3
"""
CLI Tool for Data Ingestion
Supports Excel, PDF, and JSON files with smart chunking
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Ingest.unified_ingestor import UnifiedIngestor
from Ingest.smart_chunker import SmartChunker


def main():
    parser = argparse.ArgumentParser(
        description="Ingest files into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single file
  python ingest_cli.py file data/products.xlsx
  
  # Ingest a directory
  python ingest_cli.py directory data/ --recursive
  
  # Ingest with custom collection name
  python ingest_cli.py file data/report.pdf --collection my_collection
  
  # Preview chunks without ingesting
  python ingest_cli.py preview data/sample.json --limit 5

Supported file types:
  - Excel: .xlsx, .xls, .xlsm, .csv (row-wise chunking)
  - PDF: .pdf (paragraph-wise chunking)
  - JSON: .json (block-wise chunking)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # File command
    file_parser = subparsers.add_parser('file', help='Ingest a single file')
    file_parser.add_argument('path', help='Path to file')
    file_parser.add_argument('--collection', '-c', default='business_dataset',
                            help='Collection name (default: business_dataset)')
    file_parser.add_argument('--chunk-size', type=int, default=500,
                            help='Max chunk size (default: 500)')
    file_parser.add_argument('--chunk-overlap', type=int, default=50,
                            help='Chunk overlap (default: 50)')
    
    # Directory command
    dir_parser = subparsers.add_parser('directory', help='Ingest all files in a directory')
    dir_parser.add_argument('path', help='Path to directory')
    dir_parser.add_argument('--collection', '-c', default='business_dataset',
                           help='Collection name (default: business_dataset)')
    dir_parser.add_argument('--recursive', '-r', action='store_true',
                           help='Process subdirectories')
    dir_parser.add_argument('--chunk-size', type=int, default=500,
                           help='Max chunk size (default: 500)')
    dir_parser.add_argument('--chunk-overlap', type=int, default=50,
                           help='Chunk overlap (default: 50)')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Preview chunks without ingesting')
    preview_parser.add_argument('path', help='Path to file')
    preview_parser.add_argument('--limit', '-l', type=int, default=10,
                               help='Number of chunks to show (default: 10)')
    preview_parser.add_argument('--chunk-size', type=int, default=500,
                               help='Max chunk size (default: 500)')
    preview_parser.add_argument('--chunk-overlap', type=int, default=50,
                               help='Chunk overlap (default: 50)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show collection statistics')
    stats_parser.add_argument('--collection', '-c', default='business_dataset',
                             help='Collection name (default: business_dataset)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'file':
        ingest_file(args)
    elif args.command == 'directory':
        ingest_directory(args)
    elif args.command == 'preview':
        preview_chunks(args)
    elif args.command == 'stats':
        show_stats(args)


def ingest_file(args):
    """Ingest a single file"""
    print(f"📁 Ingesting file: {args.path}")
    print(f"   Collection: {args.collection}")
    print(f"   Chunk size: {args.chunk_size}")
    print()
    
    try:
        ingestor = UnifiedIngestor(
            collection_name=args.collection,
            max_chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        result = ingestor.ingest_file(args.path)
        
        print("\n✅ Ingestion Complete!")
        print(f"   Status: {result.get('status')}")
        print(f"   File type: {result.get('file_type')}")
        print(f"   Total chunks: {result.get('total_chunks')}")
        print(f"   Points created: {result.get('points_created')}")
        print(f"   Time: {result.get('elapsed_seconds')}s")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def ingest_directory(args):
    """Ingest all files in a directory"""
    print(f"📂 Ingesting directory: {args.path}")
    print(f"   Collection: {args.collection}")
    print(f"   Recursive: {args.recursive}")
    print()
    
    try:
        ingestor = UnifiedIngestor(
            collection_name=args.collection,
            max_chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        result = ingestor.ingest_directory(args.path, recursive=args.recursive)
        
        print("\n✅ Directory Ingestion Complete!")
        print(f"   Files processed: {result.get('files_processed')}")
        print(f"   Files failed: {result.get('files_failed')}")
        print(f"   Total chunks: {result.get('total_chunks')}")
        print(f"   Total points: {result.get('total_points')}")
        print(f"   Time: {result.get('elapsed_seconds')}s")
        
        if result.get('errors'):
            print("\n⚠️ Errors:")
            for error in result['errors']:
                print(f"   - {error['file']}: {error['error']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def preview_chunks(args):
    """Preview chunks without ingesting"""
    print(f"👁️ Previewing chunks for: {args.path}")
    print()
    
    try:
        chunker = SmartChunker(
            max_chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        chunks = chunker.chunk_file(args.path)
        
        print(f"Total chunks: {len(chunks)}")
        print(f"File type: {chunks[0].file_type if chunks else 'unknown'}")
        print("-" * 60)
        
        for i, chunk in enumerate(chunks[:args.limit]):
            print(f"\n[Chunk {i + 1}]")
            print(f"  Primary index: {chunk.primary_chunk_index}")
            print(f"  Sub-chunk: {chunk.sub_chunk_index + 1}/{chunk.total_sub_chunks}")
            print(f"  Length: {len(chunk.content)} chars")
            print(f"  Content preview:")
            
            preview = chunk.content[:200]
            if len(chunk.content) > 200:
                preview += "..."
            
            for line in preview.split('\n')[:5]:
                print(f"    {line}")
            
            print()
        
        if len(chunks) > args.limit:
            print(f"... and {len(chunks) - args.limit} more chunks")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def show_stats(args):
    """Show collection statistics"""
    print(f"📊 Statistics for collection: {args.collection}")
    print()
    
    try:
        ingestor = UnifiedIngestor(collection_name=args.collection)
        stats = ingestor.get_collection_stats()
        
        if not stats:
            print("No statistics available")
            return
        
        print(f"Collection: {stats.get('collection_name')}")
        print(f"Total points: {stats.get('total_points', 0):,}")
        print(f"Vector size: {stats.get('vector_size')}")
        print(f"Status: {stats.get('status')}")
        
        if stats.get('file_type_distribution'):
            print("\nFile Type Distribution:")
            for ft, count in stats['file_type_distribution'].items():
                print(f"  - {ft}: {count}")
        
        if stats.get('source_file_distribution'):
            print("\nSource File Distribution:")
            for source, count in stats['source_file_distribution'].items():
                print(f"  - {source}: {count}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()