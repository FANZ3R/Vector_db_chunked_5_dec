"""
Enhanced Hybrid Search for Multi-File RAG
Supports searching across Excel rows, PDF paragraphs, and JSON blocks
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import logging
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Config.local_embeddings import get_local_embedding_model
from Config.qdrant_cfg import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedHybridSearch:
    """
    Enhanced hybrid search engine for multi-file type RAG
    Supports:
    - Semantic vector search
    - File type filtering (excel, pdf, json)
    - Source file filtering
    - Context expansion (combining sub-chunks)
    """
    
    def __init__(self, collection_name: str = "business_dataset"):
        """
        Initialize the hybrid search engine
        
        Args:
            collection_name: Name of the Qdrant collection to search
        """
        self.collection_name = collection_name
        self.embedding_model = get_local_embedding_model()
        self.qdrant_client = get_qdrant_client()
        
        # Verify collection exists
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                raise ValueError(f"Collection '{collection_name}' not found. Available: {collection_names}")
            
            collection_info = self.qdrant_client.get_collection(collection_name)
            self.total_points = collection_info.points_count
            
            logger.info(f"Connected to collection '{collection_name}' with {self.total_points:,} vectors")
            
        except Exception as e:
            logger.error(f"Error connecting to collection: {e}")
            raise
    
    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.3,
        file_type: Optional[str] = None,
        source_file: Optional[str] = None,
        expand_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Main search method with multiple strategies
        
        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            file_type: Filter by file type ('excel', 'pdf', 'json')
            source_file: Filter by source file name
            expand_context: Whether to expand sub-chunks to full context
            
        Returns:
            List of search results with metadata
        """
        all_contexts = []
        
        try:
            # Strategy 1: Direct semantic search
            contexts = self._semantic_search(
                query, limit * 2, score_threshold, 
                file_type, source_file, expand_context
            )
            all_contexts.extend(contexts)
            
            # Strategy 2: Multi-query approach for complex questions
            if len(all_contexts) < limit // 2:
                sub_queries = self._generate_sub_queries(query)
                for sub_query in sub_queries[:2]:
                    sub_contexts = self._semantic_search(
                        sub_query, limit, score_threshold,
                        file_type, source_file, expand_context
                    )
                    all_contexts.extend(sub_contexts)
            
            # Deduplicate and sort
            unique_contexts = self._deduplicate_contexts(all_contexts)
            
            return unique_contexts[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _semantic_search(
        self,
        query: str,
        limit: int,
        score_threshold: float,
        file_type: Optional[str],
        source_file: Optional[str],
        expand_context: bool
    ) -> List[Dict[str, Any]]:
        """Perform semantic vector search with filters"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.embed_text(query)
            
            # Build filter conditions
            filter_conditions = []
            
            if file_type:
                filter_conditions.append(
                    FieldCondition(key="file_type", match=MatchValue(value=file_type))
                )
            
            if source_file:
                filter_conditions.append(
                    FieldCondition(key="source_file", match=MatchValue(value=source_file))
                )
            
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 2 if expand_context else limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )
            
            if not expand_context:
                return [self._format_result(result) for result in search_results[:limit]]
            
            # Group and expand results
            grouped = self._group_by_primary_chunk(search_results)
            expanded = self._expand_contexts(grouped, limit)
            
            return expanded
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _group_by_primary_chunk(self, search_results) -> Dict[str, List]:
        """Group results by source file and primary chunk index"""
        grouped = defaultdict(list)
        
        for result in search_results:
            source = result.payload.get('source_file', 'unknown')
            primary_idx = result.payload.get('primary_chunk_index', 0)
            key = f"{source}_{primary_idx}"
            
            grouped[key].append({
                'id': result.id,
                'score': float(result.score),
                'payload': result.payload
            })
        
        # Sort sub-chunks within each group
        for key in grouped:
            grouped[key].sort(key=lambda x: x['payload'].get('sub_chunk_index', 0))
        
        return grouped
    
    def _expand_contexts(self, grouped: Dict, limit: int) -> List[Dict[str, Any]]:
        """Expand contexts by combining sub-chunks"""
        expanded_contexts = []
        processed = set()
        
        # Sort groups by best score
        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: max(chunk['score'] for chunk in x[1]),
            reverse=True
        )
        
        for group_key, chunks in sorted_groups:
            if len(expanded_contexts) >= limit:
                break
            
            if group_key in processed:
                continue
            
            # Get the best chunk's info
            best_chunk = max(chunks, key=lambda x: x['score'])
            source_file = best_chunk['payload'].get('source_file')
            primary_idx = best_chunk['payload'].get('primary_chunk_index')
            
            # Get full content from all sub-chunks
            full_content = self._get_full_primary_content(source_file, primary_idx)
            
            if full_content:
                context = {
                    'content': full_content['text'],
                    'score': best_chunk['score'],
                    'source_file': source_file,
                    'file_type': best_chunk['payload'].get('file_type', 'unknown'),
                    'primary_chunk_index': primary_idx,
                    'matched_sub_chunks': len(chunks),
                    'total_sub_chunks': full_content['total_sub_chunks'],
                    'metadata': best_chunk['payload'].get('metadata', {}),
                    'header': best_chunk['payload'].get('header', ''),
                    'context_type': 'expanded'
                }
                
                # Add file-type specific metadata
                self._add_file_type_metadata(context, best_chunk['payload'])
                
                expanded_contexts.append(context)
                processed.add(group_key)
        
        return expanded_contexts
    
    def _get_full_primary_content(self, source_file: str, primary_idx: int) -> Optional[Dict]:
        """Get all sub-chunks for a primary chunk and combine them"""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                    FieldCondition(key="primary_chunk_index", match=MatchValue(value=primary_idx))
                ]
            )
            
            all_sub_chunks = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=100,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not all_sub_chunks:
                return None
            
            # Sort by sub_chunk_index
            sorted_chunks = sorted(
                all_sub_chunks,
                key=lambda x: x.payload.get('sub_chunk_index', 0)
            )
            
            # Combine text
            full_text = " ".join(
                chunk.payload.get('text', '') for chunk in sorted_chunks
            ).strip()
            
            # Remove duplicate sentences from overlapping chunks
            full_text = self._remove_duplicates(full_text)
            
            return {
                'text': full_text,
                'total_sub_chunks': len(sorted_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error getting full content: {e}")
            return None
    
    def _add_file_type_metadata(self, context: Dict, payload: Dict):
        """Add file-type specific metadata to context"""
        file_type = payload.get('file_type', 'unknown')
        
        if file_type == 'excel':
            context['row_index'] = payload.get('row_index')
            context['columns'] = payload.get('columns', [])
            context['row_data'] = payload.get('row_data', {})
            
        elif file_type == 'pdf':
            context['paragraph_index'] = payload.get('paragraph_index')
            context['total_paragraphs'] = payload.get('total_paragraphs')
            
        elif file_type == 'json':
            context['block_index'] = payload.get('block_index')
            context['total_blocks'] = payload.get('total_blocks')
            context['fields_with_content'] = payload.get('fields_with_content', [])
    
    def _format_result(self, result) -> Dict[str, Any]:
        """Format a single search result"""
        return {
            'id': result.id,
            'content': result.payload.get('text', ''),
            'score': float(result.score),
            'source_file': result.payload.get('source_file', ''),
            'file_type': result.payload.get('file_type', 'unknown'),
            'primary_chunk_index': result.payload.get('primary_chunk_index', 0),
            'sub_chunk_index': result.payload.get('sub_chunk_index', 0),
            'total_sub_chunks': result.payload.get('total_sub_chunks', 1),
            'header': result.payload.get('header', ''),
            'context_type': 'chunk'
        }
    
    def _remove_duplicates(self, text: str) -> str:
        """Remove duplicate sentences from text"""
        sentences = text.split('. ')
        seen = set()
        unique = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and sentence not in seen:
                seen.add(sentence)
                unique.append(sentence)
        
        return '. '.join(unique)
    
    def _generate_sub_queries(self, query: str) -> List[str]:
        """Generate related sub-queries for better coverage"""
        sub_queries = []
        
        # Extract key terms
        stopwords = {
            'what', 'is', 'are', 'how', 'why', 'when', 'where', 'who',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'with', 'by', 'from', 'about', 'into'
        }
        
        words = re.findall(r'\w+', query.lower())
        key_terms = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Create sub-queries from individual terms
        for term in key_terms[:3]:
            sub_queries.append(term)
        
        return sub_queries
    
    def _deduplicate_contexts(self, contexts: List[Dict]) -> List[Dict]:
        """Remove duplicate contexts and sort by score"""
        seen = set()
        unique = []
        
        # Sort by score first
        contexts.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        for ctx in contexts:
            # Create hash from content prefix
            content_hash = hash(ctx.get('content', '')[:200])
            
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(ctx)
        
        return unique
    
    def search_by_file_type(
        self,
        query: str,
        file_type: str,
        limit: int = 10,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search within a specific file type"""
        return self.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            file_type=file_type,
            expand_context=True
        )
    
    def search_by_source(
        self,
        query: str,
        source_file: str,
        limit: int = 10,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search within a specific source file"""
        return self.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            source_file=source_file,
            expand_context=True
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Sample points for analysis
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=200,
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Analyze distributions
            file_type_counts = defaultdict(int)
            source_file_counts = defaultdict(int)
            primary_chunks_per_source = defaultdict(set)
            
            for point in sample_points:
                file_type = point.payload.get('file_type', 'unknown')
                source = point.payload.get('source_file', 'unknown')
                primary_idx = point.payload.get('primary_chunk_index', 0)
                
                file_type_counts[file_type] += 1
                source_file_counts[source] += 1
                primary_chunks_per_source[source].add(primary_idx)
            
            return {
                'collection_name': self.collection_name,
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'status': collection_info.status.value,
                'file_type_distribution': dict(file_type_counts),
                'source_file_distribution': dict(source_file_counts),
                'unique_primary_chunks_per_source': {
                    k: len(v) for k, v in primary_chunks_per_source.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter values"""
        try:
            sample_points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=500,
                with_payload=True,
                with_vectors=False
            )[0]
            
            file_types = set()
            source_files = set()
            
            for point in sample_points:
                file_types.add(point.payload.get('file_type', 'unknown'))
                source_files.add(point.payload.get('source_file', 'unknown'))
            
            return {
                'file_types': sorted(list(file_types)),
                'source_files': sorted(list(source_files))
            }
            
        except Exception as e:
            logger.error(f"Failed to get filters: {e}")
            return {'file_types': [], 'source_files': []}


def test_enhanced_search(collection_name: str = "test_unified_ingest"):
    """Test the enhanced hybrid search"""
    print("🚀 Testing Enhanced Hybrid Search")
    print("=" * 80)
    
    try:
        # Initialize search engine
        search_engine = EnhancedHybridSearch(collection_name)
        
        # Test 1: Basic search
        print("\n📋 Test 1: Basic Semantic Search")
        print("-" * 40)
        
        test_queries = [
            "product features",
            "enterprise solutions",
            "maintenance service"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = search_engine.search(query, limit=3)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.4f}")
                print(f"     Source: {result['source_file']} ({result['file_type']})")
                print(f"     Content: {result['content'][:100]}...")
        
        # Test 2: Collection statistics
        print(f"\n📋 Test 2: Collection Statistics")
        print("-" * 40)
        
        stats = search_engine.get_collection_stats()
        print(f"Total Points: {stats.get('total_points', 'N/A')}")
        print(f"Vector Size: {stats.get('vector_size', 'N/A')}")
        
        if stats.get('file_type_distribution'):
            print("\nFile Type Distribution:")
            for ft, count in stats['file_type_distribution'].items():
                print(f"  - {ft}: {count}")
        
        # Test 3: Available filters
        print(f"\n📋 Test 3: Available Filters")
        print("-" * 40)
        
        filters = search_engine.get_available_filters()
        print(f"File Types: {filters.get('file_types', [])}")
        print(f"Source Files: {filters.get('source_files', [])}")
        
        print("\n✅ Enhanced search testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_search()