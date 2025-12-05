"""
Enhanced RAG Chatbot with Multi-File Support
Supports data from Excel (row-wise), PDF (paragraph-wise), and JSON (block-wise)
"""

import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Retreive.enhanced_hybrid_search import EnhancedHybridSearch

# ========== CONFIG ==========
OPENROUTER_API_KEY = "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"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== OPENROUTER CLIENT ==========
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client with caching"""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )


@st.cache_resource
def init_search_engine(collection_name: str):
    """Initialize search engine with caching"""
    try:
        return EnhancedHybridSearch(collection_name)
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        return None


def format_context_for_llm(contexts: List[Dict[str, Any]]) -> str:
    """Format search contexts for the LLM prompt"""
    if not contexts:
        return "No relevant information found in the knowledge base."
    
    context_parts = []
    
    for i, ctx in enumerate(contexts[:8], 1):
        source = ctx.get('source_file', 'Unknown')
        file_type = ctx.get('file_type', 'unknown')
        score = ctx.get('score', 0)
        content = ctx.get('content', '')[:800]
        
        # Format based on file type
        if file_type == 'excel':
            row_idx = ctx.get('row_index', 'N/A')
            header = f"Excel Row {row_idx} from {source}"
        elif file_type == 'pdf':
            para_idx = ctx.get('paragraph_index', 'N/A')
            header = f"PDF Paragraph {para_idx} from {source}"
        elif file_type == 'json':
            block_idx = ctx.get('block_index', ctx.get('primary_chunk_index', 'N/A'))
            header = f"Data Block {block_idx} from {source}"
        else:
            header = f"Content from {source}"
        
        context_parts.append(
            f"[Context {i}] ({file_type.upper()}, Relevance: {score:.2f})\n"
            f"Source: {header}\n"
            f"Content: {content}\n"
        )
    
    return "\n".join(context_parts)


def generate_answer(
    question: str, 
    contexts: List[Dict[str, Any]],
    client: OpenAI,
    model: str = "meta-llama/llama-3-70b-instruct"
) -> str:
    """Generate answer using LLM with context"""
    
    context_text = format_context_for_llm(contexts)
    context_summary = f"{len(contexts)} relevant documents found" if contexts else "No context available"
    
    prompt = f"""You are an AI assistant with access to a comprehensive knowledge base containing:
- Excel/spreadsheet data (product info, inventory, records)
- PDF documents (reports, documentation, procedures)
- JSON data (structured business data, articles)

User Question: {question}

Relevant Information from Knowledge Base:
{context_text}

Instructions:
1. Provide a clear, comprehensive answer based on the available information
2. If the information directly answers the question, synthesize it into a coherent response
3. When referencing information, mention the source type (e.g., "According to the spreadsheet data...", "The PDF documentation states...")
4. If multiple sources provide related information, combine them logically
5. If no relevant information is found, explain what types of information might be helpful
6. Keep your response informative but concise (aim for 2-4 paragraphs)

Context Summary: {context_summary}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"I encountered an error while generating the response: {e}"


def display_context_details(contexts: List[Dict[str, Any]]):
    """Display context details in an expander"""
    with st.expander(f"📋 Found {len(contexts)} relevant contexts (click to expand)"):
        for i, ctx in enumerate(contexts, 1):
            score = ctx.get('score', 0)
            source = ctx.get('source_file', 'Unknown')
            file_type = ctx.get('file_type', 'unknown')
            content = ctx.get('content', '')
            
            # Create header based on file type
            if file_type == 'excel':
                row_idx = ctx.get('row_index', ctx.get('primary_chunk_index', 'N/A'))
                st.markdown(f"**Context {i}** - 📊 Excel Row {row_idx} (Score: {score:.3f})")
            elif file_type == 'pdf':
                para_idx = ctx.get('paragraph_index', ctx.get('primary_chunk_index', 'N/A'))
                st.markdown(f"**Context {i}** - 📄 PDF Paragraph {para_idx} (Score: {score:.3f})")
            elif file_type == 'json':
                block_idx = ctx.get('block_index', ctx.get('primary_chunk_index', 'N/A'))
                st.markdown(f"**Context {i}** - 📦 JSON Block {block_idx} (Score: {score:.3f})")
            else:
                st.markdown(f"**Context {i}** (Score: {score:.3f})")
            
            st.caption(f"Source: {source}")
            
            # Show content preview
            display_content = content[:400] + ("..." if len(content) > 400 else "")
            st.text(display_content)
            
            # Show file-type specific metadata
            if file_type == 'excel' and ctx.get('row_data'):
                with st.expander("View row data"):
                    st.json(ctx['row_data'])
            
            st.divider()


# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="Multi-File RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-File RAG Chatbot")
st.caption("Ask questions across Excel spreadsheets, PDF documents, and JSON data")

# Initialize client
client = init_openai_client()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Collection selection
    collection_name = st.text_input(
        "Collection Name",
        value="test_unified_ingest",
        help="Name of the Qdrant collection to search"
    )
    
    # Initialize search engine
    search_engine = init_search_engine(collection_name)
    
    if search_engine:
        st.success(f"✅ Connected to '{collection_name}'")
        
        # Show collection stats
        with st.expander("📊 Collection Statistics"):
            stats = search_engine.get_collection_stats()
            if stats:
                st.metric("Total Vectors", f"{stats.get('total_points', 0):,}")
                st.metric("Vector Size", stats.get('vector_size', 0))
                
                if stats.get('file_type_distribution'):
                    st.subheader("By File Type")
                    for ft, count in stats['file_type_distribution'].items():
                        st.text(f"• {ft}: {count}")
        
        # Get available filters
        filters = search_engine.get_available_filters()
        
        # Filter options
        st.subheader("🔍 Search Filters")
        
        file_type_filter = st.selectbox(
            "Filter by File Type",
            options=["All"] + filters.get('file_types', []),
            index=0
        )
        
        source_filter = st.selectbox(
            "Filter by Source File",
            options=["All"] + filters.get('source_files', []),
            index=0
        )
        
        # Search settings
        st.subheader("⚡ Search Settings")
        search_limit = st.slider("Max Results", 3, 15, 8)
        score_threshold = st.slider("Score Threshold", 0.0, 0.8, 0.3, 0.1)
        expand_context = st.checkbox("Expand to Full Content", True)
        
    else:
        st.error(f"❌ Failed to connect to '{collection_name}'")
        st.info("Make sure Qdrant is running and the collection exists")
    
    # Help section
    st.subheader("💡 Try asking about:")
    st.text("• Product information")
    st.text("• Inventory details")
    st.text("• Documentation content")
    st.text("• Business data")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask me about your data...")

if user_input and search_engine:
    # Save user input
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Prepare filters
    file_type = None if file_type_filter == "All" else file_type_filter
    source_file = None if source_filter == "All" else source_filter
    
    # Search for relevant contexts
    with st.spinner("Searching knowledge base..."):
        start_time = time.time()
        
        contexts = search_engine.search(
            query=user_input,
            limit=search_limit,
            score_threshold=score_threshold,
            file_type=file_type,
            source_file=source_file,
            expand_context=expand_context
        )
        
        search_time = time.time() - start_time
        st.info(f"🔍 Found {len(contexts)} relevant contexts in {search_time:.2f}s")
    
    # Generate response
    with st.spinner("Generating response..."):
        answer = generate_answer(user_input, contexts, client)
    
    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Show context details
    if contexts:
        display_context_details(contexts)

elif user_input and not search_engine:
    st.error("Cannot process questions - search engine not connected.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Footer
if not search_engine:
    st.divider()
    st.warning("⚠️ Search engine not connected. Please check configuration.")
    st.text("Troubleshooting:")
    st.text("1. Make sure Qdrant is running (docker-compose up -d)")
    st.text("2. Verify the collection exists and has data")
    st.text("3. Run the unified ingestor to populate data")