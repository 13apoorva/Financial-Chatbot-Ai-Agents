import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from main import SimpleRAGFinancialBot

load_dotenv()


API_KEY = os.getenv('GROQ_API_KEY') 

# Set page config
st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'bot_initialized' not in st.session_state:
        st.session_state.bot_initialized = False


def initialize_bot():
    """Initialize the bot"""
    if not st.session_state.bot_initialized:
        try:
            st.session_state.bot = SimpleRAGFinancialBot(API_KEY)
            st.session_state.bot_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize bot: {str(e)}")
            return False
    return True

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #1f4e79, #2e7db8);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .success-box {
            padding: 1rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            color: #155724;
        }
        .error-box {
            padding: 1rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            color: #721c24;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>💰 Financial RAG Chatbot</h1>
        <p>Analyze financial documents and web content with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_metrics():
    """Render status metrics at the top"""
    if st.session_state.processed_files:
        files_count = sum(1 for item in st.session_state.processed_files if item['type'] == 'file')
        urls_count = sum(1 for item in st.session_state.processed_files if item['type'] == 'url')
        total_chunks = sum(item['chunks'] for item in st.session_state.processed_files)
        chat_count = len(st.session_state.chat_history) // 2
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Files", files_count)
        col2.metric("🌐 URLs", urls_count)
        col3.metric("🧩 Chunks", total_chunks)
        col4.metric("💬 Conversations", chat_count)

def render_quick_start():
    """Render quick start guide for new users"""
    if not st.session_state.processed_files and not st.session_state.chat_history:
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            **Get started in 3 easy steps:**
            
            1. **📁 Add Content:**
               - Upload financial documents (CSV, Excel, PDF, Word)
               - Or add URLs from financial websites
            
            2. **💬 Ask Questions:**
               - Use example queries or write your own
               - Try different analysis types (trends, comparisons, summaries)
            
            3. **🎯 Get AI-Powered Insights:**
               - Specialized agents analyze your data
               - Get comprehensive responses with citations
            
            **Example queries to try:**
            - "What are the main financial metrics in my data?"
            - "Compare Q1 vs Q2 performance"
            - "Summarize key financial insights from all sources"
            """)

def process_uploaded_file(uploaded_file):
    """Process a single uploaded file"""
    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Process file using the bot
            result = st.session_state.bot.add_document(temp_path)
            
            if result['success']:
                st.success(f"✅ {result['message']}")
                
                # Add to processed files list
                file_info = {
                    'name': uploaded_file.name,
                    'type': 'file',
                    'size': uploaded_file.size,
                    'chunks': result['chunks_created'],
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.processed_files.append(file_info)
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass

def process_multiple_files(uploaded_files):
    """Process multiple uploaded files with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        temp_path = None
        try:
            # Save and process file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            result = st.session_state.bot.add_document(temp_path)
            
            if result['success']:
                file_info = {
                    'name': uploaded_file.name,
                    'type': 'file',
                    'size': uploaded_file.size,
                    'chunks': result['chunks_created'],
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.processed_files.append(file_info)
                st.success(f"✅ {uploaded_file.name}: {result['message']}")
            else:
                st.error(f"❌ {uploaded_file.name}: {result['error']}")
        
        except Exception as e:
            st.error(f"❌ {uploaded_file.name}: {str(e)}")
        
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    status_text.text("All files processed!")
    st.rerun()

def process_url(url):
    """Process URL content"""
    try:
        with st.spinner(f"Processing URL: {url}..."):
            result = st.session_state.bot.add_url(url)
            
            if result['success']:
                st.success(f"✅ {result['message']}")
                
                # Add to processed files list
                url_info = {
                    'name': url,
                    'type': 'url',
                    'chunks': result['chunks_created'],
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.processed_files.append(url_info)
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
                
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")

def process_batch_urls(urls):
    """Process multiple URLs with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(urls):
        status_text.text(f"Processing URL {i+1}/{len(urls)}: {url[:50]}...")
        progress_bar.progress((i + 1) / len(urls))
        
        try:
            result = st.session_state.bot.add_url(url)
            
            if result['success']:
                url_info = {
                    'name': url,
                    'type': 'url',
                    'chunks': result['chunks_created'],
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.processed_files.append(url_info)
                st.success(f"✅ Processed: {url}")
            else:
                st.error(f"❌ Failed: {url} - {result['error']}")
        
        except Exception as e:
            st.error(f"❌ Error with {url}: {str(e)}")
    
    status_text.text("All URLs processed!")
    st.rerun()

def file_upload_section():
    """Render file upload section"""
    st.write("**Drag and drop or browse for files**")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls', 'pdf', 'docx'],
        help="Supported: CSV, Excel (.xlsx, .xls), PDF, Word (.docx)",
        key="main_file_uploader"
    )
    
    if uploaded_files:
        # Show file details
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        
        file_details = []
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            file_details.append({
                'Name': file.name,
                'Size (MB)': f"{file_size_mb:.2f}",
                'Type': file.type
            })
        
        st.dataframe(pd.DataFrame(file_details), use_container_width=True)
        
        # Process files buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Process All Files", type="primary"):
                process_multiple_files(uploaded_files)
        
        with col2:
            selected_file = st.selectbox("Or select individual file:", 
                                       [f.name for f in uploaded_files])
            if st.button(f"Process Selected") and selected_file:
                selected_upload = next(f for f in uploaded_files if f.name == selected_file)
                process_uploaded_file(selected_upload)

def url_addition_section():
    """Render URL addition section"""
    st.write("**Add financial websites and reports**")
    
    # Single URL input
    url_input = st.text_input(
        "Enter URL:",
        placeholder="https://finance.yahoo.com/news/...",
        key="main_url_input"
    )
    
    if url_input:
        if st.button("🌐 Process URL", type="primary"):
            process_url(url_input)
    
    # Batch URL processing
    with st.expander("🌐 Add Multiple URLs"):
        urls_text = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com",
            height=100,
            key="batch_urls_textarea"
        )
        
        if urls_text and st.button("Process All URLs", key="batch_process_urls"):
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            if urls:
                process_batch_urls(urls)
    
    # Popular financial websites
    st.write("**Popular Financial Sources:**")
    popular_urls = {
        "Yahoo Finance News": "https://finance.yahoo.com/news/",
        "SEC Investor Info": "https://www.sec.gov/investor",
        "Investopedia": "https://www.investopedia.com/financial-analysis-4689808",
        "MarketWatch": "https://www.marketwatch.com/",
        "Bloomberg Markets": "https://www.bloomberg.com/markets"
    }
    
    cols = st.columns(2)
    for i, (name, url) in enumerate(popular_urls.items()):
        with cols[i % 2]:
            if st.button(f"📰 {name}", key=f"popular_{i}"):
                process_url(url)

def content_viewer_section():
    """Render content viewer section"""
    if not st.session_state.processed_files:
        st.info("No content processed yet. Add files or URLs to get started.")
        return
    
    st.write(f"**{len(st.session_state.processed_files)} items processed**")
    
    # Content overview metrics
    total_chunks = sum(item['chunks'] for item in st.session_state.processed_files)
    files_count = sum(1 for item in st.session_state.processed_files if item['type'] == 'file')
    urls_count = sum(1 for item in st.session_state.processed_files if item['type'] == 'url')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", total_chunks)
    col2.metric("Files", files_count)
    col3.metric("URLs", urls_count)
    
    # Detailed content list
    st.write("**Processed Content:**")
    
    for i, item in enumerate(st.session_state.processed_files):
        with st.expander(f"{'📄' if item['type'] == 'file' else '🌐'} {item['name'][:50]}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {item['type'].upper()}")
                st.write(f"**Chunks:** {item['chunks']}")
                st.write(f"**Processed:** {item['processed_at']}")
            
            with col2:
                if item['type'] == 'file' and 'size' in item:
                    st.write(f"**Size:** {item['size'] / 1024:.1f} KB")
                
                # Remove individual item
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.processed_files.pop(i)
                    st.rerun()
    
    # Clear all content
    if st.button("🗑️ Clear All Content", type="secondary"):
        # Reinitialize bot to clear vector store
        st.session_state.bot = SimpleRAGFinancialBot(API_KEY)
        st.session_state.processed_files = []
        st.success("All content cleared!")
        st.rerun()

def render_query_categories():
    """Render query categories with examples"""
    query_categories = {
        "📈 Trend Analysis": [
            "What are the revenue trends over time?",
            "Show me growth patterns in the data",
            "Identify increasing or decreasing metrics"
        ],
        "🔍 Comparative Analysis": [
            "Compare Q1 vs Q2 performance",
            "What are the differences between datasets?",
            "Compare this year vs last year"
        ],
        "📊 Statistical Analysis": [
            "Calculate average revenue and profit margins",
            "What are the key statistical insights?",
            "Show me the distribution of financial metrics"
        ],
        "📋 Document Summary": [
            "Summarize all financial documents",
            "What are the key insights from my data?",
            "Provide an overview of all processed content"
        ],
        "🌐 Web Analysis": [
            "What are the latest market developments?",
            "How do my documents relate to current market news?",
            "Compare my data with current web trends"
        ]
    }
    
    selected_category = st.selectbox(
        "Choose analysis type:",
        list(query_categories.keys()),
        help="Select the type of analysis you want to perform"
    )
    
    if selected_category:
        st.write("**Example queries:**")
        
        for i, example in enumerate(query_categories[selected_category]):
            if st.button(f"📝 {example}", key=f"cat_example_{selected_category}_{i}"):
                process_query(example)

def display_chat_history():
    """Display chat history"""
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        
        
        for message in reversed(st.session_state.chat_history[-10:]):  
            if message['type'] == 'user':
                with st.chat_message("user"):
                    st.write(f"**[{message['timestamp']}]** {message['content']}")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**[{message['timestamp']}]**")
                    st.markdown(message['content'])
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No chat history yet. Ask a question to get started!")

def process_query(query, language="auto"):
    """Process user query and display results"""
    if not st.session_state.bot:
        st.error("Bot not initialized.")
        return
    
    if not st.session_state.processed_files:
        st.warning("No content has been processed yet. Please add files or URLs first.")
        return
    
    # Add to chat history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': query,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'language': language
    })
    
    try:
        with st.spinner("🤔 Analyzing your query with specialized agents..."):
           
            response = st.session_state.bot.query(query, language)
            
            # Add response to chat history
            st.session_state.chat_history.append({
                'type': 'assistant',
                'content': response,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def chat_interface():
    """Main chat interface"""
    st.header("💬 Chat Interface")
    
    if not st.session_state.bot_initialized:
        st.info("Initializing bot... Please wait.")
        return
    
    
    display_chat_history()
    
   
    st.subheader("Ask Questions")
    
    
    render_query_categories()
    
   
    st.write("**Custom Query:**")
    
    col_query, col_lang = st.columns([3, 1])
    
    with col_lang:
        language = st.selectbox(
            "Language",
            ["auto", "English", "Spanish", "French", "German"],
            help="Select response language (auto-detect if 'auto')"
        )
    
    with col_query:
        user_query = st.text_input(
            "Enter your question:",
            placeholder="What financial insights can you provide?",
            key="user_query_input"
        )
    
    # Submit button
    if st.button("🚀 Submit Query", type="primary", disabled=not user_query):
        if user_query:
            process_query(user_query, language)

def content_management_section():
    """Content management section for files and URLs"""
    st.header("📁 Content Management")
    
    if not st.session_state.bot_initialized:
        st.info("Initializing bot...")
        return
    
    
    tab1, tab2, tab3 = st.tabs(["📄 Upload Files", "🌐 Add URLs", "📊 View Content"])
    
    with tab1:
        file_upload_section()
    
    with tab2:
        url_addition_section()
    
    with tab3:
        content_viewer_section()

def render_sidebar():
    """Render sidebar information"""
    with st.sidebar:
        st.header("ℹ System Status")
        
        # Bot status
        if st.session_state.bot:
            st.success("Bot is ready!")
            
            # Show bot statistics
            total_chunks = len(st.session_state.bot.vector_store.chunks)
            total_agents = len(st.session_state.bot.agents)
            
            col1, col2 = st.columns(2)
            col1.metric("Chunks", total_chunks)
            col2.metric("Agents", total_agents)
        
        st.header("🎯 Specialized Agents")
        
        agents_info = {
            "Trend Analyzer": "Identifies financial trends and patterns over time",
            "Comparative Analyzer": "Compares different datasets or time periods",
            "Statistical Calculator": "Performs calculations and statistical analysis",
            "Document Summarizer": "Creates comprehensive summaries",
            "Table Extractor": "Extracts and analyzes tabular data",
            "Visualization Generator": "Suggests appropriate charts and graphs",
            "Multilingual Processor": "Handles queries in multiple languages",
            "Web Content Analyzer": "Analyzes web-based financial content"
        }
        
        for agent, description in agents_info.items():
            with st.expander(f"🤖 {agent}"):
                st.write(description)
        
        st.header("📋 Supported Formats")
        st.markdown("""
        **Documents:**
        - CSV files
        - Excel files (.xlsx, .xls)
        - PDF documents
        - Word documents (.docx)
        
        **Web Content:**
        - Financial news websites
        - Company reports
        - Market analysis pages
        - Any financial web content
        """)
        
        
        st.header("📤 Export")
        if st.session_state.chat_history:
            if st.button("Export Chat History"):
                export_data = {
                    'chat_history': st.session_state.chat_history,
                    'processed_files': st.session_state.processed_files,
                    'export_timestamp': datetime.now().isoformat()
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download Chat History (JSON)",
                    data=json_str,
                    file_name=f"financial_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def main():
    """Main application function"""
    
    init_session_state()
    
    if not initialize_bot():
        st.stop()
    
    
    apply_custom_css()
    
    
    render_header()
    
    
    render_status_metrics()
    
    
    render_quick_start()
    
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        content_management_section()
        
    with col2:
        chat_interface()
    
    
    render_sidebar()


if __name__ == "__main__":
    main()