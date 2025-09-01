"""
Main Financial RAG Chatbot Application
Integrates all components: core classes, agents, and provides user interface
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from groq import Groq

# Import our custom modules
from core import DocumentChunk, VectorStore, DocumentProcessor, WebContentProcessor
from agents import (
    TrendAnalyzerAgent, ComparativeAnalyzerAgent, StatisticalCalculatorAgent,
    DocumentSummarizerAgent, TableExtractorAgent, VisualizationAgent,
    MultilingualAgent, WebContentAnalyzer
)

# Load environment variables
load_dotenv()


class SimpleRAGFinancialBot:
    """Simplified RAG-based Financial Chatbot with Specialized Agents"""
    
    def __init__(self, groq_api_key: str = None):
        # Use provided key or get from environment
        api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or provided directly")
        
        self.client = Groq(api_key=api_key)
        self.model_name = os.getenv('MODEL_NAME', "llama3-8b-8192")
        
        # Initialize core components
        embedding_model = os.getenv('EMBEDDING_MODEL', "all-MiniLM-L6-v2")
        chunk_size = int(os.getenv('CHUNK_SIZE', 800))
        
        self.vector_store = VectorStore(embedding_model)
        self.document_processor = DocumentProcessor(chunk_size)
        self.web_processor = WebContentProcessor(chunk_size)
        self.processed_files = []
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
        print("🚀 RAG Financial ChatBot with Specialized Agents initialized")
        print(f"🤖 Available agents: {', '.join(self.agents.keys())}")
    
    def _initialize_agents(self):
        """Initialize all specialized financial agents"""
        return {
            'trend_analyzer': TrendAnalyzerAgent(self.client, self.model_name),
            'comparative_analyzer': ComparativeAnalyzerAgent(self.client, self.model_name),
            'statistical_calculator': StatisticalCalculatorAgent(self.client, self.model_name),
            'document_summarizer': DocumentSummarizerAgent(self.client, self.model_name),
            'table_extractor': TableExtractorAgent(self.client, self.model_name),
            'visualization_generator': VisualizationAgent(self.client, self.model_name),
            'multilingual_processor': MultilingualAgent(self.client, self.model_name),
            'web_content_analyzer': WebContentAnalyzer(self.client, self.model_name)
        }
    
    def add_document(self, file_path: str) -> Dict:
        """Add and process a document"""
        try:
            if not os.path.exists(file_path):
                return {'success': False, 'error': f'File not found: {file_path}'}
            
            # Process file into chunks
            chunks = self.document_processor.process_file(file_path)
            
            # Add to vector store
            self.vector_store.add_chunks(chunks)
            
            # Track processed file
            file_info = {
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'chunks_created': len(chunks),
                'processed_at': datetime.now().isoformat()
            }
            self.processed_files.append(file_info)
            
            return {
                'success': True,
                'message': f"Successfully processed {file_info['file_name']} into {len(chunks)} chunks",
                'chunks_created': len(chunks)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def add_url(self, url: str) -> Dict:
        """Add and process web content with RAG"""
        try:
            print(f"🌐 Processing URL: {url}")
            
            # Fetch web content
            content = self.web_processor.fetch_web_content(url)
            
            if not content:
                return {'success': False, 'error': 'Failed to fetch content from URL'}
            
            # Process web content into chunks  
            chunks = self.web_processor.process_web_content_into_chunks(url, content)
            
            # Add chunks to vector store
            self.vector_store.add_chunks(chunks)
            
            # Store URL info
            url_info = {
                'url_id': str(uuid.uuid4()),
                'url': url,
                'title': self.web_processor.extract_title_from_content(content),
                'content_type': 'url',
                'processed_at': datetime.now().isoformat(),
                'chunk_count': len(chunks)
            }
            
            self.processed_files.append(url_info)
            
            return {
                'success': True,
                'url_id': url_info['url_id'],
                'chunks_created': len(chunks),
                'message': f"Successfully processed web content from {url} into {len(chunks)} searchable chunks"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def add_content(self, content_source: str) -> Dict:
        """Add content from file path or URL"""
        # Determine if it's a URL or file path
        if content_source.startswith(('http://', 'https://', 'www.')):
            return self.add_url(content_source)
        else:
            return self.add_document(content_source)
    
    def query(self, user_question: str, language: str = "auto") -> str:
        """Process user query with specialized agents and RAG"""
        try:
            print(f"\n Processing query: {user_question}")
            
            # Step 1: Detect language if auto
            if language == "auto":
                detected_lang = self.agents['multilingual_processor'].detect_language(user_question)
                print(f" Detected language: {detected_lang}")
            else:
                detected_lang = language
            
            # Step 2: Analyze intent and determine required agents
            required_agents = self._analyze_intent_and_select_agents(user_question)
            print(f" Required agents: {', '.join(required_agents)}")
            
            # Step 3: Search for relevant chunks
            search_k = int(os.getenv('SEARCH_K', 7))
            relevant_chunks = self.vector_store.search(user_question, k=search_k)
            print(f" Found {len(relevant_chunks)} relevant chunks")
            
            # Step 4: Execute specialized agents
            agent_results = {}
            for agent_name in required_agents:
                if agent_name in self.agents:
                    print(f"⚡ Executing {agent_name}...")
                    agent_results[agent_name] = self.agents[agent_name].execute({
                        'query': user_question,
                        'relevant_chunks': relevant_chunks,
                        'language': detected_lang
                    })
            
            # Step 5: Generate comprehensive response
            response = self._generate_comprehensive_response(
                user_question, relevant_chunks, agent_results, detected_lang
            )
            
            return response
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def _analyze_intent_and_select_agents(self, query: str) -> List[str]:
        """Analyze intent and select appropriate agents"""
        query_lower = query.lower()
        selected_agents = []
        
        # Intent-based agent selection
        if any(word in query_lower for word in ['trend', 'growth', 'increase', 'decrease', 'over time']):
            selected_agents.append('trend_analyzer')
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'between']):
            selected_agents.append('comparative_analyzer')
        
        if any(word in query_lower for word in ['average', 'mean', 'sum', 'total', 'calculate', 'statistics']):
            selected_agents.append('statistical_calculator')
        
        if any(word in query_lower for word in ['summary', 'summarize', 'overview', 'key points']):
            selected_agents.append('document_summarizer')
        
        if any(word in query_lower for word in ['table', 'extract', 'top', 'list', 'rank']):
            selected_agents.append('table_extractor')
        
        if any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize', 'show']):
            selected_agents.append('visualization_generator')
        
        # Always include multilingual for non-English queries
        if not all(ord(char) < 128 for char in query):  # Contains non-ASCII characters
            selected_agents.append('multilingual_processor')
        
        # Default to document summarizer if no specific intent detected
        if not selected_agents:
            selected_agents.append('document_summarizer')
        
        return selected_agents
    
    def _build_context(self, relevant_chunks: List[Dict]) -> str:
        """Build context from relevant chunks"""
        if not relevant_chunks:
            return "No relevant document context found."
        
        context = "=== RELEVANT DOCUMENT INFORMATION ===\n\n"
        
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"CHUNK {i} (Relevance: {chunk['score']:.2f}):\n"
            context += f"Source: {chunk['metadata'].get('file_name', 'Unknown')}\n"
            context += f"Type: {chunk['metadata'].get('chunk_type', 'Unknown')}\n"
            context += f"Content:\n{chunk['content']}\n"
            context += "-" * 50 + "\n\n"
        
        return context
    
    def _generate_comprehensive_response(self, question: str, chunks: List[Dict], 
                                       agent_results: Dict, language: str) -> str:
        """Generate comprehensive response using agent results and RAG context"""
        
        # Build context from chunks
        chunk_context = self._build_context(chunks)
        
        # Build agent results context
        agent_context = "\n=== SPECIALIZED AGENT ANALYSIS ===\n\n"
        for agent_name, result in agent_results.items():
            agent_context += f"{agent_name.upper().replace('_', ' ')} RESULTS:\n"
            if isinstance(result, dict):
                for key, value in result.items():
                    agent_context += f"  {key}: {value}\n"
            else:
                agent_context += f"  {result}\n"
            agent_context += "\n"
        
        prompt = f"""You are an expert financial analyst AI. Provide a comprehensive response to the user's question using the specialized analysis results and document context.

DOCUMENT CONTEXT:
{chunk_context}

SPECIALIZED AGENT ANALYSIS:
{agent_context}

USER QUESTION: {question}
RESPONSE LANGUAGE: {language}

INSTRUCTIONS:
- Synthesize information from both document context and agent analysis
- Use specific numbers and data points when available
- Provide actionable financial insights
- Reference source documents when citing specific data
- Respond in {language} if not English
- Be thorough but concise

COMPREHENSIVE ANALYSIS:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a financial expert. Respond in {language}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating comprehensive response: {str(e)}"
    
    def show_processed_files(self):
        """Show what files have been processed"""
        if not self.processed_files:
            print("🔭 No files processed yet")
            return
        
        print("\n📁 PROCESSED FILES:")
        for file_info in self.processed_files:
            if 'file_name' in file_info:
                print(f"  ✅ {file_info['file_name']} ({file_info['chunks_created']} chunks)")
            elif 'url' in file_info:
                print(f"  ✅ {file_info['url']} ({file_info['chunk_count']} chunks)")
    
    def show_sample_chunks(self, max_chunks: int = 3):
        """Show sample chunks for debugging"""
        if not self.vector_store.chunks:
            print("🔭 No chunks available")
            return
        
        print(f"\n📚 SAMPLE CHUNKS (showing {min(max_chunks, len(self.vector_store.chunks))}):")
        for i, chunk in enumerate(self.vector_store.chunks[:max_chunks]):
            print(f"\nCHUNK {i+1}:")
            print(f"ID: {chunk.chunk_id}")
            print(f"Type: {chunk.metadata.get('chunk_type', 'unknown')}")
            print(f"Source: {chunk.metadata.get('file_name', chunk.metadata.get('url', 'unknown'))}")
            print(f"Content Preview: {chunk.content[:200]}...")
            print("-" * 40)


# ===== TESTING FUNCTIONS =====

def test_with_content_and_queries():
    """Test with files and URLs"""
    
    # ===== CONFIGURATION =====
    API_KEY = os.getenv('GROQ_API_KEY')
    
    # ===== CONTENT TO PROCESS =====
    test_file = os.getenv('TEST_FILE_PATH', r"C:\path\to\your\financial_data.csv")
    test_urls = os.getenv('TEST_URLS', "https://finance.yahoo.com/news/,https://www.sec.gov/investor").split(',')
    
    CONTENT_SOURCES = [
        # Files (update paths as needed)
        test_file,
        # URLs (these work as examples)
        *test_urls
    ]
    
    # ===== TEST QUERIES =====
    TEST_QUERIES = [
        "What are the main financial metrics across all my sources?",
        "Compare insights from my documents with current web trends", 
        "What are the latest market developments?",
        "Summarize key financial insights from all sources",
        "How do my document findings relate to current market news?",
        "Extract the most important financial data points"
    ]
    
    # ===== INITIALIZE AND TEST =====
    print("="*70)
    print("🚀 FINANCIAL RAG CHATBOT - FILES + URLs")
    print("="*70)
    
    bot = SimpleRAGFinancialBot(API_KEY)
    
    # ===== STEP 1: PROCESS ALL CONTENT =====
    print(f"\n📤 STEP 1: Processing content sources...")
    
    processed_count = 0
    for content_source in CONTENT_SOURCES:
        print(f"\n🔥 Processing: {content_source}")
        
        result = bot.add_content(content_source)
        
        if result['success']:
            print(f"✅ {result['message']}")
            processed_count += 1
        else:
            print(f"❌ Failed: {result['error']}")
    
    if processed_count == 0:
        print("\n❌ No content was successfully processed!")
        print("\n💡 TO TEST THIS SYSTEM:")
        print("1. Update file paths in .env file")
        print("2. Or test with just URLs (they should work as-is)")
        print("3. Make sure you have internet connection for URL processing")
        return
    
    # Show what was processed
    bot.show_processed_files()
    bot.show_sample_chunks(2)
    
    # ===== STEP 2: TEST QUERIES =====
    print("\n" + "="*70)
    print("🤖 STEP 2: TESTING QUERIES WITH MULTI-SOURCE DATA")
    print("="*70)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n" + "📝" + "="*50)
        print(f"QUERY {i}: {query}")
        print("="*50)
        
        # Process query
        response = bot.query(query)
        
        print(f"\n🤖 COMPREHENSIVE RESPONSE:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Pause between queries
        if i < len(TEST_QUERIES):
            input(f"\n⏸️  Press Enter to continue to query {i+1}...")
    
    print(f"\n🎉 ALL QUERIES COMPLETED!")
    print(f"📊 Total sources processed: {processed_count}")
    print(f"🧩 Total chunks created: {len(bot.vector_store.chunks)}")
    print(f"🤖 Available agents: {len(bot.agents)}")


def quick_test_with_url():
    """Quick test specifically for URL processing"""
    
    API_KEY = os.getenv('GROQ_API_KEY')
    
    print("🌐 Quick URL Test")
    print("="*40)
    
    # Get URL from user
    url = input("🔗 Enter financial website URL: ").strip()
    
    if not url:
        print("❌ No URL provided")
        return
    
    # Initialize bot
    bot = SimpleRAGFinancialBot(API_KEY)
    
    # Process URL
    print(f"\n📤 Processing URL: {url}")
    result = bot.add_url(url)
    
    if not result['success']:
        print(f"❌ Failed: {result['error']}")
        return
    
    print(f"✅ {result['message']}")
    
    # Test query
    query = input("\n❓ Ask about the web content: ").strip()
    
    if query:
        response = bot.query(query)
        print(f"\n🤖 RESPONSE:")
        print("="*50)
        print(response)


def interactive_test_enhanced():
    """Enhanced interactive testing with file and URL support"""
    
    API_KEY = os.getenv('GROQ_API_KEY')
    
    print("💰 Enhanced Interactive Financial RAG Chatbot")
    print("="*60)
    print("📋 You can add:")
    print("  📄 Files: CSV, Excel, PDF, Word documents")
    print("  🌐 URLs: Any financial website or report")
    print("="*60)
    
    bot = SimpleRAGFinancialBot(API_KEY)
    
    # Content addition loop
    while True:
        print(f"\n🔥 ADD CONTENT:")
        content_input = input("Enter file path or URL (or 'done' to finish): ").strip().strip('"\'')
        
        if content_input.lower() in ['done', 'finish', 'stop']:
            break
        
        if content_input:
            result = bot.add_content(content_input)
            
            if result['success']:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ {result['error']}")
    
    # Check if any content was processed
    if not bot.processed_files:
        print("❌ No content was processed. Exiting...")
        return
    
    bot.show_processed_files()
    
    # Query loop
    print(f"\n💬 ASK QUESTIONS:")
    print("Examples:")
    print("  • 'What are the key financial trends?'")
    print("  • 'Compare my document data with web insights'")
    print("  • 'Summarize all financial information'")
    print("  • Type 'quit' to exit")
    
    while True:
        query = input(f"\n Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(" Goodbye!")
            break
        
        if query:
            print(f"\n Processing...")
            response = bot.query(query)
            
            print(f"\n Response:")
            print("="*50)
            print(response)
            print("="*50)


def test_with_file_and_queries():
    """Simple testing with direct file path and queries"""
    
    # ===== CONFIGURATION =====
    API_KEY = os.getenv('GROQ_API_KEY')
    FILE_PATH = os.getenv('TEST_FILE_PATH', r"C:\path\to\your\financial_data.csv")
    
    # ===== TEST QUERIES =====
    TEST_QUERIES = [
        "What are the main financial metrics in this document?",
        "Calculate the average revenue and profit margins", 
        "Show me the revenue trends over time",
        "Compare Q1 vs Q2 performance",
        "Extract the top 5 products by sales",
        "Summarize the key financial insights from this data",
        "What statistical patterns can you identify?",
        "¿Cuáles son las tendencias de ingresos?"  # Spanish test
    ]
    
    # ===== INITIALIZE AND TEST =====
    print("="*70)
    print(" FINANCIAL RAG CHATBOT WITH SPECIALIZED AGENTS")
    print("="*70)
    
    # Initialize bot
    bot = SimpleRAGFinancialBot(API_KEY)
    
    # Check file exists
    if not os.path.exists(FILE_PATH):
        print(f" File not found: {FILE_PATH}")
        print("\n TO TEST THIS SYSTEM:")
        print("1. Update TEST_FILE_PATH in .env file with your actual file path")
        print("2. Make sure the file exists and is readable")  
        print("3. Run the script again")
        print("\nExample paths:")
        print('  Windows: C:\\Users\\YourName\\Documents\\financial_data.csv')
        print('  Mac/Linux: /home/username/documents/financial_data.csv')
        return
    
    # ===== STEP 1: PROCESS FILE =====
    print(f"\n STEP 1: Processing file...")
    print(f"File: {FILE_PATH}")
    
    result = bot.add_document(FILE_PATH)
    
    if not result['success']:
        print(f"❌ Failed to process file: {result['error']}")
        return
    
    print(f" {result['message']}")
    
    # Show what was processed
    print(f"\nProcessed files:")
    bot.show_processed_files()
    
    # Show sample chunks
    print(f"\n Sample chunks created:")
    bot.show_sample_chunks(2)
    
    # ===== STEP 2: TEST QUERIES =====
    print("\n" + "="*70)
    print(" STEP 2: TESTING QUERIES WITH SPECIALIZED AGENTS")
    print("="*70)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n" + "📝" + "="*50)
        print(f"QUERY {i}: {query}")
        print("="*50)
        
        # Process query
        response = bot.query(query)
        
        print(f"\n COMPREHENSIVE RESPONSE:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Pause between queries for readability
        if i < len(TEST_QUERIES):
            input(f"\n⏸️  Press Enter to continue to query {i+1}...")
    
    print(f"\n ALL QUERIES COMPLETED!")
    print(f" Processed {len(bot.processed_files)} file(s)")
    print(f" Created {len(bot.vector_store.chunks)} searchable chunks")
    print(f" Available agents: {len(bot.agents)}")

def quick_test(file_path: str, query: str):
    """Quick single query test"""
    
    API_KEY = os.getenv('GROQ_API_KEY')
    
    print(f" Quick Test Mode")
    print(f" File: {file_path}")
    print(f" Query: {query}")
    print("-" * 50)
    
    # Initialize and process
    bot = SimpleRAGFinancialBot(API_KEY)
    
    # Add document
    result = bot.add_document(file_path)
    if not result['success']:
        print(f" Error: {result['error']}")
        return
    
    print(f" File processed: {result['chunks_created']} chunks created")
    
    # Process query
    response = bot.query(query)
    
    print(f"\n RESPONSE:")
    print("="*50)
    print(response)


def interactive_test():
    """Interactive testing mode"""
    
    API_KEY = os.getenv('GROQ_API_KEY')
    
    print(" Interactive Financial RAG Chatbot")
    print("="*50)
    
    # Get file path
    file_path = input(" Enter your financial file path: ").strip().strip('"\'')
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    # Initialize and process
    bot = SimpleRAGFinancialBot(API_KEY)
    
    print(f"\n Processing {file_path}...")
    result = bot.add_document(file_path)
    
    if not result['success']:
        print(f" Error: {result['error']}")
        return
    
    print(f"{result['message']}")
    bot.show_processed_files()
    
    # Interactive querying
    print(f"\n Ask questions about your financial data:")
    print("Examples:")
    print("  • 'What are the key metrics?'")
    print("  • 'Calculate average revenue'")
    print("  • 'Show revenue trends'")
    print("  • 'Compare Q1 vs Q2'")
    print("  • Type 'quit' to exit")
    
    while True:
        query = input(f"\n💤 Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(" Goodbye!")
            break
        
        if query:
            print(f"\n Processing...")
            response = bot.query(query)
            
            print(f"\n Response:")
            print("="*50)
            print(response)
            print("="*50)


def quick_test_legacy():
    """Legacy file-only test"""
    API_KEY = os.getenv('GROQ_API_KEY')
    
    file_path = input(" Enter file path: ").strip().strip('"\'')
    query = input("❓ Enter query: ").strip()
    
    if file_path and query:
        bot = SimpleRAGFinancialBot(API_KEY)
        result = bot.add_document(file_path)
        
        if result['success']:
            print(f" {result['message']}")
            response = bot.query(query)
            print(f"\n Response:\n{response}")
        else:
            print(f" {result['error']}")
    else:
        print(" Please provide both file path and query")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print(" Financial RAG Chatbot with Files + URLs")
    print("Choose testing mode:")
    print("1. Automatic test with files + URLs")
    print("2. Quick URL-only test")
    print("3. Enhanced interactive test")
    print("4. File-only test (original)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_with_content_and_queries()
    
    elif choice == "2":
        quick_test_with_url()
    
    elif choice == "3":
        interactive_test_enhanced()
    
    elif choice == "4":
        # Original file-only test
        quick_test_legacy()
    
    else:
        print(" Invalid choice. Please run again and choose 1-4.")
    
    print(f"\n Testing completed!")