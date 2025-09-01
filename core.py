"""
Core classes for the Financial RAG Chatbot
Contains: DocumentChunk, VectorStore, DocumentProcessor
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from bs4 import BeautifulSoup
import re
import requests


class DocumentChunk:
    """Represents a chunk of document content"""
    def __init__(self, content: str, metadata: Dict, chunk_id: str = None):
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata
        self.embedding = None


class VectorStore:
    """FAISS-based vector store for document chunks"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
            
        contents = [chunk.content for chunk in chunks]
        embeddings = self.encoder.encode(contents, normalize_embeddings=True)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        
        print(f" Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if len(self.chunks) == 0:
            return []
            
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'score': float(score)
                })
        
        return results


class DocumentProcessor:
    """Process documents into chunks"""
    
    def __init__(self, chunk_size: int = 800):
        self.chunk_size = chunk_size
        
    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """Process file and return chunks"""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        print(f"📄 Processing {file_name}...")
        
        if file_extension == '.csv':
            return self._process_csv(file_path, file_name)
        elif file_extension in ['.xlsx', '.xls']:
            return self._process_excel(file_path, file_name)
        elif file_extension == '.pdf':
            return self._process_pdf(file_path, file_name)
        elif file_extension == '.docx':
            return self._process_docx(file_path, file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported: CSV, Excel, PDF, DOCX")
    
    def _process_csv(self, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Process CSV file into meaningful chunks"""
        df = pd.read_csv(file_path)
        chunks = []
        
        # 1. Overview chunk
        overview = f"""FILE: {file_name}
TYPE: CSV Dataset
SHAPE: {df.shape[0]} rows, {df.shape[1]} columns
COLUMNS: {', '.join(df.columns.tolist())}

COLUMN TYPES:
"""
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            overview += f"- {col}: {dtype} ({non_null}/{len(df)} non-null)\n"
        
        chunks.append(DocumentChunk(
            content=overview,
            metadata={'file_name': file_name, 'chunk_type': 'overview', 'file_type': 'csv'}
        ))
        
        # 2. Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats_content = f"STATISTICAL SUMMARY for {file_name}:\n\n"
            
            for col in numeric_cols:
                stats = df[col].describe()
                stats_content += f"{col.upper()}:\n"
                stats_content += f"  Count: {stats['count']}\n"
                stats_content += f"  Mean: {stats['mean']:.2f}\n"
                stats_content += f"  Std: {stats['std']:.2f}\n"
                stats_content += f"  Min: {stats['min']:.2f}\n"
                stats_content += f"  25%: {stats['25%']:.2f}\n"
                stats_content += f"  50% (Median): {stats['50%']:.2f}\n"
                stats_content += f"  75%: {stats['75%']:.2f}\n"
                stats_content += f"  Max: {stats['max']:.2f}\n\n"
            
            chunks.append(DocumentChunk(
                content=stats_content,
                metadata={'file_name': file_name, 'chunk_type': 'statistics', 'file_type': 'csv'}
            ))
        
        # 3. Sample data chunks
        sample_size = 20  # rows per chunk
        for i in range(0, min(len(df), 100), sample_size):  # Max 100 rows, 20 per chunk
            chunk_df = df.iloc[i:i+sample_size]
            
            chunk_content = f"SAMPLE DATA from {file_name} (Rows {i+1}-{min(i+sample_size, len(df))}):\n\n"
            
            # Format as readable text instead of raw dataframe
            for _, row in chunk_df.iterrows():
                chunk_content += "ROW DATA:\n"
                for col, val in row.items():
                    chunk_content += f"  {col}: {val}\n"
                chunk_content += "\n"
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata={
                    'file_name': file_name,
                    'chunk_type': 'sample_data',
                    'file_type': 'csv',
                    'row_start': i,
                    'row_end': min(i+sample_size, len(df))
                }
            ))
        
        # 4. Column analysis chunk
        col_analysis = f"COLUMN ANALYSIS for {file_name}:\n\n"
        
        for col in df.columns:
            col_analysis += f"{col.upper()}:\n"
            col_analysis += f"  Data Type: {df[col].dtype}\n"
            col_analysis += f"  Unique Values: {df[col].nunique()}\n"
            col_analysis += f"  Missing Values: {df[col].isnull().sum()}\n"
            
            if df[col].dtype in ['object', 'string']:
                # For text columns, show most common values
                top_values = df[col].value_counts().head(3)
                col_analysis += f"  Top Values: {dict(top_values)}\n"
            elif pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, show range info
                col_analysis += f"  Range: {df[col].min()} to {df[col].max()}\n"
            
            col_analysis += "\n"
        
        chunks.append(DocumentChunk(
            content=col_analysis,
            metadata={'file_name': file_name, 'chunk_type': 'column_analysis', 'file_type': 'csv'}
        ))
        
        return chunks
    
    def _process_pdf(self, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Process PDF file into meaningful chunks"""
        try:
            import PyPDF2
            chunks = []
            
            # Extract text from PDF
            full_text = ""
            page_texts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_texts.append({
                        'page_number': page_num + 1,
                        'text': page_text
                    })
                    full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
            
            # 1. Overview chunk
            overview = f"""FILE: {file_name}
TYPE: PDF Document
PAGES: {len(page_texts)}
TOTAL_LENGTH: {len(full_text)} characters

CONTENT OVERVIEW:
"""
            # Add first 500 characters as preview
            overview += full_text[:500] + "..."
            
            chunks.append(DocumentChunk(
                content=overview,
                metadata={'file_name': file_name, 'chunk_type': 'overview', 'file_type': 'pdf'}
            ))
            
            # 2. Page-by-page chunks (for shorter documents)
            if len(page_texts) <= 10:  # Only for reasonably sized PDFs
                for page_info in page_texts:
                    if len(page_info['text'].strip()) > 100:  # Skip nearly empty pages
                        page_content = f"PAGE {page_info['page_number']} from {file_name}:\n\n{page_info['text']}"
                        
                        chunks.append(DocumentChunk(
                            content=page_content,
                            metadata={
                                'file_name': file_name,
                                'page_number': page_info['page_number'],
                                'chunk_type': 'page_content',
                                'file_type': 'pdf'
                            }
                        ))
            
            # 3. Smart text chunking for longer documents
            else:
                text_chunks = self._smart_text_chunking(full_text, self.chunk_size)
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        content=f"SECTION {i+1} from {file_name}:\n\n{chunk_text}",
                        metadata={
                            'file_name': file_name,
                            'chunk_index': i,
                            'chunk_type': 'text_section',
                            'file_type': 'pdf'
                        }
                    ))
            
            # 4. Financial data extraction chunk
            financial_extract = self._extract_financial_info_from_text(full_text, file_name)
            if financial_extract:
                chunks.append(DocumentChunk(
                    content=financial_extract,
                    metadata={'file_name': file_name, 'chunk_type': 'financial_extract', 'file_type': 'pdf'}
                ))
            
            return chunks
            
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")
    
    def _process_docx(self, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Process DOCX file into meaningful chunks"""
        try:
            from docx import Document
            chunks = []
            
            # Extract text from DOCX
            doc = Document(file_path)
            full_text = ""
            paragraphs = []
            
            for para in doc.paragraphs:
                if para.text.strip():  # Skip empty paragraphs
                    paragraphs.append(para.text)
                    full_text += para.text + "\n"
            
            # 1. Overview chunk
            overview = f"""FILE: {file_name}
TYPE: Word Document
PARAGRAPHS: {len(paragraphs)}
WORD_COUNT: {len(full_text.split())}

CONTENT OVERVIEW:
"""
            overview += full_text[:500] + "..."
            
            chunks.append(DocumentChunk(
                content=overview,
                metadata={'file_name': file_name, 'chunk_type': 'overview', 'file_type': 'docx'}
            ))
            
            # 2. Paragraph-based chunks (group paragraphs)
            chunk_size_paragraphs = 5  # 5 paragraphs per chunk
            for i in range(0, len(paragraphs), chunk_size_paragraphs):
                chunk_paragraphs = paragraphs[i:i+chunk_size_paragraphs]
                chunk_content = f"SECTION {i//chunk_size_paragraphs + 1} from {file_name}:\n\n"
                chunk_content += "\n\n".join(chunk_paragraphs)
                
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata={
                        'file_name': file_name,
                        'section_number': i//chunk_size_paragraphs + 1,
                        'chunk_type': 'document_section',
                        'file_type': 'docx'
                    }
                ))
            
            # 3. Financial data extraction chunk
            financial_extract = self._extract_financial_info_from_text(full_text, file_name)
            if financial_extract:
                chunks.append(DocumentChunk(
                    content=financial_extract,
                    metadata={'file_name': file_name, 'chunk_type': 'financial_extract', 'file_type': 'docx'}
                ))
            
            return chunks
            
        except Exception as e:
            raise Exception(f"DOCX processing error: {str(e)}")
    
    def _process_excel(self, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Process Excel file"""
        chunks = []
        excel_file = pd.ExcelFile(file_path)
        
        # Overview chunk
        overview = f"""FILE: {file_name}
TYPE: Excel Workbook
SHEETS: {', '.join(excel_file.sheet_names)}
TOTAL_SHEETS: {len(excel_file.sheet_names)}
"""
        
        chunks.append(DocumentChunk(
            content=overview,
            metadata={'file_name': file_name, 'chunk_type': 'overview', 'file_type': 'excel'}
        ))
        
        # Process each sheet (limit to first 3 sheets)
        for sheet_name in excel_file.sheet_names[:3]:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            sheet_content = f"SHEET: {sheet_name} from {file_name}\n"
            sheet_content += f"SHAPE: {df.shape[0]} rows, {df.shape[1]} columns\n"
            sheet_content += f"COLUMNS: {', '.join(df.columns.tolist())}\n\n"
            
            # Add sample data
            sheet_content += "SAMPLE DATA:\n"
            for _, row in df.head(10).iterrows():
                for col, val in row.items():
                    sheet_content += f"  {col}: {val}\n"
                sheet_content += "\n"
            
            chunks.append(DocumentChunk(
                content=sheet_content,
                metadata={'file_name': file_name, 'sheet_name': sheet_name, 'chunk_type': 'sheet_data', 'file_type': 'excel'}
            ))
        
        return chunks
    
    def _smart_text_chunking(self, text: str, chunk_size: int) -> List[str]:
        """Smart text chunking that preserves context"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _extract_financial_info_from_text(self, text: str, file_name: str) -> str:
        """Extract financial information from text content"""
        financial_extract = f"FINANCIAL DATA EXTRACTED from {file_name}:\n\n"
        
        # Extract currency amounts
        currency_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # USD
            r'€[\d,]+(?:\.\d{2})?',   # EUR  
            r'£[\d,]+(?:\.\d{2})?',   # GBP
            r'₹[\d,]+(?:\.\d{2})?',   # INR
            r'[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?)'
        ]
        
        all_currencies = []
        for pattern in currency_patterns:
            all_currencies.extend(re.findall(pattern, text, re.IGNORECASE))
        
        if all_currencies:
            financial_extract += f"CURRENCY VALUES FOUND:\n"
            for curr in all_currencies[:10]:  # Limit to first 10
                financial_extract += f"  • {curr}\n"
            financial_extract += "\n"
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        if percentages:
            financial_extract += f"PERCENTAGES FOUND:\n"
            for pct in percentages[:10]:
                financial_extract += f"  • {pct}\n"
            financial_extract += "\n"
        
        # Extract financial terms and context
        financial_terms = {
            'revenue': [],
            'profit': [],
            'loss': [],
            'expenses': [],
            'assets': [],
            'liabilities': [],
            'equity': [],
            'cash flow': [],
            'margin': [],
            'roi': [],
            'ebitda': [],
            'growth': []
        }
        
        # Find sentences containing financial terms
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for term in financial_terms.keys():
                if term in sentence_lower and len(sentence.strip()) > 20:
                    financial_terms[term].append(sentence.strip())
        
        # Add financial term contexts
        for term, contexts in financial_terms.items():
            if contexts:
                financial_extract += f"{term.upper()} MENTIONS:\n"
                for context in contexts[:3]:  # Limit to 3 per term
                    financial_extract += f"  • {context[:200]}...\n"
                financial_extract += "\n"
        
        # Only return if we found substantial financial content
        if len(all_currencies) > 0 or len(percentages) > 0 or any(financial_terms.values()):
            return financial_extract
        else:
            return None


class WebContentProcessor:
    """Process web content into chunks"""
    
    def __init__(self, chunk_size: int = 800):
        self.chunk_size = chunk_size
    
    def fetch_web_content(self, url: str) -> str:
        """Fetch content from web URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            print(f"🔥 Fetching content from {url}...")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            print(f" Successfully fetched {len(text)} characters")
            return text[:8000]  # Limit content size
            
        except Exception as e:
            raise Exception(f"Failed to fetch {url}: {str(e)}")

    def process_web_content_into_chunks(self, url: str, content: str) -> List[DocumentChunk]:
        """Process web content into searchable chunks"""
        chunks = []
        
        # 1. Overview chunk
        overview = f"""WEB CONTENT: {url}
TYPE: Web Page
CONTENT_LENGTH: {len(content)} characters
WORD_COUNT: {len(content.split())}

CONTENT_PREVIEW:
{content[:800]}...
"""
        
        chunks.append(DocumentChunk(
            content=overview,
            metadata={'url': url, 'chunk_type': 'web_overview', 'content_type': 'url'}
        ))
        
        # 2. Break content into sections
        sections = self._smart_text_chunking(content, 1000)
        
        for i, section in enumerate(sections):
            chunk_content = f"WEB SECTION {i+1} from {url}:\n\n{section}"
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata={
                    'url': url,
                    'section_number': i+1,
                    'chunk_type': 'web_section',
                    'content_type': 'url'
                }
            ))
        
        # 3. Extract financial information
        financial_extract = self._extract_financial_info_from_text(content, url)
        if financial_extract:
            chunks.append(DocumentChunk(
                content=financial_extract,
                metadata={'url': url, 'chunk_type': 'web_financial_extract', 'content_type': 'url'}
            ))
        
        return chunks

    def extract_title_from_content(self, content: str) -> str:
        """Extract title from web content"""
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 100:  # Reasonable title length
                return line
        return "Web Content"
    
    def _smart_text_chunking(self, text: str, chunk_size: int) -> List[str]:
        """Smart text chunking that preserves context"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _extract_financial_info_from_text(self, text: str, file_name: str) -> str:
        """Extract financial information from text content"""
        financial_extract = f"FINANCIAL DATA EXTRACTED from {file_name}:\n\n"
        
        # Extract currency amounts
        currency_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # USD
            r'€[\d,]+(?:\.\d{2})?',   # EUR  
            r'£[\d,]+(?:\.\d{2})?',   # GBP
            r'₹[\d,]+(?:\.\d{2})?',   # INR
            r'[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?)'
        ]
        
        all_currencies = []
        for pattern in currency_patterns:
            all_currencies.extend(re.findall(pattern, text, re.IGNORECASE))
        
        if all_currencies:
            financial_extract += f"CURRENCY VALUES FOUND:\n"
            for curr in all_currencies[:10]:  # Limit to first 10
                financial_extract += f"  • {curr}\n"
            financial_extract += "\n"
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        if percentages:
            financial_extract += f"PERCENTAGES FOUND:\n"
            for pct in percentages[:10]:
                financial_extract += f"  • {pct}\n"
            financial_extract += "\n"
        
        # Extract financial terms and context
        financial_terms = {
            'revenue': [],
            'profit': [],
            'loss': [],
            'expenses': [],
            'assets': [],
            'liabilities': [],
            'equity': [],
            'cash flow': [],
            'margin': [],
            'roi': [],
            'ebitda': [],
            'growth': []
        }
        
        # Find sentences containing financial terms
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for term in financial_terms.keys():
                if term in sentence_lower and len(sentence.strip()) > 20:
                    financial_terms[term].append(sentence.strip())
        
        # Add financial term contexts
        for term, contexts in financial_terms.items():
            if contexts:
                financial_extract += f"{term.upper()} MENTIONS:\n"
                for context in contexts[:3]:  # Limit to 3 per term
                    financial_extract += f"  • {context[:200]}...\n"
                financial_extract += "\n"
        
        # Only return if we found substantial financial content
        if len(all_currencies) > 0 or len(percentages) > 0 or any(financial_terms.values()):
            return financial_extract
        else:
            return None