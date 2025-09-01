"""
Specialized Financial Agents for RAG Chatbot
Contains all specialized agents for different financial analysis tasks
"""

import json
import re
from typing import Dict, List
from groq import Groq


class TrendAnalyzerAgent:
    """Agent specialized in financial trend analysis"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Trend Analyzer"
    
    def execute(self, input_data: Dict) -> Dict:
        """Analyze financial trends from data"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        
        # Extract time-series data from chunks
        time_series_data = self._extract_time_series_data(relevant_chunks)
        
        if not time_series_data:
            return {
                'trend_analysis': 'No time-series data found for trend analysis',
                'trends_detected': 0
            }
        
        # Analyze trends using LLM
        analysis = self._analyze_trends_with_llm(query, time_series_data)
        
        return {
            'trend_analysis': analysis,
            'data_points_analyzed': len(time_series_data),
            'trends_detected': self._count_trends(analysis)
        }
    
    def _extract_time_series_data(self, chunks: List[Dict]) -> List[str]:
        """Extract time-series related data from chunks"""
        time_data = []
        
        for chunk in chunks:
            content = chunk['content']
            # Look for date patterns and numeric data
            if any(word in content.lower() for word in ['date', 'month', 'quarter', 'year', 'q1', 'q2', 'q3', 'q4']):
                time_data.append(content)
        
        return time_data
    
    def _analyze_trends_with_llm(self, query: str, time_data: List[str]) -> str:
        """Use LLM to analyze trends"""
        data_context = "\n".join(time_data[:3])  # Limit context size
        
        prompt = f"""Analyze financial trends from this data:

DATA:
{data_context}

QUERY: {query}

Identify:
1. Time periods covered
2. Financial metrics and their changes
3. Growth/decline patterns
4. Key trend insights

Provide trend analysis:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial trend analysis expert. Your ask is to generate response the the user query based on the finacial data provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            return response.choices[0].message.content
        except:
            return "Trend analysis unavailable"
    
    def _count_trends(self, analysis: str) -> int:
        """Count number of trends mentioned"""
        trend_words = ['increase', 'decrease', 'growth', 'decline', 'rising', 'falling']
        return sum(1 for word in trend_words if word in analysis.lower())


class ComparativeAnalyzerAgent:
    """Agent for comparative financial analysis"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Comparative Analyzer"
    
    def execute(self, input_data: Dict) -> Dict:
        """Perform comparative analysis"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        
        # Extract comparable data
        comparable_data = self._extract_comparable_data(relevant_chunks, query)
        
        if len(comparable_data) < 2:
            return {
                'comparison_result': 'Insufficient data for comparison',
                'comparisons_made': 0
            }
        
        # Perform comparison analysis
        comparison = self._perform_comparison(query, comparable_data)
        
        return {
            'comparison_result': comparison,
            'data_sources_compared': len(comparable_data),
            'comparisons_made': 1
        }
    
    def _extract_comparable_data(self, chunks: List[Dict], query: str) -> List[str]:
        """Extract data suitable for comparison"""
        comparable = []
        
        # Look for different time periods, entities, or categories
        for chunk in chunks:
            content = chunk['content']
            if any(word in content.lower() for word in ['q1', 'q2', 'q3', 'q4', '2023', '2024', 'year', 'month']):
                comparable.append(content)
        
        return comparable[:4]  # Limit to 4 for comparison
    
    def _perform_comparison(self, query: str, data: List[str]) -> str:
        """Perform comparison using LLM"""
        data_context = "\n\n".join([f"DATA SOURCE {i+1}:\n{d}" for i, d in enumerate(data)])
        
        prompt = f"""Compare the financial data from different sources:

        {data_context}

        USER REQUEST: {query}

        Provide comparative analysis:
        1. Key differences between data sources
        2. Percentage changes or variations
        3. Which performs better and why
        4. Actionable insights from comparison

        Comparison analysis:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial comparison expert. Compare and analyse between different components of the data provided as per the user query."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except:
            return "Comparison analysis unavailable"


class StatisticalCalculatorAgent:
    """Agent for statistical calculations and analysis"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Statistical Calculator"
    
    def execute(self, input_data: Dict) -> Dict:
        """Perform statistical calculations"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        
        # Extract numerical data
        numerical_data = self._extract_numerical_data(relevant_chunks)
        
        # Perform calculations
        calculations = self._perform_calculations(query, numerical_data)
        
        return {
            'statistical_results': calculations,
            'data_points_used': len(numerical_data),
            'calculations_performed': len(calculations.split('\n')) if calculations else 0
        }
    
    def _extract_numerical_data(self, chunks: List[Dict]) -> Dict:
        """Extract numerical data from chunks"""
        numerical_data = {}
        
        for chunk in chunks:
            content = chunk['content']
            
            # Look for statistical summaries
            if 'statistics' in chunk['metadata'].get('chunk_type', '').lower():
                # Parse statistical information
                lines = content.split('\n')
                for line in lines:
                    if ':' in line and any(stat in line.lower() for stat in ['mean', 'average', 'sum', 'total', 'min', 'max']):
                        parts = line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            try:
                                value = float(parts[1].strip().replace(',', ''))
                                numerical_data[key] = value
                            except:
                                pass
        
        return numerical_data
    
    def _perform_calculations(self, query: str, data: Dict) -> str:
        """Perform statistical calculations"""
        if not data:
            return "No numerical data available for calculations"
        
        prompt = f"""Perform statistical analysis on this financial data as asked by the user:

AVAILABLE DATA:
{json.dumps(data, indent=2)}

USER REQUEST: {query}

Calculate and provide:
1. Requested specific metrics (average, sum, etc.)
2. Additional relevant statistics
3. Data interpretation and insights
4. Mathematical reasoning

Statistical analysis:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial statistics expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            return response.choices[0].message.content
        except:
            return "Statistical calculation unavailable"


class DocumentSummarizerAgent:
    """Agent for document summarization"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Document Summarizer"
    
    def execute(self, input_data: Dict) -> Dict:
        """Generate document summaries"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        language = input_data.get('language', 'English')
        
        # Categorize chunks by type
        chunk_categories = self._categorize_chunks(relevant_chunks)
        
        # Generate summaries for each category
        summaries = {}
        for category, chunks in chunk_categories.items():
            if chunks:
                summaries[category] = self._generate_category_summary(chunks, language)
        
        return {
            'document_summaries': summaries,
            'documents_processed': len(relevant_chunks),
            'summary_categories': list(summaries.keys())
        }
    
    def _categorize_chunks(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize chunks by type"""
        categories = {
            'overview': [],
            'statistics': [],
            'sample_data': [],
            'analysis': []
        }
        
        for chunk in chunks:
            chunk_type = chunk['metadata'].get('chunk_type', 'analysis')
            if chunk_type in categories:
                categories[chunk_type].append(chunk)
            else:
                categories['analysis'].append(chunk)
        
        return categories
    
    def _generate_category_summary(self, chunks: List[Dict], language: str) -> str:
        """Generate summary for chunk category"""
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks])
        
        prompt = f"""Summarize this financial information concisely:

CONTENT:
{combined_content[:1500]}

Requirements:
- Provide key insights and main points
- Focus on financial metrics and important data
- Be concise but comprehensive
- Respond in {language}

Summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a financial document summarization expert. Respond in {language}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600
            )
            return response.choices[0].message.content
        except:
            return "Summary unavailable"


class TableExtractorAgent:
    """Agent for extracting and querying tabular data"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Table Extractor"
    
    def execute(self, input_data: Dict) -> Dict:
        """Extract and analyze tabular data"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        
        # Find table-like data
        table_data = self._extract_table_data(relevant_chunks)
        
        if not table_data:
            return {
                'extraction_result': 'No tabular data found',
                'tables_extracted': 0
            }
        
        # Process table query
        result = self._process_table_query(query, table_data)
        
        return {
            'extraction_result': result,
            'tables_extracted': len(table_data),
            'rows_processed': sum(len(table.split('\n')) for table in table_data)
        }
    
    def _extract_table_data(self, chunks: List[Dict]) -> List[str]:
        """Extract table-like data from chunks"""
        tables = []
        
        for chunk in chunks:
            content = chunk['content']
            chunk_type = chunk['metadata'].get('chunk_type', '')
            
            # Look for structured data
            if any(indicator in chunk_type for indicator in ['sample_data', 'statistics', 'sheet_data']):
                tables.append(content)
        
        return tables
    
    def _process_table_query(self, query: str, table_data: List[str]) -> str:
        """Process query against table data"""
        tables_context = "\n\n".join([f"TABLE {i+1}:\n{table}" for i, table in enumerate(table_data)])
        
        prompt = f"""Extract specific information from these tables based on the user query:

TABLES:
{tables_context}

USER QUERY: {query}

Tasks:
1. Identify relevant table(s)
2. Extract requested data
3. Perform any sorting, filtering, or ranking
4. Present results clearly

Extraction result:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data extraction expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except:
            return "Table extraction unavailable"


class VisualizationAgent:
    """Agent for generating visualization descriptions and code"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Visualization Generator"
    
    def execute(self, input_data: Dict) -> Dict:
        """Generate visualization recommendations"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        
        # Analyze data for visualization potential
        viz_data = self._analyze_visualization_potential(relevant_chunks)
        
        # Generate visualization recommendations
        recommendations = self._generate_viz_recommendations(query, viz_data)
        
        return {
            'visualization_recommendations': recommendations,
            'chart_types_suggested': self._extract_chart_types(recommendations),
            'data_suitable_for_viz': len(viz_data) > 0
        }
    
    def _analyze_visualization_potential(self, chunks: List[Dict]) -> List[str]:
        """Analyze what data can be visualized"""
        viz_suitable = []
        
        for chunk in chunks:
            content = chunk['content']
            # Look for numerical data that can be visualized
            if any(indicator in content.lower() for indicator in ['mean', 'total', 'sum', 'revenue', 'profit', 'statistics']):
                viz_suitable.append(content)
        
        return viz_suitable
    
    def _generate_viz_recommendations(self, query: str, data: List[str]) -> str:
        """Generate visualization recommendations"""
        data_context = "\n".join(data[:2])  # Limit context
        
        prompt = f"""Based on this financial data, recommend appropriate visualizations:

DATA:
{data_context}

USER REQUEST: {query}

Provide:
1. Best chart types for this data
2. What should be on X and Y axes
3. Key insights the visualization would show
4. Python code example (matplotlib/plotly)

Visualization recommendations:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            return response.choices[0].message.content
        except:
            return "Visualization recommendations unavailable"
    
    def _extract_chart_types(self, recommendations: str) -> List[str]:
        """Extract chart types from recommendations"""
        chart_types = []
        charts = ['line chart', 'bar chart', 'pie chart', 'scatter plot', 'histogram', 'box plot']
        
        for chart in charts:
            if chart in recommendations.lower():
                chart_types.append(chart)
        
        return chart_types


class MultilingualAgent:
    """Agent for multilingual processing"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Multilingual Processor"
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple language detection
        if any(char in text for char in 'áéíóúñü¿¡'):
            return 'Spanish'
        elif any(char in text for char in 'àâäéèêëïîôöùûüÿç'):
            return 'French'
        elif any(char in text for char in 'äöüß'):
            return 'German'
        else:
            return 'English'
    
    def execute(self, input_data: Dict) -> Dict:
        """Process multilingual queries"""
        query = input_data['query']
        language = input_data.get('language', 'English')
        
        if language == 'English':
            return {
                'translation_needed': False,
                'detected_language': 'English',
                'processing_notes': 'No translation required'
            }
        
        # Translate query to English for processing
        translated_query = self._translate_to_english(query, language)
        
        return {
            'translation_needed': True,
            'detected_language': language,
            'original_query': query,
            'translated_query': translated_query,
            'processing_notes': f'Translated from {language} for processing'
        }
    
    def _translate_to_english(self, text: str, source_language: str) -> str:
        """Translate text to English"""
        prompt = f"""Translate this {source_language} financial query to English:

{source_language} text: {text}

English translation:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial terminology translator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content
        except:
            return text  # Fallback to original


class WebContentAnalyzer:
    """Agent for analyzing web-based financial content"""
    
    def __init__(self, client: Groq, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = "Web Content Analyzer"
    
    def execute(self, input_data: Dict) -> Dict:
        """Analyze web content from URLs in query"""
        query = input_data['query']
        relevant_chunks = input_data.get('relevant_chunks', [])
        
        # Extract URLs from query
        urls = self._extract_urls_from_query(query)
        
        if not urls:
            return {
                'web_analysis': 'No URLs detected in query for web content analysis',
                'sources_analyzed': 0,
                'content_extracted': False,
                'suggestion': 'To analyze web content, please provide URLs in your query'
            }
        
        # Analyze web content from existing chunks (if any web content was already processed)
        web_chunks = [chunk for chunk in relevant_chunks 
                     if chunk['metadata'].get('content_type') == 'url']
        
        if web_chunks:
            # Use already processed web content
            analysis = self._analyze_existing_web_chunks(web_chunks, query)
            
            return {
                'web_analysis': analysis,
                'sources_analyzed': len(web_chunks),
                'content_extracted': True,
                'urls_found': urls
            }
        
        else:
            # No web content in chunks, suggest adding URLs
            return {
                'web_analysis': f'URLs detected: {", ".join(urls)}. To analyze web content, please add these URLs first using add_url() method.',
                'sources_analyzed': 0,
                'content_extracted': False,
                'urls_found': urls
            }
    
    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extract URLs from user query"""
        # URL patterns
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
        ]
        
        urls = []
        for pattern in url_patterns:
            found_urls = re.findall(pattern, query)
            urls.extend(found_urls)
        
        # Clean URLs
        cleaned_urls = []
        for url in urls:
            url = url.rstrip('.,;!?)')  # Remove trailing punctuation
            if not url.startswith('http'):
                url = 'https://' + url
            cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  
    
    def _analyze_existing_web_chunks(self, web_chunks: List[Dict], query: str) -> str:
        """Analyze already processed web content chunks"""
        
        analysis = f"WEB CONTENT ANALYSIS:\n\n"
        
        # Group chunks by URL
        url_groups = {}
        for chunk in web_chunks:
            url = chunk['metadata'].get('url', 'Unknown URL')
            if url not in url_groups:
                url_groups[url] = []
            url_groups[url].append(chunk)
        
        # Analyze each URL's content
        for url, chunks in url_groups.items():
            analysis += f"SOURCE: {url}\n"
            
            # Combine content from chunks
            combined_content = "\n".join([chunk['content'] for chunk in chunks])
            
            # Use LLM to analyze this URL's content
            url_analysis = self._analyze_web_financial_content(combined_content[:2000], url, query)
            analysis += f"ANALYSIS: {url_analysis}\n\n"
        
        return analysis
    
    def _analyze_web_financial_content(self, content: str, url: str, query: str) -> str:
        """Analyze web content for financial information"""
        
        prompt = f"""Analyze this web content for financial information relevant to the user's query:

URL: {url}
USER QUERY: {query}

WEB CONTENT:
{content}

Provide analysis focusing on:
1. Key financial metrics mentioned
2. Market trends and insights
3. Relevant data points
4. How this relates to the user's query
5. Current market conditions or news

Financial analysis of web content:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a web financial content analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except:
            return f"Analysis unavailable for {url}"