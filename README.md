# Financial-Chatbot-Ai-Agents

## Financial RAG Chatbot with Specialized Agents

A sophisticated Retrieval-Augmented Generation (RAG) system designed for comprehensive financial document analysis and intelligent querying. This chatbot processes multiple document types, web content, and uses specialized AI agents to provide expert-level financial insights.

---

## Features

### 🎯 **Multi-Format Document Support**

* **CSV/Excel**: Statistical analysis, data visualization recommendations
* **PDF Documents**: Text extraction, financial data mining
* **Word Documents**: Content analysis, structured processing
* **Web Content**: Real-time financial news and data integration

### 🤖 **8 Specialized AI Agents**

* **Trend Analyzer**: Identifies financial trends and time-series patterns
* **Comparative Analyzer**: Performs cross-entity, cross-period comparisons
* **Statistical Calculator**: Advanced mathematical analysis and metrics
* **Document Summarizer**: Intelligent content summarization
* **Table Extractor**: Structured data extraction and querying
* **Visualization Generator**: Chart and graph recommendations with code
* **Multilingual Processor**: Multi-language query support
* **Web Content Analyzer**: Real-time web data integration

### 🔍 **Intelligent Search & Processing**

* Vector-based semantic search using FAISS
* Context-aware chunking strategies
* Financial keyword and pattern extraction
* Cross-document insight synthesis

---

## Installation

### Prerequisites

* Python 3.10+
* Virtual environment recommended

### Setup

```bash
# Clone repo
git clone <repository-url>
cd Financial-Chatbot-Ai-Agents

# Create and activate venv (Linux)
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The bot will automatically load keys from environment variables.

---

## Quick Start
run > streamlit run stream_app.py
