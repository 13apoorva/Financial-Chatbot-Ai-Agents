"""
Flask API for Financial RAG Chatbot
Single file REST API with all endpoints
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging
from functools import wraps
import traceback

from main import SimpleRAGFinancialBot

load_dotenv()

app = Flask(__name__)
CORS(app)  


API_KEY = os.getenv('GROQ_API_KEY', 'your-default-api-key')
UPLOAD_FOLDER = tempfile.gettempdir()
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


bot_instances = {}


ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'pdf', 'docx', 'txt'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    return decorated_function

def get_bot_instance(session_id='default'):
    """Get or create bot instance for session"""
    if session_id not in bot_instances:
        try:
            bot_instances[session_id] = SimpleRAGFinancialBot(API_KEY)
            logger.info(f"Created new bot instance for session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to create bot instance: {str(e)}")
            raise
    
    return bot_instances[session_id]

# ===== HEALTH CHECK ENDPOINTS =====

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Financial RAG Chatbot API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status', methods=['GET'])
@error_handler
def get_status():
    """Get system status"""
    session_id = request.args.get('session_id', 'default')
    
    try:
        bot = get_bot_instance(session_id)
        
        return jsonify({
            'success': True,
            'status': {
                'bot_initialized': True,
                'total_chunks': len(bot.vector_store.chunks),
                'available_agents': list(bot.agents.keys()),
                'session_id': session_id
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'status': {
                'bot_initialized': False,
                'error': str(e)
            },
            'timestamp': datetime.now().isoformat()
        }), 500

# ===== SESSION MANAGEMENT =====

@app.route('/session/create', methods=['POST'])
@error_handler
def create_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    
    try:
        bot = get_bot_instance(session_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session created successfully',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to create session: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/session/<session_id>/reset', methods=['POST'])
@error_handler
def reset_session(session_id):
    """Reset a session (clear all data)"""
    try:
        # Remove existing bot instance
        if session_id in bot_instances:
            del bot_instances[session_id]
        
        # Create new bot instance
        bot = get_bot_instance(session_id)
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to reset session: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

# ===== FILE UPLOAD ENDPOINTS =====

@app.route('/upload/file', methods=['POST'])
@error_handler
def upload_file():
    """Upload and process a single file"""
    session_id = request.form.get('session_id', 'default')
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
    file.save(temp_path)
    
    try:
        bot = get_bot_instance(session_id)
        result = bot.add_document(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'file_info': {
                    'name': filename,
                    'chunks_created': result['chunks_created'],
                    'processed_at': datetime.now().isoformat()
                },
                'session_id': session_id
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
    
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

@app.route('/upload/files', methods=['POST'])
@error_handler
def upload_multiple_files():
    """Upload and process multiple files"""
    session_id = request.form.get('session_id', 'default')
    
    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files provided'
        }), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({
            'success': False,
            'error': 'No files selected'
        }), 400
    
    bot = get_bot_instance(session_id)
    results = []
    processed_files = []
    
    for file in files:
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'success': False,
                'error': f'File type not allowed'
            })
            continue
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            result = bot.add_document(temp_path)
            
            if result['success']:
                processed_files.append({
                    'name': filename,
                    'chunks_created': result['chunks_created'],
                    'processed_at': datetime.now().isoformat()
                })
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'message': result['message'],
                    'chunks_created': result['chunks_created']
                })
            else:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': result['error']
                })
        
        except Exception as e:
            results.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    successful_uploads = len([r for r in results if r['success']])
    
    return jsonify({
        'success': successful_uploads > 0,
        'message': f'Processed {successful_uploads}/{len(files)} files successfully',
        'results': results,
        'processed_files': processed_files,
        'session_id': session_id
    })

# ===== URL PROCESSING ENDPOINTS =====

@app.route('/process/url', methods=['POST'])
@error_handler
def process_url():
    """Process a single URL"""
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({
            'success': False,
            'error': 'URL is required'
        }), 400
    
    url = data['url']
    session_id = data.get('session_id', 'default')
    
    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        return jsonify({
            'success': False,
            'error': 'Invalid URL format. Must start with http:// or https://'
        }), 400
    
    bot = get_bot_instance(session_id)
    result = bot.add_url(url)
    
    if result['success']:
        return jsonify({
            'success': True,
            'message': result['message'],
            'url_info': {
                'url': url,
                'chunks_created': result['chunks_created'],
                'processed_at': datetime.now().isoformat()
            },
            'session_id': session_id
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 400

@app.route('/process/urls', methods=['POST'])
@error_handler
def process_multiple_urls():
    """Process multiple URLs"""
    data = request.get_json()
    
    if not data or 'urls' not in data:
        return jsonify({
            'success': False,
            'error': 'URLs list is required'
        }), 400
    
    urls = data['urls']
    session_id = data.get('session_id', 'default')
    
    if not isinstance(urls, list) or not urls:
        return jsonify({
            'success': False,
            'error': 'URLs must be a non-empty list'
        }), 400
    
    bot = get_bot_instance(session_id)
    results = []
    processed_urls = []
    
    for url in urls:
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            results.append({
                'url': url,
                'success': False,
                'error': 'Invalid URL format'
            })
            continue
        
        try:
            result = bot.add_url(url)
            
            if result['success']:
                processed_urls.append({
                    'url': url,
                    'chunks_created': result['chunks_created'],
                    'processed_at': datetime.now().isoformat()
                })
                
                results.append({
                    'url': url,
                    'success': True,
                    'message': result['message'],
                    'chunks_created': result['chunks_created']
                })
            else:
                results.append({
                    'url': url,
                    'success': False,
                    'error': result['error']
                })
        
        except Exception as e:
            results.append({
                'url': url,
                'success': False,
                'error': str(e)
            })
    
    successful_urls = len([r for r in results if r['success']])
    
    return jsonify({
        'success': successful_urls > 0,
        'message': f'Processed {successful_urls}/{len(urls)} URLs successfully',
        'results': results,
        'processed_urls': processed_urls,
        'session_id': session_id
    })

# ===== QUERY PROCESSING ENDPOINTS =====

@app.route('/query', methods=['POST'])
@error_handler
def process_query():
    """Process a query and return analysis"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400
    
    query = data['query'].strip()
    session_id = data.get('session_id', 'default')
    language = data.get('language', 'auto')
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Query cannot be empty'
        }), 400
    
    bot = get_bot_instance(session_id)
    
    # Check if any content is processed
    if not bot.vector_store.chunks:
        return jsonify({
            'success': False,
            'error': 'No content has been processed yet. Please upload files or add URLs first.'
        }), 400
    
    try:
        response = bot.query(query, language)
        
        return jsonify({
            'success': True,
            'query': query,
            'response': response,
            'language': language,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'total_chunks': len(bot.vector_store.chunks),
                'agents_used': 'auto-selected based on query'
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Query processing failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/query/batch', methods=['POST'])
@error_handler
def process_batch_queries():
    """Process multiple queries in batch"""
    data = request.get_json()
    
    if not data or 'queries' not in data:
        return jsonify({
            'success': False,
            'error': 'Queries list is required'
        }), 400
    
    queries = data['queries']
    session_id = data.get('session_id', 'default')
    language = data.get('language', 'auto')
    
    if not isinstance(queries, list) or not queries:
        return jsonify({
            'success': False,
            'error': 'Queries must be a non-empty list'
        }), 400
    
    bot = get_bot_instance(session_id)
    
    # Check if any content is processed
    if not bot.vector_store.chunks:
        return jsonify({
            'success': False,
            'error': 'No content has been processed yet. Please upload files or add URLs first.'
        }), 400
    
    results = []
    
    for i, query in enumerate(queries):
        query = query.strip()
        
        if not query:
            results.append({
                'query_index': i,
                'query': query,
                'success': False,
                'error': 'Empty query'
            })
            continue
        
        try:
            response = bot.query(query, language)
            
            results.append({
                'query_index': i,
                'query': query,
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            results.append({
                'query_index': i,
                'query': query,
                'success': False,
                'error': str(e)
            })
    
    successful_queries = len([r for r in results if r['success']])
    
    return jsonify({
        'success': successful_queries > 0,
        'message': f'Processed {successful_queries}/{len(queries)} queries successfully',
        'results': results,
        'session_id': session_id,
        'language': language,
        'batch_timestamp': datetime.now().isoformat()
    })

# ===== CONTENT MANAGEMENT ENDPOINTS =====

@app.route('/content/list', methods=['GET'])
@error_handler
def list_content():
    """List all processed content"""
    session_id = request.args.get('session_id', 'default')
    
    bot = get_bot_instance(session_id)
    
    # Get processed files info
    processed_files = getattr(bot, 'processed_files', [])
    
    # Get chunk statistics
    chunks = bot.vector_store.chunks
    chunk_stats = {}
    
    for chunk in chunks:
        source = chunk.metadata.get('file_name') or chunk.metadata.get('url', 'unknown')
        chunk_type = chunk.metadata.get('chunk_type', 'unknown')
        
        if source not in chunk_stats:
            chunk_stats[source] = {}
        
        chunk_stats[source][chunk_type] = chunk_stats[source].get(chunk_type, 0) + 1
    
    return jsonify({
        'success': True,
        'content': {
            'processed_files': processed_files,
            'chunk_statistics': chunk_stats,
            'total_chunks': len(chunks),
            'total_sources': len(chunk_stats)
        },
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/content/clear', methods=['DELETE'])
@error_handler
def clear_content():
    """Clear all processed content"""
    session_id = request.args.get('session_id', 'default')
    
    # Remove existing bot instance and create new one
    if session_id in bot_instances:
        del bot_instances[session_id]
    
    bot = get_bot_instance(session_id)
    
    return jsonify({
        'success': True,
        'message': 'All content cleared successfully',
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/content/stats', methods=['GET'])
@error_handler
def get_content_stats():
    """Get detailed content statistics"""
    session_id = request.args.get('session_id', 'default')
    
    bot = get_bot_instance(session_id)
    
    chunks = bot.vector_store.chunks
    processed_files = getattr(bot, 'processed_files', [])
    
    # Calculate statistics
    stats = {
        'total_chunks': len(chunks),
        'total_files': len([f for f in processed_files if f.get('file_name')]),
        'total_urls': len([f for f in processed_files if f.get('url')]),
        'agents_available': len(bot.agents),
        'chunk_types': {},
        'file_types': {},
        'processing_timeline': []
    }
    
    # Analyze chunks
    for chunk in chunks:
        chunk_type = chunk.metadata.get('chunk_type', 'unknown')
        file_type = chunk.metadata.get('file_type', 'unknown')
        
        stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
    
    # Processing timeline
    for file_info in processed_files:
        stats['processing_timeline'].append({
            'name': file_info.get('file_name') or file_info.get('url', 'unknown'),
            'type': 'file' if file_info.get('file_name') else 'url',
            'chunks_created': file_info.get('chunks_created', 0),
            'processed_at': file_info.get('processed_at')
        })
    
    return jsonify({
        'success': True,
        'statistics': stats,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    })

# ===== AGENT INFORMATION ENDPOINTS =====

@app.route('/agents/list', methods=['GET'])
@error_handler
def list_agents():
    """List all available agents"""
    session_id = request.args.get('session_id', 'default')
    
    bot = get_bot_instance(session_id)
    
    agents_info = {
        'trend_analyzer': {
            'name': 'Trend Analyzer',
            'description': 'Identifies financial trends and patterns over time',
            'capabilities': ['time_series_analysis', 'pattern_recognition', 'trend_detection']
        },
        'comparative_analyzer': {
            'name': 'Comparative Analyzer',
            'description': 'Compares different datasets or time periods',
            'capabilities': ['data_comparison', 'percentage_changes', 'performance_analysis']
        },
        'statistical_calculator': {
            'name': 'Statistical Calculator',
            'description': 'Performs calculations and statistical analysis',
            'capabilities': ['mathematical_operations', 'statistical_summaries', 'data_insights']
        },
        'document_summarizer': {
            'name': 'Document Summarizer',
            'description': 'Creates comprehensive summaries',
            'capabilities': ['document_analysis', 'key_insights_extraction', 'content_synthesis']
        },
        'table_extractor': {
            'name': 'Table Extractor',
            'description': 'Extracts and analyzes tabular data',
            'capabilities': ['table_parsing', 'data_extraction', 'structured_analysis']
        },
        'visualization_generator': {
            'name': 'Visualization Generator',
            'description': 'Suggests appropriate charts and graphs',
            'capabilities': ['chart_recommendations', 'visualization_code', 'data_presentation']
        },
        'multilingual_processor': {
            'name': 'Multilingual Processor',
            'description': 'Handles queries in multiple languages',
            'capabilities': ['language_detection', 'translation', 'multilingual_responses']
        },
        'web_content_analyzer': {
            'name': 'Web Content Analyzer',
            'description': 'Analyzes web-based financial content',
            'capabilities': ['web_scraping', 'content_analysis', 'market_insights']
        }
    }
    
    return jsonify({
        'success': True,
        'agents': agents_info,
        'total_agents': len(agents_info),
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    })

# ===== EXPORT ENDPOINTS =====

@app.route('/export/session', methods=['GET'])
@error_handler
def export_session_data():
    """Export session data"""
    session_id = request.args.get('session_id', 'default')
    
    bot = get_bot_instance(session_id)
    processed_files = getattr(bot, 'processed_files', [])
    
    export_data = {
        'session_id': session_id,
        'export_timestamp': datetime.now().isoformat(),
        'processed_files': processed_files,
        'total_chunks': len(bot.vector_store.chunks),
        'available_agents': list(bot.agents.keys()),
        'statistics': {
            'chunks_by_type': {},
            'files_by_type': {}
        }
    }
    
    # Calculate statistics for export
    for chunk in bot.vector_store.chunks:
        chunk_type = chunk.metadata.get('chunk_type', 'unknown')
        file_type = chunk.metadata.get('file_type', 'unknown')
        
        export_data['statistics']['chunks_by_type'][chunk_type] = \
            export_data['statistics']['chunks_by_type'].get(chunk_type, 0) + 1
        export_data['statistics']['files_by_type'][file_type] = \
            export_data['statistics']['files_by_type'].get(file_type, 0) + 1
    
    return jsonify({
        'success': True,
        'export_data': export_data,
        'timestamp': datetime.now().isoformat()
    })

# ===== ERROR HANDLERS =====

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': f'Maximum file size is {MAX_CONTENT_LENGTH // (1024*1024)}MB',
        'timestamp': datetime.now().isoformat()
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

# ===== API DOCUMENTATION ENDPOINT =====

@app.route('/docs', methods=['GET'])
def api_documentation():
    """API documentation endpoint"""
    docs = {
        'title': 'Financial RAG Chatbot API',
        'version': '1.0.0',
        'description': 'REST API for processing financial documents and querying with AI',
        'endpoints': {
            'Health Check': {
                'GET /health': 'Check API health status',
                'GET /status': 'Get system status and bot information'
            },
            'Session Management': {
                'POST /session/create': 'Create a new session',
                'POST /session/{session_id}/reset': 'Reset a session'
            },
            'File Processing': {
                'POST /upload/file': 'Upload and process a single file',
                'POST /upload/files': 'Upload and process multiple files'
            },
            'URL Processing': {
                'POST /process/url': 'Process a single URL',
                'POST /process/urls': 'Process multiple URLs'
            },
            'Query Processing': {
                'POST /query': 'Process a single query',
                'POST /query/batch': 'Process multiple queries'
            },
            'Content Management': {
                'GET /content/list': 'List all processed content',
                'GET /content/stats': 'Get detailed content statistics',
                'DELETE /content/clear': 'Clear all processed content'
            },
            'Agent Information': {
                'GET /agents/list': 'List all available agents'
            },
            'Export': {
                'GET /export/session': 'Export session data'
            }
        },
        'supported_file_types': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024*1024),
        'example_usage': {
            'create_session': 'POST /session/create',
            'upload_file': 'POST /upload/file (form-data with file)',
            'process_url': 'POST /process/url {"url": "https://example.com", "session_id": "your-session-id"}',
            'query': 'POST /query {"query": "What are the key financial metrics?", "session_id": "your-session-id"}'
        }
    }
    
    return jsonify(docs)

# ===== MAIN APPLICATION =====

if __name__ == '__main__':
  
    print("=" * 60)
    print(" Financial RAG Chatbot API Starting...")
    print("=" * 60)
    print(f" Health Check: http://localhost:5000/health")
    print(f" Documentation: http://localhost:5000/docs")
    print(f" Status Check: http://localhost:5000/status")
    print("=" * 60)
    
    # Run the Flask app
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000))
    )