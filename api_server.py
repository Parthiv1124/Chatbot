from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import hashlib
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Import your existing core modules
from core.embeddings import GeminiEmbedder
from core.vector_store import VectorStore
from core.llm import GeminiLLM
from core.retrieval import (
    is_generic_query, make_context, build_prompt, 
    add_inline_citations, minimal_extractive_fallback
)
from core.pdf_utils import build_chunks
from core.config import (
    VS_BASE, UPLOAD_DIR, LOW_CONFIDENCE_THRESH, 
    TOPK_DENSE, TOPK_FINAL, load_env, ensure_dirs,
    collection_id_from_file_infos
)

app = Flask(__name__)
CORS(app)  # Enable CORS for web integration

# Global state
ensure_dirs()
cfg = load_env()
embedder = GeminiEmbedder()
llm = GeminiLLM(cfg["GEMINI_MODEL"])

# In-memory session storage (replace with Redis/DB for production)
sessions = {}

def get_or_create_session(session_id=None):
    """Get existing session or create new one"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    
    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "id": new_id,
        "messages": [],
        "collection_id": None,
        "created_at": datetime.now().isoformat()
    }
    return new_id, sessions[new_id]

def conversation_summary(session):
    """Generate conversation summary from last N messages"""
    messages = session.get("messages", [])
    if not messages:
        return "No prior conversation."
    
    last_n = messages[-10:]  # Last 10 messages
    summary_parts = []
    for msg in last_n:
        role = "User" if msg["role"] == "user" else "Assistant"
        summary_parts.append(f"{role}: {msg['content'][:100]}")
    
    return " | ".join(summary_parts)

def get_all_collections():
    """Get list of all collection folders in VS_BASE"""
    if not os.path.exists(VS_BASE):
        return []
    
    collections = []
    for folder_name in os.listdir(VS_BASE):
        folder_path = os.path.join(VS_BASE, folder_name)
        if os.path.isdir(folder_path):
            meta_path = os.path.join(folder_path, "meta.json")
            if os.path.exists(meta_path):
                collections.append({
                    "id": folder_name,
                    "path": folder_path
                })
    return collections

def search_all_collections(query, embedder, topk_dense=20, final_k=5):
    """
    Search across all uploaded document collections.
    Returns combined results from all vector stores.
    """
    all_collections = get_all_collections()
    
    if not all_collections:
        print("DEBUG: No collections found in backend")
        return [], {}, 0.0
    
    print(f"DEBUG: Found {len(all_collections)} collection(s) to search")
    
    # Collect results from all collections
    all_results = []
    combined_meta = {}
    max_score = 0.0
    
    for collection in all_collections:
        try:
            print(f"DEBUG: Loading collection: {collection['id']}")
            vs = VectorStore(embedder)
            vs.load(collection['path'])
            
            print(f"DEBUG: Collection has {len(vs.meta)} chunks")
            
            # Search this collection
            results = vs.search_hybrid(query, topk_dense=topk_dense, final_k=final_k)
            print(f"DEBUG: Found {len(results)} results in this collection")
            
            # Add to combined results with collection prefix
            for chunk_id, score in results:
                # Create unique ID across all collections
                global_id = f"{collection['id']}::{chunk_id}"
                all_results.append((global_id, score))
                
                # Copy metadata with global ID
                if chunk_id in vs.meta:
                    combined_meta[global_id] = vs.meta[chunk_id].copy()
            
            # Track max score for confidence check
            if results:
                top_score = vs.top_dense_score(query)
                max_score = max(max_score, top_score)
                
        except Exception as e:
            print(f"DEBUG: Error loading collection {collection['id']}: {e}")
            continue
    
    # Sort all results by score and take top final_k
    all_results.sort(key=lambda x: x[1], reverse=True)
    top_results = all_results[:final_k]
    
    print(f"DEBUG: Combined results: {len(all_results)} total, returning top {len(top_results)}")
    print(f"DEBUG: Max confidence score: {max_score}")
    
    return top_results, combined_meta, max_score

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "UniMate API is running"})

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create a new chat session"""
    session_id, session = get_or_create_session()
    return jsonify({
        "session_id": session_id,
        "created_at": session["created_at"]
    })

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Upload PDF documents and create vector store"""
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    session_id, session = get_or_create_session(session_id)
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files provided"}), 400
    
    try:
        file_infos = []
        files_bytes = []
        
        for file in files:
            if file.filename.endswith('.pdf'):
                data = file.read()
                filepath = os.path.join(UPLOAD_DIR, file.filename)
                with open(filepath, "wb") as f:
                    f.write(data)
                file_infos.append((file.filename, len(data)))
                files_bytes.append((file.filename, data))
        
        if not file_infos:
            return jsonify({"error": "No valid PDF files"}), 400
        
        # Create collection ID
        collection_id = collection_id_from_file_infos(file_infos)
        session["collection_id"] = collection_id
        
        print(f"DEBUG UPLOAD: Set collection_id in session: {collection_id}")
        print(f"DEBUG UPLOAD: Session ID: {session_id}")
        
        vs_folder = os.path.join(VS_BASE, collection_id)
        idx_path = os.path.join(vs_folder, "index.faiss")
        meta_path = os.path.join(vs_folder, "meta.json")
        
        vs = VectorStore(embedder)
        
        # Check if already indexed
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            vs.load(vs_folder)
            status = "loaded_cache"
        else:
            # Build new index
            chunks = build_chunks(files_bytes)
            if not chunks:
                return jsonify({"error": "No text extracted from PDFs"}), 400
            
            vs.build(chunks)
            vs.save(vs_folder)
            status = "indexed"
        
        # Get document stats
        by_doc = {}
        for m in vs.meta.values():
            by_doc.setdefault(m["doc"], set()).add(int(m["page"]))
        
        doc_stats = [{"filename": doc, "pages": len(pages)} for doc, pages in by_doc.items()]
        
        return jsonify({
            "status": status,
            "collection_id": collection_id,
            "documents": doc_stats,
            "total_chunks": len(vs.meta)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Process a user query"""
    data = request.json
    session_id = data.get('session_id')
    user_query = data.get('query', '').strip()
    
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    if not user_query:
        return jsonify({"error": "query cannot be empty"}), 400
    
    session_id, session = get_or_create_session(session_id)
    
    # Add user message to history
    session["messages"].append({
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # Get conversation context
        conv_summary = conversation_summary(session)
        
        print(f"\n=== DEBUG START ===")
        print(f"DEBUG: User query: {user_query}")
        
        # Determine if generic query
        generic = is_generic_query(user_query)
        print(f"DEBUG: Query is generic: {generic}")
        
        context_block = ""
        pages = []
        source_meta = []
        use_general = generic
        
        # Search ALL collections (not just current session)
        if not generic:
            print("DEBUG: Searching ALL document collections...")
            
            fused, combined_meta, max_score = search_all_collections(
                user_query, 
                embedder, 
                topk_dense=TOPK_DENSE, 
                final_k=TOPK_FINAL
            )
            
            if fused and combined_meta:
                print(f"DEBUG: Found {len(fused)} results across all collections")
                print(f"DEBUG: Max confidence score: {max_score}")
                
                context_block, pages, source_meta = make_context(fused, combined_meta)
                print(f"DEBUG: Context block length: {len(context_block)}")
                print(f"DEBUG: Number of sources: {len(source_meta)}")
                
                # Fall back to general if low confidence
                use_general = (max_score < LOW_CONFIDENCE_THRESH) or (len(source_meta) == 0)
                print(f"DEBUG: Confidence threshold check - use_general: {use_general}")
            else:
                print("DEBUG: No documents found in backend")
                use_general = True
        else:
            print("DEBUG: Skipping document search (generic query)")
        
        print(f"DEBUG: Final decision - use_general: {use_general}")
        print(f"=== DEBUG END ===\n")
        
        # Build prompt
        prompt = build_prompt(user_query, conv_summary, context_block, general=use_general)
        
        # Generate answer
        raw_answer = llm.generate(prompt)
        print(f"DEBUG: Raw answer from LLM: {raw_answer[:200]}")

        # Handle errors
        if raw_answer.startswith("__LLM_ERROR__"):
            print(f"DEBUG: LLM Error detected: {raw_answer}")
            if (not use_general) and source_meta:
                final_answer = minimal_extractive_fallback([(list(combined_meta.keys())[0], 0.0)], combined_meta)
            else:
                final_answer = "Sorry, I couldn't generate an answer right now."
        else:
            final_answer = raw_answer.strip()
            
            # Retry with general if no answer found
            if (not use_general) and ("Not found in the document" in final_answer or len(final_answer) < 4):
                alt = llm.generate(build_prompt(user_query, conv_summary, context_block, general=True))
                if not alt.startswith("__LLM_ERROR__"):
                    final_answer = alt.strip()
                    pages = []
        
        # Add citations if using documents
        if (not use_general) and pages:
            final_answer = add_inline_citations(final_answer, pages)
        
        # Add assistant message to history
        session["messages"].append({
            "role": "assistant",
            "content": final_answer,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "answer": final_answer,
            "sources": source_meta if not use_general else [],
            "used_documents": not use_general,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"ERROR in query: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 400
    
    session = sessions[session_id]
    return jsonify({
        "messages": session["messages"],
        "collection_id": session.get("collection_id")
    })

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear session history"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 400
    
    sessions[session_id]["messages"] = []
    return jsonify({"status": "cleared"})

@app.route('/api/collections', methods=['GET'])
def list_collections():
    """List all available document collections"""
    collections = get_all_collections()
    
    collection_info = []
    for collection in collections:
        try:
            vs = VectorStore(embedder)
            vs.load(collection['path'])
            
            # Get document names
            docs = set()
            for meta in vs.meta.values():
                docs.add(meta['doc'])
            
            collection_info.append({
                "id": collection['id'],
                "documents": list(docs),
                "total_chunks": len(vs.meta)
            })
        except Exception as e:
            print(f"Error loading collection {collection['id']}: {e}")
            continue
    
    return jsonify({
        "collections": collection_info,
        "total_collections": len(collection_info)
    })

if __name__ == '__main__':
    print("🚀 UniMate API Server Starting...")
    print("📚 Make sure your .env file contains GOOGLE_API_KEY")
    app.run(host='0.0.0.0', port=5000, debug=True)