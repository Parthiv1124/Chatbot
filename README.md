<<<<<<< HEAD
## 📚 UniMate – AI University Assistant

UniMate is an AI-powered assistant that lets you query your PDF documents in natural language. Upload reports, notes, or resumes; ask a question; and get a clear, well-formatted answer grounded in your files. If the answer is not in your PDFs, UniMate falls back to a high‑quality general answer using Google Gemini.

## ✨ Features

- 📂 Multi-PDF support across all uploaded documents
- 🔍 Hybrid retrieval (FAISS dense vectors + BM25 lexical)
- 🧠 LLM answers via Google Gemini 1.5‑Flash; `text-embedding-004` for embeddings
- 💬 Chat-style Streamlit UI with conversational history
- 📝 Clean formatting with numbered lists, bullets, and inline citations (page numbers)
- 🗂 Topic management with persistent history
- ⚡ Vector index caching for previously uploaded PDFs
- 🔐 Secure .env-based configuration (keys not tracked in git)

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – web app framework
- [Google Gemini API](https://ai.google.dev/) – LLM + embeddings
- [FAISS](https://github.com/facebookresearch/faiss) – dense vector search
- [rank-bm25](https://pypi.org/project/rank-bm25/) – lexical retrieval
- [PyMuPDF](https://pymupdf.readthedocs.io/) – PDF parsing
- [regex](https://pypi.org/project/regex/) – text cleanup and formatting

## 📦 Project Structure

```
unimate-ai-assistant/
├─ app.py                   # Streamlit entrypoint
├─ core/                    # Core logic (backend code)
│  ├─ config.py             # Constants, paths, env loader
│  ├─ pdf_utils.py          # PDF parsing + chunking
│  ├─ embeddings.py         # Gemini embeddings
│  ├─ vector_store.py       # FAISS + BM25 hybrid search
│  ├─ llm.py                # Gemini text generation
│  ├─ retrieval.py          # Prompting & context builder
│  ├─ formatting.py         # Answer formatting utilities
│  └─ history.py            # Persistent chat history/topics
├─ backend/                 # Stored data
│  ├─ vector_store/         # Cached FAISS indices
│  └─ history/              # Saved chat threads
├─ data/
│  └─ uploaded_files/       # Uploaded PDFs
├─ requirements.txt         # Python dependencies
└─ .env                     # Your Gemini API key (not tracked in git)
```

## 🚀 Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/anshtrivediaiml/UniMate-AI-University-Assistant.git
cd UniMate-AI-University-Assistant
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On Linux / macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5) Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## 🔒 Security

- Never commit `.env` files or API keys.
- The repository’s `.gitignore` is configured to keep secrets and cache files out of Git.
=======
# Chatbot
>>>>>>> 57cd13efd4472f58a20c09860084c49bd82a452f
