from __future__ import annotations
import os
import hashlib
from dotenv import load_dotenv
import google.generativeai as genai

# App
APP_TITLE = "UniMate – AI University Assistant"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS_BASE = os.path.join(BASE_DIR, "backend", "vector_store")
HIST_BASE = os.path.join(BASE_DIR, "backend", "history")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploaded_files")

# Models / constants
EMB_MODEL_NAME = "models/text-embedding-004"
GEMINI_DEFAULT = "gemini-2.5-flash"
TOPK_DENSE = 20
TOPK_FINAL = 5
LOW_CONFIDENCE_THRESH = 0.25
MAX_SESSION_SUMMARY_TURNS = 10
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4

THREADS_PATH = os.path.join(HIST_BASE, "threads.json")

def ensure_dirs() -> None:
    for p in (VS_BASE, HIST_BASE, UPLOAD_DIR):
        os.makedirs(p, exist_ok=True)

def load_env() -> dict:
    load_dotenv(override=True)

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in .env")
    genai.configure(api_key=api_key)
    return {
        "GOOGLE_API_KEY": api_key,
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", GEMINI_DEFAULT),
    }

def collection_id_from_file_infos(file_infos: list[tuple[str, int]]) -> str:
    """Hash of (filename, size) pairs -> stable collection id."""
    h = hashlib.sha256()
    for name, size in sorted(file_infos):
        h.update(name.encode("utf-8"))
        h.update(str(size).encode("utf-8"))
    return h.hexdigest()[:16]
