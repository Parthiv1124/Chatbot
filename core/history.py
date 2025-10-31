from __future__ import annotations
from typing import List, Dict, Optional
import json, os, time, uuid
from .config import THREADS_PATH, MAX_SESSION_SUMMARY_TURNS

def _read_threads() -> List[Dict]:
    if not os.path.exists(THREADS_PATH):
        return []
    try:
        with open(THREADS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_threads(data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(THREADS_PATH), exist_ok=True)
    with open(THREADS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_thread(title: str, collection_id: Optional[str]) -> str:
    threads = _read_threads()
    tid = uuid.uuid4().hex[:12]
    threads.append({"id": tid, "title": title, "collection_id": collection_id, "messages": [], "created": time.time()})
    _write_threads(threads)
    return tid

def append_message(tid: str, role: str, content: str) -> None:
    threads = _read_threads()
    for t in threads:
        if t["id"] == tid:
            t["messages"].append({"role": role, "content": content, "ts": time.time()})
            break
    _write_threads(threads)

def get_thread(tid: str) -> Optional[Dict]:
    for t in _read_threads():
        if t["id"] == tid:
            return t
    return None

def set_thread_title(tid: str, new_title: str) -> None:
    threads = _read_threads()
    for t in threads:
        if t["id"] == tid:
            t["title"] = new_title
            break
    _write_threads(threads)

def set_thread_collection(tid: str, collection_id: Optional[str]) -> None:
    threads = _read_threads()
    for t in threads:
        if t["id"] == tid:
            t["collection_id"] = collection_id
            break
    _write_threads(threads)

def list_threads() -> List[Dict]:
    return sorted(_read_threads(), key=lambda x: x.get("created", 0), reverse=True)

def update_thread_title_if_empty(tid: str, fallback_title: str) -> None:
    threads = _read_threads()
    for t in threads:
        if t["id"] == tid and (not t["title"] or t["title"].strip().lower() == "new topic"):
            t["title"] = fallback_title
            break
    _write_threads(threads)

def conversation_summary_for_prompt(tid: str) -> str:
    t = get_thread(tid)
    if not t:
        return ""
    msgs = [m for m in t["messages"] if m["role"] in ("user", "assistant")]
    items: List[str] = []
    for m in msgs[-MAX_SESSION_SUMMARY_TURNS:]:
        role = "Q" if m["role"] == "user" else "A"
        snippet = m["content"].replace("\n", " ")[:80]
        items.append(f"{role}: {snippet}")
    return " ; ".join(items)
