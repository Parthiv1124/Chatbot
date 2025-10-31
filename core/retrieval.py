from __future__ import annotations
from typing import List, Tuple, Dict
import re

SYSTEM_PROMPT = (
    "You are UniMate, a precise academic assistant.\n"
    'Use ONLY the provided sources to answer the user. If the answer is not contained in the sources, say: "Not found in the document."\n'
    "Return a short, direct answer in clean **Markdown**.\n"
    "If you must list multiple items, format them as a multi-line **numbered list** (1., 2., 3.).\n"
    "If a numbered item contains multiple sub-points, render them as **sub-bullets**.\n"
    "Avoid inline lists and long single paragraphs.\n"
    "Cite sources as (page X)."
)

GENERAL_PROMPT_PREFIX = (
    "You are UniMate, a helpful academic assistant. The answer may not be in the user's PDFs; "
    "provide a concise, high-quality general answer. Use clean Markdown with multi-line lists when applicable."
)

GENERIC_PATTERNS = re.compile(
    r"\b(what\s+is|who\s+is|define|definition\s+of|when\s+is|where\s+is|capital\s+of|pm\s+of|president\s+of|meaning\s+of)\b",
    re.IGNORECASE,
)

def is_generic_query(q: str) -> bool:
    # If query mentions document/pdf/file/uploaded, it's NOT generic
    if re.search(r"\b(document|pdf|file|uploaded|software_testing|according to|in the)\b", q, re.IGNORECASE):
        return False
    
    if GENERIC_PATTERNS.search(q):
        if re.search(r"\b(according to|in the (document|report|pdf|paper|syllabus))\b", q, re.IGNORECASE):
            return False
        return True
    return len(q.split()) <= 5 and q.strip().endswith("?")

def make_context(ranked: List[Tuple[str, float]], meta: Dict[str, Dict], token_cap: int = 1800):
    ctx_parts: List[str] = []
    pages: List[int] = []
    metas: List[Dict] = []
    char_budget = token_cap * 3
    used = 0
    for cid, score in ranked:
        m = meta[cid]
        snippet = m["text"].strip()
        block = f"[Source | page {m['page']} | {m['doc']}]\n{snippet}\n"
        if used + len(block) > char_budget:
            break
        used += len(block)
        ctx_parts.append(block)
        pages.append(int(m["page"]))
        metas.append({"page": m["page"], "doc": m["doc"], "score": round(float(score), 4)})
    return "\n\n".join(ctx_parts), pages, metas

def build_prompt(user_q: str, conversation_summary: str, context_block: str, general: bool = False) -> str:
    if general:
        return (
            f"{GENERAL_PROMPT_PREFIX}\n\n"
            f"Conversation summary: {conversation_summary}\n\n"
            f"Question: {user_q}\n"
        )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Conversation summary: {conversation_summary}\n\n"
        f"Sources:\n{context_block}\n\n"
        f"Question: {user_q}\n"
        f"Answer:"
    )

def add_inline_citations(answer: str, pages: List[int]) -> str:
    pages = sorted(set(int(p) for p in pages))
    if not pages or "(page" in answer:
        return answer
    if len(pages) == 1:
        return f"{answer} (page {pages[0]})"
    return f"{answer} (pages {', '.join(map(str, pages))})"

def minimal_extractive_fallback(ranked: List[Tuple[str, float]], meta: Dict[str, Dict]) -> str:
    if not ranked:
        return "Sorry, I couldn't generate an answer right now. Please try again."
    best_id = ranked[0][0]
    m = meta[best_id]
    snippet = m["text"].strip()
    pg = m["page"]
    return f"Not found in the document. Closest relevant excerpt:\n\n{snippet}\n\n(page {pg})"
