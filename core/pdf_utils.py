from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
from .config import UPLOAD_DIR

@dataclass
class Chunk:
    id: str
    text: str
    page: int
    doc: str

def extract_pdf_text(file_bytes: bytes) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text("text")))
    return pages

def paragraph_chunk(page_text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buff = ""
    for p in paras:
        if len(buff) + len(p) + 1 <= max_chars:
            buff = (buff + "\n\n" + p).strip() if buff else p
        else:
            if buff:
                chunks.append(buff)
            carry = buff[-overlap:] if buff else ""
            buff = (carry + "\n\n" + p).strip()
            while len(buff) > max_chars:
                chunks.append(buff[:max_chars])
                buff = (buff[max_chars - overlap :]).strip()
    if buff:
        chunks.append(buff)
    return chunks

def build_chunks(files: List[Tuple[str, bytes]]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for fname, fbytes in files:
        for page_no, txt in extract_pdf_text(fbytes):
            for i, piece in enumerate(paragraph_chunk(txt)):
                cid = hashlib.md5((fname + str(page_no) + str(i)).encode()).hexdigest()[:12]
                all_chunks.append(Chunk(id=cid, text=piece, page=page_no, doc=fname))
    return all_chunks

def render_pdf_page_image(doc_name: str, page_no: int, zoom: float = 1.5) -> Optional[bytes]:
    path = f"{UPLOAD_DIR}/{doc_name}"
    try:
        with fitz.open(path) as d:
            idx = max(0, min(page_no - 1, d.page_count - 1))
            pix = d.load_page(idx).get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return pix.tobytes("png")
    except Exception:
        return None
