from __future__ import annotations
from typing import Tuple, List, Dict
import os, json, numpy as np, faiss
from rank_bm25 import BM25Okapi
from .embeddings import GeminiEmbedder
from .pdf_utils import Chunk
from .config import DENSE_WEIGHT, BM25_WEIGHT

class VectorStore:
    def __init__(self, embedder: GeminiEmbedder):
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.ids: list[str] = []
        self.meta: Dict[str, Dict] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_tokens: list[list[str]] = []

    def build(self, chunks: List[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embs = self.embedder.encode(texts)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.ids = [c.id for c in chunks]
        self.meta = {c.id: {"text": c.text, "page": c.page, "doc": c.doc} for c in chunks}
        self._bm25_tokens = [m["text"].lower().split() for m in self.meta.values()]
        self._bm25 = BM25Okapi(self._bm25_tokens)

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index not built")
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "meta": self.meta}, f, ensure_ascii=False)

    def load(self, folder: str) -> None:
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "meta.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.meta = data["meta"]
        self._bm25_tokens = [m["text"].lower().split() for m in self.meta.values()]
        self._bm25 = BM25Okapi(self._bm25_tokens)

    def _dense(self, q: str, k: int) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        D, I = self.index.search(self.embedder.encode([q]), k)
        return list(zip(I[0].tolist(), [float(s) for s in D[0].tolist()]))

    def _bm25_search(self, q: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(q.lower().split())
        idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idxs]

    @staticmethod
    def _minmax(m: Dict[int, float]) -> Dict[int, float]:
        if not m:
            return {}
        vals = list(m.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 0.0 for k in m}
        return {k: (v - mn) / (mx - mn) for k, v in m.items()}

    def search_hybrid(self, q: str, topk_dense=20, final_k=5) -> list[tuple[str, float]]:
        d = self._dense(q, topk_dense)
        b = self._bm25_search(q, topk_dense)
        d_map, b_map = {i: s for i, s in d}, {i: s for i, s in b}
        d_norm, b_norm = self._minmax(d_map), self._minmax(b_map)
        cand = set(d_map) | set(b_map)
        fused = [(i, 0.0 + DENSE_WEIGHT * d_norm.get(i, 0) + BM25_WEIGHT * b_norm.get(i, 0)) for i in cand]
        fused.sort(key=lambda x: x[1], reverse=True)
        res: list[tuple[str, float]] = []
        for idx, score in fused[:final_k]:
            if 0 <= idx < len(self.ids):
                res.append((self.ids[idx], float(score)))
        return res

    def top_dense_score(self, q: str) -> float:
        hits = self._dense(q, 1)
        return hits[0][1] if hits else 0.0
