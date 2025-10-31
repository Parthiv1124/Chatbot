from __future__ import annotations
import numpy as np
import faiss
import google.generativeai as genai
from .config import EMB_MODEL_NAME

class GeminiEmbedder:
    def __init__(self, model_name: str = EMB_MODEL_NAME):
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for t in texts:
            if not t or not t.strip():
                vecs.append(np.zeros(768, dtype="float32"))
                continue
            try:
                out = genai.embed_content(model=self.model_name, content=t)
                v = np.array(out["embedding"], dtype="float32")
                vecs.append(v)
            except Exception:
                vecs.append(np.zeros(768, dtype="float32"))
        arr = np.vstack(vecs)
        faiss.normalize_L2(arr)
        return arr
