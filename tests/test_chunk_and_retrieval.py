# tests/test_chunk_and_retrieval.py
# Minimal deterministic tests without heavy models
# We stub embeddings to avoid downloading models in CI.

import json
import numpy as np
import faiss

from app import paragraph_chunk, VectorStore, Chunk


class FakeEmb:
    def encode(
        self,
        texts,
        batch_size=0,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        # Deterministic hash -> vector
        vecs = []
        for t in texts:
            h = abs(hash(t)) % (10**6)
            rng = np.random.default_rng(h)
            v = rng.random(384).astype("float32")
            if normalize_embeddings:
                faiss.normalize_L2(v.reshape(1, -1))
            vecs.append(v)
        return np.stack(vecs)


def test_paragraph_chunk_deterministic():
    txt = ("Para1\n\n" + "x" * 500 + "\n\n" + "Para2 " * 50)
    a = paragraph_chunk(txt, max_chars=200, overlap=50)
    b = paragraph_chunk(txt, max_chars=200, overlap=50)
    assert a == b
    assert all(len(c) <= 200 for c in a)


def test_vectorstore_search_stable(tmp_path):
    # Build tiny collection
    chunks = [
        Chunk(id="c1", text="machine learning introduction", page=1, doc="a.pdf"),
        Chunk(id="c2", text="deep learning with neural networks", page=2, doc="a.pdf"),
        Chunk(id="c3", text="classical statistics and probability", page=3, doc="b.pdf"),
    ]
    vs = VectorStore(FakeEmb())
    vs.build(chunks)

    # Save/Load roundtrip
    folder = tmp_path / "vs"
    folder.mkdir()
    faiss.write_index(vs.index, str(folder / "index.faiss"))
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"ids": vs.ids, "meta": vs.meta}, f)

    # Reload
    vs2 = VectorStore(FakeEmb())
    vs2.load(str(folder))

    res = vs2.search("neural networks", topk=2)
    # Expect c2 among top results
    ids = [cid for cid, _ in res]
    assert "c2" in ids
