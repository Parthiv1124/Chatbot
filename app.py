from __future__ import annotations
import os, time
import streamlit as st

from core.config import (
    APP_TITLE, ensure_dirs, load_env, VS_BASE, UPLOAD_DIR,
    LOW_CONFIDENCE_THRESH, TOPK_DENSE, TOPK_FINAL, collection_id_from_file_infos
)
# NOTE: page preview removed — no import from pdf_utils
from core.embeddings import GeminiEmbedder
from core.vector_store import VectorStore
from core.llm import GeminiLLM
from core.retrieval import (
    is_generic_query, make_context, build_prompt, add_inline_citations, minimal_extractive_fallback
)
from core.formatting import prettify_answer
from core.history import (
    list_threads, create_thread, get_thread, set_thread_title, set_thread_collection,
    append_message, update_thread_title_if_empty, conversation_summary_for_prompt
)

# ---------- helpers ----------
def set_active_topic(tid: str | None):
    st.session_state["active_tid"] = tid
    try:
        if tid is None:
            st.query_params.from_dict({k: v for k, v in dict(st.query_params).items() if k != "tid"})
        else:
            st.query_params["tid"] = tid
    except Exception:
        pass


def message_card(role: str, content: str):
    """Render one chat message as a rounded card; content is Markdown."""
    is_user = (role == "user")
    icon = "🧑‍💻" if is_user else "🤖"
    role_label = "User" if is_user else "Assistant"
    bg = "#0B1220" if is_user else "#0F172A"
    border = "#1F2A44" if is_user else "#263046"

    st.markdown(
        f"""
<div class="msg-card" style="
  margin: 10px 0 16px 0;
  border: 1px solid {border};
  background: {bg};
  border-radius: 14px;
  padding: 10px 14px 6px 14px;">
  <div style="opacity:.85;margin-bottom:6px">{icon} {role_label}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    # content as real Markdown so lists render
    st.markdown(content)
    # a little spacer to separate Q/A pairs
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ---------- app ----------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    ensure_dirs()

    try:
        cfg = load_env()
    except RuntimeError as e:
        st.error(str(e)); st.stop()

    # Global CSS
    st.markdown(
        """
<style>
.block-container { padding-top: 1.0rem; max-width: 1120px; }
.topic-card { background:#0b1220; border:1px solid #1E293B; color:#cbd5e1;
              border-radius:10px; padding:.15rem .35rem; margin:.35rem 0; }
.topic-card.active { background:#111827; border-color:#334155; color:#e5e7eb; }
/* Fix title overlap with navbar */
h2 { margin-top: 2rem !important; padding-top: 1rem !important; }
/* Ensure proper spacing for the main title */
.main-title { margin-top: 2rem !important; padding-top: 1rem !important; }
/* Additional spacing to prevent navbar overlap */
.stApp > header { visibility: hidden; }
.stApp > div:first-child { padding-top: 0rem; }
/* Ensure main content has proper top margin */
.main .block-container { padding-top: 2rem !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h2 class='main-title' style='margin-bottom:0;'>📚 UniMate – AI University Assistant</h2>"
        "<p style='color:#9ca3af;margin-top:4px'>Ask concise questions about your PDFs. "
        "If not in the PDFs, UniMate provides a high-quality general answer.</p>",
        unsafe_allow_html=True,
    )

    # Session init
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        url_tid = None
        try:
            url_tid = st.query_params.get("tid")
        except Exception:
            pass
        st.session_state["active_tid"] = url_tid if url_tid else None
        st.session_state.setdefault("rename_open_tid", None)
        st.session_state.setdefault("last_sources", [])
        st.session_state.setdefault("last_sources_tid", None)

    # Embedder (cached)
    @st.cache_resource(show_spinner=False)
    def get_embedder():
        return GeminiEmbedder()

    embedder = get_embedder()

    # ---------- Sidebar: topics ----------
    with st.sidebar:
        st.subheader("📁 Chats")
        threads = list_threads()

        if st.button("➕ New Topic", use_container_width=True):
            new_tid = create_thread("New topic", None)
            set_active_topic(new_tid)
            st.rerun()

        rename_open_tid = st.session_state.get("rename_open_tid")

        for t in threads:
            is_active = (t["id"] == st.session_state.get("active_tid"))
            css_class = "topic-card active" if is_active else "topic-card"
            with st.container():
                st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                c1, c2 = st.columns([0.82, 0.18], vertical_alignment="center")
                with c1:
                    if st.button(t["title"] or "New topic", key=f"topic_{t['id']}", use_container_width=True):
                        set_active_topic(t["id"]); st.rerun()
                with c2:
                    if st.button("✎", key=f"edit_{t['id']}", help="Rename", use_container_width=True):
                        st.session_state["rename_open_tid"] = (None if rename_open_tid == t["id"] else t["id"])
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

                if st.session_state.get("rename_open_tid") == t["id"]:
                    with st.form(f"rename_form_{t['id']}"):
                        new_name = st.text_input("Rename topic", value=t["title"] or "New topic", label_visibility="collapsed")
                        col1, col2 = st.columns(2)
                        save = col1.form_submit_button("Save")
                        cancel = col2.form_submit_button("Cancel")
                    if save and new_name.strip():
                        set_thread_title(t["id"], new_name.strip())
                        st.session_state["rename_open_tid"] = None; st.rerun()
                    if cancel:
                        st.session_state["rename_open_tid"] = None; st.rerun()

    # ---------- Layout ----------
    left, right = st.columns([0.66, 0.34], gap="large")

    # ---------- Right: documents ----------
    with right:
        st.subheader("📄 Documents")
        tid = st.session_state.get("active_tid")
        thread = get_thread(tid) if tid else None
        active_collection = thread["collection_id"] if thread else None

        nonce_key = f"upl_nonce_{tid or 'none'}"
        nonce = st.session_state.get(nonce_key, 0)
        uploads = st.file_uploader(
            "Upload 1..N PDFs (attached to this topic)",
            type=["pdf"], accept_multiple_files=True,
            key=f"uploader_{tid or 'none'}_{nonce}"
        )

        if uploads and tid:
            try:
                file_infos, files_bytes = [], []
                for up in uploads:
                    data = up.read()
                    with open(os.path.join(UPLOAD_DIR, up.name), "wb") as f:
                        f.write(data)
                    file_infos.append((up.name, len(data)))
                    files_bytes.append((up.name, data))

                collection_id = collection_id_from_file_infos(file_infos)
                set_thread_collection(tid, collection_id)
                vs_folder = os.path.join(VS_BASE, collection_id)

                vs = VectorStore(embedder)
                idx_path = os.path.join(vs_folder, "index.faiss")
                meta_path = os.path.join(vs_folder, "meta.json")

                with st.status("Indexing your documents…", expanded=True) as s:
                    if os.path.exists(idx_path) and os.path.exists(meta_path):
                        s.write("Loading cached vector index…")
                        vs.load(vs_folder)
                    else:
                        s.write("Extracting text & chunking…")
                        from core.pdf_utils import build_chunks  # local import to avoid confusion
                        chunks = build_chunks(files_bytes)
                        if not chunks:
                            st.error("No text extracted from PDFs.")
                            s.update(label="Failed to index", state="error")
                            return
                        s.write("Embedding & building FAISS…")
                        vs.build(chunks)
                        s.write("Saving index for reuse…")
                        vs.save(vs_folder)
                    s.update(label="Documents processed ✅", state="complete", expanded=False)

                st.toast("Documents processed ✅")
            finally:
                st.session_state[nonce_key] = nonce + 1
                st.rerun()

        if active_collection:
            vs_folder = os.path.join(VS_BASE, active_collection)
            if os.path.exists(os.path.join(vs_folder, "meta.json")):
                tmp_vs = VectorStore(embedder); tmp_vs.load(vs_folder)
                by_doc: dict[str, set[int]] = {}
                for m in tmp_vs.meta.values():
                    by_doc.setdefault(m["doc"], set()).add(int(m["page"]))
                for doc, pages in by_doc.items():
                    st.write(f"- **{doc}** · {len(pages)} pages indexed")
            else:
                st.caption("No documents indexed for this topic yet.")
        else:
            st.caption("No documents indexed for this topic yet.")

        # Page preview REMOVED as requested

    # ---------- Left: chat ----------
    with left:
        st.subheader("💬 Chat")
        tid = st.session_state.get("active_tid")
        thread = get_thread(tid) if tid else None

        if thread and thread["messages"]:
            # Render messages as spaced cards
            for m in thread["messages"]:
                message_card(m["role"], m["content"])
        else:
            st.caption("This topic is empty. Ask the first question to begin.")

        # Sources (no preview buttons)
        if st.session_state.get("last_sources_tid") == tid and st.session_state.get("last_sources"):
            with st.expander("Sources (pages / documents)"):
                for m in st.session_state["last_sources"]:
                    st.write(f"- **{m['doc']}** (page {m['page']}) · score={m['score']}")
            st.session_state["last_sources"] = []
            st.session_state["last_sources_tid"] = None

        # Input at the bottom (always stays below messages)
        with st.form("ask_form", clear_on_submit=True):
            user_q = st.text_area("Ask a question", placeholder="Type your question…", height=90, disabled=(tid is None))
            send = st.form_submit_button("Send ➤", disabled=(tid is None))

        if tid is None and send:
            st.warning("Create or select a topic first.")
        elif send and user_q.strip():
            append_message(tid, "user", user_q)
            conv_summary = conversation_summary_for_prompt(tid)

            # Load VS for this topic (if exists)
            vs = None; pages = []; source_meta = []; context_block = ""
            active = get_thread(tid)
            collection_id = active["collection_id"] if active else None
            if collection_id:
                vs_folder = os.path.join(VS_BASE, collection_id)
                if os.path.exists(os.path.join(vs_folder, "meta.json")):
                    @st.cache_resource(show_spinner=False)
                    def load_vs_cached(path: str):
                        v = VectorStore(embedder); v.load(path); return v
                    vs = load_vs_cached(vs_folder)

            generic = is_generic_query(user_q)
            use_general = generic

            if (not generic) and vs is not None:
                with st.status("🔎 Searching your documents…", expanded=True) as s:
                    s.write("Hybrid search • BM25 + dense")
                    fused = vs.search_hybrid(user_q, topk_dense=TOPK_DENSE, final_k=TOPK_FINAL)
                    top_dense = vs.top_dense_score(user_q)
                    context_block, pages, source_meta = make_context(fused, vs.meta)
                    s.update(label="Search complete ✅", state="complete", expanded=False)
                use_general = (top_dense < LOW_CONFIDENCE_THRESH) or (len(source_meta) == 0)

            prompt = build_prompt(user_q, conv_summary, context_block, general=use_general)
            llm = GeminiLLM(cfg["GEMINI_MODEL"])

            with st.status("💡 Generating answer…", expanded=False):
                raw = llm.generate(prompt)

            if raw.startswith("__LLM_ERROR__"):
                if (not use_general) and vs is not None and source_meta:
                    final = minimal_extractive_fallback([(list(vs.meta.keys())[0], 0.0)], vs.meta)
                else:
                    final = "Sorry, I couldn't generate an answer right now."
            else:
                final = raw.strip()
                if (not use_general) and ("Not found in the document" in final or len(final) < 4):
                    alt = llm.generate(build_prompt(user_q, conv_summary, context_block, general=True))
                    if not alt.startswith("__LLM_ERROR__"):
                        final = alt.strip(); pages = []

            if (not use_general) and pages:
                final = add_inline_citations(final, pages)
            final = prettify_answer(final)

            append_message(tid, "assistant", final)
            update_thread_title_if_empty(tid, fallback_title=user_q[:48])

            st.session_state["last_sources"] = source_meta if (not use_general) else []
            st.session_state["last_sources_tid"] = tid
            st.rerun()

    st.markdown("---")
    st.caption("Powered by FAISS + BM25 + Gemini (text-embedding-004 + 1.5-flash)")


if __name__ == "__main__":
    main()