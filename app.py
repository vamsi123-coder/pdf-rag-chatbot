import os
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from rag_utils import process_multiple_pdfs
from vector_store import vector_store, FAISS_INDEX_DIR
from langgraph_agent import ask

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocChat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready: bool = False
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: List[str] = []
if "input_key" not in st.session_state:
    # Incrementing this forces the text_input to re-render empty after send
    st.session_state.input_key: int = 0

if not st.session_state.index_ready:
    if vector_store.load(FAISS_INDEX_DIR):
        st.session_state.index_ready = True


def parse_response(raw: str) -> Tuple[str, str]:
    if "[SOURCES]" in raw and "[/SOURCES]" in raw:
        parts = raw.split("[SOURCES]", 1)
        return parts[0].strip(), parts[1].replace("[/SOURCES]", "").strip()
    return raw.strip(), ""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-label">Documents</span>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Build index", use_container_width=True):
            with st.spinner("Building index…"):
                temp_paths: List[str] = []
                try:
                    for uf in uploaded_files:
                        suffix = Path(uf.name).suffix or ".pdf"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.read())
                            temp_paths.append(tmp.name)

                    documents = process_multiple_pdfs(temp_paths)

                    if not documents:
                        st.error("No extractable text found in the uploaded PDFs.")
                    else:
                        vector_store.build_from_documents(documents)
                        vector_store.save(FAISS_INDEX_DIR)
                        st.session_state.index_ready = True
                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                        st.success(f"Done — {len(documents)} chunks indexed.")

                except Exception as exc:
                    logger.error("Index build failed: %s", exc)
                    st.error(f"Failed: {exc}")
                finally:
                    for tp in temp_paths:
                        try:
                            os.unlink(tp)
                        except OSError:
                            pass

    st.markdown("---")
    st.markdown('<span class="sidebar-label">Status</span>', unsafe_allow_html=True)

    if st.session_state.index_ready:
        st.markdown('<span class="pill pill-green">● Index ready</span>', unsafe_allow_html=True)
        if st.session_state.indexed_files:
            st.markdown("<br>", unsafe_allow_html=True)
            for fname in st.session_state.indexed_files:
                st.markdown(f'<div class="file-chip">📄 {fname}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-yellow">○ No index</span>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Clear chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-row">
    <div class="title-icon">💬</div>
    <h1 class="page-title">DocChat</h1>
</div>
<p class="page-subtitle">Ask questions about your uploaded PDF documents</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Chat history ───────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    if not st.session_state.index_ready:
        st.markdown('<div class="empty-state"><div class="icon">📂</div><p>Upload PDFs in the sidebar and click <strong>Build index</strong> to get started.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><div class="icon">💬</div><p>Index is ready. Ask a question below.</p></div>', unsafe_allow_html=True)
else:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f'<div class="msg-user"><div class="bubble-u">{content}</div></div>', unsafe_allow_html=True)
        else:
            answer_body, sources_text = parse_response(content)

            source_chips_html = ""
            if sources_text:
                for line in sources_text.split("\n"):
                    line = line.strip()
                    if line:
                        source_chips_html += f'<div class="source-chip"><span class="source-chip-icon">📄</span>{line}</div>'

            st.markdown(
                f'<div class="msg-ai"><div><div class="bubble-b">{answer_body}</div>{source_chips_html}</div></div>',
                unsafe_allow_html=True,
            )

# ── Chat input ─────────────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_btn = st.columns([5, 1])

with col_input:
    # Key changes on every send → forces Streamlit to re-render the input empty
    user_input = st.text_input(
        "Question",
        key=f"user_input_{st.session_state.input_key}",
        placeholder="Ask a question about your documents…",
        label_visibility="collapsed",
    )

with col_btn:
    send_clicked = st.button("Send", use_container_width=True)

if send_clicked and user_input and user_input.strip():
    if not st.session_state.index_ready:
        st.warning("Please upload PDFs and build the index first.")
    else:
        question = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            try:
                raw_answer = ask(question)
            except Exception as exc:
                logger.error("Error: %s", exc)
                raw_answer = f"An error occurred: {exc}"

        st.session_state.chat_history.append({"role": "assistant", "content": raw_answer})

        # Increment key → text_input re-renders blank on next run
        st.session_state.input_key += 1
        st.rerun()
