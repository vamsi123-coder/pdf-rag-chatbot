<div align="center">

# 💬 DocChat — PDF RAG Chatbot

**Ask questions about your PDF documents using the power of AI**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

</div>

---

## 📌 Overview

**DocChat** is a Retrieval-Augmented Generation (RAG) chatbot that lets you upload one or more PDF documents and ask questions about their content. It uses **Google Gemini** for language generation, **FAISS** for fast vector similarity search, **HuggingFace embeddings** for semantic understanding, and **LangGraph** to orchestrate the pipeline as a stateful agent graph.

Answers are grounded strictly in the content of your documents — if the answer isn't there, the model says so.

---

## ✨ Features

- 📄 **Multi-PDF support** — upload and query multiple PDFs at once
- 🔍 **Semantic search** — HuggingFace `all-MiniLM-L6-v2` embeddings for accurate retrieval
- 🤖 **Gemini 2.5 Flash** — fast, grounded answers via Google's latest model
- 🗺️ **LangGraph pipeline** — three-node graph: `retrieve → build_context → generate`
- 💾 **Persistent index** — FAISS index saved to disk and auto-loaded on startup
- 📎 **Source citations** — every answer includes the source file and page number
- 🚫 **Hallucination guardrails** — relevance-score filtering keeps answers grounded in retrieved context
- 🎨 **Clean UI** — Streamlit chat interface with auto-clearing input

---

## 🏗️ Architecture

```
               User Question
                    │
                    ▼
┌──────────────────────────────────────────────┐
│                LangGraph Agent               │
│                                              │
│  retrieve  →  build_context  →  generate     │
│     │                              │         │
│  FAISS search                  Gemini API    │
│  (HuggingFace embeddings)                    │
└──────────────────────────────────────────────┘
                    │
                    ▼
        Answer + Source Citations
```

**Pipeline steps**

1. **Upload** — PDFs are uploaded via the Streamlit sidebar
2. **Process** — text is extracted, cleaned, and split into chunks (`RecursiveCharacterTextSplitter`)
3. **Embed** — chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`
4. **Index** — embeddings are stored in a FAISS index and persisted to disk
5. **Retrieve** — top-K similar chunks are fetched per query and filtered by relevance score
6. **Generate** — Gemini 2.5 Flash answers using only the retrieved context
7. **Cite** — source filename and page number are appended below the answer

---

## 🗂️ Project Structure

```
pdf-rag-chatbot/
├── app.py                # Streamlit UI — chat interface, sidebar, layout
├── rag_utils.py          # PDF loading, text cleaning, chunking
├── vector_store.py       # FAISS index creation, saving, loading, search
├── langgraph_agent.py    # LangGraph pipeline — retrieve, context, generate
├── style.css             # Custom CSS (UI theme)
├── .streamlit/
│   └── config.toml       # Streamlit theme configuration
├── requirements.txt      # Python dependencies
├── .env.example          # Sample environment file (copy to .env)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/vamsi123-coder/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Copy `.env.example` to `.env` and add your key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ Usage

1. **Upload PDFs** — drag and drop or browse for files in the sidebar
2. **Build Index** — click **Build index** to process and embed the documents
3. **Ask Questions** — type a question in the chat input and press **Send**
4. **View Sources** — each answer shows its source file and page number
5. **Clear Chat** — use the **Clear chat** button to start a new session

> The FAISS index is saved automatically after building and reloaded on the next startup, so you don't need to re-index every run.

---

## ⚙️ Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `CHUNK_SIZE` | `rag_utils.py` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `rag_utils.py` | `200` | Overlap between chunks |
| `EMBEDDING_MODEL_NAME` | `vector_store.py` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `TOP_K_RESULTS` | `vector_store.py` | `5` | Chunks retrieved per query |
| `MIN_RELEVANCE_SCORE` | `langgraph_agent.py` | `0.2` | Minimum cosine similarity threshold |
| `GEMINI_MODEL` | `langgraph_agent.py` | `gemini-2.5-flash` | Google Gemini model name |
| `MAX_CONTEXT_CHARS` | `langgraph_agent.py` | `12000` | Max characters sent to Gemini |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM | [Google Gemini 2.5 Flash](https://ai.google.dev) |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Embeddings | [HuggingFace `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector Store | [FAISS](https://faiss.ai) |
| PDF Parsing | [PyPDF](https://pypdf.readthedocs.io) |
| Text Splitting | [LangChain `RecursiveCharacterTextSplitter`](https://python.langchain.com) |
| Env Management | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 🧩 Known Limitations

- Answer quality depends on PDF text extractability — scanned/image-only PDFs are not OCR'd
- Large documents may require raising `MAX_CONTEXT_CHARS` or increasing `TOP_K_RESULTS`
- The FAISS index is local to the machine; it isn't shared across deployments

## 🗺️ Roadmap

- [ ] OCR support for scanned PDFs
- [ ] Support for additional LLM providers
- [ ] Docker deployment guide

---


<div align="center">
  Built with ❤️ using LangChain · LangGraph · Google Gemini · FAISS · Streamlit
</div>