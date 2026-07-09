import logging
import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from vector_store import vector_store

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
MAX_CONTEXT_CHARS = 12_000
MIN_RELEVANCE_SCORE = 0.2

_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env file.")
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=1024,
        )
    return _llm


class AgentState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    context: str
    answer: str


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    question = state["question"]
    if not vector_store.is_ready:
        return {"retrieved_docs": [], "context": ""}

    raw_results = vector_store.similarity_search(question)
    filtered_docs = [doc for doc, score in raw_results if score >= MIN_RELEVANCE_SCORE]
    logger.info("[retrieve] %d/%d chunks passed filter.", len(filtered_docs), len(raw_results))
    return {"retrieved_docs": filtered_docs}


def build_context_node(state: AgentState) -> Dict[str, Any]:
    docs = state["retrieved_docs"]
    if not docs:
        return {"context": ""}

    parts: List[str] = []
    total_chars = 0

    for doc in docs:
        header = f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]"
        entry = f"{header}\n{doc.page_content}"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(entry[:remaining])
            break

        parts.append(entry)
        total_chars += len(entry)

    context = "\n\n---\n\n".join(parts)
    logger.info("[context] %d chars built.", len(context))
    return {"context": context}


def _format_sources(docs: List[Document]) -> str:
    seen = set()
    lines = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"{src} — Page {page}")
    return "\n".join(lines)


def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    question = state["question"]
    context = state["context"]
    retrieved_docs = state["retrieved_docs"]

    if not context:
        return {"answer": "The uploaded documents don't contain enough information to answer your question."}

    prompt = f"""You are a precise AI assistant. Answer ONLY using the CONTEXT below.
If the answer is not in the context, respond: "The uploaded documents don't contain enough information to answer this question."
Be clear and concise. Do NOT mention sources in your answer body — they will be shown separately.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        response = get_llm().invoke(prompt)
        answer = response.content.strip()
    except Exception as exc:
        logger.error("[generate] Gemini call failed: %s", exc)
        return {"answer": f"An error occurred: {exc}"}

    # Append formatted sources block after the answer
    if retrieved_docs:
        sources_str = _format_sources(retrieved_docs)
        answer = f"{answer}\n[SOURCES]{sources_str}[/SOURCES]"

    logger.info("[generate] Answer: %d chars.", len(answer))
    return {"answer": answer}


def build_rag_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


rag_graph = build_rag_graph()


def ask(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a valid question."

    initial_state: AgentState = {
        "question": question.strip(),
        "retrieved_docs": [],
        "context": "",
        "answer": "",
    }

    try:
        final_state = rag_graph.invoke(initial_state)
        return final_state.get("answer", "No answer was generated.")
    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc)
        return f"An unexpected error occurred: {exc}"
