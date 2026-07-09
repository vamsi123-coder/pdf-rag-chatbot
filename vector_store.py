import logging
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 5
FAISS_INDEX_DIR = "faiss_index"

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _embeddings


class VectorStore:
    def __init__(self) -> None:
        self._store: Optional[FAISS] = None

    def build_from_documents(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("Document list is empty.")
        logger.info("Building FAISS index from %d chunks…", len(documents))
        self._store = FAISS.from_documents(documents, get_embeddings())
        logger.info("FAISS index built.")

    def save(self, directory: str = FAISS_INDEX_DIR) -> None:
        if self._store is None:
            raise RuntimeError("No FAISS store to save.")
        os.makedirs(directory, exist_ok=True)
        self._store.save_local(directory)
        logger.info("FAISS index saved to '%s'.", directory)

    def load(self, directory: str = FAISS_INDEX_DIR) -> bool:
        index_file = os.path.join(directory, "index.faiss")
        if not os.path.exists(index_file):
            return False
        try:
            self._store = FAISS.load_local(
                directory, get_embeddings(), allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded from '%s'.", directory)
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    def similarity_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Tuple[Document, float]]:
        if self._store is None:
            raise RuntimeError("Vector store not initialised. Upload PDFs first.")
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        logger.info("Retrieved %d chunks for query: '%s'", len(results), query[:80])
        return results

    def get_relevant_documents(self, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
        return [doc for doc, _ in self.similarity_search(query, k=k)]

    @property
    def is_ready(self) -> bool:
        return self._store is not None


vector_store = VectorStore()
