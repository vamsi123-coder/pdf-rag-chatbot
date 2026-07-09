import re
import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 50

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def load_pdf(file_path: str) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages: List[str] = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                pages.append(text)
        except Exception as exc:
            logger.warning("Could not extract page %d from %s: %s", page_num, file_path, exc)

    if not pages:
        raise ValueError(f"No extractable text found in: {file_path}")

    logger.info("Loaded %d page(s) from '%s'", len(pages), path.name)
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(pages: List[str], source_name: str = "unknown") -> List[Document]:
    documents: List[Document] = []

    for page_num, raw_text in enumerate(pages, start=1):
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        chunks = _text_splitter.split_text(cleaned)

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < MIN_CHUNK_LENGTH:
                continue
            documents.append(Document(
                page_content=chunk,
                metadata={"source": source_name, "page": page_num, "chunk": chunk_idx},
            ))

    logger.info("Chunked '%s' → %d chunks across %d page(s)", source_name, len(documents), len(pages))
    return documents


def process_pdf(file_path: str) -> List[Document]:
    source_name = Path(file_path).name
    pages = load_pdf(file_path)
    return chunk_text(pages, source_name=source_name)


def process_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            all_docs.extend(process_pdf(fp))
        except Exception as exc:
            logger.error("Failed to process '%s': %s", fp, exc)

    logger.info("Total chunks from %d PDF(s): %d", len(file_paths), len(all_docs))
    return all_docs
