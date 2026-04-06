from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def extract_metadata(pdf_path: Path) -> dict[str, Any]:
    parts = pdf_path.stem.split(" - ", maxsplit=2)
    if len(parts) < 3:
        return {
            "publisher": "Unknown",
            "year": None,
            "title": pdf_path.stem,
            "source": pdf_path.name,
        }

    year_match = re.search(r"\b(\d{4})\b", parts[1])
    year = int(year_match.group(1)) if year_match else None

    return {
        "publisher": parts[0].strip(),
        "year": year,
        "title": parts[2].strip(),
        "source": pdf_path.name,
    }


def index_exists(index_dir: Path, index_name: str = "index") -> bool:
    return (index_dir / f"{index_name}.faiss").exists() and (index_dir / f"{index_name}.pkl").exists()


def load_finance_documents(finance_dir: Path) -> list[Document]:
    pdf_paths = sorted(finance_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {finance_dir}")

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)

        if not pages:
            continue

        metadata = extract_metadata(pdf_path)
        metadata["page_count"] = len(reader.pages)

        documents.append(
            Document(
                page_content="\n\n".join(pages),
                metadata=metadata,
            )
        )

    if not documents:
        raise ValueError(f"Unable to extract text from PDFs in {finance_dir}")

    return documents


def build_embeddings_client(embedding_model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=embedding_model)


def build_vectorstore(
    finance_dir: Path,
    index_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    index_name: str = "index",
) -> FAISS:
    documents = load_finance_documents(finance_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". "],
    )
    split_documents = splitter.split_documents(documents)
    if not split_documents:
        raise ValueError("No document chunks were created for the embeddings index.")

    embeddings = build_embeddings_client(embedding_model)
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir), index_name=index_name)
    return vectorstore


def load_vectorstore(index_dir: Path, embedding_model: str, index_name: str = "index") -> FAISS:
    embeddings = build_embeddings_client(embedding_model)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
        index_name=index_name,
    )


def ensure_vectorstore(
    finance_dir: Path,
    index_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
    index_name: str = "index",
) -> FAISS:
    if rebuild or not index_exists(index_dir, index_name=index_name):
        return build_vectorstore(
            finance_dir=finance_dir,
            index_dir=index_dir,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_name=index_name,
        )

    return load_vectorstore(index_dir=index_dir, embedding_model=embedding_model, index_name=index_name)


def build_retriever(vectorstore: FAISS, k: int):
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
