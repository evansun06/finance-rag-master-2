from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_metadata(pdf_path: Path) -> dict[str, Any]:
    parts = pdf_path.stem.split(" - ", maxsplit=2)
    if len(parts) < 3:
        print(
            f"Warning: Could not properly parse metadata for {pdf_path.stem}. "
            "Expected format '<Publisher> - <Year> - <Title>'"
        )
        return {"publisher": "Unknown", "year": None, "title": pdf_path.stem}

    year_match = re.search(r"\b(\d{4})\b", parts[1])
    year = int(year_match.group(1)) if year_match else None

    return {
        "publisher": parts[0].strip(),
        "year": year,
        "title": parts[2].strip(),
    }


def index_exists(index_dir: Path, index_name: str = "index") -> bool:
    return (index_dir / f"{index_name}.faiss").exists() and (index_dir / f"{index_name}.pkl").exists()


def load_finance_documents(finance_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Any]:
    pdf_paths = sorted(finance_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {finance_dir}")

    all_documents: list[Any] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". "],
    )

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(file_path=str(pdf_path))
        documents = loader.load()
        metadata = extract_metadata(pdf_path)
        for document in documents:
            document.metadata.update(metadata)

        split_documents = text_splitter.split_documents(documents)
        all_documents.extend(split_documents)

    if not all_documents:
        raise ValueError(f"Unable to extract text from PDFs in {finance_dir}")

    return all_documents


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
    documents = load_finance_documents(
        finance_dir=finance_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embeddings = build_embeddings_client(embedding_model)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
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
