from __future__ import annotations

from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_function

from config import EMBEDDINGS_INDEX_DIR, FINANCE_FILES_DIR, LOGGER
from utils.csv_utils import parse_llm_response
from utils.embedder import build_retriever, ensure_vectorstore, index_exists
from utils.odean_pipeline import (
    EMBEDDING_CHUNK_OVERLAP,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_NAME,
    MODEL_NAME,
    RETRIEVAL_K,
    TOP_P,
    TRANSCRIPT_RAG_PROMPT_6,
    TEMPERATURE,
    TRUNCATE_CHARS,
    TranscriptAnalysisResult,
    build_fallback_parsed,
    build_output_row,
    retrieve_context,
)


def prepare_prompt6_retriever(rebuild_index: bool) -> Any:
    if rebuild_index or not index_exists(EMBEDDINGS_INDEX_DIR, index_name=INDEX_NAME):
        LOGGER.info("Building embeddings index from %s ...", FINANCE_FILES_DIR)
    else:
        LOGGER.info("Loading embeddings index from %s ...", EMBEDDINGS_INDEX_DIR)

    vectorstore = ensure_vectorstore(
        finance_dir=FINANCE_FILES_DIR,
        index_dir=EMBEDDINGS_INDEX_DIR,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=EMBEDDING_CHUNK_SIZE,
        chunk_overlap=EMBEDDING_CHUNK_OVERLAP,
        rebuild=rebuild_index,
        index_name=INDEX_NAME,
    )
    return build_retriever(vectorstore, RETRIEVAL_K)


def render_prompt6_text(context: str, transcript_text: str) -> str:
    return TRANSCRIPT_RAG_PROMPT_6.format(
        context=context,
        input=transcript_text[:TRUNCATE_CHARS],
    )


def build_prompt6_response_format() -> dict[str, Any]:
    schema = dict(convert_to_openai_function(TranscriptAnalysisResult, strict=True))
    schema["schema"] = schema.pop("parameters")
    return {
        "type": "json_schema",
        "json_schema": schema,
    }


def build_prompt6_chat_completion_body(context: str, transcript_text: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "messages": [
            {
                "role": "user",
                "content": render_prompt6_text(context, transcript_text),
            }
        ],
        "response_format": build_prompt6_response_format(),
    }


def extract_chat_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def parse_prompt6_chat_completion_response(response_body: dict[str, Any]) -> dict[str, Any]:
    choices = response_body.get("choices") or []
    if not choices:
        return {}

    message = choices[0].get("message") or {}
    if message.get("refusal"):
        return {}

    raw_text = extract_chat_message_text(message)
    payload = parse_llm_response(raw_text)
    if not payload:
        return {}

    validated = TranscriptAnalysisResult.model_validate(payload)
    return validated.model_dump()


def build_prompt6_result_row(
    video_id: str,
    transcript_text: str,
    document_names: list[str],
    parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = parsed or build_fallback_parsed(transcript_text[:TRUNCATE_CHARS])
    return build_output_row(video_id, len(transcript_text), document_names, normalized)


__all__ = [
    "MODEL_NAME",
    "TEMPERATURE",
    "TOP_P",
    "build_prompt6_chat_completion_body",
    "build_prompt6_result_row",
    "parse_prompt6_chat_completion_response",
    "prepare_prompt6_retriever",
    "render_prompt6_text",
    "retrieve_context",
]
