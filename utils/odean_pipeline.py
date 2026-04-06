from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDINGS_INDEX_DIR,
    FINANCE_FILES_DIR,
    LOGGER,
    OPENAI_API_KEY,
)
from utils.csv_utils import parse_llm_response
from utils.embedder import build_retriever, ensure_vectorstore, index_exists


MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
TOP_P = 0.0

# The legacy Odean workflow used OpenAIEmbeddings() without an explicit model.
# Existing experiment logs in Research-RAG-master2 show that default resolved to text-embedding-ada-002.
EMBEDDING_MODEL = "text-embedding-ada-002"

RETRIEVAL_K = 5
RETRIEVAL_QUERY_CHARS = 3000
TRUNCATE_CHARS = 100_000

EMBEDDING_CHUNK_SIZE = 800
EMBEDDING_CHUNK_OVERLAP = 100
DOCUMENT_SEPARATOR = "\n\n\n"
INDEX_NAME = "index"


CSV_COLUMNS = [
    "video_ID",
    "text_length",
    "is_personal_finance",
    "finance_topic",
    "summary_of_text",
    "is_bad_advice",
    "RAG_docs",
    "advice_quality",
    "advice_quality_explanation",
    "complexity_rating",
    "complexity_rating_explanation",
    "RAG_consistency",
    "RAG_consistency_explanation",
    "customized_specificity",
    "customized_specificity_explanation",
    "jargon_depth_score",
    "jargon_depth_explanation",
    "decision_complexity_score",
    "decision_complexity_explanation",
]


TRANSCRIPT_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth and all scoring must be grounded in them.\n\n"
        "Return only a single valid JSON object with no markdown or extra text.\n\n"
        "JSON OUTPUT FORMAT:\n"
        "{{\n"
        '  "is_personal_finance": <true | false>,\n'
        '  "finance_topic": "<5-word keyword summary of the main finance topic>",\n'
        '  "summary_of_text": "<1-2 sentence summary focused on personal finance content>",\n'
        '  "is_bad_advice": <true | false>,\n'
        '  "advice_quality": <integer 1-5>,\n'
        '  "advice_quality_explanation": "<2 sentences explaining the advice_quality score relative to normative finance>",\n'
        '  "complexity_rating": <integer 1-5>,\n'
        '  "complexity_rating_explanation": "<2 sentences explaining how difficult the advice is for an average household to understand and implement>",\n'
        '  "RAG_consistency": <integer 1-5>,\n'
        '  "RAG_consistency_explanation": "<2 sentences explaining how consistent the transcript advice is with the retrieved RAG documents>",\n'
        '  "customized_specificity": <integer 1-5>,\n'
        '  "customized_specificity_explanation": "<2 sentences explaining why this specificity score was assigned>",\n'
        '  "jargon_depth_score": <integer 1-5>,\n'
        '  "jargon_depth_explanation": "<2 sentences explaining the jargon_depth_score and the highest-tier terms present>",\n'
        '  "decision_complexity_score": <integer 1-5>,\n'
        '  "decision_complexity_explanation": "<2 sentences explaining the decision_complexity_score across its core dimensions>"\n'
        "}}\n\n"
        "Scoring rubrics:\n"
        "  advice_quality: 1 = clearly harmful or incorrect, 3 = mixed or neutral, 5 = excellent and evidence-based.\n"
        "  complexity_rating: 1 = simple concepts any adult can grasp, 5 = specialist knowledge or complex implementation.\n"
        "  RAG_consistency: 1 = directly contradicts the RAG docs, 3 = loosely aligned, 5 = fully supported by the RAG docs.\n"
        "  customized_specificity: 1 = generic platitudes, 5 = highly specific strategies, instruments, or circumstances.\n"
        "  jargon_depth_score: 1 = basic everyday money terms only, 5 = sophisticated instruments or technical finance vocabulary.\n"
        "  decision_complexity_score: 1 = one simple action with little uncertainty, 5 = many variables, conditions, trade-offs, and probabilistic reasoning.\n\n"
        "---\n"
        "RAG Documents:\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    ),
)


def require_base_configuration() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and set the key before running the script."
        )
    if not FINANCE_FILES_DIR.exists():
        raise FileNotFoundError(f"Finance files directory does not exist: {FINANCE_FILES_DIR}")


def stringify_message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def retrieve_context(retriever: Any, transcript_text: str) -> tuple[str, list[str]]:
    query = transcript_text[:RETRIEVAL_QUERY_CHARS]
    documents = retriever.invoke(query)

    context_pieces = []
    document_names = []
    for document in documents:
        page_content = document.page_content.strip()
        if page_content:
            context_pieces.append(page_content)

        metadata = getattr(document, "metadata", {})
        document_name = metadata.get("title") or metadata.get("source") or "unknown-document"
        document_names.append(str(document_name))

    return DOCUMENT_SEPARATOR.join(context_pieces), document_names


def build_output_row(video_id: str, text_length: int, document_names: list[str], parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_ID": video_id,
        "text_length": text_length,
        "is_personal_finance": parsed.get("is_personal_finance", ""),
        "finance_topic": parsed.get("finance_topic", ""),
        "summary_of_text": parsed.get("summary_of_text", ""),
        "is_bad_advice": parsed.get("is_bad_advice", ""),
        "RAG_docs": " | ".join(document_names),
        "advice_quality": parsed.get("advice_quality", ""),
        "advice_quality_explanation": parsed.get("advice_quality_explanation", ""),
        "complexity_rating": parsed.get("complexity_rating", ""),
        "complexity_rating_explanation": parsed.get("complexity_rating_explanation", ""),
        "RAG_consistency": parsed.get("RAG_consistency", ""),
        "RAG_consistency_explanation": parsed.get("RAG_consistency_explanation", ""),
        "customized_specificity": parsed.get("customized_specificity", ""),
        "customized_specificity_explanation": parsed.get("customized_specificity_explanation", ""),
        "jargon_depth_score": parsed.get("jargon_depth_score", ""),
        "jargon_depth_explanation": parsed.get("jargon_depth_explanation", ""),
        "decision_complexity_score": parsed.get("decision_complexity_score", ""),
        "decision_complexity_explanation": parsed.get("decision_complexity_explanation", ""),
    }


def prepare_analysis_resources(rebuild_index: bool) -> tuple[Any, Any]:
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
    retriever = build_retriever(vectorstore, RETRIEVAL_K)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        model_kwargs={"top_p": TOP_P},
    )
    chain = TRANSCRIPT_RAG_PROMPT | llm
    return retriever, chain


def analyze_transcript(video_id: str, transcript_text: str, retriever: Any, chain: Any) -> dict[str, Any]:
    text_length = len(transcript_text)

    try:
        context, document_names = retrieve_context(retriever, transcript_text)
    except Exception as exc:
        LOGGER.warning("Retrieval failed for %s: %s", video_id, exc)
        context, document_names = "", []

    truncated_text = transcript_text[:TRUNCATE_CHARS]
    try:
        response = chain.invoke({"context": context, "input": truncated_text})
        response_text = stringify_message_content(response)
    except Exception as exc:
        LOGGER.warning("LLM call failed for %s: %s", video_id, exc)
        response_text = ""

    parsed = parse_llm_response(response_text)
    if not parsed:
        LOGGER.warning("Invalid JSON response for %s. Writing a blank analysis row.", video_id)

    return build_output_row(video_id, text_length, document_names, parsed)
