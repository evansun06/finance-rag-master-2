from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, field_validator

from config import (
    EMBEDDINGS_INDEX_DIR,
    FINANCE_FILES_DIR,
    LOGGER,
    OPENAI_API_KEY,
)
from utils.embedder import build_retriever, ensure_vectorstore, index_exists


MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
TOP_P = 1.0

# The legacy Odean workflow used OpenAIEmbeddings() without an explicit model.
# Existing experiment logs in Research-RAG-master2 show that default resolved to text-embedding-ada-002.
EMBEDDING_MODEL = "text-embedding-ada-002"

RETRIEVAL_K = 5
RETRIEVAL_QUERY_CHARS = 3000
TRUNCATE_CHARS = 100_000
FALLBACK_PREVIEW_WORDS = 32

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

SCORE_FIELDS = (
    "advice_quality",
    "complexity_rating",
    "RAG_consistency",
    "customized_specificity",
    "jargon_depth_score",
    "decision_complexity_score",
)

EXPLANATION_FIELDS = (
    "advice_quality_explanation",
    "complexity_rating_explanation",
    "RAG_consistency_explanation",
    "customized_specificity_explanation",
    "jargon_depth_explanation",
    "decision_complexity_explanation",
)


class TranscriptAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_personal_finance: bool
    finance_topic: str
    summary_of_text: str
    is_bad_advice: bool
    advice_quality: int
    advice_quality_explanation: str
    complexity_rating: int
    complexity_rating_explanation: str
    RAG_consistency: int
    RAG_consistency_explanation: str
    customized_specificity: int
    customized_specificity_explanation: str
    jargon_depth_score: int
    jargon_depth_explanation: str
    decision_complexity_score: int
    decision_complexity_explanation: str

    @field_validator(*SCORE_FIELDS)
    @classmethod
    def validate_score_range(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("All score fields must be integers from 1 to 5.")
        return value

    @field_validator("finance_topic", "summary_of_text", *EXPLANATION_FIELDS)
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        return value.strip()

    @field_validator("finance_topic", "summary_of_text")
    @classmethod
    def require_core_text_fields(cls, value: str) -> str:
        if not value:
            raise ValueError("finance_topic and summary_of_text must not be blank.")
        return value

    @field_validator(*EXPLANATION_FIELDS)
    @classmethod
    def require_explanations(cls, value: str) -> str:
        if not value:
            raise ValueError("Explanation fields must not be blank.")
        return value

TRANSCRIPT_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth -- all scoring must be grounded in them.\n\n"

        "Return ONLY a single valid JSON object -- no markdown, no explanation, no extra text.\n\n"

        "JSON OUTPUT FORMAT:\n"
        "{{\n"
        '  "is_personal_finance": <true | false>,\n'
        '  "finance_topic": "<5-word keyword summary of the main finance topic>",\n'
        '  "summary_of_text": "<1–2 sentence summary focused on personal finance content>",\n'
        '  "is_bad_advice": <true | false — true ONLY if the transcript contains advice that contradicts established financial/economic principles>,\n'
        '  "advice_quality": <integer 1–5>,\n'
        '  "advice_quality_explanation": "<2 sentences explaining the advice_quality score relative to normative finance>",\n'
        '  "complexity_rating": <integer 1–5>,\n'
        '  "complexity_rating_explanation": "<2 sentences explaining how difficult the advice is for an average household to understand and implement>",\n'
        '  "RAG_consistency": <integer 1–5>,\n'
        '  "RAG_consistency_explanation": "<2 sentences explaining how consistent the transcript advice is with the 5 retrieved RAG documents>",\n'
        '  "customized_specificity": <integer 1–5 — 5 = very specific/niche/tailored, 1 = very generic>,\n'
        '  "customized_specificity_explanation": "<2 sentences explaining why this specificity score was assigned>",\n'
        '  "jargon_depth_score": <integer 1–5>,\n'
        '  "jargon_depth_explanation": "<2 sentences explaining the jargon_depth_score, identifying the highest-tier terms present>",\n'
        '  "decision_complexity_score": <integer 1–5>,\n'
        '  "decision_complexity_explanation": "<2 sentences explaining the decision_complexity_score across its four sub-dimensions>"\n'
        "}}\n\n"

        "Scoring rubrics (apply consistently):\n"
        "  advice_quality      - 1: clearly harmful/incorrect, 3: mixed/neutral, 5: excellent, evidence-based advice\n"
        "  complexity_rating   - 1: simple concepts any adult grasps, 5: requires specialist knowledge or complex implementation\n"
        "  RAG_consistency     - 1: directly contradicts RAG docs, 3: loosely aligned, 5: fully supported by RAG docs\n"
        "  customized_specificity - 1: generic platitudes ('save more, spend less'), 5: highly specific strategies, instruments, or circumstances\n"
        "  jargon_depth_score     - 1: basic everyday money terms only, 3: risk/diversification concepts, 5: sophisticated instruments (derivatives, Sharpe ratio, factor models, options Greeks); score = highest Lusardi-Mitchell literacy tier reached\n"
        "  decision_complexity_score - 1: single action, no conditions, one time horizon, certain outcomes; 5: many simultaneous variables, deeply conditional, multi-period trade-offs, explicit probabilistic reasoning; average across all four rational-choice sub-dimensions\n\n"

        "---\n"
        "RAG Documents (treat as ground truth):\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    )
)

TRANSCRIPT_RAG_PROMPT_2 = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth -- all scoring must be grounded in them.\n\n"

        "Evaluate conservatively and literally. Do not infer missing caveats, safeguards, nuance, suitability limits, "
        "or risk warnings that the speaker did not actually state. If the advice is vague, overconfident, anecdotal, "
        "incomplete, one-sided, impractical, or missing major risks, trade-offs, or household suitability constraints, "
        "lower the score.\n\n"

        "Score against what a careful, evidence-based financial educator should tell a typical household, not against "
        "the average quality of online finance content.\n\n"

        "Populate every output field. Never use 0, null, or blank explanation fields.\n\n"

        "If the transcript is not personal finance:\n"
        "  - set is_personal_finance to false\n"
        "  - set is_bad_advice to false\n"
        "  - use an empty string for finance_topic and summary_of_text if there is no genuine finance content\n"
        "  - still assign all six score fields an integer from 1 to 5\n"
        "  - for non-finance transcripts, use 1 for each score and explicitly state that there is no personal finance advice to evaluate\n\n"

        "Scoring rubrics (apply consistently):\n"
        "  advice_quality - overall normative quality of the financial advice for a typical household, judged against established personal finance principles and the RAG documents.\n"
        "    1: materially harmful, reckless, false, or strongly contradicted by RAG; critical omissions could plausibly cause harm\n"
        "    2: substantially weak; partially correct but misleading in effect because of overgeneralization, overconfidence, impracticality, or missing major caveats, trade-offs, or suitability constraints\n"
        "    3: mixed; some sound points, but incomplete, generic, weakly justified, or reliable only under limited unstated conditions\n"
        "    4: mostly sound and useful; minor omissions or mild overgeneralization, broadly aligned with RAG, and unlikely to mislead most households\n"
        "    5: highly sound, balanced, evidence-based, clearly qualified, and decision-useful; strongly aligned with RAG and clearly communicates risks, trade-offs, and household suitability\n"
        "    Important: advice can deserve a low score even if it is not explicitly false. Penalize omission, lack of caveats, lack of suitability guidance, overconfidence, and one-sided framing. Assign 4 or 5 only when the transcript itself provides strong positive evidence.\n"
        "  complexity_rating - how difficult the advice is for an average household to understand and implement.\n"
        "    1: simple everyday concept or single basic action with minimal interpretation required\n"
        "    2: somewhat simple; limited terminology or a few straightforward implementation steps\n"
        "    3: moderate; requires some financial literacy, comparison, or multi-step execution\n"
        "    4: fairly complex; several interacting concepts, calculations, or implementation constraints\n"
        "    5: highly complex; specialist knowledge, technical judgment, or difficult execution for most households\n"
        "  RAG_consistency - how well the transcript's advice matches the retrieved RAG documents.\n"
        "    1: directly contradicted by RAG on important claims or recommendations\n"
        "    2: materially in tension with RAG, or supported only by cherry-picked fragments while omitting major conflicts\n"
        "    3: mixed or partial alignment; some overlap with RAG but also unsupported, overstated, or weakly grounded claims\n"
        "    4: mostly aligned with RAG; minor gaps or overstatements, but the core advice matches the documents\n"
        "    5: strongly and specifically supported by RAG; key claims, caveats, and recommendations clearly align\n"
        "  customized_specificity - how tailored, detailed, and circumstance-specific the advice is.\n"
        "    1: generic platitudes or broad slogans with little operational detail\n"
        "    2: somewhat general; includes a few concrete points but remains broadly applicable and unspecific\n"
        "    3: moderately specific; identifies a target situation, tactic, or conditional recommendation\n"
        "    4: specific and tailored; includes meaningful constraints, examples, thresholds, or audience distinctions\n"
        "    5: highly specific or niche; tightly tailored to particular instruments, households, conditions, or decision contexts\n"
        "  jargon_depth_score - highest financial-literacy tier reached by the terminology actually used.\n"
        "    1: basic everyday money terms only\n"
        "    2: common personal finance terms such as credit score, emergency fund, APR, or index fund\n"
        "    3: intermediate concepts such as diversification, duration, marginal tax rate, or sequence risk\n"
        "    4: advanced technical language such as tax-loss harvesting, factor exposure, convexity, or Monte Carlo analysis\n"
        "    5: sophisticated specialist terms such as derivatives, options Greeks, Sharpe ratio, or factor models\n"
        "  decision_complexity_score - complexity of the decisions the viewer would need to make, averaging across variables, conditionality, time horizon, and uncertainty.\n"
        "    1: single action, no meaningful conditions, one time horizon, and mostly certain outcomes\n"
        "    2: a small number of factors or mild trade-offs, but still mostly straightforward\n"
        "    3: multiple relevant variables, some conditional decisions, or moderate multi-period trade-offs\n"
        "    4: many interacting factors, substantial conditionality, and meaningful long-term uncertainty or sequencing\n"
        "    5: deeply conditional, multi-stage decision-making with explicit probabilistic reasoning or complex cross-period trade-offs\n\n"

        "---\n"
        "RAG Documents (treat as ground truth):\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    )
)

TRANSCRIPT_RAG_PROMPT_3 = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth -- all scoring must be grounded in them.\n\n"

        "Evaluate conservatively and literally. Do not infer missing caveats, safeguards, nuance, suitability limits, "
        "or risk warnings that the speaker did not actually state. If the advice is vague, overconfident, anecdotal, "
        "incomplete, one-sided, impractical, or missing major risks, trade-offs, or applicability constraints, "
        "lower the score.\n\n"

        "Score against what a careful, evidence-based finance educator should say about general finance advice, not against "
        "the average quality of online finance content.\n\n"

        "Populate every output field. Never use 0, null, or blank explanation fields.\n\n"

        "If the transcript is not personal finance:\n"
        "  - set is_personal_finance to false\n"
        "  - set is_bad_advice to false\n"
        "  - still provide a non-empty finance_topic and summary_of_text describing what the transcript is actually about\n"
        "  - still assign all six score fields an integer from 1 to 5\n"
        "  - if the content is finance-related but not genuinely applicable as advice, you may use advice_quality = 1 and explain that it is informational, technical, exam-prep, theoretical, or otherwise not applicable as general finance advice\n"
        "  - if the transcript is nonsensical, garbled, repetitive, or too corrupted to interpret reliably, set is_personal_finance to false, set is_bad_advice to false, provide a finance_topic and summary_of_text that explicitly say the transcript is incoherent/corrupted, and assign 1 to all six score fields with explanations that the content cannot be meaningfully evaluated\n\n"

        "Scoring rubrics (apply consistently):\n"
        "  advice_quality - overall normative quality of the finance advice in the transcript, judged against established financial principles and the RAG documents.\n"
        "    1: materially harmful, reckless, false, strongly contradicted by RAG, nonsensical/incoherent, or not genuinely applicable as finance advice\n"
        "    2: substantially weak; partially correct but misleading in effect because of overconfidence, one-sided framing, promotional tone, impracticality, or missing major caveats, trade-offs, or applicability constraints that materially affect the recommendation\n"
        "    3: directionally sound or partly useful, but materially incomplete, generic, weakly justified, or reliable only under limited unstated conditions\n"
        "    4: sound and useful finance guidance; broadly aligned with RAG and unlikely to mislead despite some missing nuance or minor caveats\n"
        "    5: excellent general finance advice; clearly correct, evidence-based, strongly aligned with RAG, and genuinely decision-useful\n"
        "    Important: do not withhold a 5 solely because the transcript is concise, introductory, non-technical, or not maximally tailored. A transcript can earn 5 when the core advice is mainstream, correct, and safely framed even if it does not cover every edge case or provide a complete individualized financial plan.\n"
        "    Important: for standard evidence-based finance guidance such as budgeting, emergency funds, insurance, fee awareness, diversification, long-term index investing, credit hygiene, debt repayment, retirement saving, and basic risk management, a correct and safe presentation should usually land in the 4-5 range.\n"
        "    Important: lower the score to 3 or below when missing caveats materially change the evaluation, or when the advice is hype-driven, cherry-picked, unrealistically certain, product-pushing, or one-size-fits-all in a misleading way. advice_quality measures the quality and applicability of the finance advice, not direct household benefit.\n"
        "  complexity_rating - how difficult the content is for an average viewer to understand and implement.\n"
        "    1: simple everyday concept or single basic action with minimal interpretation required\n"
        "    2: somewhat simple; limited terminology or a few straightforward implementation steps\n"
        "    3: moderate; requires some financial literacy, comparison, or multi-step execution\n"
        "    4: fairly complex; several interacting concepts, calculations, or implementation constraints\n"
        "    5: highly complex; specialist knowledge, technical judgment, or difficult execution for most viewers\n"
        "  RAG_consistency - how well the transcript's advice matches the retrieved RAG documents.\n"
        "    1: directly contradicted by RAG on important claims or recommendations\n"
        "    2: materially in tension with RAG, or supported only by cherry-picked fragments while omitting major conflicts\n"
        "    3: mixed or partial alignment; some overlap with RAG but also unsupported, overstated, or weakly grounded claims\n"
        "    4: mostly aligned with RAG; minor gaps or overstatements, but the core advice matches the documents\n"
        "    5: strongly and specifically supported by RAG; key claims, caveats, and recommendations clearly align\n"
        "  customized_specificity - how tailored, detailed, and circumstance-specific the advice is.\n"
        "    1: generic platitudes or broad slogans with little operational detail\n"
        "    2: somewhat general; includes a few concrete points but remains broadly applicable and unspecific\n"
        "    3: moderately specific; identifies a target situation, tactic, or conditional recommendation\n"
        "    4: specific and tailored; includes meaningful constraints, examples, thresholds, or audience distinctions\n"
        "    5: highly specific or niche; tightly tailored to particular instruments, audiences, conditions, or decision contexts\n"
        "  jargon_depth_score - highest financial-literacy tier reached by the terminology actually used.\n"
        "    1: basic everyday money terms only\n"
        "    2: common personal finance terms such as credit score, emergency fund, APR, or index fund\n"
        "    3: intermediate concepts such as diversification, duration, marginal tax rate, or sequence risk\n"
        "    4: advanced technical language such as tax-loss harvesting, factor exposure, convexity, or Monte Carlo analysis\n"
        "    5: sophisticated specialist terms such as derivatives, options Greeks, Sharpe ratio, or factor models\n"
        "  decision_complexity_score - complexity of the decisions implied by the content, averaging across variables, conditionality, time horizon, and uncertainty.\n"
        "    1: single action, no meaningful conditions, one time horizon, and mostly certain outcomes\n"
        "    2: a small number of factors or mild trade-offs, but still mostly straightforward\n"
        "    3: multiple relevant variables, some conditional decisions, or moderate multi-period trade-offs\n"
        "    4: many interacting factors, substantial conditionality, and meaningful long-term uncertainty or sequencing\n"
        "    5: deeply conditional, multi-stage decision-making with explicit probabilistic reasoning or complex cross-period trade-offs\n\n"

        "---\n"
        "RAG Documents (treat as ground truth):\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    )
)

TRANSCRIPT_RAG_PROMPT_4 = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth -- all scoring must be grounded in them.\n\n"

        "Evaluate conservatively and literally. Do not infer missing caveats, safeguards, nuance, suitability limits, "
        "or risk warnings that the speaker did not actually state. If the advice is vague, overconfident, anecdotal, "
        "incomplete, one-sided, impractical, or missing major risks, trade-offs, or applicability constraints, "
        "lower the score.\n\n"

        "Use prompt-3 style calibration as the baseline: content that is sound, aligned with mainstream or accepted financial "
        "principles, clear, practical or conceptually useful, and unlikely to mislead should usually land in the 4-5 range. "
        "Do not judge quality primarily by household applicability; judge the quality of the finance reasoning, explanation, "
        "and guidance itself.\n\n"

        "Populate every output field. Never use 0, null, or blank explanation fields.\n\n"

        "If the transcript is not personal finance:\n"
        "  - set is_personal_finance to false\n"
        "  - set is_bad_advice to false\n"
        "  - still provide a non-empty finance_topic and summary_of_text describing what the transcript is actually about\n"
        "  - still assign all six score fields an integer from 1 to 5\n"
        "  - coherent finance education, technical explanation, mathematical demonstration, market analysis, or cautionary discussion should still be evaluated for correctness and usefulness within scope; do not default those to 1 merely because they are educational, descriptive, technical, or not broad household advice\n"
        "  - reserve advice_quality = 1 for content that is genuinely non-finance, materially false, harmful, incoherent, corrupted, or otherwise devoid of meaningful finance insight\n"
        "  - if the transcript is nonsensical, garbled, repetitive, or too corrupted to interpret reliably, set is_personal_finance to false, set is_bad_advice to false, provide a finance_topic and summary_of_text that explicitly say the transcript is incoherent/corrupted, and assign 1 to all six score fields with explanations that the content cannot be meaningfully evaluated\n\n"

        "Scoring rubrics (apply consistently):\n"
        "  advice_quality - overall quality of the finance advice, analysis, or reasoning in the transcript, judged against established financial principles and the RAG documents.\n"
        "    1: materially harmful, reckless, false, strongly contradicted by RAG, genuinely non-finance, or nonsensical/incoherent\n"
        "    2: substantially weak; partially correct but misleading in effect because of overconfidence, one-sided framing, promotional tone, impracticality, or missing major caveats, trade-offs, or applicability constraints that materially affect the recommendation\n"
        "    3: directionally sound or partly useful, but materially incomplete, generic, weakly justified, or reliable only under limited unstated conditions\n"
        "    4: high-quality finance guidance or analysis; accurate, useful, clear, and aligned with mainstream or accepted financial principles, with only limited missing nuance or caveats\n"
        "    5: exceptional quality; clearly correct, rigorous, evidence-based, strongly aligned with RAG, and genuinely excellent as finance advice, explanation, or analysis\n"
        "    Important: advice_quality is not a popularity or broad-applicability score. Correct technical, mathematical, theoretical, or advanced finance content should not be downgraded merely for being specialized or educational.\n"
        "    Important: if the transcript presents correct financial mathematics, valuation logic, annuity or perpetuity reasoning, portfolio theory, risk-return analysis, optimization logic, or other quantitative finance reasoning accurately and responsibly, that usually merits a 5.\n"
        "    Important: descriptive or educational content can still be excellent if it provides correct, useful, and non-misleading financial insight within its stated scope. Educational format alone is not a reason to lower the score.\n"
        "    Important: complexity belongs in complexity_rating, not as a penalty to advice_quality. Lower the score only for actual inaccuracies, misleading framing, unsupported conclusions, overconfidence, or omitted assumptions/caveats that materially change the interpretation.\n"
        "    Important: lower the score when the content is inaccurate, materially incomplete, overconfident, misleading, cherry-picked, or missing caveats that materially change the interpretation. A transcript should not receive 5 merely for sounding sophisticated.\n"
        "  complexity_rating - how difficult the content is for an average viewer to understand and implement.\n"
        "    1: simple everyday concept or single basic action with minimal interpretation required\n"
        "    2: somewhat simple; limited terminology or a few straightforward implementation steps\n"
        "    3: moderate; requires some financial literacy, comparison, or multi-step execution\n"
        "    4: fairly complex; several interacting concepts, calculations, or implementation constraints\n"
        "    5: highly complex; specialist knowledge, technical judgment, or difficult execution for most viewers\n"
        "  RAG_consistency - how well the transcript's advice matches the retrieved RAG documents.\n"
        "    1: directly contradicted by RAG on important claims or recommendations\n"
        "    2: materially in tension with RAG, or supported only by cherry-picked fragments while omitting major conflicts\n"
        "    3: mixed or partial alignment; some overlap with RAG but also unsupported, overstated, or weakly grounded claims\n"
        "    4: mostly aligned with RAG; minor gaps or overstatements, but the core advice matches the documents\n"
        "    5: strongly and specifically supported by RAG; key claims, caveats, and recommendations clearly align\n"
        "  customized_specificity - how tailored, detailed, and circumstance-specific the advice is.\n"
        "    1: generic platitudes or broad slogans with little operational detail\n"
        "    2: somewhat general; includes a few concrete points but remains broadly applicable and unspecific\n"
        "    3: moderately specific; identifies a target situation, tactic, or conditional recommendation\n"
        "    4: specific and tailored; includes meaningful constraints, examples, thresholds, or audience distinctions\n"
        "    5: highly specific or niche; tightly tailored to particular instruments, audiences, conditions, or decision contexts\n"
        "  jargon_depth_score - highest financial-literacy tier reached by the terminology actually used.\n"
        "    1: basic everyday money terms only\n"
        "    2: common personal finance terms such as credit score, emergency fund, APR, or index fund\n"
        "    3: intermediate concepts such as diversification, duration, marginal tax rate, or sequence risk\n"
        "    4: advanced technical language such as tax-loss harvesting, factor exposure, convexity, or Monte Carlo analysis\n"
        "    5: sophisticated specialist terms such as derivatives, options Greeks, Sharpe ratio, or factor models\n"
        "  decision_complexity_score - complexity of the decisions implied by the content, averaging across variables, conditionality, time horizon, and uncertainty.\n"
        "    1: single action, no meaningful conditions, one time horizon, and mostly certain outcomes\n"
        "    2: a small number of factors or mild trade-offs, but still mostly straightforward\n"
        "    3: multiple relevant variables, some conditional decisions, or moderate multi-period trade-offs\n"
        "    4: many interacting factors, substantial conditionality, and meaningful long-term uncertainty or sequencing\n"
        "    5: deeply conditional, multi-stage decision-making with explicit probabilistic reasoning or complex cross-period trade-offs\n\n"

        "---\n"
        "RAG Documents (treat as ground truth):\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    )
)

TRANSCRIPT_RAG_PROMPT_5 = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth -- all scoring must be grounded in them.\n\n"

        "Evaluate conservatively and literally. Do not infer missing caveats, safeguards, nuance, suitability limits, "
        "or risk warnings that the speaker did not actually state. If the advice is vague, overconfident, anecdotal, "
        "incomplete, one-sided, impractical, or missing major risks, trade-offs, or applicability constraints, "
        "lower the score.\n\n"

        "Score against what a careful, evidence-based finance educator should say about general finance advice, not against "
        "the average quality of online finance content.\n\n"

        "Populate every output field. Never use 0, null, or blank explanation fields.\n\n"

        "If the transcript is not personal finance:\n"
        "  - set is_personal_finance to false\n"
        "  - set is_bad_advice to false\n"
        "  - still provide a non-empty finance_topic and summary_of_text describing what the transcript is actually about\n"
        "  - still assign all six score fields an integer from 1 to 5\n"
        "  - if the content is finance-related but not genuinely applicable as advice, you may use advice_quality = 1 and explain that it is informational, technical, exam-prep, theoretical, or otherwise not applicable as general finance advice\n"
        "  - if the transcript is nonsensical, garbled, repetitive, or too corrupted to interpret reliably, set is_personal_finance to false, set is_bad_advice to false, provide a finance_topic and summary_of_text that explicitly say the transcript is incoherent/corrupted, and assign 1 to all six score fields with explanations that the content cannot be meaningfully evaluated\n\n"

        "Scoring rubrics (apply consistently):\n"
        "  advice_quality - overall normative quality of the finance reasoning, guidance, or instruction. Rigorous finance education counts as advice if it would improve a viewer's financial decisions or understanding.\n"
        "    1: unreliable or not useful as finance advice; this includes garbled, fragmented, rote exam-prep, definition-dumping, promotional or hype-driven content, clearly false claims, news reaction or career commentary with little finance judgment, or content that cannot be evaluated reliably\n"
        "    2: coherent but substantially weak; partially correct yet materially misleading because of overconfidence, one-sided framing, risky implications, or missing caveats that would change the recommendation\n"
        "    3: directionally sound but mid-quality; generic, listicle-style, anecdotal, materially incomplete, weakly justified, or missing context that matters\n"
        "    4: mostly correct and useful mainstream advice; safe and practical, but lighter on rigor, evidence, mechanism, or boundaries\n"
        "    5: coherent, rigorous, mechanism-first finance instruction or analysis that a careful professor or fiduciary educator would endorse; accurate, non-promotional, and genuinely decision-improving\n"
        "    Important: judge transcript integrity first. If the transcript is garbled, fragmented, rote exam-prep, definition-dumping, promotional, hype-driven, or too noisy to recover a stable line of reasoning, use 1 even if finance terms appear.\n"
        "    Important: correct behavioral-finance explanations, insurance or retirement guidance, diversification or fee discussions, and mathematically sound finance walkthroughs can earn 5.\n"
        "    Important: use 2 sparingly. If the main problem is unreliability, transcript corruption, or absence of real decision-useful content, prefer 1.\n"
        "    Important: use 5 more often than a normal internet-video rubric would. Do not withhold 5 merely because the content is educational, technical, specialized, concise, or not personalized.\n"
        "    Important: downgrade from 5 only for actual defects such as factual error, unsupported leaps, product-pushing, or omitted assumptions that materially change the conclusion.\n"
        "    Important: if the speaker recommends a product, ranking, tactic, or forecast, require stronger evidence and clearer limits; otherwise cap the score at 3 or 4.\n"
        "  complexity_rating - how difficult the content is for an average viewer to understand and implement.\n"
        "    1: simple everyday concept or single basic action with minimal interpretation required\n"
        "    2: somewhat simple; limited terminology or a few straightforward implementation steps\n"
        "    3: moderate; requires some financial literacy, comparison, or multi-step execution\n"
        "    4: fairly complex; several interacting concepts, calculations, or implementation constraints\n"
        "    5: highly complex; specialist knowledge, technical judgment, or difficult execution for most viewers\n"
        "  RAG_consistency - how well the transcript's advice matches the retrieved RAG documents.\n"
        "    1: directly contradicted by RAG on important claims or recommendations\n"
        "    2: materially in tension with RAG, or supported only by cherry-picked fragments while omitting major conflicts\n"
        "    3: mixed or partial alignment; some overlap with RAG but also unsupported, overstated, or weakly grounded claims\n"
        "    4: mostly aligned with RAG; minor gaps or overstatements, but the core advice matches the documents\n"
        "    5: strongly and specifically supported by RAG; key claims, caveats, and recommendations clearly align\n"
        "  customized_specificity - how tailored, detailed, and circumstance-specific the advice is.\n"
        "    1: generic platitudes or broad slogans with little operational detail\n"
        "    2: somewhat general; includes a few concrete points but remains broadly applicable and unspecific\n"
        "    3: moderately specific; identifies a target situation, tactic, or conditional recommendation\n"
        "    4: specific and tailored; includes meaningful constraints, examples, thresholds, or audience distinctions\n"
        "    5: highly specific or niche; tightly tailored to particular instruments, audiences, conditions, or decision contexts\n"
        "  jargon_depth_score - highest financial-literacy tier reached by the terminology actually used.\n"
        "    1: basic everyday money terms only\n"
        "    2: common personal finance terms such as credit score, emergency fund, APR, or index fund\n"
        "    3: intermediate concepts such as diversification, duration, marginal tax rate, or sequence risk\n"
        "    4: advanced technical language such as tax-loss harvesting, factor exposure, convexity, or Monte Carlo analysis\n"
        "    5: sophisticated specialist terms such as derivatives, options Greeks, Sharpe ratio, or factor models\n"
        "  decision_complexity_score - complexity of the decisions implied by the content, averaging across variables, conditionality, time horizon, and uncertainty.\n"
        "    1: single action, no meaningful conditions, one time horizon, and mostly certain outcomes\n"
        "    2: a small number of factors or mild trade-offs, but still mostly straightforward\n"
        "    3: multiple relevant variables, some conditional decisions, or moderate multi-period trade-offs\n"
        "    4: many interacting factors, substantial conditionality, and meaningful long-term uncertainty or sequencing\n"
        "    5: deeply conditional, multi-stage decision-making with explicit probabilistic reasoning or complex cross-period trade-offs\n\n"

        "---\n"
        "RAG Documents (treat as ground truth):\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    )
)

TRANSCRIPT_RAG_PROMPT_6 = PromptTemplate(
    input_variables=["context", "input"],
    template=(
        "You are a rigorous financial analyst with deep expertise in personal finance and behavioral economics. "
        "Your task is to evaluate a YouTube video transcript against a set of authoritative academic financial documents "
        "retrieved via RAG. The RAG documents are ground truth -- all scoring must be grounded in them.\n\n"

        "Return ONLY a single valid JSON object -- no markdown, no explanation, no extra text.\n\n"

        "JSON OUTPUT FORMAT:\n"
        "{{\n"
        '  "is_personal_finance": <true | false>,\n'
        '  "finance_topic": "<5-word keyword summary of the main finance topic>",\n'
        '  "summary_of_text": "<1–2 sentence summary focused on personal finance content>",\n'
        '  "is_bad_advice": <true | false — true ONLY if the transcript contains advice that contradicts established financial/economic principles>,\n'
        '  "advice_quality": <integer 1–5>,\n'
        '  "advice_quality_explanation": "<2 sentences explaining the advice_quality score relative to normative finance>",\n'
        '  "complexity_rating": <integer 1–5>,\n'
        '  "complexity_rating_explanation": "<2 sentences explaining how difficult the advice is for an average household to understand and implement>",\n'
        '  "RAG_consistency": <integer 1–5>,\n'
        '  "RAG_consistency_explanation": "<2 sentences explaining how consistent the transcript advice is with the 5 retrieved RAG documents>",\n'
        '  "customized_specificity": <integer 1–5 — 5 = very specific/niche/tailored, 1 = very generic>,\n'
        '  "customized_specificity_explanation": "<2 sentences explaining why this specificity score was assigned>",\n'
        '  "jargon_depth_score": <integer 1–5>,\n'
        '  "jargon_depth_explanation": "<2 sentences explaining the jargon_depth_score, identifying the highest-tier terms present>",\n'
        '  "decision_complexity_score": <integer 1–5>,\n'
        '  "decision_complexity_explanation": "<2 sentences explaining the decision_complexity_score across its four sub-dimensions>"\n'
        "}}\n\n"

        "Scoring rubrics (apply consistently):\n"
        "  advice_quality      - 1: clearly harmful/incorrect, 3: mixed/neutral, 5: excellent, evidence-based advice\n"
        "    Important: use advice_quality = 1 when the transcript is garbled, fragmented, repetitive, corrupted, or too noisy to recover a stable line of reasoning, even if finance terms appear.\n"
        "    Important: use advice_quality = 1 when the content is primarily exam-prep, definition-dumping, news/political/economic commentary, or career commentary rather than decision-useful finance guidance for the viewer.\n"
        "    Important: do not lower advice_quality merely because the content is educational, introductory, concise, or lecture-style if it is coherent and genuinely useful.\n"
        "  complexity_rating   - 1: simple concepts any adult grasps, 5: requires specialist knowledge or complex implementation\n"
        "  RAG_consistency     - 1: directly contradicts RAG docs, 3: loosely aligned, 5: fully supported by RAG docs\n"
        "  customized_specificity - 1: generic platitudes ('save more, spend less'), 5: highly specific strategies, instruments, or circumstances\n"
        "  jargon_depth_score     - 1: basic everyday money terms only, 3: risk/diversification concepts, 5: sophisticated instruments (derivatives, Sharpe ratio, factor models, options Greeks); score = highest Lusardi-Mitchell literacy tier reached\n"
        "  decision_complexity_score - 1: single action, no conditions, one time horizon, certain outcomes; 5: many simultaneous variables, deeply conditional, multi-period trade-offs, explicit probabilistic reasoning; average across all four rational-choice sub-dimensions\n\n"

        "---\n"
        "RAG Documents (treat as ground truth):\n{context}\n\n"
        "---\n"
        "YouTube Transcript to Analyze:\n{input}\n"
    )
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


def parse_structured_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict) and {"parsed", "raw", "parsing_error"}.issubset(response):
        parsing_error = response.get("parsing_error")
        parsed = response.get("parsed")
        if parsing_error is not None or parsed is None:
            raw_preview = stringify_message_content(response.get("raw", ""))[:500]
            if parsing_error is not None:
                LOGGER.warning("Structured output parsing failed: %s", parsing_error)
            if raw_preview:
                LOGGER.warning("Structured output raw preview: %s", raw_preview)
            return {}
        response = parsed

    if isinstance(response, BaseModel):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {}


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


def normalize_text_preview(text: str, max_words: int = FALLBACK_PREVIEW_WORDS) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    words = cleaned.split(" ")
    preview = " ".join(words[:max_words]).strip()
    if len(words) > max_words:
        preview += " ..."
    return preview


def build_fallback_parsed(transcript_text: str) -> dict[str, Any]:
    preview = normalize_text_preview(transcript_text)

    if not preview:
        finance_topic = "Empty or unusable transcript"
        summary = "The transcript text is empty or unusable, so the content could not be evaluated as finance advice."
        explanation = "The transcript is empty or unusable, so it cannot be meaningfully evaluated as finance advice. A conservative fallback score of 1 was assigned to keep all fields populated."
    else:
        finance_topic = "Structured output fallback"
        summary = (
            "The transcript could not be converted into the required structured evaluation format. "
            f"Transcript preview: {preview}"
        )
        explanation = (
            "The model did not return a valid structured evaluation for this transcript, so a conservative fallback score of 1 was assigned. "
            "This row is populated to avoid blank outputs and should be treated as a pipeline fallback rather than a trusted content judgment."
        )

    return {
        "is_personal_finance": False,
        "finance_topic": finance_topic,
        "summary_of_text": summary,
        "is_bad_advice": False,
        "advice_quality": 1,
        "advice_quality_explanation": explanation,
        "complexity_rating": 1,
        "complexity_rating_explanation": explanation,
        "RAG_consistency": 1,
        "RAG_consistency_explanation": explanation,
        "customized_specificity": 1,
        "customized_specificity_explanation": explanation,
        "jargon_depth_score": 1,
        "jargon_depth_explanation": explanation,
        "decision_complexity_score": 1,
        "decision_complexity_explanation": explanation,
    }


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
    structured_llm = llm.with_structured_output(
        TranscriptAnalysisResult,
        include_raw=True,
        strict=True,
    )
    chain = TRANSCRIPT_RAG_PROMPT_6 | structured_llm
    return retriever, chain


def analyze_transcript(video_id: str, transcript_text: str, retriever: Any, chain: Any) -> dict[str, Any]:
    text_length = len(transcript_text)

    try:
        context, document_names = retrieve_context(retriever, transcript_text)
    except Exception as exc:
        LOGGER.warning("Retrieval failed for %s: %s", video_id, exc)
        context, document_names = "", []

    truncated_text = transcript_text[:TRUNCATE_CHARS]
    parsed: dict[str, Any] = {}
    for attempt in range(1, 3):
        try:
            response = chain.invoke({"context": context, "input": truncated_text})
            parsed = parse_structured_response(response)
        except Exception as exc:
            LOGGER.warning("LLM call failed for %s on attempt %s/2: %s", video_id, attempt, exc)
            parsed = {}

        if parsed:
            break

        LOGGER.warning("Structured response invalid for %s on attempt %s/2.", video_id, attempt)

    if not parsed:
        LOGGER.warning("Invalid structured response for %s. Writing a populated fallback analysis row.", video_id)
        parsed = build_fallback_parsed(truncated_text)

    return build_output_row(video_id, text_length, document_names, parsed)
