from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "in"
OUTPUT_DIR = DATA_DIR / "out"
load_dotenv(ROOT_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value:
        return Path(raw_value).expanduser()
    return default


def _configure_logger() -> logging.Logger:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger = logging.getLogger("rag_master2")
    logger.setLevel(log_level)
    return logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LOGGER = _configure_logger()

FINANCE_FILES_DIR = _path_from_env("FINANCE_FILES_DIR", ROOT_DIR / "finance-files")
EMBEDDINGS_INDEX_DIR = _path_from_env("EMBEDDINGS_INDEX_DIR", ROOT_DIR / "embeddings_index")
ODEAN_INPUT_DIR = _path_from_env("ODEAN_INPUT_DIR", INPUT_DIR)
ODEAN_OUTPUT_FILE = _path_from_env("ODEAN_OUTPUT_FILE", OUTPUT_DIR / "odean_analysis_results.csv")
MATCHES_BEST_INPUT_FILE = _path_from_env(
    "MATCHES_BEST_INPUT_FILE",
    INPUT_DIR / "best_advice_match_per_odean_screened.csv",
)
MATCHES_WORST_INPUT_FILE = _path_from_env(
    "MATCHES_WORST_INPUT_FILE",
    INPUT_DIR / "worst_advice_match_per_odean_screened.csv",
)
MATCHES_SENTENCE_PANEL_FILE = _path_from_env(
    "MATCHES_SENTENCE_PANEL_FILE",
    INPUT_DIR / "google_sentence_panel_batch1_20260128.csv",
)
MATCHES_BEST_OUTPUT_FILE = _path_from_env(
    "MATCHES_BEST_OUTPUT_FILE",
    OUTPUT_DIR / "odean_matches_best_analysis_results.csv",
)
MATCHES_WORST_OUTPUT_FILE = _path_from_env(
    "MATCHES_WORST_OUTPUT_FILE",
    OUTPUT_DIR / "odean_matches_worst_analysis_results.csv",
)
