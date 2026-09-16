"""Shared configuration for the one-shot Gemini 2.0 vs 3.1 control sample.

Paths and cloud project identifiers are read from environment variables so this
package can be redistributed without embedding deployment-specific details.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"

# Optional local facts directory (inline fallback when GCS is unavailable).
FACTS_DIR = Path(
    os.environ.get(
        "CONTROL_SAMPLE_FACTS_DIR",
        str(REPO_ROOT / "logs_from_study" / "Facts"),
    )
)

# Study session export (JSON). Not shipped with this anonymous code package.
SESSION_JSON_PATH = Path(
    os.environ.get(
        "CONTROL_SAMPLE_SESSION_JSON",
        str(REPO_ROOT / "logs_from_study" / "session_export.json"),
    )
)

# Vertex access token file. Provide via CONTROL_SAMPLE_TOKEN_PATH.
TOKEN_PATH = Path(
    os.environ.get(
        "CONTROL_SAMPLE_TOKEN_PATH",
        str(PACKAGE_ROOT / "secrets" / "vertex_token.txt"),
    )
)

# Cohort / model proxy: before this date = Site A / gemini-2.0-flash-001
SWITCH_DATE = datetime(2026, 6, 5, tzinfo=timezone.utc)

MODEL_ORIGINAL = "gemini-2.0-flash-001"
MODEL_NEW = "gemini-3.1-flash-lite"
MODEL_JUDGE_DEFAULT = "gemini-3.5-flash"

VERTEX_PROJECT_ID = os.environ.get("CONTROL_SAMPLE_VERTEX_PROJECT_ID", "YOUR_VERTEX_PROJECT_ID")
VERTEX_LOCATION_ID = os.environ.get("CONTROL_SAMPLE_VERTEX_LOCATION_ID", "eu")
VERTEX_API_ENDPOINT = os.environ.get(
    "CONTROL_SAMPLE_VERTEX_API_ENDPOINT",
    "aiplatform.eu.rep.googleapis.com",
)
GCS_FACTS_BUCKET = os.environ.get("CONTROL_SAMPLE_GCS_FACTS_BUCKET", "YOUR_GCS_FACTS_BUCKET")

MAX_OUTPUT_TOKENS = 8000
TEMPERATURE = 0.0
TOP_P = 0.95

# UI default for answer-length preference
DEFAULT_ANSWERS_LENGTH_PREFERENCE = 2

DEFAULT_SAMPLE_N = 50
DEFAULT_SAMPLE_SEED = 42

ALL_TURNS_CSV = DATA_DIR / "all_turns.csv"
SAMPLE_CSV = DATA_DIR / "sample_n.csv"
REGENERATED_CSV = DATA_DIR / "regenerated.csv"
JUDGE_SCORES_JSONL = DATA_DIR / "judge_scores.jsonl"
STATS_SUMMARY_CSV = DATA_DIR / "stats_summary.csv"
FINAL_REPORT_MD = DATA_DIR / "final_report.md"

QUESTION_BLOCKLIST = (
    "test",
)

# Skip failed avatar replies
RESPONSE_ERROR_SUBSTRING = "An unexpected error has occurred. Please try again later"

# Skip near-duplicate intro prompts (same scripted greeting across participants)
INTRO_HELLO_PATTERN = (
    r"^\s*hello\b.*please\s+tell\s+me\s+about\s+yourself\b[\s.!?]*$"
)
GREETING_ONLY_QUESTIONS = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "hello there",
        "hi there",
    }
)

PERSON_DISPLAY_NAMES = {
    "BenLesser": "Ben Lesser",
    "MarkelMordechai": "Mordechai Markel",
}

LOCAL_FACTS_FILES = {
    "BenLesser": FACTS_DIR / "BenLesser_Facts.txt",
    "MarkelMordechai": FACTS_DIR / "MarkelMordechai_Facts.txt",
}

STAGE_PERSON_MAP = {
    "ben_lesser_chat_task": ("BenLesser", "chat"),
    "ben_lesser_avatar_task": ("BenLesser", "avatar"),
    "markel_chat_task": ("MarkelMordechai", "chat"),
    "markel_avatar_task": ("MarkelMordechai", "avatar"),
}
