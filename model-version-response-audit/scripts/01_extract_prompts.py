#!/usr/bin/env python3
"""Step 1: Extract reconstructable LLM turns from study session JSON into CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from config import (  # noqa: E402
    ALL_TURNS_CSV,
    DATA_DIR,
    MODEL_NEW,
    MODEL_ORIGINAL,
    PERSON_DISPLAY_NAMES,
    SESSION_JSON_PATH,
    STAGE_PERSON_MAP,
    SWITCH_DATE,
)
from prompt_builder import buildFullPrompt, extractAnswerText, wordCount  # noqa: E402


def parseTimestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def getSessionCountry(session: dict) -> str:
    data = session.get("data") or {}
    dem = data.get("demographics_1") or {}
    country = dem.get("country") or dem.get("Country") or ""
    if isinstance(country, str):
        return country.strip().lower()
    return ""


def inferCohortAndModel(ts: datetime | None, country: str) -> tuple[str, str]:
    """Return (inferred_cohort, inferred_model). Country overrides date when present."""
    if country in ("israel", "germany"):
        if country == "israel":
            return "israel", MODEL_ORIGINAL
        return "germany", MODEL_NEW

    if ts is None:
        return "unknown", "unknown"

    if ts < SWITCH_DATE:
        return "israel", MODEL_ORIGINAL
    return "germany", MODEL_NEW


def pairTurnsForStage(events: list[dict]) -> list[dict]:
    """Pair user_message_sent with following agent_message_received on same stage."""
    pairs: list[dict] = []
    i = 0
    while i < len(events):
        event = events[i]
        payload = event.get("payload") or {}
        if payload.get("custom_event_type") != "user_message_sent":
            i += 1
            continue

        question = (payload.get("userQuestionText") or "").strip()
        inputMethod = payload.get("inputMethod") or ""
        userTs = event.get("server_timestamp") or event.get("client_timestamp")

        agentText = ""
        agentTs = None
        if i + 1 < len(events):
            nextPayload = events[i + 1].get("payload") or {}
            if nextPayload.get("custom_event_type") == "agent_message_received":
                agentText = nextPayload.get("agentResponseText") or ""
                agentTs = events[i + 1].get("server_timestamp") or events[i + 1].get(
                    "client_timestamp"
                )
                i += 2
            else:
                i += 1
        else:
            i += 1

        pairs.append(
            {
                "user_question": question,
                "agent_response_raw": agentText,
                "input_method": inputMethod,
                "user_timestamp": userTs,
                "agent_timestamp": agentTs,
            }
        )
    return pairs


def extractAllTurns(sessionJsonPath: Path) -> list[dict]:
    with sessionJsonPath.open(encoding="utf-8") as handle:
        data = json.load(handle)

    rows: list[dict] = []
    for session in data.get("sessions") or []:
        sessionId = session.get("session_id") or ""
        country = getSessionCountry(session)

        eventsByStage: dict[str, list[dict]] = {}
        for event in session.get("events") or []:
            if event.get("event_type") != "external_task_log":
                continue
            stageId = event.get("stage_id") or ""
            if stageId not in STAGE_PERSON_MAP:
                continue
            eventsByStage.setdefault(stageId, []).append(event)

        for stageId, stageEvents in eventsByStage.items():
            personNameKey, modality = STAGE_PERSON_MAP[stageId]
            personPromptName = PERSON_DISPLAY_NAMES.get(personNameKey, personNameKey)

            stageEventsSorted = sorted(
                stageEvents,
                key=lambda e: e.get("server_timestamp")
                or e.get("client_timestamp")
                or "",
            )
            pairs = pairTurnsForStage(stageEventsSorted)
            priorTurns: list[tuple[str, str]] = []

            for turnIndex, pair in enumerate(pairs):
                question = pair["user_question"]
                rawResponse = pair["agent_response_raw"]
                cleanResponse = extractAnswerText(rawResponse)
                ts = parseTimestamp(pair["agent_timestamp"] or pair["user_timestamp"])
                cohort, model = inferCohortAndModel(ts, country)

                fullPrompt, chatHistoryText = buildFullPrompt(
                    personNameKey=personNameKey,
                    priorTurns=list(priorTurns),
                    newQuestion=question,
                    personPromptName=personPromptName,
                )

                trialId = f"{sessionId}__{stageId}__{turnIndex}"
                rows.append(
                    {
                        "trial_id": trialId,
                        "session_id": sessionId,
                        "stage_id": stageId,
                        "person_name_key": personNameKey,
                        "modality": modality,
                        "turn_index": turnIndex,
                        "timestamp": ts.isoformat() if ts else "",
                        "inferred_cohort": cohort,
                        "inferred_model": model,
                        "country": country,
                        "user_question": question,
                        "chat_history_text": chatHistoryText,
                        "reconstructed_full_prompt": fullPrompt,
                        "original_response_raw": rawResponse,
                        "original_response": cleanResponse,
                        "original_word_count": wordCount(cleanResponse),
                        "input_method": pair["input_method"],
                    }
                )

                if question and cleanResponse:
                    priorTurns.append((question, cleanResponse))
                elif question and rawResponse:
                    priorTurns.append((question, extractAnswerText(rawResponse) or rawResponse))

    return rows


CSV_FIELDS = [
    "trial_id",
    "session_id",
    "stage_id",
    "person_name_key",
    "modality",
    "turn_index",
    "timestamp",
    "inferred_cohort",
    "inferred_model",
    "country",
    "user_question",
    "chat_history_text",
    "reconstructed_full_prompt",
    "original_response_raw",
    "original_response",
    "original_word_count",
    "input_method",
]


def writeCsv(rows: list[dict], outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with outputPath.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LLM turns from study logs")
    parser.add_argument(
        "--input",
        type=Path,
        default=SESSION_JSON_PATH,
        help="Path to session export JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ALL_TURNS_CSV,
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = extractAllTurns(args.input)
    writeCsv(rows, args.output)

    cohortCounts: dict[str, int] = {}
    modelCounts: dict[str, int] = {}
    for row in rows:
        cohortCounts[row["inferred_cohort"]] = cohortCounts.get(row["inferred_cohort"], 0) + 1
        modelCounts[row["inferred_model"]] = modelCounts.get(row["inferred_model"], 0) + 1

    print(f"Wrote {len(rows)} turns → {args.output}")
    print(f"  cohorts: {cohortCounts}")
    print(f"  models:  {modelCounts}")


if __name__ == "__main__":
    main()
