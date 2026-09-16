#!/usr/bin/env python3
"""Step 4: Blind LLM-as-judge, paired stats, and final Markdown report."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from config import (  # noqa: E402
    FINAL_REPORT_MD,
    JUDGE_SCORES_JSONL,
    MODEL_JUDGE_DEFAULT,
    MODEL_NEW,
    MODEL_ORIGINAL,
    REGENERATED_CSV,
    STATS_SUMMARY_CSV,
    SWITCH_DATE,
)
from gemini_client import generateContent  # noqa: E402
from prompt_builder import extractAnswerText, wordCount  # noqa: E402

METRICS = [
    "constraint_adherence",
    "reasoning_depth",
    "tone_formality",
    "task_success",
]


JUDGE_SYSTEM = """You are an expert evaluator of Holocaust-survivor conversational agents.
Score two anonymized model responses (A and B) to the SAME user prompt.
Return ONLY valid JSON with this schema:
{
  "a": {
    "constraint_adherence": <int 1-5>,
    "reasoning_depth": <int 1-5>,
    "tone_formality": <int 1-5>,
    "task_success": <int 1-5>,
    "word_count": <int>,
    "notes": "<short string>"
  },
  "b": {
    "constraint_adherence": <int 1-5>,
    "reasoning_depth": <int 1-5>,
    "tone_formality": <int 1-5>,
    "task_success": <int 1-5>,
    "word_count": <int>,
    "notes": "<short string>"
  },
  "qualitative_drift": "<one sentence comparing A vs B>"
}

Scoring rubrics (1=poor, 5=excellent unless noted):
1) constraint_adherence: follows role/survivor constraints, length limits, no stage directions, answer-focused.
2) reasoning_depth: shows contextual planning vs jumping to a shallow answer (still natural speech).
3) tone_formality: 1=overly compliant/servile, 3=neutral/appropriate survivor voice, 5=overly assertive/out-of-character. Prefer mid-range authenticity.
4) task_success: answers the user's underlying question usefully and in-character.
"""


def buildJudgePrompt(userQuestion: str, responseA: str, responseB: str) -> str:
    return (
        JUDGE_SYSTEM
        + "\n\nUSER_QUESTION:\n"
        + userQuestion
        + "\n\nRESPONSE_A:\n"
        + responseA
        + "\n\nRESPONSE_B:\n"
        + responseB
        + '\n\nReturn JSON only, format: {"answer": <the json object above as a string OR nested object>}'
    )


def coerceScore(value, default=3) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(1, min(5, score))


def parseJudgeJson(rawText: str) -> dict:
    text = (rawText or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    def unwrap(obj):
        if isinstance(obj, str):
            return unwrap(json.loads(obj))
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict, got {type(obj)}")
        if "a" in obj and "b" in obj:
            return obj
        if "answer" in obj:
            return unwrap(obj["answer"])
        raise ValueError("Judge JSON missing a/b scores")

    try:
        return unwrap(json.loads(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return unwrap(json.loads(text[start : end + 1]))
        raise ValueError(f"Could not parse judge JSON: {text[:300]}")


def pairedCohensD(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.mean(diff) / sd)


def runMetricStats(scores20: np.ndarray, scores31: np.ndarray, metricName: str) -> dict:
    diff = scores31 - scores20
    normal = True
    if len(diff) >= 3 and np.std(diff) > 0:
        try:
            _, pNorm = stats.shapiro(diff)
            normal = bool(pNorm >= 0.05)
        except Exception:
            normal = True

    if len(diff) < 2:
        testName = "n_too_small"
        tStat, pValue = 0.0, 1.0
    elif normal and np.std(diff) > 0:
        testName = "paired_t"
        tStat, pValue = stats.ttest_rel(scores31, scores20)
    else:
        testName = "wilcoxon"
        if np.allclose(diff, 0):
            tStat, pValue = 0.0, 1.0
        else:
            tStat, pValue = stats.wilcoxon(scores31, scores20)

    return {
        "metric": metricName,
        "mean_2_0": float(np.mean(scores20)),
        "sd_2_0": float(np.std(scores20, ddof=1)) if len(scores20) > 1 else 0.0,
        "mean_3_1": float(np.mean(scores31)),
        "sd_3_1": float(np.std(scores31, ddof=1)) if len(scores31) > 1 else 0.0,
        "mean_diff_3_1_minus_2_0": float(np.mean(diff)),
        "test": testName,
        "p_value": float(pValue),
        "cohens_d": pairedCohensD(scores31, scores20),
        "n": int(len(diff)),
    }


def loadExistingJudgeScores(path: Path) -> dict[str, dict]:
    byId: dict[str, dict] = {}
    if not path.exists():
        return byId
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            byId[str(obj["trial_id"])] = obj
    return byId


def appendJudgeScore(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def judgeAll(
    df: pd.DataFrame,
    judgeModel: str,
    scoresPath: Path,
    seed: int,
) -> list[dict]:
    existing = loadExistingJudgeScores(scoresPath)
    results = list(existing.values())
    done = set(existing.keys())
    pending = df[~df["trial_id"].astype(str).isin(done)]
    rng = random.Random(seed)

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="judge"):
        trialId = str(row["trial_id"])
        resp20 = str(row["original_response"] or "")
        resp31 = str(row["new_3_1_response"] or "")
        question = str(row["user_question"] or "")

        swap = rng.random() < 0.5
        if swap:
            responseA, responseB = resp31, resp20
            mapA, mapB = "3_1", "2_0"
        else:
            responseA, responseB = resp20, resp31
            mapA, mapB = "2_0", "3_1"

        prompt = buildJudgePrompt(question, responseA, responseB)
        # Judge does not need survivor facts file; use BenLesser key only for API shape,
        # but prefer no GCS by using inline empty — pass person from row with preferGcs.
        raw, httpStatus, error, factsMode = generateContent(
            promptText=prompt,
            modelId=judgeModel,
            personNameKey=str(row["person_name_key"]),
            preferGcsFacts=False,
        )

        record: dict = {
            "trial_id": trialId,
            "swap": swap,
            "map_a": mapA,
            "map_b": mapB,
            "judge_model": judgeModel,
            "judge_http_status": httpStatus,
            "judge_error": error,
            "facts_mode": factsMode,
            "qualitative_drift": "",
        }

        if error or not raw:
            record["parse_error"] = error or "empty judge response"
            for metric in METRICS:
                record[f"{metric}_2_0"] = None
                record[f"{metric}_3_1"] = None
            record["word_count_2_0"] = wordCount(resp20)
            record["word_count_3_1"] = wordCount(resp31)
        else:
            try:
                parsed = parseJudgeJson(raw)
                aScores = parsed.get("a") or {}
                bScores = parsed.get("b") or {}
                scoresByLabel = {"A": aScores, "B": bScores}
                labelToModel = {"A": mapA, "B": mapB}

                for label in ("A", "B"):
                    modelKey = labelToModel[label]
                    side = scoresByLabel[label]
                    for metric in METRICS:
                        record[f"{metric}_{modelKey}"] = coerceScore(side.get(metric))
                    wc = side.get("word_count")
                    try:
                        record[f"word_count_{modelKey}"] = int(wc)
                    except (TypeError, ValueError):
                        record[f"word_count_{modelKey}"] = wordCount(
                            resp20 if modelKey == "2_0" else resp31
                        )
                record["qualitative_drift"] = str(parsed.get("qualitative_drift") or "")
                record["judge_raw"] = raw[:2000]
            except Exception as exc:  # noqa: BLE001
                record["parse_error"] = str(exc)
                record["judge_raw"] = (raw or "")[:2000]
                for metric in METRICS:
                    record[f"{metric}_2_0"] = None
                    record[f"{metric}_3_1"] = None
                record["word_count_2_0"] = wordCount(resp20)
                record["word_count_3_1"] = wordCount(resp31)

        appendJudgeScore(scoresPath, record)
        results.append(record)

    return results


def buildReport(statsDf: pd.DataFrame, judgeRecords: list[dict], n: int) -> str:
    significant = []
    for _, row in statsDf.iterrows():
        if row["p_value"] < 0.05 and abs(row["cohens_d"]) >= 0.2:
            significant.append(row["metric"])

    if not significant:
        execSummary = (
            "The model shift from gemini-2.0-flash-001 to gemini-3.1-flash-lite did **not** "
            "introduce a statistically and practically significant behavioral confound on the "
            "evaluated dimensions (no metric with p < 0.05 and |Cohen's d| ≥ 0.2). "
            "Site A vs Site B cohort comparisons can proceed without model-shift covariate adjustment "
            "for these agent-behavior metrics, though residual unmeasured differences remain possible."
        )
        mitigation = (
            "The variance is statistically/practically insignificant on core metrics "
            "(no p < 0.05 with |d| ≥ 0.2), so the cohorts can be compared directly on "
            "downstream outcomes. Optionally report model ID as a study limitation."
        )
    else:
        execSummary = (
            "The model shift **did** alter agent behavior on: "
            + ", ".join(significant)
            + ". This threatens unadjusted Site B vs Site A comparisons for those dimensions; "
            "treat model version as a confounder."
        )
        mitigation = (
            "Apply a statistical covariate (or stratified analysis) for model version / cohort "
            f"split date ({SWITCH_DATE.date().isoformat()}), and/or restrict primary claims to "
            "outcomes insensitive to the drifted metrics: "
            + ", ".join(significant)
            + "."
        )

    # Qualitative notes
    drifts = [
        str(r.get("qualitative_drift") or "").strip()
        for r in judgeRecords
        if str(r.get("qualitative_drift") or "").strip()
    ]
    # Simple frequency of shorter/more direct language
    shorterCount = sum(
        1
        for r in judgeRecords
        if (r.get("word_count_3_1") is not None and r.get("word_count_2_0") is not None)
        and r["word_count_3_1"] < r["word_count_2_0"]
    )
    withWc = sum(
        1
        for r in judgeRecords
        if r.get("word_count_3_1") is not None and r.get("word_count_2_0") is not None
    )
    shorterPct = (100.0 * shorterCount / withWc) if withWc else 0.0
    driftSample = drifts[:8]

    tableLines = [
        "| Metric | Mean 2.0 (SD) | Mean 3.1 (SD) | p-value | Cohen's d | Test |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in statsDf.iterrows():
        tableLines.append(
            "| {metric} | {m20:.2f} ({s20:.2f}) | {m31:.2f} ({s31:.2f}) | {p:.4f} | {d:.3f} | {t} |".format(
                metric=row["metric"],
                m20=row["mean_2_0"],
                s20=row["sd_2_0"],
                m31=row["mean_3_1"],
                s31=row["sd_3_1"],
                p=row["p_value"],
                d=row["cohens_d"],
                t=row["test"],
            )
        )

    qualitative = (
        f"Across {withWc} judged pairs, the 3.1 response was shorter in {shorterPct:.1f}% of cases. "
        "Representative judge notes:\n"
        + ("\n".join(f"- {d}" for d in driftSample) if driftSample else "- (no qualitative notes parsed)")
    )

    return f"""# Control Sample Report: Gemini 2.0 vs 3.1 Model Shift

## 1. Executive Summary

{execSummary}

## 2. Methodology

- **Source logs:** Study session export; chat turns from `external_task_log` events.
- **Cohort proxy:** timestamps before `{SWITCH_DATE.date().isoformat()}` labeled Site A / `{MODEL_ORIGINAL}`; on/after that date Site B / `{MODEL_NEW}` (demographics `country` overrides when present).
- **Sample:** N={n} Site A / 2.0 turns with reconstructed prompts matching avatar PHP role text + JS chat-history format; facts via Vertex GCS file (local inline fallback).
- **Regeneration:** identical prompts re-sent to `{MODEL_NEW}` with temperature=0.0, topP=0.95, maxOutputTokens=8000.
- **Evaluation:** blind pairwise LLM-as-judge scores (1–5) on Constraint Adherence, Reasoning Depth, Tone/Formality, Task Success, plus word counts; A/B order randomized per trial.
- **Statistics:** paired t-test or Wilcoxon on paired differences; Cohen's d on paired differences.

## 3. Quantitative Results

{chr(10).join(tableLines)}

## 4. Qualitative Drift Analysis

{qualitative}

## 5. Mitigation Recommendation

{mitigation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge + analyze regenerated control sample")
    parser.add_argument("--input", type=Path, default=REGENERATED_CSV)
    parser.add_argument("--judge-model", type=str, default=MODEL_JUDGE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores-out", type=Path, default=JUDGE_SCORES_JSONL)
    parser.add_argument("--stats-out", type=Path, default=STATS_SUMMARY_CSV)
    parser.add_argument("--report-out", type=Path, default=FINAL_REPORT_MD)
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Only recompute stats/report from existing judge_scores.jsonl",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Missing {args.input}. Run scripts/03_regenerate_responses.py first."
        )

    df = pd.read_csv(args.input)
    df = df[df["new_3_1_response"].fillna("").astype(str).str.strip() != ""].copy()
    if df.empty:
        raise SystemExit("No successful regenerations to judge.")

    if args.skip_judge:
        judgeRecords = list(loadExistingJudgeScores(args.scores_out).values())
    else:
        # Rewrite scores file cleanly when starting fresh partial merge is handled in judgeAll
        judgeRecords = judgeAll(df, args.judge_model, args.scores_out, args.seed)

    judgeDf = pd.DataFrame(judgeRecords)
    # Keep rows with all metric scores
    usable = judgeDf.dropna(subset=[f"{m}_2_0" for m in METRICS] + [f"{m}_3_1" for m in METRICS])
    if usable.empty:
        raise SystemExit("No successfully judged rows with full metric scores.")

    statsRows = []
    for metric in METRICS:
        statsRows.append(
            runMetricStats(
                usable[f"{metric}_2_0"].astype(float).to_numpy(),
                usable[f"{metric}_3_1"].astype(float).to_numpy(),
                metric,
            )
        )

    # Word count from judge if present else from regenerated CSV merge
    wc20 = usable["word_count_2_0"].astype(float).to_numpy()
    wc31 = usable["word_count_3_1"].astype(float).to_numpy()
    statsRows.append(runMetricStats(wc20, wc31, "word_count"))

    statsDf = pd.DataFrame(statsRows)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    statsDf.to_csv(args.stats_out, index=False)

    report = buildReport(statsDf, judgeRecords, n=len(usable))
    args.report_out.write_text(report, encoding="utf-8")

    print(f"Judged usable rows: {len(usable)}")
    print(f"Stats → {args.stats_out}")
    print(f"Report → {args.report_out}")
    print(statsDf.to_string(index=False))


if __name__ == "__main__":
    main()
