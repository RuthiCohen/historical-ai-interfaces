#!/usr/bin/env python3
"""Step 2: Select N Site A / gemini-2.0-flash-001 turns for the control sample."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from config import (  # noqa: E402
    ALL_TURNS_CSV,
    DEFAULT_SAMPLE_N,
    DEFAULT_SAMPLE_SEED,
    GREETING_ONLY_QUESTIONS,
    INTRO_HELLO_PATTERN,
    MODEL_ORIGINAL,
    QUESTION_BLOCKLIST,
    RESPONSE_ERROR_SUBSTRING,
    SAMPLE_CSV,
)

_INTRO_HELLO_RE = re.compile(INTRO_HELLO_PATTERN, re.IGNORECASE | re.DOTALL)


def normalizeQuestion(question: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", (question or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def isBlockedQuestion(question: str) -> bool:
    lowered = (question or "").lower()
    if any(token in lowered for token in QUESTION_BLOCKLIST):
        return True
    if _INTRO_HELLO_RE.match(question or ""):
        return True
    if normalizeQuestion(question) in GREETING_ONLY_QUESTIONS:
        return True
    return False


def isErrorResponse(response: str) -> bool:
    return RESPONSE_ERROR_SUBSTRING.lower() in (response or "").lower()


def stratifiedSample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratify by person_name_key × modality; half first-turn, half later turns."""
    if len(df) <= n:
        out = df.copy()
        out["sample_reason"] = "all_eligible"
        return out

    rng = seed
    first = df[df["turn_index"] == 0]
    later = df[df["turn_index"] >= 1]
    targetFirst = n // 2
    targetLater = n - targetFirst

    # Adjust if one pool is too small
    if len(first) < targetFirst:
        targetLater = n - len(first)
        targetFirst = len(first)
    if len(later) < targetLater:
        targetFirst = n - len(later)
        targetLater = len(later)

    def takeStratified(pool: pd.DataFrame, count: int, reasonPrefix: str) -> pd.DataFrame:
        if count <= 0 or pool.empty:
            return pd.DataFrame(columns=list(pool.columns) + ["sample_reason"])
        strata = list(pool.groupby(["person_name_key", "modality"], dropna=False))
        if not strata:
            picked = pool.sample(n=min(count, len(pool)), random_state=rng)
            picked = picked.copy()
            picked["sample_reason"] = reasonPrefix
            return picked

        base = count // len(strata)
        remainder = count % len(strata)
        chunks: list[pd.DataFrame] = []
        for idx, (_, group) in enumerate(strata):
            want = base + (1 if idx < remainder else 0)
            want = min(want, len(group))
            if want <= 0:
                continue
            chunk = group.sample(n=want, random_state=rng + idx)
            chunk = chunk.copy()
            chunk["sample_reason"] = reasonPrefix
            chunks.append(chunk)

        picked = pd.concat(chunks, ignore_index=True) if chunks else pool.iloc[0:0].copy()

        # Top up if strata undershot
        if len(picked) < count:
            remaining = pool.loc[~pool["trial_id"].isin(picked["trial_id"])]
            need = count - len(picked)
            if not remaining.empty and need > 0:
                extra = remaining.sample(n=min(need, len(remaining)), random_state=rng + 99)
                extra = extra.copy()
                extra["sample_reason"] = reasonPrefix + "_topup"
                picked = pd.concat([picked, extra], ignore_index=True)
        return picked.head(count)

    partFirst = takeStratified(first, targetFirst, "first_turn")
    partLater = takeStratified(later, targetLater, "later_turn")
    combined = pd.concat([partFirst, partLater], ignore_index=True)

    if len(combined) < n:
        remaining = df.loc[~df["trial_id"].isin(combined["trial_id"])]
        need = n - len(combined)
        if not remaining.empty:
            extra = remaining.sample(n=min(need, len(remaining)), random_state=rng + 7)
            extra = extra.copy()
            extra["sample_reason"] = "fill"
            combined = pd.concat([combined, extra], ignore_index=True)

    return combined.head(n).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample N control-sample turns")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_N, help="Sample size (default 50)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--cohort",
        type=str,
        default="israel",
        choices=["israel", "germany", "all"],
        help="Which inferred cohort to sample from (default: israel = Site A / 2.0)",
    )
    parser.add_argument("--input", type=Path, default=ALL_TURNS_CSV)
    parser.add_argument("--output", type=Path, default=SAMPLE_CSV)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Missing {args.input}. Run scripts/01_extract_prompts.py first."
        )

    df = pd.read_csv(args.input)
    eligible = df[
        (df["user_question"].fillna("").str.strip() != "")
        & (df["original_response"].fillna("").str.strip() != "")
    ].copy()

    if args.cohort != "all":
        eligible = eligible[eligible["inferred_cohort"] == args.cohort]

    # Prefer explicit original model when sampling israel control set
    if args.cohort == "israel":
        eligible = eligible[eligible["inferred_model"] == MODEL_ORIGINAL]

    beforeBlock = len(eligible)
    eligible = eligible[~eligible["user_question"].map(isBlockedQuestion)]
    afterQuestionFilter = len(eligible)
    eligible = eligible[
        ~eligible["original_response"].fillna("").map(isErrorResponse)
        & ~eligible["original_response_raw"].fillna("").map(isErrorResponse)
    ]
    afterErrorFilter = len(eligible)

    if eligible.empty:
        raise SystemExit("No eligible rows after filters.")

    sample = stratifiedSample(eligible, args.n, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    print(f"Eligible before blocklists: {beforeBlock}")
    print(
        f"After question filters (hello intros / greetings / blocklist): {afterQuestionFilter}"
    )
    print(f"After error-response filter: {afterErrorFilter}")
    print(f"Wrote sample n={len(sample)} → {args.output}")
    print(sample.groupby(["person_name_key", "modality", "sample_reason"]).size().to_string())
    # Sanity: confirm excluded patterns are gone
    helloLeft = sample["user_question"].fillna("").map(
        lambda q: bool(_INTRO_HELLO_RE.match(q))
        or normalizeQuestion(q) in GREETING_ONLY_QUESTIONS
    ).sum()
    errLeft = sample["original_response"].fillna("").map(isErrorResponse).sum()
    print(f"Sanity hello/error remaining in sample: hello={helloLeft}, errors={errLeft}")


if __name__ == "__main__":
    main()
