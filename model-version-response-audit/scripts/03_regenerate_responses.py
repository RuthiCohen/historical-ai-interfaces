#!/usr/bin/env python3
"""Step 3: Regenerate sample prompts with gemini-3.1-flash-lite (checkpointed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from config import MODEL_NEW, REGENERATED_CSV, SAMPLE_CSV  # noqa: E402
from gemini_client import generateContent  # noqa: E402
from prompt_builder import extractAnswerText, wordCount  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate sample turns with 3.1 Flash Lite")
    parser.add_argument("--input", type=Path, default=SAMPLE_CSV)
    parser.add_argument("--output", type=Path, default=REGENERATED_CSV)
    parser.add_argument("--model", type=str, default=MODEL_NEW)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap for debugging (0 = all rows)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Missing {args.input}. Run scripts/02_sample_examples.py first."
        )

    sample = pd.read_csv(args.input)
    if args.limit and args.limit > 0:
        sample = sample.head(args.limit).copy()

    doneIds: set[str] = set()
    existingRows: list[dict] = []
    if args.output.exists():
        existing = pd.read_csv(args.output)
        # Keep successful regenerations; retry failures on re-run
        okMask = existing["new_3_1_response"].fillna("").astype(str).str.strip() != ""
        existingOk = existing[okMask]
        doneIds = set(existingOk["trial_id"].astype(str))
        existingRows = existingOk.to_dict(orient="records")
        print(f"Resuming: {len(doneIds)} already regenerated")

    results = list(existingRows)
    pending = sample[~sample["trial_id"].astype(str).isin(doneIds)]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="regen"):
        prompt = row["reconstructed_full_prompt"]
        personNameKey = row["person_name_key"]
        rawText, httpStatus, error, factsMode = generateContent(
            promptText=str(prompt),
            modelId=args.model,
            personNameKey=str(personNameKey),
        )
        clean = extractAnswerText(rawText) if rawText else ""
        out = row.to_dict()
        out.update(
            {
                "new_3_1_response_raw": rawText or "",
                "new_3_1_response": clean,
                "new_3_1_word_count": wordCount(clean),
                "regen_model": args.model,
                "regen_http_status": httpStatus if httpStatus is not None else "",
                "regen_error": error or "",
                "facts_mode": factsMode,
            }
        )
        results.append(out)
        pd.DataFrame(results).to_csv(args.output, index=False)

    print(f"Wrote {len(results)} rows → {args.output}")
    failed = sum(1 for r in results if not str(r.get("new_3_1_response") or "").strip())
    if failed:
        print(f"WARNING: {failed} rows missing new_3_1_response (re-run to retry)")


if __name__ == "__main__":
    main()
