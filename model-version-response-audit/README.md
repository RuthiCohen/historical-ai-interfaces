# Historical Control Sample: Gemini 2.0 vs. Gemini 3.1

Archived code package for the post hoc paired model-version diagnostic
reported in the manuscript Appendix.

This pipeline documents a historical N=50 audit that was executed on an
earlier snapshot of the study data. At the time of that audit, the eligible
historical control pool used by the pipeline corresponded to interactions
generated with `gemini-2.0-flash-001`. Additional participants were enrolled
after this audit was conducted.

Final-sample model-version assignment is reconstructed separately in the main
analysis notebook from deployment cutovers and participant timestamps.

The archived audit reconstructed 50 eligible historical prompts, regenerated
them with `gemini-3.1-flash-lite` using the same generation settings, and
evaluated the paired responses in randomized A/B order with a blinded
`gemini-3.5-flash` judge.

This export does not include the historical raw session logs, API credentials,
or per-trial judge outputs. It is included to document the audit procedure and
its aggregate results; it is not intended to be rerun from the released final
study dataset. 


## Contents

```
config.py                 # models, sampling defaults, env-based paths
prompt_builder.py         # reconstructs live-system prompt payloads
gemini_client.py          # Vertex generateContent client
scripts/
  01_extract_prompts.py   # session JSON → all_turns.csv
  02_sample_examples.py   # sample N Site-A / 2.0 turns (default 50)
  03_regenerate_responses.py
  04_judge_and_analyze.py # LLM-as-judge + paired stats + report
SUMMARY.md                # aggregate results summary (no raw transcripts)
data/                     # empty placeholder (outputs would be written here)
requirements.txt
```

## Setup

```bash
cd model-version-response-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure environment (placeholders only in this package):

```bash
export CONTROL_SAMPLE_SESSION_JSON=/path/to/session_export.json
export CONTROL_SAMPLE_TOKEN_PATH=/path/to/vertex_token.txt
export CONTROL_SAMPLE_VERTEX_PROJECT_ID=YOUR_VERTEX_PROJECT_ID
export CONTROL_SAMPLE_GCS_FACTS_BUCKET=YOUR_GCS_FACTS_BUCKET
# optional:
# export CONTROL_SAMPLE_FACTS_DIR=/path/to/Facts
# export CONTROL_SAMPLE_VERTEX_LOCATION_ID=eu
# export CONTROL_SAMPLE_VERTEX_API_ENDPOINT=aiplatform.eu.rep.googleapis.com
```

## Run steps (one at a time)

```bash
python scripts/01_extract_prompts.py
python scripts/02_sample_examples.py --n 50
python scripts/03_regenerate_responses.py
python scripts/04_judge_and_analyze.py
```

Outputs (when run with private inputs) land in `data/`:
`all_turns.csv`, `sample_n.csv`, `regenerated.csv`, `judge_scores.jsonl`,
`stats_summary.csv`, `final_report.md`.

Steps 3–4 checkpoint/resume: re-run to continue after failures. Delete the
output file to force a full redo.

## Method notes

- Prompt reconstruction mirrors the production avatar request builder and
  front-end chat-history formatting.
- Facts may be attached via GCS (`gs://$CONTROL_SAMPLE_GCS_FACTS_BUCKET/{person}_Facts.txt`)
  with optional local inline fallback.
- Generation defaults: temperature 0.0, topP 0.95, maxOutputTokens 8000,
  `answersLengthPreference=2`.
- Cohort split: before `2026-06-05` → Site A / `gemini-2.0-flash-001`;
  on/after → Site B / `gemini-3.1-flash-lite` (explicit country fields override
  when present).
- Default judge model: `gemini-3.5-flash`.

## Anonymization

Removed from this package:

- Study session logs and derived CSV/JSONL outputs
- API tokens and cloud project / bucket identifiers
- Absolute deployment paths
- Virtualenv and Python caches
