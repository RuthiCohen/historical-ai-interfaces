# Control Sample Summary: Gemini 2.0 → 3.1

One-shot audit of whether the mid-study backend change (`gemini-2.0-flash-001` → `gemini-3.1-flash-lite` on **2026-06-05**) confounds the Avatar–Chat / cross-site study.

## What we did

We ran a paired control sample on historical Site A / `gemini-2.0-flash-001` chat turns: reconstructed each full avatar prompt from the study logs, using the same generation settings as the live system (temperature 0.0, top‑p 0.95). We then sampled **N=50** eligible turns (excluding scripted Hello intros), re-sent those identical prompts to `gemini-3.1-flash-lite`, and scored each 2.0 vs 3.1 response pair with a blind LLM-as-judge on constraint adherence, reasoning depth, **tone/formality**, task success, and word count, followed by paired statistical tests.

Raw transcripts and per-trial judge files are **not** included in this anonymous code package.

## Results (N=50)

| Metric               | 2.0 mean | 3.1 mean | p        | Cohen's d |
| -------------------- | -------- | -------- | -------- | --------- |
| **Tone/formality**   | 2.96     | 3.02     | **0.26** | **0.16**  |
| Constraint adherence | 4.44     | 4.82     | 0.009    | 0.39      |
| Reasoning depth      | 3.60     | 4.18     | 0.007    | 0.42      |
| Task success         | 4.14     | 4.62     | 0.013    | 0.37      |
| Word count           | 30.6     | 39.2     | <0.001   | 1.01      |

## Interpretation for this paper

The historical audit did not detect a difference in judge-rated response-style tone/formality between the two model versions, while Gemini 3.1 responses were longer and received higher ratings on several other judge-rated dimensions. This was a post hoc diagnostic based on 50 reconstructed prompts and unadjusted tests, so it cannot rule out subtler model-version effects.

Because each participant completed both Avatar and Chat conditions under the same model version, the primary within-participant Avatar--Chat comparison is not directly confounded by the migration. However, model version was largely aligned with research cohort, so model-version effects cannot be separated from cohort effects in cross-cohort comparisons. These comparisons should therefore retain the cautious, exploratory interpretation used in the manuscript.
