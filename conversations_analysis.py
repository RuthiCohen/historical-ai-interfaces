#!/usr/bin/env python3
"""
Cross-country analysis of the three LLM annotations.

Primary inference is participant-level:
1) Calculate, for each participant, the proportion of their messages assigned
   to each label.
2) Compare Israel, Germany, and China using Kruskal-Wallis.
3) Correct the 18 substantive label tests (tone, address form, and question
   style) within each judge using BH-FDR.
4) For each label, run pairwise Mann-Whitney U tests and BH-correct the
   three country comparisons.
5) Also calculate descriptive message-level percentages.
6) Reproduce the human-validation agreement statistics from the two human
   annotation files, including Cohen's kappa and agreement with the LLM
   majority label where a two-of-three majority exists.

This avoids treating message-level observations as statistically independent.
"""

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import scipy.stats as stats

MODELS = ["gemini-flash", "gpt-54", "claude-sonnet-5"]
FIELDS = ["tone", "addressForm", "questionStyle"]
COUNTRIES = ["Israel", "Germany", "China"]
PAIRS = [("Israel", "Germany"), ("Israel", "China"), ("Germany", "China")]

HUMAN_COLUMNS = {
    "tone": ("tone", "human_tone"),
    "addressForm": ("human_address_form", "human_address_form"),
    "questionStyle": ("human_question_style", "human_question_style"),
}

EXPECTED_LABELS = {
    "tone": {
        "casual", "confrontational", "emotional", "neutral",
        "respectful", "uncertain",
    },
    "addressForm": {
        "formal_you", "informal_you", "named_honorific",
        "no_direct_address", "other",
    },
    "questionStyle": {
        "evaluative", "experiential", "factual_closed", "factual_open",
        "other", "procedural", "relational",
    },
}

def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [1.0] * n
    running = 1.0
    for pos in range(n - 1, -1, -1):
        i = order[pos]
        rank = pos + 1
        val = pvals[i] * n / rank
        running = min(running, val)
        adjusted[i] = min(1.0, running)
    return adjusted

def majority_label(row, field):
    vals = [row[f"{m}__{field}"] for m in MODELS]
    counts = Counter(vals)
    label, n = counts.most_common(1)[0]
    return label if n >= 2 else "no_majority"

def get_label(row, source, field):
    return majority_label(row, field) if source == "majority" else row[f"{source}__{field}"]

def clean_label(value):
    return (value or "").strip()

def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for paired categorical labels, excluding missing pairs."""
    pairs = [
        (clean_label(a), clean_label(b))
        for a, b in zip(labels_a, labels_b)
        if clean_label(a) and clean_label(b)
    ]
    if not pairs:
        return float("nan")

    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    labels = set(count_a) | set(count_b)
    expected = sum((count_a[label] / n) * (count_b[label] / n) for label in labels)

    if expected == 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return (observed - expected) / (1.0 - expected)

def agreement_record(field, comparison, labels_a, labels_b,
                     n_total, n_excluded_no_majority=0):
    """Create one reproducible agreement-statistics row."""
    pairs = [
        (clean_label(a), clean_label(b))
        for a, b in zip(labels_a, labels_b)
        if clean_label(a) and clean_label(b)
    ]
    n_used = len(pairs)
    agreement_n = sum(a == b for a, b in pairs)
    agreement_percent = 100 * agreement_n / n_used if n_used else float("nan")
    return {
        "field": field,
        "comparison": comparison,
        "n_total_validation_messages": n_total,
        "n_used": n_used,
        "n_excluded_no_majority": n_excluded_no_majority,
        "agreement_n": agreement_n,
        "agreement_percent": agreement_percent,
        "cohen_kappa": cohen_kappa(
            [a for a, _ in pairs],
            [b for _, b in pairs],
        ),
    }

def compute_human_validation(llm_rows, human1_csv, human2_csv, validation_out):
    """Reproduce human-human and human-LLM agreement for the validation sample."""
    with open(human1_csv, encoding="utf-8-sig", newline="") as f:
        human1_rows = list(csv.DictReader(f))
    with open(human2_csv, encoding="utf-8-sig", newline="") as f:
        human2_rows = list(csv.DictReader(f))

    def index_unique(rows, label):
        indexed = {}
        for row in rows:
            key = clean_label(row.get("source_row_index"))
            if not key:
                raise ValueError(f"Missing source_row_index in {label}")
            if key in indexed:
                raise ValueError(f"Duplicate source_row_index {key} in {label}")
            indexed[key] = row
        return indexed

    h1 = index_unique(human1_rows, "human1")
    h2 = index_unique(human2_rows, "human2")
    llm = index_unique(llm_rows, "LLM coding corpus")

    if set(h1) != set(h2):
        only_h1 = sorted(set(h1) - set(h2))
        only_h2 = sorted(set(h2) - set(h1))
        raise ValueError(
            "Human annotation files do not contain the same validation rows: "
            f"only human1={only_h1[:5]}, only human2={only_h2[:5]}"
        )

    missing_from_llm = sorted(set(h1) - set(llm))
    if missing_from_llm:
        raise ValueError(
            "Human-validation rows are missing from the LLM coding corpus: "
            f"{missing_from_llm[:10]}"
        )

    keys = sorted(h1, key=lambda x: int(x) if x.isdigit() else x)
    n_total = len(keys)

    # Integrity check: human validation text should point to the same LLM row.
    text_mismatches = 0
    for key in keys:
        human_text = clean_label(h1[key].get("user_message_to_code"))
        llm_text = clean_label(llm[key].get("user_message"))
        if human_text != llm_text:
            text_mismatches += 1
    if text_mismatches:
        raise ValueError(
            f"Validation text mismatch for {text_mismatches} row(s) after whitespace trimming."
        )

    records = []
    for field in FIELDS:
        h1_col, h2_col = HUMAN_COLUMNS[field]
        h1_vals = [h1[key].get(h1_col, "") for key in keys]
        h2_vals = [h2[key].get(h2_col, "") for key in keys]

        records.append(agreement_record(
            field, "human1_vs_human2", h1_vals, h2_vals, n_total
        ))

        # Human versus each individual LLM judge.
        for model in MODELS:
            model_vals = [llm[key].get(f"{model}__{field}", "") for key in keys]
            records.append(agreement_record(
                field, f"human1_vs_{model}", h1_vals, model_vals, n_total
            ))
            records.append(agreement_record(
                field, f"human2_vs_{model}", h2_vals, model_vals, n_total
            ))

        # Human versus majority: exclude rows with no actual two-of-three majority.
        majority_vals = [majority_label(llm[key], field) for key in keys]
        resolved_idx = [i for i, value in enumerate(majority_vals) if value != "no_majority"]
        excluded = n_total - len(resolved_idx)
        resolved_majority = [majority_vals[i] for i in resolved_idx]
        resolved_h1 = [h1_vals[i] for i in resolved_idx]
        resolved_h2 = [h2_vals[i] for i in resolved_idx]

        records.append(agreement_record(
            field, "human1_vs_majority", resolved_h1, resolved_majority,
            n_total, excluded
        ))
        records.append(agreement_record(
            field, "human2_vs_majority", resolved_h2, resolved_majority,
            n_total, excluded
        ))

    with open(validation_out, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "field", "comparison", "n_total_validation_messages", "n_used",
            "n_excluded_no_majority", "agreement_n", "agreement_percent",
            "cohen_kappa",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    return records

def safe_kruskal(groups):
    flat = [x for g in groups for x in g]
    if not flat or len(set(flat)) == 1:
        return 0.0, 1.0
    try:
        return stats.kruskal(*groups)
    except ValueError:
        return 0.0, 1.0

def analyze(input_csv, rates_out, omnibus_out, pairwise_out, summary_out,
            human1_csv, human2_csv, human_validation_out):
    with open(input_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # Discover the substantive label set from the three LLMs.
    labels = {}
    for field in FIELDS:
        vals = set()
        for row in rows:
            for model in MODELS:
                v = row[f"{model}__{field}"]
                if v:
                    vals.add(v)
        labels[field] = sorted(vals)

        if vals != EXPECTED_LABELS[field]:
            raise ValueError(
                f"Unexpected label set for {field}: {sorted(vals)}; "
                f"expected {sorted(EXPECTED_LABELS[field])}"
            )

    n_substantive_tests = sum(len(labels[field]) for field in FIELDS)
    if n_substantive_tests != 18:
        raise ValueError(
            f"Semantic inferential family should contain 18 tests, got {n_substantive_tests}."
        )

    # ---------- Human validation ----------
    validation_rows = compute_human_validation(
        rows, human1_csv, human2_csv, human_validation_out
    )

    # Group messages by participant.
    by_user = defaultdict(list)
    for row in rows:
        by_user[(row["user_id"], row["country_group"])].append(row)

    sources = MODELS + ["majority"]

    # ---------- Descriptive message-level percentages ----------
    rate_rows = []
    for source in sources:
        for field in FIELDS:
            output_labels = labels[field] + (["no_majority"] if source == "majority" else [])
            for country in COUNTRIES:
                country_rows = [r for r in rows if r["country_group"] == country]
                vals = [get_label(r, source, field) for r in country_rows]
                for label in output_labels:
                    n = sum(v == label for v in vals)
                    rate_rows.append({
                        "source": source,
                        "field": field,
                        "label": label,
                        "country": country,
                        "n_messages": n,
                        "total_messages_country": len(vals),
                        "message_rate_percent": 100 * n / len(vals) if vals else 0.0,
                    })

    with open(rates_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rate_rows[0].keys()))
        w.writeheader()
        w.writerows(rate_rows)

    # Helper: participant-level rates for one label.
    def participant_rates(source, field, label):
        out = defaultdict(list)
        for (uid, country), user_rows in by_user.items():
            vals = [get_label(r, source, field) for r in user_rows]
            out[country].append(sum(v == label for v in vals) / len(vals))
        return out

    # ---------- Omnibus participant-level tests ----------
    test_rows = []
    for source in sources:
        for field in FIELDS:
            # Exclude "no_majority" from inferential multiplicity family:
            # it measures judge disagreement, not message content.
            for label in labels[field]:
                rates = participant_rates(source, field, label)
                groups = [rates[c] for c in COUNTRIES]
                H, p = safe_kruskal(groups)
                test_rows.append({
                    "source": source,
                    "field": field,
                    "label": label,
                    "israel_participant_mean_percent": 100 * statistics.mean(rates["Israel"]),
                    "germany_participant_mean_percent": 100 * statistics.mean(rates["Germany"]),
                    "china_participant_mean_percent": 100 * statistics.mean(rates["China"]),
                    "kruskal_H": H,
                    "p_value": p,
                })

    # BH-FDR across the 18 substantive label tests within each source.
    for source in sources:
        idx = [i for i, r in enumerate(test_rows) if r["source"] == source]
        qs = bh_adjust([test_rows[i]["p_value"] for i in idx])
        for i, q in zip(idx, qs):
            test_rows[i]["bh_q_within_source"] = q
            test_rows[i]["significant_q_lt_05"] = q < 0.05

    with open(omnibus_out, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "source", "field", "label",
            "israel_participant_mean_percent",
            "germany_participant_mean_percent",
            "china_participant_mean_percent",
            "kruskal_H", "p_value", "bh_q_within_source",
            "significant_q_lt_05",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(test_rows)

    # ---------- Pairwise participant-level tests ----------
    pair_rows = []
    for source in sources:
        for field in FIELDS:
            for label in labels[field]:
                rates = participant_rates(source, field, label)
                tmp = []
                for a, b in PAIRS:
                    U, p = stats.mannwhitneyu(rates[a], rates[b], alternative="two-sided")
                    tmp.append((a, b, U, p))
                qs = bh_adjust([x[3] for x in tmp])
                for (a, b, U, p), q in zip(tmp, qs):
                    pair_rows.append({
                        "source": source,
                        "field": field,
                        "label": label,
                        "country_a": a,
                        "country_b": b,
                        "mean_a_percent": 100 * statistics.mean(rates[a]),
                        "mean_b_percent": 100 * statistics.mean(rates[b]),
                        "mannwhitney_U": U,
                        "p_value": p,
                        "bh_q_across_3_pairs": q,
                        "significant_q_lt_05": q < 0.05,
                    })

    with open(pairwise_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    # ---------- Automatic summary ----------
    robust = []
    for field in FIELDS:
        for label in labels[field]:
            model_rows = [
                next(r for r in test_rows
                     if r["source"] == m and r["field"] == field and r["label"] == label)
                for m in MODELS
            ]
            if all(r["bh_q_within_source"] < 0.05 for r in model_rows):
                robust.append((field, label, model_rows))

    majority_sig = [
        r for r in test_rows
        if r["source"] == "majority" and r["bh_q_within_source"] < 0.05
    ]
    majority_sig.sort(key=lambda r: r["bh_q_within_source"])

    user_counts = Counter(country for _, country in by_user.keys())
    msg_counts = Counter(r["country_group"] for r in rows)

    with open(summary_out, "w", encoding="utf-8") as f:
        f.write("LLM COUNTRY ANALYSIS - FULL CORPUS\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"Messages: {len(rows)}\n")
        f.write(f"Participants: {len(by_user)}\n")
        f.write(f"Message counts: {dict(msg_counts)}\n")
        f.write(f"Participant counts: {dict(user_counts)}\n\n")
        f.write("Primary inference: participant-level category rates + "
                f"Kruskal-Wallis; BH-FDR across {n_substantive_tests} "
                "semantic-category tests within each judge.\n\n")

        f.write("HUMAN VALIDATION (two independent human coders):\n")
        for field in FIELDS:
            hh = next(r for r in validation_rows
                      if r["field"] == field and r["comparison"] == "human1_vs_human2")
            h1m = next(r for r in validation_rows
                       if r["field"] == field and r["comparison"] == "human1_vs_majority")
            h2m = next(r for r in validation_rows
                       if r["field"] == field and r["comparison"] == "human2_vs_majority")
            f.write(
                f"- {field}: human-human={hh['agreement_percent']:.1f}% "
                f"(kappa={hh['cohen_kappa']:.3f}); "
                f"human1-majority={h1m['agreement_percent']:.1f}% "
                f"(kappa={h1m['cohen_kappa']:.3f}, n={h1m['n_used']}); "
                f"human2-majority={h2m['agreement_percent']:.1f}% "
                f"(kappa={h2m['cohen_kappa']:.3f}, n={h2m['n_used']})\n"
            )
        f.write("\n")

        f.write("ROBUST ACROSS ALL THREE INDIVIDUAL LLM JUDGES (q < .05 in each):\n")
        for field, label, model_rows in robust:
            f.write(f"- {field} / {label}\n")
            for r in model_rows:
                f.write(
                    f"  {r['source']}: Israel={r['israel_participant_mean_percent']:.1f}%, "
                    f"Germany={r['germany_participant_mean_percent']:.1f}%, "
                    f"China={r['china_participant_mean_percent']:.1f}%, "
                    f"q={r['bh_q_within_source']:.4g}\n"
                )

        f.write("\nSIGNIFICANT MAJORITY-VOTE RESULTS:\n")
        for r in majority_sig:
            f.write(
                f"- {r['field']} / {r['label']}: "
                f"Israel={r['israel_participant_mean_percent']:.1f}%, "
                f"Germany={r['germany_participant_mean_percent']:.1f}%, "
                f"China={r['china_participant_mean_percent']:.1f}%, "
                f"q={r['bh_q_within_source']:.4g}\n"
            )

        f.write("\nInterpretation rule: strongest findings are those that "
                "are directionally consistent and FDR-significant in all "
                "three individual judges, not merely in the pooled majority.\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input_csv",
        nargs="?",
        default="data/coded-messages-three-models.csv"
    )
    ap.add_argument("--prefix", default="llm_country")
    ap.add_argument(
        "--human1",
        default="human_annotations/human-annotation-human1.csv"
    )
    ap.add_argument(
        "--human2",
        default="human_annotations/human-annotation-human2.csv"
    )
    args = ap.parse_args()

    # Folder for all output files
    output_dir = Path("conversations_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze(
        args.input_csv,
        output_dir / f"{args.prefix}_message_rates.csv",
        output_dir / f"{args.prefix}_participant_tests.csv",
        output_dir / f"{args.prefix}_pairwise_tests.csv",
        output_dir / f"{args.prefix}_summary.txt",
        args.human1,
        args.human2,
        output_dir / f"{args.prefix}_human_validation.csv",
    )


if __name__ == "__main__":
    main()
