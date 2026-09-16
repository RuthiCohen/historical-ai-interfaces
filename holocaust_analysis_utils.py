#!/usr/bin/env python3
"""
This module contains deterministic data preparation, scoring, statistical, and plotting helpers used by analysis notebook. 
Dependencies: pandas, numpy, scipy, statsmodels, matplotlib, openpyxl.
"""

from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:
    raise SystemExit(
        "Missing statsmodels. Install dependencies with:\n"
        "pip install pandas numpy scipy statsmodels matplotlib openpyxl"
    ) from exc

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


AUTO_INTRO_PATTERNS = [
    r"^\s*hello[,! ]+my name is .+?[,! ]+please tell me about yourself[.!?]?\s*$",
    r"^\s*hello[,! ]+i(?:'|’)m .+?[,! ]+please tell me about yourself[.!?]?\s*$",
]

# Conservative pattern: objective surface forms only, not semantic categories.
QUESTION_START_RE = re.compile(
    r"^\s*(who|what|when|where|why|which|whose|whom|how|"
    r"do|does|did|is|are|was|were|can|could|would|will|"
    r"have|has|had|may|might|should)\b",
    flags=re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*")

# Minimal English stopword list for reproducible lexical metrics.
STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","that","this","these","those",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your",
    "his","its","our","their","mine","yours","hers","ours","theirs","is","are","was",
    "were","be","been","being","do","does","did","have","has","had","can","could",
    "will","would","shall","should","may","might","must","to","of","in","on","at",
    "for","from","with","without","about","as","by","into","over","under","after",
    "before","during","through","up","down","out","off","again","further","not","no",
    "yes","so","very","too","also","just","what","who","when","where","why","which",
    "how","tell","please"
}

CHARACTER_ALIASES = {
    "ben_lesser": "Ben Lesser",
    "markel": "Markel",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokens(text: object) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall("" if pd.isna(text) else str(text))]


def content_tokens(text: object) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


def normalize_country(value: object) -> str:
    """Normalize the explicit research-group country value.

    Only the three study groups are accepted. Missing or unexpected values are
    returned as an empty string and rejected by ``assign_research_groups``.
    """
    text = normalize_text(value)
    mapping = {
        "israel": "israel",
        "germany": "germany",
        "china": "china",
    }
    return mapping.get(text, "")


def normalize_gender(value: object) -> str:
    """Normalize participant-reported gender for descriptive analyses."""
    text = normalize_text(value)
    mapping = {
        "female": "female", "woman": "female", "women": "female", "f": "female",
        "male": "male", "man": "male", "men": "male", "m": "male",
        "other": "other", "non-binary": "other", "nonbinary": "other",
        "prefer not to say": "other", "prefer_not_to_say": "other",
    }
    return mapping.get(text, text if text else "unknown")


def add_normalized_gender(data: pd.DataFrame) -> pd.DataFrame:
    """Add `gender_normalized` without discarding non-binary/other responses."""
    out = data.copy()
    if "gender" in out.columns:
        out["gender_normalized"] = out["gender"].map(normalize_gender)
    else:
        out["gender_normalized"] = "unknown"
    return out


def assign_research_groups(
    study: pd.DataFrame,
    messages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach the explicit Israel/Germany/China research-group field.
    """
    study = study.copy()
    messages = messages.copy()

    country_rows = study.loc[
        study["field_id"].eq("country"), ["user_id", "value", "completed_at"]
    ].copy()
    country_rows["completed_at"] = pd.to_datetime(
        country_rows["completed_at"], errors="coerce", utc=True
    )
    country_rows["country_from_data"] = country_rows["value"].map(normalize_country)
    country_rows = (
        country_rows.sort_values("completed_at")
        .drop_duplicates("user_id", keep="last")
    )

    all_users = pd.Index(pd.unique(pd.concat(
        [study["user_id"], messages["user_id"]],
        ignore_index=True,
    )))
    country_map = country_rows.set_index("user_id")["country_from_data"]
    assigned = pd.Series(all_users, index=all_users).map(country_map)

    invalid_mask = assigned.isna() | assigned.eq("")
    if invalid_mask.any():
        raise ValueError(
            "Every participant must have an explicit country value of "
            "Israel, Germany, or China in the conference dataset."
        )

    assignment_df = pd.DataFrame({
        "user_id": all_users.to_numpy(),
        "country": assigned.to_numpy(),
        "country_source": "study_country_field",
    })

    study = (
        study.drop(columns=["country"], errors="ignore")
        .merge(assignment_df, on="user_id", how="left", validate="many_to_one")
    )
    messages = (
        messages.drop(columns=["country"], errors="ignore")
        .merge(assignment_df, on="user_id", how="left", validate="many_to_one")
    )
    return study, messages, assignment_df



def character_from_stage(stage: object) -> str:
    value = normalize_text(stage)
    if "ben_lesser" in value:
        return "Ben Lesser"
    if "markel" in value:
        return "Markel"
    return "Unknown"


def condition_from_stage(stage: object) -> str:
    value = normalize_text(stage)
    if "avatar" in value:
        return "Avatar"
    if "chat" in value:
        return "Chat"
    return "Unknown"


def is_auto_intro(text: object) -> bool:
    value = normalize_text(text)
    return any(re.match(pattern, value, flags=re.IGNORECASE) for pattern in AUTO_INTRO_PATTERNS)


def clean_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group()) if match else np.nan


def shannon_entropy(values: pd.Series) -> float:
    counts = values.dropna().value_counts()
    if counts.empty:
        return np.nan
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def standardized_mean_difference_paired(x: pd.Series, y: pd.Series) -> float:
    delta = (x - y).dropna()
    if len(delta) < 2 or delta.std(ddof=1) == 0:
        return np.nan
    return float(delta.mean() / delta.std(ddof=1))


def rank_biserial_from_wilcoxon(x: pd.Series, y: pd.Series) -> float:
    delta = (x - y).dropna()
    delta = delta[delta != 0]
    if delta.empty:
        return np.nan
    ranks = stats.rankdata(abs(delta))
    positive = ranks[delta > 0].sum()
    negative = ranks[delta < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust_frame(frame: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = frame.copy()
    out["p_holm"] = np.nan
    mask = out[p_col].notna()
    if mask.any():
        out.loc[mask, "p_holm"] = multipletests(
            out.loc[mask, p_col].astype(float), method="holm"
        )[1]
    return out


def ci_mean(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan, np.nan
    mean = s.mean()
    sem = stats.sem(s)
    width = stats.t.ppf((1 + confidence) / 2, len(s) - 1) * sem
    return float(mean - width), float(mean + width)


def flatten_model(model, model_name: str, outcome: str) -> pd.DataFrame:
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    rows = []
    for term in params.index:
        rows.append({
            "model": model_name,
            "outcome": outcome,
            "term": term,
            "estimate": params[term],
            "std_error": model.bse.get(term, np.nan),
            "statistic": model.tvalues.get(term, np.nan),
            "p_value": pvals.get(term, np.nan),
            "ci_low": conf.loc[term, 0] if term in conf.index else np.nan,
            "ci_high": conf.loc[term, 1] if term in conf.index else np.nan,
            "n_obs": int(model.nobs),
            "aic": getattr(model, "aic", np.nan),
            "bic": getattr(model, "bic", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conversation extraction from the main long-format file
# ---------------------------------------------------------------------------

def extract_conversation_events(study: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract user/agent messages from JSON stored in `_event_payloads`.

    Returns
    -------
    user_turns:
        One row per user message, enriched with the next agent response when available.
    all_events:
        One row per extracted user/agent event for auditing and sequence analysis.
    """
    task_rows = study.loc[
        study["field_id"].eq("_event_payloads")
        & study["stage_id"].astype(str).str.contains("_task", na=False),
        ["user_id", "session_id", "stage_id", "value"],
    ].copy()

    event_rows = []
    parse_errors = []
    for row_idx, row in task_rows.iterrows():
        try:
            events = json.loads(row["value"])
        except Exception as exc:
            parse_errors.append({"row_index": row_idx, "error": str(exc)})
            continue
        if not isinstance(events, list):
            continue
        for event_index, event in enumerate(events):
            if event.get("event_type") != "external_task_log":
                continue
            payload = event.get("payload") or {}
            kind = payload.get("custom_event_type")
            if kind not in {"user_message_sent", "agent_message_received"}:
                continue
            connection = payload.get("connectionQuality") or {}
            event_rows.append({
                "event_id": f"{row['session_id']}::{row['stage_id']}::{event_index}",
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "stage_id": row["stage_id"],
                "event_index": event_index,
                "event_kind": "user" if kind == "user_message_sent" else "agent",
                "client_timestamp": event.get("timestamp"),
                "text": payload.get("userQuestionText") if kind == "user_message_sent"
                        else payload.get("agentResponseText"),
                "input_method": payload.get("inputMethod"),
                "connection_level": connection.get("level"),
                "connection_label": connection.get("label"),
                "avg_ping_ms": connection.get("avgPingMs"),
            })

    all_events = pd.DataFrame(event_rows)
    if all_events.empty:
        raise ValueError("No user/agent conversation events were found in _event_payloads.")

    all_events["client_timestamp"] = safe_datetime(all_events["client_timestamp"])
    keys = ["user_id", "session_id", "stage_id"]
    all_events = all_events.sort_values(keys + ["client_timestamp", "event_index"])
    all_events["sequence_index"] = all_events.groupby(keys).cumcount() + 1
    all_events["text"] = all_events["text"].fillna("").astype(str)
    all_events["word_count"] = all_events["text"].map(lambda x: len(tokens(x)))
    all_events["char_count"] = all_events["text"].str.len()

    # Explicitly mark the automatic user greeting and the first agent reply that follows it.
    # Both remain in the full audit table, but the paired user turn is removed later by
    # prepare_messages(), so neither side contributes to interaction metrics.
    all_events["is_auto_intro"] = (
        all_events["event_kind"].eq("user")
        & all_events["text"].map(is_auto_intro)
    )
    all_events["is_auto_intro_agent_response"] = False
    for _, idx in all_events.groupby(keys, dropna=False).groups.items():
        ordered_idx = list(all_events.loc[idx].sort_values(
            ["client_timestamp", "event_index"]
        ).index)
        for pos, row_idx in enumerate(ordered_idx):
            if not bool(all_events.at[row_idx, "is_auto_intro"]):
                continue
            for next_idx in ordered_idx[pos + 1:]:
                if all_events.at[next_idx, "event_kind"] == "agent":
                    all_events.at[next_idx, "is_auto_intro_agent_response"] = True
                    break
                if all_events.at[next_idx, "event_kind"] == "user":
                    break

    user_records = []
    for _, group in all_events.groupby(keys, dropna=False):
        group = group.sort_values(["client_timestamp", "event_index"]).reset_index(drop=True)
        for i, event in group.iterrows():
            if event["event_kind"] != "user":
                continue
            following = group.iloc[i + 1:]
            next_agent = following.loc[following["event_kind"].eq("agent")].head(1)
            if not next_agent.empty:
                agent = next_agent.iloc[0]
                latency = (agent["client_timestamp"] - event["client_timestamp"]).total_seconds()
                if latency < 0:
                    latency = np.nan
                agent_text = agent["text"]
                agent_words = agent["word_count"]
                agent_chars = agent["char_count"]
                agent_time = agent["client_timestamp"]
            else:
                latency = np.nan
                agent_text = np.nan
                agent_words = np.nan
                agent_chars = np.nan
                agent_time = pd.NaT

            user_records.append({
                "event_id": event["event_id"],
                "user_id": event["user_id"],
                "session_id": event["session_id"],
                "stage_id": event["stage_id"],
                "client_timestamp": event["client_timestamp"],
                "user_question_text": event["text"],
                "input_method": normalize_text(event["input_method"]) or "unknown",
                "connection_level": pd.to_numeric(event["connection_level"], errors="coerce"),
                "connection_label": event["connection_label"],
                "avg_ping_ms": pd.to_numeric(event["avg_ping_ms"], errors="coerce"),
                "agent_response_text": agent_text,
                "agent_response_timestamp": agent_time,
                "agent_response_latency_seconds": latency,
                "agent_response_word_count": agent_words,
                "agent_response_char_count": agent_chars,
            })

    user_turns = pd.DataFrame(user_records)
    return user_turns, all_events



def prepare_messages(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_columns(
        raw,
        ["user_id", "session_id", "stage_id", "client_timestamp",
         "user_question_text", "input_method"],
        "Messages file",
    )

    df = raw.copy()
    df["client_timestamp"] = safe_datetime(df["client_timestamp"])
    df["text_original"] = df["user_question_text"].fillna("").astype(str)
    df["text_normalized"] = df["text_original"].map(normalize_text)
    df["character"] = df["stage_id"].map(character_from_stage)
    df["condition"] = df["stage_id"].map(condition_from_stage)
    df["is_auto_intro"] = df["text_original"].map(is_auto_intro)
    df["is_blank"] = df["text_normalized"].eq("")
    df["word_count"] = df["text_original"].map(lambda x: len(tokens(x)))
    df["content_word_count"] = df["text_original"].map(lambda x: len(content_tokens(x)))
    df["char_count"] = df["text_original"].str.len()
    df["unique_word_count"] = df["text_original"].map(lambda x: len(set(tokens(x))))
    df["unique_content_word_count"] = df["text_original"].map(
        lambda x: len(set(content_tokens(x)))
    )
    df["type_token_ratio"] = np.where(
        df["word_count"] > 0, df["unique_word_count"] / df["word_count"], np.nan
    )
    df["question_mark_count"] = df["text_original"].str.count(r"\?")
    df["contains_question_mark"] = df["question_mark_count"].gt(0)
    df["starts_like_question"] = df["text_original"].str.match(QUESTION_START_RE, na=False)
    df["objective_question_form"] = (
        df["contains_question_mark"] | df["starts_like_question"]
    )
    df["is_very_short"] = df["word_count"].le(2)
    df["is_single_token"] = df["word_count"].eq(1)
    df["input_method"] = df["input_method"].fillna("unknown").astype(str).str.lower().str.strip()
    df["is_voice"] = df["input_method"].eq("voice")
    df["is_typed"] = df["input_method"].eq("typed")
    df["is_unknown_input"] = ~df["input_method"].isin(["voice", "typed"])

    for col in ["agent_response_latency_seconds", "agent_response_word_count",
                "agent_response_char_count", "avg_ping_ms", "connection_level"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")


    removed = df.loc[df["is_auto_intro"] | df["is_blank"]].copy()
    clean = df.loc[~df["is_auto_intro"] & ~df["is_blank"]].copy()

    interaction_keys = ["user_id", "session_id", "stage_id"]
    clean = clean.sort_values(interaction_keys + ["client_timestamp", "event_id"], na_position="last")
    clean["message_index"] = clean.groupby(interaction_keys).cumcount() + 1
    clean["previous_input_method"] = clean.groupby(interaction_keys)["input_method"].shift()
    clean["input_method_switch"] = (
        clean["previous_input_method"].notna()
        & clean["input_method"].ne(clean["previous_input_method"])
    )
    clean["seconds_since_previous_user_message"] = (
        clean.groupby(interaction_keys)["client_timestamp"].diff().dt.total_seconds()
    )
    clean.loc[
        clean["seconds_since_previous_user_message"] < 0,
        "seconds_since_previous_user_message"
    ] = np.nan

    # Objective repeated/near-template indicators based on exact normalized text.
    exact_counts = clean["text_normalized"].value_counts()
    clean["exact_text_frequency"] = clean["text_normalized"].map(exact_counts)
    clean["is_exact_repeated_question"] = clean["exact_text_frequency"].gt(1)

    return clean, removed



def filter_complete_interface_sample(
    study: pd.DataFrame,
    clean_messages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retain only participant-session pairs with real messages in both interfaces.

    Eligibility rule
    ----------------
    A participant-session pair is retained only when, after automatic greetings
    and blank messages have been removed, it contains:

    - at least one real participant message in Chat, and
    - at least one real participant message in Avatar.

    Returns
    -------
    filtered_study:
        Main long-format study data restricted to eligible participant sessions.
    filtered_messages:
        Clean message data restricted to eligible participant sessions.
    eligibility:
        Audit table with real-message counts in Chat and Avatar and eligibility.
    excluded:
        Subset of the audit table containing excluded participant sessions.
    """
    ensure_columns(
        study,
        ["user_id", "session_id"],
        "Study data for complete-interface filtering",
    )
    ensure_columns(
        clean_messages,
        ["user_id", "session_id", "condition"],
        "Clean messages for complete-interface filtering",
    )

    participant_sessions = (
        study[["user_id", "session_id"]]
        .drop_duplicates()
        .copy()
    )

    counts = (
        clean_messages.loc[
            clean_messages["condition"].isin(["Chat", "Avatar"])
        ]
        .groupby(["user_id", "session_id", "condition"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Chat", "Avatar"], fill_value=0)
        .reset_index()
        .rename(columns={
            "Chat": "n_real_chat_messages",
            "Avatar": "n_real_avatar_messages",
        })
    )

    eligibility = participant_sessions.merge(
        counts,
        on=["user_id", "session_id"],
        how="left",
        validate="one_to_one",
    )
    for col in ["n_real_chat_messages", "n_real_avatar_messages"]:
        eligibility[col] = (
            pd.to_numeric(eligibility[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    eligibility["eligible_both_interfaces"] = (
        eligibility["n_real_chat_messages"].ge(1)
        & eligibility["n_real_avatar_messages"].ge(1)
    )

    eligibility["exclusion_reason"] = np.select(
        [
            eligibility["eligible_both_interfaces"],
            eligibility["n_real_chat_messages"].eq(0)
            & eligibility["n_real_avatar_messages"].eq(0),
            eligibility["n_real_chat_messages"].eq(0),
            eligibility["n_real_avatar_messages"].eq(0),
        ],
        [
            "retained",
            "no_real_messages_in_either_interface",
            "no_real_chat_message",
            "no_real_avatar_message",
        ],
        default="incomplete_interaction",
    )

    eligible_keys = eligibility.loc[
        eligibility["eligible_both_interfaces"],
        ["user_id", "session_id"],
    ].copy()

    filtered_study = study.merge(
        eligible_keys,
        on=["user_id", "session_id"],
        how="inner",
        validate="many_to_one",
    )
    filtered_messages = clean_messages.merge(
        eligible_keys,
        on=["user_id", "session_id"],
        how="inner",
        validate="many_to_one",
    )
    excluded = eligibility.loc[
        ~eligibility["eligible_both_interfaces"]
    ].copy()

    # Strong validation: each retained participant-session must have both conditions.
    retained_condition_counts = (
        filtered_messages.groupby(["user_id", "session_id"])["condition"]
        .nunique()
    )
    if not retained_condition_counts.empty:
        assert retained_condition_counts.eq(2).all(), (
            "At least one retained participant-session does not have real "
            "messages in both Chat and Avatar."
        )

    return filtered_study, filtered_messages, eligibility, excluded



def build_interaction_metrics(clean: pd.DataFrame) -> pd.DataFrame:
    keys = ["user_id", "session_id", "stage_id", "character", "condition", "country"]

    def aggregate(group: pd.DataFrame) -> pd.Series:
        ordered = group.sort_values("client_timestamp")
        timestamps = ordered["client_timestamp"].dropna()
        duration = (
            (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
            if len(timestamps) >= 2 else 0.0
        )

        all_tokens = [token for text in ordered["text_original"] for token in tokens(text)]
        all_content = [
            token for text in ordered["text_original"] for token in content_tokens(text)
        ]


        gaps = ordered["seconds_since_previous_user_message"].dropna()
        gaps = gaps[gaps >= 0]


        return pd.Series({
            "n_messages": len(ordered),
            "n_voice_messages": int(ordered["is_voice"].sum()),
            "n_typed_messages": int(ordered["is_typed"].sum()),
            "n_unknown_input_messages": int(ordered["is_unknown_input"].sum()),
            "known_input_method_rate": float((~ordered["is_unknown_input"]).mean()),
            "voice_message_rate": float(ordered["is_voice"].mean()),
            "voice_rate_among_known_inputs": (
                float(ordered.loc[~ordered["is_unknown_input"], "is_voice"].mean())
                if (~ordered["is_unknown_input"]).any() else np.nan
            ),
            "used_voice": bool(ordered["is_voice"].any()),
            "used_typed": bool(ordered["is_typed"].any()),
            "input_mode_pattern": (
                "mixed" if ordered["is_voice"].any() and ordered["is_typed"].any()
                else "voice_only" if ordered["is_voice"].any()
                else "typed_only"
            ),
            "n_input_switches": int(ordered["input_method_switch"].sum()),
            "input_method_entropy": shannon_entropy(ordered["input_method"]),
            "total_words": len(all_tokens),
            "total_content_words": len(all_content),
            "mean_words_per_message": float(ordered["word_count"].mean()),
            "median_words_per_message": float(ordered["word_count"].median()),
            "max_words_in_message": int(ordered["word_count"].max()),
            "total_characters": int(ordered["char_count"].sum()),
            "mean_characters_per_message": float(ordered["char_count"].mean()),
            "unique_words": len(set(all_tokens)),
            "unique_content_words": len(set(all_content)),
            "conversation_type_token_ratio": (
                len(set(all_tokens)) / len(all_tokens) if all_tokens else np.nan
            ),
            "root_type_token_ratio": (
                len(set(all_tokens)) / math.sqrt(len(all_tokens))
                if all_tokens else np.nan
            ),
            "n_question_marks": int(ordered["question_mark_count"].sum()),
            "n_messages_with_question_mark": int(ordered["contains_question_mark"].sum()),
            "n_objective_question_forms": int(ordered["objective_question_form"].sum()),
            "question_form_rate": float(ordered["objective_question_form"].mean()),
            "n_very_short_messages": int(ordered["is_very_short"].sum()),
            "very_short_message_rate": float(ordered["is_very_short"].mean()),
            "n_single_token_messages": int(ordered["is_single_token"].sum()),
            "n_exact_repeated_questions": int(ordered["is_exact_repeated_question"].sum()),
            "exact_repeated_question_rate": float(
                ordered["is_exact_repeated_question"].mean()
            ),
            "interaction_duration_seconds": float(duration),
            "interaction_duration_minutes": float(duration / 60),
            "messages_per_minute": (
                len(ordered) / (duration / 60) if duration > 0 else np.nan
            ),
            "median_inter_message_gap_seconds": (
                float(gaps.median()) if not gaps.empty else np.nan
            ),
            "mean_inter_message_gap_seconds": (
                float(gaps.mean()) if not gaps.empty else np.nan
            ),
            "n_agent_responses": int(ordered["agent_response_text"].notna().sum()) if "agent_response_text" in ordered else 0,
            "agent_response_coverage": float(ordered["agent_response_text"].notna().mean()) if "agent_response_text" in ordered else np.nan,
            "total_agent_words": float(ordered["agent_response_word_count"].sum(min_count=1)),
            "mean_agent_words_per_response": float(ordered["agent_response_word_count"].mean()),
            "median_agent_response_latency_seconds": float(ordered["agent_response_latency_seconds"].median()),
            "mean_agent_response_latency_seconds": float(ordered["agent_response_latency_seconds"].mean()),
            "median_avg_ping_ms": float(ordered["avg_ping_ms"].median()),
            "mean_avg_ping_ms": float(ordered["avg_ping_ms"].mean()),
            "user_to_agent_word_ratio": (
                len(all_tokens) / ordered["agent_response_word_count"].sum()
                if ordered["agent_response_word_count"].sum() > 0 else np.nan
            ),
            "first_real_message_time": timestamps.min() if not timestamps.empty else pd.NaT,
            "last_real_message_time": timestamps.max() if not timestamps.empty else pd.NaT,
        })

    metrics = clean.groupby(keys, dropna=False).apply(aggregate).reset_index()
    return metrics



def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha using complete rows across the supplied items."""
    frame = items.apply(pd.to_numeric, errors="coerce").dropna()
    k = frame.shape[1]
    if k < 2 or len(frame) < 2:
        return np.nan
    total = frame.sum(axis=1)
    total_var = total.var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return np.nan
    return float((k / (k - 1)) * (
        1 - frame.var(axis=0, ddof=1).sum() / total_var
    ))


def spearman_brown_two_item(items: pd.DataFrame) -> float:
    """Spearman-Brown reliability for a two-item scale."""
    frame = items.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.shape[1] != 2 or len(frame) < 3:
        return np.nan
    r = frame.iloc[:, 0].corr(frame.iloc[:, 1], method="pearson")
    if pd.isna(r) or np.isclose(1 + r, 0):
        return np.nan
    return float((2 * r) / (1 + r))



def extract_importance_ratings(study: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the initial Holocaust-learning importance rating and post-character ratings.

    The post score is linked to the character and later to the randomized interface
    condition. Change is post-character rating minus the initial rating.
    """
    relevant = study.loc[
        study["field_id"].eq("response")
        & study["stage_id"].isin([
            "initial_importance_rating_likert",
            "ben_lesser_importance_rating_likert",
            "markel_importance_rating_likert",
        ]),
        ["user_id", "session_id", "stage_id", "value", "completed_at"],
    ].copy()
    relevant["numeric_value"] = relevant["value"].map(clean_numeric)
    relevant = relevant.sort_values("completed_at").drop_duplicates(
        ["user_id", "session_id", "stage_id"], keep="last"
    )

    initial = relevant.loc[
        relevant["stage_id"].eq("initial_importance_rating_likert"),
        ["user_id", "session_id", "numeric_value"],
    ].rename(columns={"numeric_value": "initial_importance"})

    post = relevant.loc[
        relevant["stage_id"].isin([
            "ben_lesser_importance_rating_likert",
            "markel_importance_rating_likert",
        ])
    ].copy()
    post["character"] = post["stage_id"].map(character_from_stage)
    post = post.rename(columns={"numeric_value": "post_importance"})
    post = post.merge(initial, on=["user_id", "session_id"], how="left")
    post["importance_change"] = post["post_importance"] - post["initial_importance"]

    return relevant, post[[
        "user_id", "session_id", "character",
        "initial_importance", "post_importance", "importance_change",
    ]]


def questionnaire_reliability(
    scale_items: pd.DataFrame,
) -> pd.DataFrame:
    """Compute reliability for satisfaction and character connection."""
    rows = []

    if not scale_items.empty:
        for (family, character), frame in scale_items.groupby(
            ["scale_family", "character"], dropna=False
        ):
            item_wide = frame.pivot_table(
                index=["user_id", "session_id"],
                columns="field_id",
                values="numeric_value",
                aggfunc="first",
            )
            if item_wide.shape[1] < 2:
                continue
            rows.append({
                "scale": family,
                "character": character,
                "n_items": int(item_wide.shape[1]),
                "n_complete": int(item_wide.dropna().shape[0]),
                "cronbach_alpha": cronbach_alpha(item_wide),
                "spearman_brown": (
                    spearman_brown_two_item(item_wide)
                    if item_wide.shape[1] == 2 else np.nan
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Study long-format outcomes
# ---------------------------------------------------------------------------

def extract_assignments(study: pd.DataFrame) -> pd.DataFrame:
    assignment = study.loc[study["stage_id"].eq("_assignments")].copy()
    if assignment.empty:
        return pd.DataFrame(columns=["user_id", "session_id"])

    wide = assignment.pivot_table(
        index=["user_id", "session_id"],
        columns="field_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    if "people_tasks_phase" in wide:
        wide["character_order_raw"] = wide["people_tasks_phase"]
        wide["first_character"] = wide["people_tasks_phase"].map(
            lambda x: character_from_stage(str(x).split(",")[0]) if pd.notna(x) else np.nan
        )
    else:
        wide["character_order_raw"] = np.nan
        wide["first_character"] = np.nan

    return wide


def extract_demographics(study: pd.DataFrame) -> pd.DataFrame:
    demographic_fields = [
        "age", "gender", "country", "educational_background",
        "professional_background", "ai_experience_years",
        "explanation_type_preference", "interface_detail_preference",
        "participation_consent",
    ]
    subset = study.loc[study["field_id"].isin(demographic_fields)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["user_id", "session_id"])

    wide = subset.pivot_table(
        index=["user_id", "session_id"],
        columns="field_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    if "age" in wide:
        wide["age_numeric"] = wide["age"].map(clean_numeric)
    if "ai_experience_years" in wide:
        wide["ai_experience_years_numeric"] = wide["ai_experience_years"].map(clean_numeric)
    return wide


def extract_knowledge_items(study: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    qmask = (
        study["field_id"].eq("response")
        & study["correct_answer"].notna()
        & study["stage_id"].astype(str).str.match(r"^(ben_lesser|markel)_q\d+$", na=False)
    )
    items = study.loc[qmask, [
        "user_id", "session_id", "stage_id", "value", "correct_answer", "completed_at"
    ]].copy()
    if items.empty:
        return items, pd.DataFrame()

    items["character"] = items["stage_id"].map(character_from_stage)
    items["question_number"] = pd.to_numeric(
        items["stage_id"].str.extract(r"_q(\d+)$")[0], errors="coerce"
    )
    items["response_normalized"] = items["value"].map(normalize_text)
    items["correct_answer_normalized"] = items["correct_answer"].map(normalize_text)
    items["is_correct"] = (
        items["response_normalized"].eq(items["correct_answer_normalized"])
        & items["response_normalized"].ne("")
    ).astype(int)
    items["completed_at"] = safe_datetime(items["completed_at"])

    scores = items.groupby(
        ["user_id", "session_id", "character"], dropna=False
    ).agg(
        knowledge_correct=("is_correct", "sum"),
        knowledge_n_items=("is_correct", "size"),
        knowledge_accuracy=("is_correct", "mean"),
        knowledge_completed_at=("completed_at", "max"),
    ).reset_index()

    return items, scores


def extract_character_scales(study: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the exact post-interaction satisfaction and connection items.

    Satisfaction:
      - three items per character
    Character connection:
      - two items per character

    Importance ratings are intentionally excluded here and are handled by
    extract_importance_ratings(), because they represent a single repeated item
    with a meaningful pre-to-post change score rather than a multi-item scale.
    """
    subset = study.copy()
    subset["numeric_value"] = subset["value"].map(clean_numeric)

    definitions = {
        "Ben Lesser": {
            "satisfaction": [
                "ben_lesser_satisfaction_q1",
                "ben_lesser_satisfaction_q2",
                "ben_lesser_satisfaction_q3",
            ],
            "character_connection": [
                "ben_lesser_character_connection_q1",
                "ben_lesser_character_connection_q2",
            ],
        },
        "Markel": {
            "satisfaction": [
                "markel_satisfaction_q1",
                "markel_satisfaction_q2",
                "markel_satisfaction_q3",
            ],
            "character_connection": [
                "markel_character_connection_q1",
                "markel_character_connection_q2",
            ],
        },
    }

    records = []
    item_records = []
    for character, families in definitions.items():
        for family, fields in families.items():
            frame = subset.loc[
                subset["field_id"].isin(fields)
                & subset["numeric_value"].notna(),
                [
                    "user_id", "session_id", "stage_id", "field_id",
                    "numeric_value", "completed_at",
                ],
            ].copy()
            if frame.empty:
                continue
            frame = frame.sort_values("completed_at").drop_duplicates(
                ["user_id", "session_id", "field_id"], keep="last"
            )
            frame["character"] = character
            frame["scale_family"] = family
            item_records.append(frame)

            expected_n = len(fields)
            agg = frame.groupby(["user_id", "session_id"]).agg(
                scale_mean=("numeric_value", "mean"),
                scale_sum=("numeric_value", "sum"),
                scale_n_items=("numeric_value", "size"),
                scale_sd=("numeric_value", "std"),
            ).reset_index()
            # Require all planned items for the confirmatory score.
            agg.loc[agg["scale_n_items"] < expected_n, ["scale_mean", "scale_sum"]] = np.nan
            agg["character"] = character
            agg["scale_family"] = family
            records.append(agg)

    scale_items = pd.concat(item_records, ignore_index=True) if item_records else pd.DataFrame()
    scale_long = pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    if not scale_long.empty:
        scale_wide = scale_long.pivot_table(
            index=["user_id", "session_id", "character"],
            columns="scale_family",
            values="scale_mean",
            aggfunc="first",
        ).reset_index()
        scale_wide.columns.name = None
    else:
        scale_wide = pd.DataFrame(columns=["user_id", "session_id", "character"])

    return scale_items, scale_wide

def attach_design_information(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    out = interactions.merge(assignments, on=["user_id", "session_id"], how="left")
    out = out.merge(demographics, on=["user_id", "session_id"], how="left")

    out["condition_order"] = np.where(
        out["first_character"].eq(out["character"]),
        "first",
        np.where(out["first_character"].notna(), "second", None),
    )
    out["condition_position"] = out["condition_order"].map({"first": 1, "second": 2})
    return out


# ---------------------------------------------------------------------------
# Statistical summaries and tests
# ---------------------------------------------------------------------------

def descriptive_by_condition(data: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["condition"]
    for outcome in outcomes:
        if outcome not in data:
            continue
        for keys, group in data.groupby(group_cols, dropna=False):
            series = pd.to_numeric(group[outcome], errors="coerce").dropna()
            if series.empty:
                continue
            low, high = ci_mean(series)
            rows.append({
                "outcome": outcome,
                "condition": keys if isinstance(keys, str) else keys[0],
                "n": len(series),
                "mean": series.mean(),
                "sd": series.std(ddof=1),
                "median": series.median(),
                "iqr": series.quantile(.75) - series.quantile(.25),
                "min": series.min(),
                "max": series.max(),
                "ci95_low": low,
                "ci95_high": high,
            })
    return pd.DataFrame(rows)


def descriptive_by_country_condition(data: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    if "country" not in data.columns:
        return pd.DataFrame()
    for outcome in outcomes:
        if outcome not in data.columns:
            continue
        for (country, condition), group in data.groupby(["country", "condition"], dropna=False):
            series = pd.to_numeric(group[outcome], errors="coerce").dropna()
            if series.empty:
                continue
            low, high = ci_mean(series)
            rows.append({
                "outcome": outcome, "country": country, "condition": condition,
                "n": len(series), "mean": series.mean(), "sd": series.std(ddof=1),
                "median": series.median(),
                "iqr": series.quantile(.75) - series.quantile(.25),
                "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(rows)


def paired_tests_by_country(data: pd.DataFrame, outcomes: list[str], min_group_n: int) -> pd.DataFrame:
    frames = []
    if "country" not in data.columns:
        return pd.DataFrame()
    for country, subset in data.groupby("country"):
        result = paired_condition_tests(subset, outcomes, min_group_n)
        if not result.empty:
            result.insert(0, "country", country)
            frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_condition_tests(
    data: pd.DataFrame,
    outcomes: list[str],
    min_group_n: int,
) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in data.columns:
            continue
        wide = data.pivot_table(
            index=["user_id", "session_id"],
            columns="condition",
            values=outcome,
            aggfunc="mean",
        )
        if not {"Chat", "Avatar"}.issubset(wide.columns):
            continue
        paired = wide[["Avatar", "Chat"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(paired) < min_group_n:
            continue

        delta = paired["Avatar"] - paired["Chat"]
        try:
            t_stat, t_p = stats.ttest_rel(
                paired["Avatar"], paired["Chat"], nan_policy="omit"
            )
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = stats.wilcoxon(
                paired["Avatar"], paired["Chat"], zero_method="wilcox"
            )
        except Exception:
            w_stat, w_p = np.nan, np.nan
        try:
            shapiro_w, shapiro_p = (
                stats.shapiro(delta) if 3 <= len(delta) <= 5000 else (np.nan, np.nan)
            )
        except Exception:
            shapiro_w, shapiro_p = np.nan, np.nan

        low, high = ci_mean(delta)
        rows.append({
            "outcome": outcome,
            "n_pairs": len(paired),
            "avatar_mean": paired["Avatar"].mean(),
            "avatar_sd": paired["Avatar"].std(ddof=1),
            "chat_mean": paired["Chat"].mean(),
            "chat_sd": paired["Chat"].std(ddof=1),
            "mean_difference_avatar_minus_chat": delta.mean(),
            "difference_ci95_low": low,
            "difference_ci95_high": high,
            "paired_t": t_stat,
            "p_value": t_p,
            "cohens_dz": standardized_mean_difference_paired(
                paired["Avatar"], paired["Chat"]
            ),
            "wilcoxon_w": w_stat,
            "wilcoxon_p": w_p,
            "rank_biserial": rank_biserial_from_wilcoxon(
                paired["Avatar"], paired["Chat"]
            ),
            "difference_shapiro_w": shapiro_w,
            "difference_shapiro_p": shapiro_p,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = holm_adjust_frame(result, "p_value")
        mask = result["wilcoxon_p"].notna()
        result["wilcoxon_p_holm"] = np.nan
        if mask.any():
            result.loc[mask, "wilcoxon_p_holm"] = multipletests(
                result.loc[mask, "wilcoxon_p"], method="holm"
            )[1]
    return result



def fit_knowledge_item_model(
    knowledge_items: pd.DataFrame,
    analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if knowledge_items.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "model": "knowledge_item_logistic",
            "error": "No knowledge items found.",
        }])

    design_cols = [
        "user_id", "session_id", "character", "condition", "condition_position",
        "country", "voice_message_rate", "n_messages", "total_words",
    ]
    available = [c for c in design_cols if c in analysis.columns]
    design = analysis[available].drop_duplicates(
        ["user_id", "session_id", "character"]
    )
    frame = knowledge_items.merge(
        design, on=["user_id", "session_id", "character"], how="left"
    )
    frame = frame.dropna(subset=["is_correct", "condition", "character", "user_id"])
    if frame.empty or frame["condition"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame([{
            "model": "knowledge_item_logistic",
            "error": "Insufficient merged knowledge/design data.",
        }])

    terms = [
        "C(condition)", "C(character)", "C(stage_id)",
    ]
    if "country" in frame and frame["country"].notna().sum() >= 10 and frame["country"].nunique() >= 2:
        terms += ["C(country)", "C(condition):C(country)"]
    if "condition_position" in frame and frame["condition_position"].notna().sum() >= 10:
        terms += ["condition_position"]
    if "voice_message_rate" in frame:
        terms += ["voice_message_rate"]
    if "n_messages" in frame:
        terms += ["n_messages"]

    formula = "is_correct ~ " + " + ".join(terms)

    try:
        model = smf.gee(
            formula,
            groups="user_id",
            data=frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        result = flatten_model(model, f"Binomial GEE: {formula}", "is_correct")
        result["odds_ratio"] = np.exp(result["estimate"])
        result["or_ci_low"] = np.exp(result["ci_low"])
        result["or_ci_high"] = np.exp(result["ci_high"])
        result = holm_adjust_frame(result, "p_value")
        return result, pd.DataFrame()
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame([{
            "model": f"Binomial GEE: {formula}",
            "error": str(exc),
        }])


def build_paired_difference_dataset(
    data: pd.DataFrame,
    outcomes: list[str],
    participant_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create one row per participant-session with Avatar-minus-Chat differences."""
    participant_columns = participant_columns or []
    keys = ["user_id", "session_id"]

    metadata_cols = [
        c for c in participant_columns
        if c in data.columns and c not in keys
    ]
    metadata = data[keys + metadata_cols].drop_duplicates(keys)

    result = metadata.copy()
    for outcome in outcomes:
        if outcome not in data.columns:
            continue
        wide = data.pivot_table(
            index=keys,
            columns="condition",
            values=outcome,
            aggfunc="mean",
        )
        if not {"Avatar", "Chat"}.issubset(wide.columns):
            continue
        difference = (
            pd.to_numeric(wide["Avatar"], errors="coerce")
            - pd.to_numeric(wide["Chat"], errors="coerce")
        ).rename(f"{outcome}_avatar_minus_chat").reset_index()
        result = result.merge(difference, on=keys, how="left")
    return result


def mcnemar_condition_test(
    data: pd.DataFrame,
    binary_outcome: str,
) -> pd.DataFrame:
    """McNemar test for a paired binary outcome in Chat versus Avatar."""
    if binary_outcome not in data.columns:
        return pd.DataFrame()

    wide = data.pivot_table(
        index=["user_id", "session_id"],
        columns="condition",
        values=binary_outcome,
        aggfunc="max",
    )
    if not {"Avatar", "Chat"}.issubset(wide.columns):
        return pd.DataFrame()

    paired = wide[["Avatar", "Chat"]].dropna().astype(bool)
    chat_no_avatar_yes = int((~paired["Chat"] & paired["Avatar"]).sum())
    chat_yes_avatar_no = int((paired["Chat"] & ~paired["Avatar"]).sum())

    discordant = chat_no_avatar_yes + chat_yes_avatar_no
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        # Exact two-sided binomial form of McNemar's test.
        p_value = float(
            stats.binomtest(
                min(chat_no_avatar_yes, chat_yes_avatar_no),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        statistic = (
            (abs(chat_no_avatar_yes - chat_yes_avatar_no) - 1) ** 2
            / discordant
        )

    return pd.DataFrame([{
        "outcome": binary_outcome,
        "n_pairs": len(paired),
        "chat_no_avatar_yes": chat_no_avatar_yes,
        "chat_yes_avatar_no": chat_yes_avatar_no,
        "discordant_pairs": discordant,
        "mcnemar_statistic_continuity_corrected": statistic,
        "p_value": p_value,
    }])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figures(data: pd.DataFrame, out_dir: Path) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("n_messages", "Number of participant messages"),
        ("total_words", "Total participant words"),
        ("interaction_duration_minutes", "Interaction duration (minutes)"),
        ("voice_message_rate", "Proportion of voice messages"),
        ("knowledge_accuracy", "Knowledge accuracy"),
        ("satisfaction", "Satisfaction"),
        ("character_connection", "Character connection"),
        ("post_importance", "Post-interaction Holocaust-learning importance"),
        ("importance_change", "Change in Holocaust-learning importance"),
    ]

    for metric, label in metrics:
        if metric not in data.columns:
            continue
        plot = data[["condition", metric]].copy()
        plot[metric] = pd.to_numeric(plot[metric], errors="coerce")
        plot = plot.dropna()
        if plot.empty or plot["condition"].nunique() < 2:
            continue

        summaries = plot.groupby("condition")[metric].agg(["mean", "sem", "count"])
        order = [x for x in ["Chat", "Avatar"] if x in summaries.index]
        summaries = summaries.loc[order]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(
            summaries.index,
            summaries["mean"],
            yerr=1.96 * summaries["sem"],
            capsize=5,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("Condition")
        ax.set_title(f"{label} by condition (mean ± 95% CI)")
        fig.tight_layout()
        fig.savefig(figures / f"{metric}_by_condition.png", dpi=200)
        plt.close(fig)

    # Participant-level paired difference plots.
    paired_metrics = ["knowledge_accuracy", "n_messages", "total_words",
                      "interaction_duration_minutes", "character_connection",
                      "satisfaction", "post_importance", "importance_change"]
    for metric in paired_metrics:
        if metric not in data:
            continue
        wide = data.pivot_table(
            index=["user_id", "session_id"],
            columns="condition",
            values=metric,
            aggfunc="mean",
        )
        if not {"Chat", "Avatar"}.issubset(wide.columns):
            continue
        wide = wide[["Chat", "Avatar"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(wide) < 3:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for _, row in wide.iterrows():
            ax.plot(["Chat", "Avatar"], [row["Chat"], row["Avatar"]], alpha=0.25)
        ax.plot(
            ["Chat", "Avatar"],
            [wide["Chat"].mean(), wide["Avatar"].mean()],
            marker="o",
            linewidth=3,
            label="Mean",
        )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Within-participant change: {metric.replace('_', ' ')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{metric}_paired.png", dpi=200)
        plt.close(fig)
