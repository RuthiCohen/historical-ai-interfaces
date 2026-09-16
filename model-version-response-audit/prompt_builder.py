"""Reconstruct avatar PHP + JS prompt payloads for historical turns."""

from __future__ import annotations

import json
import re
from typing import Iterable

from config import DEFAULT_ANSWERS_LENGTH_PREFERENCE, PERSON_DISPLAY_NAMES


def buildRoleInstructions(personNameKey: str, personPromptName: str | None = None) -> str:
    """Mirror makeExternalGeminiRequest_v1.php role branches for study persons."""
    display = personPromptName or PERSON_DISPLAY_NAMES.get(personNameKey, personNameKey)

    if personNameKey == "BenLesser":
        return (
            "You are playing the role of Holocaust Survivor Ben Lesser. "
            "A user will ask you questions and you should answer them as Ben Lesser."
            " You are getting the whole chat history, try to maintain a dialogue with the user."
            " Answer all questions like Ben Lesser, you can use provided interviews with Ben Lesser. "
            "Each answer can be short or long, but not be more than 5 sentences."
            " Here are some facts about Ben Lesser's life, stories and other data, "
            "use them as a source of inspiration and basic knowledge,"
            " but you may search for missing facts across the web while trying to mimic Ben Lesser's speech style."
            " Here is the whole chat history with you, use it only to understand the user's new question "
            "in the context of the chat history,"
            " but you should not repeat parts of your previous answers in the next one."
            " At the end, there will be a new question for you, answer only this question."
        )

    # Markel and any other non-special-cased survivor use the generic PHP branch
    return (
        f"You are playing the role of Holocaust Survivor {display}. "
        f"A user will ask you questions and you should answer them as {display}"
        " You are getting the whole chat history, try to maintain a dialogue with the user."
        f" Answer all questions like {display}, you can use provided interviews with {display}. "
        "Each answer can be short or long, but not be more than 5 sentences."
        f" Here are some facts about {display}'s life, stories and other data, "
        "use them as a source of inspiration and basic knowledge,"
        f" but you may search for missing facts across the web while trying to mimic {display}'s speech style."
        " Here is the whole chat history with you, use it only to understand the user's new question "
        "in the context of the chat history,"
        " but you should not repeat parts of your previous answers in the next one."
        " Also, do not repeat user's name in each answer."
        " At the end, there will be a new question for you, answer only this question."
    )


def buildLengthPreferenceLine(answersLengthPreference: int = DEFAULT_ANSWERS_LENGTH_PREFERENCE) -> str:
    if answersLengthPreference == 3:
        return (
            "\n- Prioritize short answers:  Your answers should be 2-3 sentences "
            "(should not exceed 3 sentences)."
        )
    if answersLengthPreference == 1:
        return (
            "\n- Provide a very short answer:  Your answers should not exceed 1 sentence "
            "(strict limit)."
        )
    if answersLengthPreference == 2:
        return (
            "\n- Provide short answers:  Your answers should not exceed 2 sentences "
            "(strict limit)."
        )
    if answersLengthPreference == 4:
        return "\n- Prioritize not very long answers:  Your answers should not exceed 4 sentences."
    return ""


def buildFormatRules() -> str:
    return (
        "\n- Be specific and concise in your answers"
        "\n- Do not include in the answer body movements or other not related text that cannot be spoken"
        "\n- Do not include in the answer html tags or other characters like stars or parentheses"
        "\n- Provide an output in json format that contains an answer"
        '\n- Strict to the following format: {"answer": ""}'
        "\n- Do not use json or html unsafe characters"
        "\n- Use html markdown for texts, lists and fonts"
    )


def buildChatHistoryBlock(
    priorTurns: Iterable[tuple[str, str]],
    newQuestion: str,
) -> str:
    """Mirror js/main.js history string (current question is NOT in history yet)."""
    parts = ["**Here is the whole chat history with you:**", ""]
    for userQuestion, agentAnswer in priorTurns:
        parts.append("User question:")
        parts.append(userQuestion)
        parts.append("Your answer:")
        parts.append(agentAnswer)
    parts.append("")
    parts.append("**New Question:**")
    parts.append("")
    parts.append(newQuestion)
    return "\n".join(parts)


def buildFullPrompt(
    personNameKey: str,
    priorTurns: list[tuple[str, str]],
    newQuestion: str,
    answersLengthPreference: int = DEFAULT_ANSWERS_LENGTH_PREFERENCE,
    personPromptName: str | None = None,
) -> tuple[str, str]:
    """Return (reconstructed_full_prompt, chat_history_text)."""
    chatHistoryText = buildChatHistoryBlock(priorTurns, newQuestion)
    fullPrompt = (
        buildRoleInstructions(personNameKey, personPromptName)
        + buildLengthPreferenceLine(answersLengthPreference)
        + buildFormatRules()
        + "\n"
        + chatHistoryText
    )
    return fullPrompt, chatHistoryText


def extractAnswerText(rawResponse: str) -> str:
    """Strip {"answer": "..."} wrapper when present; otherwise return cleaned text."""
    if rawResponse is None:
        return ""
    text = str(rawResponse).strip()
    if not text:
        return ""

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            answer = parsed["answer"]
            if isinstance(answer, (dict, list)):
                return json.dumps(answer, ensure_ascii=False)
            return str(answer).strip()
    except json.JSONDecodeError:
        pass

    # Soft extract from partially-valid JSON-ish output
    match = re.search(r'"answer"\s*:\s*"(.*)"\s*}?\s*$', text, re.DOTALL)
    if match:
        candidate = match.group(1)
        candidate = candidate.replace("\\n", "\n").replace('\\"', '"')
        return candidate.strip()

    return text


def wordCount(text: str) -> int:
    if not text or not str(text).strip():
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))
