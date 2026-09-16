"""Vertex AI generateContent client matching GoogleGeminiApiController.php."""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path
from typing import Any

import requests

from config import (
    GCS_FACTS_BUCKET,
    LOCAL_FACTS_FILES,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    TOKEN_PATH,
    TOP_P,
    VERTEX_API_ENDPOINT,
    VERTEX_LOCATION_ID,
    VERTEX_PROJECT_ID,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY_MS = 1000
CONNECT_TIMEOUT_SECONDS = 10
RESPONSE_TIMEOUT_SECONDS = 60


class GeminiClientError(Exception):
    def __init__(self, message: str, httpStatus: int | None = None, body: str | None = None):
        super().__init__(message)
        self.httpStatus = httpStatus
        self.body = body


def readToken(tokenPath: Path = TOKEN_PATH) -> str:
    for _ in range(10):
        if tokenPath.exists():
            token = tokenPath.read_text(encoding="utf-8").strip()
            if token:
                return token
        time.sleep(1)
    return ""


def waitForFreshToken(tokenPath: Path = TOKEN_PATH) -> str:
    """Pause until the user refreshes the Vertex token file and presses Enter."""
    while True:
        print(
            "\n[AUTH] Gemini token missing/expired/rejected.\n"
            f"  Update: {tokenPath}\n"
            "  Then press Enter to retry (or Ctrl+C to abort)..."
        )
        try:
            input()
        except EOFError:
            time.sleep(5)
        token = readToken(tokenPath)
        if token:
            return token


def buildRequestBody(
    mainRequestText: str,
    personNameKey: str,
    useGcsFacts: bool = True,
    useInlineFactsFallback: bool = False,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": mainRequestText}]

    if useGcsFacts and not useInlineFactsFallback:
        parts.append(
            {
                "fileData": {
                    "fileUri": f"gs://{GCS_FACTS_BUCKET}/{personNameKey}_Facts.txt",
                    "mimeType": "text/plain",
                }
            }
        )
    elif useInlineFactsFallback:
        factsPath = LOCAL_FACTS_FILES.get(personNameKey)
        if factsPath and factsPath.exists():
            encoded = base64.b64encode(factsPath.read_bytes()).decode("ascii")
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "text/plain",
                        "data": encoded,
                    }
                }
            )

    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "topP": TOP_P,
        },
    }


def parseResponseText(responseStr: str) -> str:
    responseArr = json.loads(responseStr)
    if "error" in responseArr:
        raise GeminiClientError(
            responseArr["error"].get("message", "Vertex API error"),
            httpStatus=responseArr["error"].get("code"),
            body=responseStr,
        )
    text = responseArr["candidates"][0]["content"]["parts"][0]["text"].strip()
    text = text.replace("```json", "").replace("```", "")
    return text


def isAuthError(httpStatus: int | None, body: str) -> bool:
    if httpStatus in (401, 403):
        return True
    try:
        parsed = json.loads(body) if body else {}
        code = parsed.get("error", {}).get("code")
        status = str(parsed.get("error", {}).get("status", "")).upper()
        if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
            return True
    except json.JSONDecodeError:
        pass
    return False


def isRetryableError(httpStatus: int | None, body: str) -> bool:
    if httpStatus in (429, 500, 503):
        return True
    if not body:
        return True
    try:
        parsed = json.loads(body)
        code = parsed.get("error", {}).get("code")
        return code in (429, 500, 503)
    except json.JSONDecodeError:
        return False


def generateContent(
    promptText: str,
    modelId: str,
    personNameKey: str,
    tokenPath: Path = TOKEN_PATH,
    preferGcsFacts: bool = True,
) -> tuple[str, int | None, str | None, str]:
    """
    Call Vertex generateContent.

    Returns: (response_text, http_status, error_message, facts_mode)
    facts_mode is 'gcs' or 'inline'.
    """
    url = (
        f"https://{VERTEX_API_ENDPOINT}/v1/projects/{VERTEX_PROJECT_ID}/"
        f"locations/{VERTEX_LOCATION_ID}/publishers/google/models/{modelId}:generateContent"
    )

    useInline = False
    factsMode = "gcs" if preferGcsFacts else "none"

    while True:
        token = readToken(tokenPath)
        if not token:
            token = waitForFreshToken(tokenPath)

        body = buildRequestBody(
            promptText,
            personNameKey,
            useGcsFacts=preferGcsFacts and not useInline,
            useInlineFactsFallback=useInline,
        )
        if preferGcsFacts and useInline:
            factsMode = "inline"
        elif preferGcsFacts:
            factsMode = "gcs"
        else:
            factsMode = "none"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        lastStatus: int | None = None
        lastBody = ""

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                delayMs = RETRY_BASE_DELAY_MS * (2 ** (attempt - 1)) + random.randint(0, 500)
                time.sleep(delayMs / 1000.0)

            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=(CONNECT_TIMEOUT_SECONDS, RESPONSE_TIMEOUT_SECONDS),
                )
                lastStatus = response.status_code
                lastBody = response.text
            except requests.RequestException as exc:
                lastStatus = None
                lastBody = str(exc)
                if attempt < MAX_RETRIES:
                    continue
                raise GeminiClientError(f"Request failed: {exc}") from exc

            if isAuthError(lastStatus, lastBody):
                waitForFreshToken(tokenPath)
                break  # restart outer while with new token

            if lastStatus == 200:
                try:
                    text = parseResponseText(lastBody)
                    return text, lastStatus, None, factsMode
                except (GeminiClientError, KeyError, IndexError, json.JSONDecodeError) as exc:
                    # GCS file attachment may fail — retry once with inline facts
                    if preferGcsFacts and not useInline and "file" in str(exc).lower():
                        useInline = True
                        factsMode = "inline"
                        break
                    if isRetryableError(lastStatus, lastBody) and attempt < MAX_RETRIES:
                        continue
                    return "", lastStatus, str(exc), factsMode

            # Non-200: maybe GCS file issue → fallback to inline
            if (
                preferGcsFacts
                and not useInline
                and lastStatus in (400, 404)
                and ("file" in lastBody.lower() or "not found" in lastBody.lower())
            ):
                useInline = True
                factsMode = "inline"
                break

            if isRetryableError(lastStatus, lastBody) and attempt < MAX_RETRIES:
                continue

            errMsg = lastBody[:500] if lastBody else f"HTTP {lastStatus}"
            return "", lastStatus, errMsg, factsMode
        else:
            # exhausted retries without auth break / facts fallback break
            errMsg = lastBody[:500] if lastBody else f"HTTP {lastStatus}"
            return "", lastStatus, errMsg, factsMode
