import json
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from app.config import (
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
)


@dataclass
class GeminiApiError(RuntimeError):
    message: str
    status_code: int = 0
    status: str = "UNKNOWN"

    @property
    def resource_exhausted(self) -> bool:
        return self.status_code == 429 or self.status == "RESOURCE_EXHAUSTED"

    def __str__(self) -> str:
        if self.status_code:
            return f"{self.status_code} {self.status}: {self.message}"
        return f"{self.status}: {self.message}"


def _api_key(explicit_api_key: str | None = None) -> str:
    api_key = (explicit_api_key or os.getenv(GEMINI_API_KEY_ENV) or "").strip()
    if not api_key:
        raise GeminiApiError(
            f"未設定 {GEMINI_API_KEY_ENV}",
            status="MISSING_API_KEY",
        )
    return api_key


def _extract_error(http_error: error.HTTPError) -> GeminiApiError:
    raw_body = http_error.read().decode("utf-8", errors="replace")
    status = "HTTP_ERROR"
    message = raw_body or http_error.reason

    try:
        payload = json.loads(raw_body)
        error_payload = payload.get("error", {})
        status = error_payload.get("status") or status
        message = error_payload.get("message") or message
    except json.JSONDecodeError:
        pass

    return GeminiApiError(
        message=message,
        status_code=http_error.code,
        status=status,
    )


def _extract_text(payload: dict[str, Any]) -> str:
    text_parts: list[str] = []

    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                text_parts.append(text.strip())

    answer = "\n".join(part for part in text_parts if part)
    if not answer:
        raise GeminiApiError("Gemini API 回傳空白內容", status="EMPTY_RESPONSE")

    return answer


def generate_with_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    api_key: str | None = None,
    base_url: str = GEMINI_API_BASE_URL,
    timeout: int = GEMINI_TIMEOUT_SECONDS,
) -> str:
    key = _api_key(api_key)
    model_name = model.strip() or GEMINI_MODEL
    url = (
        f"{base_url.rstrip('/')}/v1beta/models/"
        f"{parse.quote(model_name, safe='')}:generateContent?"
        f"{parse.urlencode({'key': key})}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    gemini_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(gemini_request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise _extract_error(exc) from exc
    except (error.URLError, TimeoutError, socket.timeout) as exc:
        raise GeminiApiError(str(exc), status="NETWORK_ERROR") from exc

    try:
        response_payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise GeminiApiError(
            "Gemini API 回傳內容不是有效 JSON",
            status="INVALID_RESPONSE",
        ) from exc

    return _extract_text(response_payload)
