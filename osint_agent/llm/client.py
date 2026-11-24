"""LLM client abstractions with multi-provider support, budget guards, retries, and strict JSON parsing."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


class LLMBudgetExceededError(RuntimeError):
    """Raised when LLM call budget has been exhausted."""
    pass


class LLMResult(dict):
    """Dictionary containing parsed JSON payload plus token usage metadata."""

    def __init__(self, data: dict[str, Any], token_usage: int = 0) -> None:
        super().__init__(data)
        self.token_usage = token_usage
        self["_token_usage"] = token_usage


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client with JSON output and token tracking."""

    def complete_json(self, system: str, user: str, max_tokens: int = 1000) -> dict[str, Any]:
        """Generate structured JSON response and track token usage."""
        ...


def strip_code_fences(text: str) -> str:
    """Strip markdown triple-backtick code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        # Match ```json ... ``` or ``` ... ```
        pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
        match = pattern.match(text)
        if match:
            return match.group(1).strip()
    return text


class BaseClient:
    """Base client with unified call budget tracking and retry wrapper."""

    def __init__(self, call_budget: int = 20, timeout: float = 10.0) -> None:
        self.call_budget = call_budget
        self.timeout = timeout
        self.calls_made = 0
        self.total_tokens = 0

    def _check_budget(self) -> None:
        if self.calls_made >= self.call_budget:
            logger.warning("LLM call budget exhausted (%d/%d)", self.calls_made, self.call_budget)
            raise LLMBudgetExceededError(f"LLM call budget of {self.call_budget} exhausted.")

    def _parse_payload(self, raw_text: str, token_count: int) -> LLMResult:
        cleaned = strip_code_fences(raw_text)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON from LLM: {exc}") from exc

        self.calls_made += 1
        self.total_tokens += token_count
        return LLMResult(parsed, token_usage=token_count)


class AnthropicClient(BaseClient):
    """Anthropic Claude API client using direct HTTP with retries and budget enforcement."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        call_budget: int = 20,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(call_budget=call_budget, timeout=timeout)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    def complete_json(self, system: str, user: str, max_tokens: int = 1000) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured in environment.")

        self._check_budget()
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        last_exc: Exception | None = None
        for attempt in range(2):  # One retry on failure
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_text = "".join(
                        c.get("text", "") for c in data.get("content", []) if c.get("type") == "text"
                    )
                    usage = data.get("usage", {})
                    token_count = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    return self._parse_payload(raw_text, token_count)
            except Exception as exc:
                last_exc = exc
                logger.warning("Anthropic request attempt %d failed: %s", attempt + 1, exc)

        raise RuntimeError(f"Anthropic completion failed after 1 retry: {last_exc}") from last_exc


class GeminiClient(BaseClient):
    """Google Gemini API client using REST generateContent with retries and budget enforcement."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        call_budget: int = 20,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(call_budget=call_budget, timeout=timeout)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def complete_json(self, system: str, user: str, max_tokens: int = 1000) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured in environment.")

        self._check_budget()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": max_tokens,
            },
        }

        last_exc: Exception | None = None
        for attempt in range(2):  # One retry on failure
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    raw_text = ""
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        raw_text = "".join(p.get("text", "") for p in parts)
                    usage = data.get("usageMetadata", {})
                    token_count = usage.get("totalTokenCount", 0)
                    return self._parse_payload(raw_text, token_count)
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini request attempt %d failed: %s", attempt + 1, exc)

        raise RuntimeError(f"Gemini completion failed after 1 retry: {last_exc}") from last_exc


class FakeClient(BaseClient):
    """Deterministic mock client for testing with scripted responses and call tracking."""

    def __init__(
        self,
        responses: list[dict[str, Any] | str] | None = None,
        call_budget: int = 20,
        tokens_per_call: int = 120,
        fail_attempts: int = 0,
    ) -> None:
        super().__init__(call_budget=call_budget)
        self.responses = list(responses or [])
        self.tokens_per_call = tokens_per_call
        self.history: list[dict[str, Any]] = []
        self.fail_attempts = fail_attempts
        self._current_attempt = 0

    def complete_json(self, system: str, user: str, max_tokens: int = 1000) -> dict[str, Any]:
        self._check_budget()
        self.history.append({"system": system, "user": user, "max_tokens": max_tokens})

        if self._current_attempt < self.fail_attempts:
            self._current_attempt += 1
            raise httpx.RequestError("Simulated network failure on attempt")

        if not self.responses:
            # Fallback default response
            raw = "{}"
        else:
            item = self.responses.pop(0)
            raw = json.dumps(item) if isinstance(item, dict) else str(item)

        return self._parse_payload(raw, self.tokens_per_call)


def get_llm_client(provider: str | None = None, call_budget: int = 20) -> LLMClient:
    """Factory creating configured LLM client instance."""
    chosen = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    if chosen == "anthropic":
        return AnthropicClient(call_budget=call_budget)
    if chosen == "gemini":
        return GeminiClient(call_budget=call_budget)
    if chosen == "fake":
        return FakeClient(call_budget=call_budget)
    raise ValueError(f"Unknown LLM provider '{provider}'. Choose 'anthropic', 'gemini', or 'fake'.")
