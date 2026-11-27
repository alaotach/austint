"""LLM client abstractions with multi-provider support, budget guards, retries, and strict JSON parsing."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Truncated-JSON auto-repair
# ---------------------------------------------------------------------------

def _repair_truncated_json(text: str) -> str:
    """Best-effort repair of a truncated JSON object by closing open strings and braces.

    Only called after a primary parse failure to give a second chance. Returns
    the input unchanged if the heuristic cannot produce valid JSON.
    """
    t = text.strip()
    if not t:
        return t

    # Count unclosed structural characters
    # Walk through respecting string escaping
    in_string = False
    escape_next = False
    stack: list[str] = []  # '{' or '['

    for ch in t:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch in ("}", "]"):
            if stack:
                stack.pop()

    close_map = {"{": "}", "[": "]"}
    # If we're mid-string, close it first
    if in_string:
        t += '"'
    # Close any trailing comma before closing braces
    t = re.sub(r",\s*$", "", t.rstrip())
    # Close open structural elements in reverse order
    for opener in reversed(stack):
        t += close_map[opener]

    return t


def _load_dotenv() -> None:
    """Populate os.environ from local .env if present (without overriding existing vars)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


_load_dotenv()


class LLMBudgetExceededError(RuntimeError):
    """Raised when LLM call budget has been exhausted."""
    pass


class LLMRateLimitError(RuntimeError):
    """Raised immediately when provider returns 429 so callers fall back to ML without spamming."""
    pass


_RATE_LIMIT_COOLDOWN_UNTIL: float = 0.0
import threading as _threading
_ERYZA_LOCK = _threading.Lock()
_ERYZA_LAST_CALL: float = 0.0
_ERYZA_MIN_INTERVAL_SEC: float = 2.0  # <= 30 RPM (safely under 40 RPM limit)


def reset_rate_limit_cooldown() -> None:
    """Reset shared rate-limit cooldown (primarily for tests)."""
    global _RATE_LIMIT_COOLDOWN_UNTIL
    _RATE_LIMIT_COOLDOWN_UNTIL = 0.0


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

        if not cleaned:
            raise ValueError("Failed to decode JSON from LLM: empty response body")

        # Pass 1: direct parse
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc_primary:
            # Pass 2: extract first {...} block if wrapped in prose
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            parsed = None
            if start != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass

            # Pass 3: auto-repair truncated JSON (handles unterminated strings / missing braces)
            if parsed is None:
                candidate = _repair_truncated_json(cleaned if start == -1 else cleaned[start:])
                try:
                    parsed = json.loads(candidate)
                    logger.info("JSON auto-repair succeeded (%d chars repaired)", len(candidate) - len(cleaned))
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to decode JSON from LLM: {exc_primary}") from exc_primary

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

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
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 15))
                        logger.warning("Anthropic 429 rate-limited. Waiting %ds.", wait)
                        time.sleep(wait)
                        continue
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
                if attempt == 0:
                    time.sleep(2)

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
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 15))
                        logger.warning("Gemini 429 rate-limited. Waiting %ds.", wait)
                        time.sleep(wait)
                        continue
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
                if attempt == 0:
                    time.sleep(2)

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


class OpenRouterClient(BaseClient):
    """OpenAI-compatible / OpenRouter client (supports Hack Club proxy and multimodal image URLs)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        call_budget: int = 20,
        timeout: float = 45.0,
    ) -> None:
        super().__init__(call_budget=call_budget, timeout=timeout)
        _load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
        self.base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL",
            "https://ai.hackclub.com/proxy/v1/chat/completions",
        )

    def complete_json(self, system: str, user: str, max_tokens: int = 2048) -> dict[str, Any]:
        return self.complete_multimodal_json(system=system, user=user, image_urls=None, max_tokens=max_tokens)

    def complete_multimodal_json(
        self,
        system: str,
        user: str,
        image_urls: list[str] | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        global _RATE_LIMIT_COOLDOWN_UNTIL
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured in environment.")

        now = time.monotonic()
        if now < _RATE_LIMIT_COOLDOWN_UNTIL:
            remaining = int(_RATE_LIMIT_COOLDOWN_UNTIL - now)
            raise LLMRateLimitError(
                f"OpenRouter cooldown active ({remaining}s remaining); skipping LLM call."
            )

        self._check_budget()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Filter out image formats that vision providers commonly reject with 400
        valid_images = [
            u for u in (image_urls or [])[:3]
            if isinstance(u, str)
            and (
                (u.startswith(("http://", "https://")) and not u.lower().split("?")[0].endswith((".svg", ".ico")))
                or u.startswith("data:image/")
            )
        ]

        if valid_images:
            user_content: Any = [{"type": "text", "text": user}]
            for img_url in valid_images:
                user_content.append({"type": "image_url", "image_url": {"url": img_url}})
        else:
            user_content = user

        # Reasoning models on openrouter/free (e.g. ernie-5.5, nemotron-reasoning, deepseek-r1)
        # consume output tokens in `message.reasoning` before emitting `message.content`.
        # Ensure at least 8192 tokens so reasoning doesn't starve the JSON output.
        effective_max_tokens = max(max_tokens, 8192)

        chosen_model = self.model
        if valid_images and "eryzalabs" in (self.base_url or "").lower():
            chosen_model = os.getenv("ERYZALABS_VISION_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")

        payload: dict[str, Any] = {
            "model": chosen_model,
            "max_tokens": effective_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }

        # Fail fast on 429; allow at most 1 quick retry (1.5s) on transient errors
        _RETRY_DELAYS = [0.0, 1.5]
        _MAX_ATTEMPTS = len(_RETRY_DELAYS)

        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS):
            if delay > 0:
                time.sleep(delay)
            if "eryzalabs" in (self.base_url or "").lower():
                global _ERYZA_LAST_CALL
                with _ERYZA_LOCK:
                    wait_s = _ERYZA_MIN_INTERVAL_SEC - (time.monotonic() - _ERYZA_LAST_CALL)
                    if wait_s > 0:
                        time.sleep(wait_s)
                    _ERYZA_LAST_CALL = time.monotonic()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.base_url, json=payload, headers=headers)

                    # 429 rate-limit: trip circuit breaker immediately — NEVER spam retries
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", 0) or 0)
                        cooldown = max(retry_after, 90)
                        _RATE_LIMIT_COOLDOWN_UNTIL = time.monotonic() + cooldown
                        logger.warning(
                            "OpenRouter 429 rate-limited — tripping %ds cooldown and falling back to ML.",
                            cooldown,
                        )
                        raise LLMRateLimitError("429 Too Many Requests — circuit breaker tripped")

                    # 400 Bad Request: often caused by an unreachable/unsupported image_url
                    # or a text-only model selected by the router; strip images and fall back.
                    if resp.status_code == 400 and isinstance(payload["messages"][1]["content"], list):
                        logger.warning(
                            "OpenRouter 400 with multimodal images (attempt %d/%d): %s — falling back to text-only",
                            attempt + 1, _MAX_ATTEMPTS, resp.text[:200],
                        )
                        payload["messages"][1]["content"] = user
                        payload["model"] = self.model
                        last_exc = httpx.HTTPStatusError(
                            f"400 Bad Request (multimodal fallback): {resp.text[:150]}",
                            request=resp.request,
                            response=resp,
                        )
                        continue

                    resp.raise_for_status()

                    data = resp.json()
                    choices = data.get("choices") or []
                    raw_text = ""
                    finish_reason = ""
                    has_reasoning = False
                    if choices:
                        finish_reason = str(choices[0].get("finish_reason") or "")
                        msg = choices[0].get("message") or {}
                        raw_text = str(msg.get("content") or "")
                        has_reasoning = bool(msg.get("reasoning") or msg.get("reasoning_content"))

                    # Detect empty content body early (often reasoning model hit max_tokens)
                    if not raw_text.strip():
                        if finish_reason == "length" or has_reasoning:
                            payload["max_tokens"] = min(int(payload["max_tokens"]) * 2, 16384)
                            raise ValueError(
                                f"LLM returned empty content (model={data.get('model')}, "
                                f"finish_reason={finish_reason!r}, reasoning_exhausted={has_reasoning}); "
                                f"bumping max_tokens to {payload['max_tokens']}"
                            )
                        raise ValueError(
                            f"LLM returned empty content body (model={data.get('model')}, finish_reason={finish_reason!r})"
                        )

                    usage = data.get("usage") or {}
                    token_count = int(
                        usage.get("total_tokens")
                        or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                    )
                    return self._parse_payload(raw_text, token_count)

            except LLMRateLimitError:
                raise
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "OpenRouter request attempt %d/%d failed: %s",
                    attempt + 1, _MAX_ATTEMPTS, exc,
                )

        raise RuntimeError(
            f"OpenRouter completion failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc


class G4FClient(BaseClient):
    """Local gpt4free client with automatic OpenAI-compatible fallback (EryzaLabs) on failure."""

    def __init__(
        self,
        provider_name: str | None = None,
        model: str | None = None,
        call_budget: int = 20,
        timeout: float = 45.0,
    ) -> None:
        super().__init__(call_budget=call_budget, timeout=timeout)
        _load_dotenv()
        self.provider_name = provider_name or os.getenv("G4F_PROVIDER", "GithubCopilot")
        self.model = model or os.getenv("G4F_MODEL", "")
        self.fallback_base_url = os.getenv(
            "ERYZALABS_BASE_URL",
            "https://llm.eryzalabs.cloud/v1/chat/completions",
        )
        self.fallback_api_key = os.getenv("ERYZALABS_API_KEY", "")
        self.fallback_model = os.getenv("ERYZALABS_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.fallback_vision_model = os.getenv(
            "ERYZALABS_VISION_MODEL",
            "nvidia/nemotron-3.5-super-vl-120b-a12b",
        )

    def _fallback_client(self, *, multimodal: bool = False) -> OpenRouterClient | None:
        if not self.fallback_api_key:
            return None
        remaining = max(1, self.call_budget - self.calls_made)
        return OpenRouterClient(
            api_key=self.fallback_api_key,
            model=self.fallback_vision_model if multimodal else self.fallback_model,
            base_url=self.fallback_base_url,
            call_budget=remaining,
            timeout=self.timeout,
        )

    def complete_json(self, system: str, user: str, max_tokens: int = 2048) -> dict[str, Any]:
        return self.complete_multimodal_json(system=system, user=user, image_urls=None, max_tokens=max_tokens)

    def complete_multimodal_json(
        self,
        system: str,
        user: str,
        image_urls: list[str] | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        self._check_budget()
        if not image_urls:
            try:
                import g4f
                from g4f.client import Client

                prov = getattr(g4f.Provider, self.provider_name, None)
                if prov is None:
                    raise ValueError(f"Unknown g4f provider: {self.provider_name}")
                model_name = self.model or getattr(prov, "default_model", "") or "gpt-4"
                g4f_client = Client(provider=prov)
                resp = g4f_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                raw_text = str((resp.choices[0].message.content if resp.choices else "") or "")
                usage = getattr(resp, "usage", None)
                token_count = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                if token_count <= 0:
                    token_count = max(1, (len(system) + len(user) + len(raw_text)) // 4)
                self.last_tokens = token_count
                return self._parse_payload(raw_text, token_count)
            except Exception as exc:
                fb = self._fallback_client(multimodal=False)
                if fb is None:
                    raise
                logger.warning("g4f failed (%s); falling back to %s", exc, self.fallback_base_url)
                res = fb.complete_json(system=system, user=user, max_tokens=max_tokens)
                self.calls_made += 1
                self.last_tokens = int(res.get("_token_usage", 0))
                return res

        fb = self._fallback_client(multimodal=True)
        if fb is None:
            return self.complete_json(system=system, user=user, max_tokens=max_tokens)
        res = fb.complete_multimodal_json(
            system=system,
            user=user,
            image_urls=image_urls,
            max_tokens=max_tokens,
        )
        self.calls_made += 1
        self.last_tokens = int(res.get("_token_usage", 0))
        return res


def get_llm_client(provider: str | None = None, call_budget: int = 20) -> LLMClient:
    """Factory creating configured LLM client instance."""
    _load_dotenv()
    default_prov = os.getenv("LLM_PROVIDER") or (
        "eryzalabs" if os.getenv("ERYZALABS_API_KEY") else "anthropic"
    )
    chosen = (provider or default_prov).lower()
    if chosen in ("eryzalabs", "eryza"):
        return OpenRouterClient(
            api_key=os.getenv("ERYZALABS_API_KEY", ""),
            model=os.getenv("ERYZALABS_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"),
            base_url=os.getenv("ERYZALABS_BASE_URL", "https://llm.eryzalabs.cloud/v1/chat/completions"),
            call_budget=call_budget,
        )
    if chosen in ("openrouter", "openai", "hackclub"):
        return OpenRouterClient(call_budget=call_budget)
    if chosen == "g4f":
        if os.getenv("G4F_BASE_URL"):
            return OpenRouterClient(
                api_key=os.getenv("G4F_API_KEY", "g4f"),
                model=os.getenv("G4F_MODEL", "auto"),
                base_url=os.environ["G4F_BASE_URL"],
                call_budget=call_budget,
            )
        return G4FClient(call_budget=call_budget)
    if chosen == "anthropic":
        return AnthropicClient(call_budget=call_budget)
    if chosen == "gemini":
        return GeminiClient(call_budget=call_budget)
    if chosen == "fake":
        return FakeClient(call_budget=call_budget)
    raise ValueError(f"Unknown LLM provider '{provider}'. Choose 'openrouter', 'anthropic', 'gemini', or 'fake'.")
