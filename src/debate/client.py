"""Thin wrapper around the OpenAI SDK pointed at Ollama's compatible endpoint.

Kept deliberately small, mirroring project-01's soa/client.py: this module's
only job is turning (system_prompt, user_prompt) into raw completion text,
retrying on *transport* failures (connection drops, timeouts, 5xx) via
tenacity. Schema-validation retries are a different concern, handled in
structured.py, not hidden in here.

Every debate is 8-10 sequential calls against a *shared* local Ollama server
(other builders are hitting it too), so timeouts here are generous and
raw socket TimeoutError is caught alongside the SDK's own timeout exception.
"""

from __future__ import annotations

import socket

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from debate import config

_TRANSPORT_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
    socket.timeout,
    TimeoutError,
)


class OllamaClient:
    """Wraps a chat-completion call to a local Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> None:
        self.model = model or config.MODEL
        self._client = openai.OpenAI(
            base_url=base_url or config.OLLAMA_BASE_URL,
            api_key=api_key or config.API_KEY,
            timeout=timeout or config.REQUEST_TIMEOUT_SECONDS,
        )
        self.temperature = temperature

    @retry(
        retry=retry_if_exception_type(_TRANSPORT_EXCEPTIONS),
        stop=stop_after_attempt(config.TRANSPORT_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Run one chat completion. Returns (raw_text, total_tokens).

        Requests JSON-object mode so the model is nudged toward emitting a
        single JSON object rather than prose; this is best-effort on a small
        local model, which is exactly why structured.py's validate/repair
        loop exists.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        total_tokens = response.usage.total_tokens if response.usage else 0
        return text, total_tokens
