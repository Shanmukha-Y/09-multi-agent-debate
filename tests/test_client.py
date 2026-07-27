"""Regression test for a real bug another builder hit on this project's
shared Ollama server: some client stacks can raise httpx.ReadTimeout from
deep inside a request, and httpx.ReadTimeout is NOT a TimeoutError
subclass — a wrapper that only catches builtin TimeoutError (and library
exceptions that subclass it) lets that exception sail through uncaught and
crash mid-debate.

OllamaClient.complete is decorated with a tenacity retry over
_TRANSPORT_EXCEPTIONS, which explicitly includes httpx.TimeoutException
(the parent of ReadTimeout/ConnectTimeout/WriteTimeout/PoolTimeout) — see
client.py's module docstring for why the openai SDK's own wrapping doesn't
make this belt-and-suspenders catch redundant. No network: the underlying
openai client is monkeypatched to raise directly.
"""

from __future__ import annotations

import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from debate.client import OllamaClient


def _fake_completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="fake",
        object="chat.completion",
        created=0,
        model="qwen3.5:9b",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TimeoutThenSucceed:
    """Raises httpx.ReadTimeout on the first N calls, then returns a fake
    completion — simulating a flaky shared server that eventually responds."""

    def __init__(self, fail_times: int, content: str = '{"ok": true}'):
        self.fail_times = fail_times
        self.content = content
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise httpx.ReadTimeout("simulated read timeout under load")
        return _fake_completion(self.content)


class AlwaysTimeout:
    def __call__(self, *args, **kwargs):
        raise httpx.ReadTimeout("simulated persistent read timeout")


class TestHttpxReadTimeoutIsCaught:
    def test_read_timeout_is_retried_and_eventually_succeeds(self, monkeypatch):
        client = OllamaClient(model="qwen3.5:9b", base_url="http://localhost:11434/v1")
        fake_create = TimeoutThenSucceed(fail_times=2)  # fails twice, succeeds on 3rd (within TRANSPORT_MAX_ATTEMPTS=3)
        monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

        text, tokens = client.complete("system", "user")

        assert text == '{"ok": true}'
        assert tokens == 15
        assert fake_create.call_count == 3

    def test_persistent_read_timeout_raises_httpx_readtimeout_not_silently_swallowed(self, monkeypatch):
        client = OllamaClient(model="qwen3.5:9b", base_url="http://localhost:11434/v1")
        monkeypatch.setattr(client._client.chat.completions, "create", AlwaysTimeout())

        # It must be caught, retried per TRANSPORT_MAX_ATTEMPTS, and THEN
        # re-raised (reraise=True) -- never silently swallowed or hung.
        with pytest.raises(httpx.ReadTimeout):
            client.complete("system", "user")

    def test_read_timeout_is_an_instance_of_the_caught_transport_exceptions(self):
        # The specific bug shape: httpx.ReadTimeout is NOT a TimeoutError,
        # so a wrapper catching only TimeoutError would miss it entirely.
        from debate.client import _TRANSPORT_EXCEPTIONS

        exc = httpx.ReadTimeout("x")
        assert not isinstance(exc, TimeoutError)
        assert isinstance(exc, _TRANSPORT_EXCEPTIONS)

    @pytest.mark.parametrize(
        "exc_cls", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout]
    )
    def test_all_httpx_timeout_subclasses_are_caught(self, exc_cls):
        from debate.client import _TRANSPORT_EXCEPTIONS

        assert isinstance(exc_cls("x"), _TRANSPORT_EXCEPTIONS)
