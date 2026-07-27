"""The validate/repair retry loop every agent call in this system goes
through (Project 01 pattern): call the model in JSON mode, validate the
result against a Pydantic schema, and on failure retry once with the prior
bad output plus its field-level errors folded into the prompt. This is what
turns a 9B model's shaky first-try JSON compliance into reliably-typed
messages between debate stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from debate import config

T = TypeVar("T", bound=BaseModel)


class CompletionClient(Protocol):
    """Structural type for anything call_structured can drive — real or fake."""

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int]: ...


class SchemaEnforcementError(Exception):
    """Raised when max_attempts is exhausted without producing valid output."""

    def __init__(self, schema_name: str, attempts: int, last_digest: str, last_raw: str):
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_digest = last_digest
        self.last_raw = last_raw
        super().__init__(
            f"Failed to produce a valid {schema_name} after {attempts} attempt(s). "
            f"Last errors:\n{last_digest}"
        )


@dataclass
class StructuredResult:
    model: BaseModel
    attempts: int
    total_tokens: int


def _extract_json_candidate(raw: str) -> str:
    """Best-effort: pull the first {...} block out of a response, in case the
    model wraps JSON in prose or a markdown fence despite json_object mode."""
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


def _format_error_digest(errors: list[dict]) -> str:
    lines = []
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        err_type = err.get("type", "")
        input_repr = repr(err.get("input"))
        if len(input_repr) > 80:
            input_repr = input_repr[:77] + "..."
        lines.append(f"- `{loc}`: {msg}, got {input_repr} ({err_type})")
    return "\n".join(lines)


def _validate(raw: str, schema: type[T]) -> tuple[T | None, str]:
    candidate = _extract_json_candidate(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"- JSON parsing failed: {exc.msg} at line {exc.lineno}, column {exc.colno}"
    try:
        return schema.model_validate(parsed), ""
    except ValidationError as exc:
        return None, _format_error_digest(exc.errors())


def _build_repair_prompt(original_user_prompt: str, schema: type[BaseModel], previous_output: str, digest: str) -> str:
    return (
        f"{original_user_prompt}\n\n"
        "Your previous response was invalid. Fix it and return ONLY a corrected "
        f"JSON object matching this schema: {schema.__name__}.\n\n"
        f"Previous (invalid) response:\n{previous_output}\n\n"
        f"Validation errors:\n{digest}\n\n"
        "Return only the corrected JSON object, no other text."
    )


def call_structured(
    client: CompletionClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
) -> StructuredResult:
    """Call `client.complete` and validate the result against `schema`,
    retrying with a repair prompt on validation failure. Raises
    SchemaEnforcementError if `max_attempts` is exhausted."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    total_tokens = 0
    previous_output: str | None = None
    previous_digest: str | None = None

    for attempt in range(1, max_attempts + 1):
        prompt = (
            user_prompt
            if attempt == 1
            else _build_repair_prompt(user_prompt, schema, previous_output or "", previous_digest or "")
        )
        raw, tokens = client.complete(system_prompt, prompt)
        total_tokens += tokens

        model, digest = _validate(raw, schema)
        if model is not None:
            return StructuredResult(model=model, attempts=attempt, total_tokens=total_tokens)

        previous_output = raw
        previous_digest = digest

    raise SchemaEnforcementError(schema.__name__, max_attempts, previous_digest or "", previous_output or "")
