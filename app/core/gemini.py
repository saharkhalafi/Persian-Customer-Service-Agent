import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.core.config import GEMINI_RETRY_ATTEMPTS, GEMINI_TIMEOUT_MS
from app.core.observability import (
    estimate_llm_cost_usd,
    get_llm_purpose,
    observe,
)


load_dotenv()


@dataclass
class GeminiJsonResult:
    data: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class GeminiClient:

    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIM = 768
    EMBED_BATCH_SIZE = 16

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        # Bounded timeout + at most a couple of transient network retries.
        # Product Search keeps its own shorter timeout and deterministic fallback.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=GEMINI_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=GEMINI_RETRY_ATTEMPTS,
                    initial_delay=0.4,
                    max_delay=2.0,
                    exp_base=2.0,
                    jitter=0.1,
                ),
            ),
        )

        self.model = os.environ.get("AGENT_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        self.thinking_budget = _thinking_budget_from_env()

    def thinking_config(self) -> types.ThinkingConfig:
        return _thinking_config(self.thinking_budget)

    def embed(
        self,
        texts: str | list[str],
        task_type: str,
    ) -> list[list[float]]:
        if isinstance(texts, str):
            items = [texts]
        else:
            items = [text for text in texts if text and text.strip()]

        if not items:
            return []

        vectors: list[list[float]] = []

        with observe(
            "llm_call",
            model=self.EMBEDDING_MODEL,
            purpose="embedding",
            task_type=task_type,
            batch_count=len(items),
        ) as extras:
            extras["timeout"] = False
            for start in range(0, len(items), self.EMBED_BATCH_SIZE):
                batch = items[start:start + self.EMBED_BATCH_SIZE]
                response = self.client.models.embed_content(
                    model=self.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.EMBEDDING_DIM,
                    ),
                )

                for embedding in response.embeddings or []:
                    values = list(embedding.values or [])
                    vectors.append(_l2_normalize(values))

        return vectors

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        model: str | None = None,
        purpose: str | None = None,
    ) -> GeminiJsonResult:
        resolved_model = model or self.model
        resolved_purpose = purpose or get_llm_purpose() or "generate_json"
        with observe(
            "llm_call",
            model=resolved_model,
            purpose=resolved_purpose,
        ) as extras:
            extras["timeout"] = False
            try:
                response = self.client.models.generate_content(
                    model=resolved_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=schema,
                        thinking_config=_thinking_config(0),
                    ),
                )
            except Exception as exc:
                extras["timeout"] = _is_timeout(exc)
                raise
            payload = _parse_json_response(getattr(response, "text", "") or "")
            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            extras["input_tokens"] = input_tokens
            extras["output_tokens"] = output_tokens
            extras["estimated_cost_usd"] = estimate_llm_cost_usd(
                input_tokens,
                output_tokens,
                resolved_model,
            )
            return GeminiJsonResult(
                data=payload,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


def _thinking_budget_from_env() -> int:
    raw = os.environ.get("AGENT_THINKING_BUDGET", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _thinking_config(budget: int | None) -> types.ThinkingConfig:
    return types.ThinkingConfig(thinking_budget=0 if budget is None else budget)


def _is_timeout(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "deadline" in text


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON must be an object")
    return payload


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))

    if not norm:
        return values

    return [value / norm for value in values]