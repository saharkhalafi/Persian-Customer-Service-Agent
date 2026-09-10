import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class GeminiClient:

    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIM = 768
    EMBED_BATCH_SIZE = 16

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        # Longer timeout + extra retries for Windows socket errors
        # (WinError 10061/10053) talking to generativelanguage.googleapis.com.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=180_000,
                retry_options=types.HttpRetryOptions(
                    attempts=5,
                    initial_delay=1.0,
                    max_delay=16.0,
                    exp_base=2.0,
                    jitter=0.2,
                ),
            ),
        )

        self.model = "gemini-2.5-flash"

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


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))

    if not norm:
        return values

    return [value / norm for value in values]