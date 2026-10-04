"""Google AI (Gemini API): embeddings for Chroma and answer generation.

The client reads the key from GOOGLE_API_KEY (or GEMINI_API_KEY). Models can
be overridden with GOOGLE_EMBED_MODEL / GOOGLE_GEN_MODEL.
"""

from __future__ import annotations

import os
from functools import lru_cache

EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.getenv("GOOGLE_GEN_MODEL", "gemini-3.8-flash")
_BATCH = 100  # max texts per embed_content request

# The model answers exactly this when the context does not support an answer.
NO_EVIDENCE = "NO_EVIDENCE"

SYSTEM_PROMPT = f"""Eres un asistente que responde preguntas sobre cómo configurar \
el ambiente de admin page. Responde SIEMPRE en español, aunque el contexto esté en \
otro idioma, y usa SOLO la información de los fragmentos numerados del contexto: \
no agregues conocimiento propio. Cita los fragmentos que uses con su número, \
p. ej. [1] o [2].
Si el contexto no contiene evidencia suficiente para responder, responde \
exactamente {NO_EVIDENCE} y nada más."""


def has_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


class GoogleAIError(RuntimeError):
    """Missing key or a failed call to the Gemini API."""


@lru_cache(maxsize=1)
def _client():
    from google import genai
    from google.genai import types

    if not has_key():
        raise GoogleAIError("falta la variable de entorno GOOGLE_API_KEY")
    # Retry rate limits and "high demand" 503s with exponential backoff.
    retry = types.HttpRetryOptions(attempts=5, initial_delay=2, max_delay=20, http_status_codes=[429, 500, 503])
    return genai.Client(http_options=types.HttpOptions(retry_options=retry))


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """task_type: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    from google.genai import errors, types

    config = types.EmbedContentConfig(task_type=task_type)
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), _BATCH):
            resp = _client().models.embed_content(
                model=EMBED_MODEL, contents=texts[i : i + _BATCH], config=config
            )
            vectors.extend(list(e.values) for e in resp.embeddings)
    except errors.APIError as exc:
        raise GoogleAIError(f"embeddings fallaron: {exc}") from exc
    return vectors


def generate(prompt: str) -> str:
    from google.genai import errors, types

    try:
        resp = _client().models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
        )
    except errors.APIError as exc:
        raise GoogleAIError(f"generación falló: {exc}") from exc
    return (resp.text or "").strip()
