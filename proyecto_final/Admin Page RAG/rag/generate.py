"""Build the prompt and produce an answer from the retrieved context.

There is no LLM here. The "generator" is extractive: it copies the sentences
that overlap the question the most and cites which chunk they came from.
If the best chunk is below `min_score`, it abstains instead of inventing.
This isolates the RAG plumbing from any particular model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.embed import Embedder
from rag.retrieve import Retrieved

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[int, ...]
    grounded: bool
    prompt: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def build_prompt(query: str, retrieved: list[Retrieved]) -> str:
    lines = [
        "Sistema: responde SOLO con el contexto. Si falta evidencia, dilo.",
        "",
        "Contexto:",
    ]
    if retrieved:
        for r in retrieved:
            lines.append(f"[{r.rank}] ({r.chunk.source}) {r.chunk.text}")
    else:
        lines.append("(sin resultados)")
    lines += ["", f"Pregunta: {query}"]
    return "\n".join(lines)


def answer(
    query: str, retrieved: list[Retrieved], embedder: Embedder, min_score: float
) -> Answer:
    prompt = build_prompt(query, retrieved)

    if not retrieved or retrieved[0].score < min_score:
        return Answer(
            text="No tengo evidencia suficiente en el corpus para responder.",
            citations=(),
            grounded=False,
            prompt=prompt,
        )

    query_tokens = set(embedder.content_tokens(query))
    scored: list[tuple[int, int, str]] = []
    for r in retrieved:
        for sentence in split_sentences(r.chunk.text):
            overlap = len(query_tokens & set(embedder.content_tokens(sentence)))
            if overlap:
                scored.append((overlap, r.rank, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))

    picked: list[str] = []
    citations: list[int] = []
    seen: set[str] = set()
    for _, rank, sentence in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        picked.append(f"{sentence} [{rank}]")
        if rank not in citations:
            citations.append(rank)
        if len(picked) >= 2:
            break

    if not picked:  # relevant chunk, but no sentence shares a word: quote its start
        top = retrieved[0]
        first = split_sentences(top.chunk.text)[0]
        picked = [f"{first} [{top.rank}]"]
        citations = [top.rank]

    return Answer(text=" ".join(picked), citations=tuple(citations), grounded=True, prompt=prompt)
