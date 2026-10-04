"""Turn a question into a vector and pull the nearest chunks from the store."""

from __future__ import annotations

from dataclasses import dataclass

from rag.chunk import Chunk
from rag.embed import Embedder
from rag.store import VectorStore


@dataclass(frozen=True)
class Retrieved:
    rank: int
    chunk: Chunk
    score: float


def retrieve(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    k: int,
    source: str | None = None,
) -> list[Retrieved]:
    query_vec = embedder.embed(query)
    hits = store.search(query_vec, k, source=source)
    return [Retrieved(rank=i + 1, chunk=entry.chunk, score=score) for i, (entry, score) in enumerate(hits)]
