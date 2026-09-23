"""A minimal vector database: vectors + the chunk they came from.

`search` is exact k-NN: it scores the query against every stored vector.
A production database swaps this loop for an approximate index (HNSW, IVF),
but returns the same kind of ranked neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.chunk import Chunk
from rag.embed import Embedder
from rag.vectors import Vector, cosine


@dataclass(frozen=True)
class Entry:
    chunk: Chunk
    vector: tuple[float, ...]


@dataclass
class VectorStore:
    entries: list[Entry] = field(default_factory=list)

    def add(self, chunk: Chunk, vector: Vector) -> None:
        self.entries.append(Entry(chunk=chunk, vector=tuple(vector)))

    def __len__(self) -> int:
        return len(self.entries)

    def search(
        self, query_vec: Vector, k: int, source: str | None = None
    ) -> list[tuple[Entry, float]]:
        scored: list[tuple[Entry, float]] = []
        for entry in self.entries:
            if source is not None and entry.chunk.source != source:
                continue
            scored.append((entry, cosine(query_vec, entry.vector)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk.id))
        return scored[:k]


def build_store(chunks: list[Chunk], embedder: Embedder) -> VectorStore:
    store = VectorStore()
    for chunk in chunks:
        store.add(chunk, embedder.embed(chunk.text))
    return store
