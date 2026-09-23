"""Bag-of-words embeddings: one count per vocabulary word.

Real systems learn embeddings from data; here the vector is a plain word
count so every number can be checked by hand. The mechanics that follow
(cosine, nearest neighbors, prompt) are identical to a production RAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.chunk import Chunk
from rag.tokenize import tokenize
from rag.vectors import Vector


@dataclass
class Embedder:
    vocab: tuple[str, ...]
    stopwords: frozenset[str]
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {tok: i for i, tok in enumerate(self.vocab)}

    def content_tokens(self, text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in self.stopwords]

    def embed(self, text: str) -> Vector:
        vec = [0.0] * len(self.vocab)
        for tok in self.content_tokens(text):
            i = self._index.get(tok)
            if i is not None:  # words outside the vocabulary are ignored
                vec[i] += 1.0
        return vec

    def nonzero(self, vec: Vector) -> list[tuple[str, float]]:
        pairs = [(self.vocab[i], v) for i, v in enumerate(vec) if v != 0.0]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs


def build_embedder(chunks: list[Chunk], stopwords: frozenset[str]) -> Embedder:
    vocab: set[str] = set()
    for chunk in chunks:
        for tok in tokenize(chunk.text):
            if tok not in stopwords:
                vocab.add(tok)
    return Embedder(vocab=tuple(sorted(vocab)), stopwords=stopwords)
