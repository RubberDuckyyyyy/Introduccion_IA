"""Wire the stages together: chunk -> embed -> store."""

from __future__ import annotations

from rag.chunk import Chunk, chunk_corpus
from rag.data import Corpus
from rag.embed import Embedder, build_embedder
from rag.store import VectorStore, build_store


def build_index(corpus: Corpus, chunk_words: int, overlap: int) -> tuple[list[Chunk], Embedder, VectorStore]:
    chunks = chunk_corpus(corpus, chunk_words, overlap)
    embedder = build_embedder(chunks, corpus.stopwords)
    store = build_store(chunks, embedder)
    return chunks, embedder, store
