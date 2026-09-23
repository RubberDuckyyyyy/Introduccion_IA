#!/usr/bin/env python3
"""Program 2: chunk the documents and build the vector database.

Stage 2 of RAG. Split every document into overlapping chunks, embed each one,
and store the vectors next to their metadata (source). This is the offline
"indexing" path you run once, before any question arrives.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.cli import build_parser, load, settings  # noqa: E402
from rag.format import format_chunks, format_corpus, format_index  # noqa: E402
from rag.pipeline import build_index  # noqa: E402


def main() -> None:
    args = build_parser("Chunk documents and build the vector store.").parse_args()
    corpus = load(args)
    cfg = settings(args, corpus)

    print(format_corpus(corpus))
    print(f"\nChunking: {cfg.chunk_words} palabras por chunk, {cfg.overlap} de solapamiento.\n")

    chunks, embedder, store = build_index(corpus, cfg.chunk_words, cfg.overlap)
    print(format_chunks(chunks))
    print()
    print(format_index(store, embedder))

    print("\nCada fila es un vector buscable + su fuente. La consulta llega en el programa 03.")


if __name__ == "__main__":
    main()
