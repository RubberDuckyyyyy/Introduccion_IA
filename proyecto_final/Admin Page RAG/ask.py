#!/usr/bin/env python3
"""Interactive Q&A: build the index once, then ask questions in a loop.

Same retrieve -> answer pipeline as 04_answer.py, just without re-indexing
per question and without needing --query for every ask.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.cli import DEFAULT_DATA, settings  # noqa: E402
from rag.data import load_corpus  # noqa: E402
from rag.format import format_answer, format_retrieved  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.pipeline import build_index  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402


def main() -> None:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    corpus = load_corpus(data_path)
    cfg = settings(SimpleNamespace(chunk_words=None, overlap=None, top_k=None), corpus)

    _, embedder, store = build_index(corpus, cfg.chunk_words, cfg.overlap)
    print(f"Índice listo: {len(store)} chunks de '{corpus.name}'.")
    print("Escribe tu pregunta (línea vacía o Ctrl+C para salir).\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break

        results = retrieve(query, embedder, store, cfg.top_k)
        ans = answer(query, results, embedder, corpus.min_score)
        print(format_retrieved(results))
        print(format_answer(ans))
        print()


if __name__ == "__main__":
    main()
