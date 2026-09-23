#!/usr/bin/env python3
"""Program 4: the whole RAG loop, end to end.

Stage 4 of RAG. For each question: retrieve context, assemble the prompt, and
produce a grounded answer with citations. When the best chunk is below
min_score (the paella question), the system abstains instead of inventing.
The "generator" here is extractive, not an LLM, so every word is traceable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.cli import build_parser, load, queries_from, settings  # noqa: E402
from rag.format import format_answer, format_retrieved  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.pipeline import build_index  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402


def main() -> None:
    args = build_parser("Run the full RAG pipeline and answer with citations.").parse_args()
    corpus = load(args)
    cfg = settings(args, corpus)

    _, embedder, store = build_index(corpus, cfg.chunk_words, cfg.overlap)

    for query in queries_from(args, corpus):
        print("=" * 72)
        print(f"Pregunta: {query}\n")

        results = retrieve(query, embedder, store, cfg.top_k, source=args.source)
        print("1) Recuperación")
        print(format_retrieved(results))

        ans = answer(query, results, embedder, corpus.min_score)
        print("\n2) Prompt enviado al modelo")
        for line in ans.prompt.splitlines():
            print(f"   | {line}")

        print("\n3) " + format_answer(ans))
        print()

    print("=" * 72)
    print("RAG = recuperar evidencia + generar con ella. Sin evidencia, se abstiene.")


if __name__ == "__main__":
    main()
