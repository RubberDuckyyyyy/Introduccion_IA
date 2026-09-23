#!/usr/bin/env python3
"""Program 3: retrieve the nearest chunks for a question.

Stage 3 of RAG. Embed the query with the same embedder, then run exact k-NN
over the store and show the ranked neighbors with their cosine scores.
Use --query to ask your own, --source to filter by metadata, --all for every
example query in the YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.cli import build_parser, load, queries_from, settings  # noqa: E402
from rag.format import format_retrieved  # noqa: E402
from rag.pipeline import build_index  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402


def main() -> None:
    args = build_parser("Embed a query and retrieve the nearest chunks.").parse_args()
    corpus = load(args)
    cfg = settings(args, corpus)

    _, embedder, store = build_index(corpus, cfg.chunk_words, cfg.overlap)
    if args.source:
        print(f"Filtro de metadatos: source = {args.source}")
    print(f"Recuperando top-{cfg.top_k} por similitud del coseno.\n")

    for query in queries_from(args, corpus):
        print(f"Q: {query}")
        results = retrieve(query, embedder, store, cfg.top_k, source=args.source)
        print(format_retrieved(results))
        print()

    print("El score depende del embedding y del chunking. Prueba --top-k 3 o --chunk-words 20.")


if __name__ == "__main__":
    main()
