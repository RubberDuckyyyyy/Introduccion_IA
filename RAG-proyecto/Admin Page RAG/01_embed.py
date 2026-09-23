#!/usr/bin/env python3
"""Program 1: turn text into vectors and compare them with cosine.

Stage 1 of RAG. Build the vocabulary from the corpus, embed a few example
texts, and print the cosine similarity between every pair. Similar sentences
score high; an out-of-vocabulary text (the paella) scores zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.cli import build_parser, load, settings  # noqa: E402
from rag.format import format_vector  # noqa: E402
from rag.pipeline import build_index  # noqa: E402
from rag.vectors import cosine  # noqa: E402


def main() -> None:
    args = build_parser("Embed text and compare with cosine similarity.").parse_args()
    corpus = load(args)
    cfg = settings(args, corpus)

    _, embedder, _ = build_index(corpus, cfg.chunk_words, cfg.overlap)
    print(f"Vocabulario: {len(embedder.vocab)} palabras de contenido (sin stopwords).\n")

    texts = list(corpus.examples) or [d.text for d in corpus.documents[:3]]
    vectors = [embedder.embed(t) for t in texts]

    for i, (text, vec) in enumerate(zip(texts, vectors)):
        print(f"[{i}] {text}")
        print(f"    {format_vector(embedder, vec)}\n")

    print("Coseno entre cada par (1 = idéntico en palabras, 0 = sin solapamiento):")
    header = "     " + " ".join(f"[{j}]" for j in range(len(texts)))
    print(header)
    for i, a in enumerate(vectors):
        row = " ".join(f"{cosine(a, b):.2f}" for b in vectors)
        print(f"[{i}]  {row}")

    print("\nUn embedding es una coordenada: la geometría (distancia) codifica el parecido.")
    print("Edita data/curso.yaml (o usa --data) y vuelve a ejecutar.")


if __name__ == "__main__":
    main()
