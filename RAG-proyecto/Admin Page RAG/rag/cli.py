"""CLI helpers shared by the four programs."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from rag.data import Corpus, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "config.yaml"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YAML corpus")
    parser.add_argument("--query", type=str, default=None, help="Ask your own question")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k", help="How many chunks to retrieve")
    parser.add_argument("--chunk-words", type=int, default=None, dest="chunk_words", help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=None, help="Overlapping words between chunks")
    parser.add_argument("--source", type=str, default=None, help="Only search chunks from this source")
    parser.add_argument("--all", action="store_true", help="Run every example query in the YAML")
    return parser


def load(args: argparse.Namespace) -> Corpus:
    return load_corpus(args.data)


def settings(args: argparse.Namespace, corpus: Corpus) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_words=args.chunk_words or corpus.chunk_words,
        overlap=corpus.overlap if args.overlap is None else args.overlap,
        top_k=args.top_k or corpus.top_k,
    )


def queries_from(args: argparse.Namespace, corpus: Corpus) -> list[str]:
    if args.query:
        return [args.query]
    if args.all:
        return list(corpus.queries)
    return list(corpus.queries[:1])
