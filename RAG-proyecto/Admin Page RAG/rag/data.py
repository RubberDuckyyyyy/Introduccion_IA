"""The corpus loaded from YAML: documents, example queries, and settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Document:
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Corpus:
    name: str
    documents: tuple[Document, ...]
    queries: tuple[str, ...]
    examples: tuple[str, ...]
    chunk_words: int
    overlap: int
    top_k: int
    min_score: float
    stopwords: frozenset[str]
    source: Path | None = None


_TEXT_EXTS = {".md", ".markdown", ".txt"}
_PDF_EXTS = {".pdf"}


def load_corpus(path: str | Path) -> Corpus:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_corpus(raw, source=path)


def _read_pdf(file: Path) -> str:
    from pypdf import PdfReader  # imported lazily: only needed if a PDF is present

    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _load_docs_dir(dir_path: Path) -> list[Document]:
    """Read every .md/.txt/.pdf file in dir_path (recursively) as one document each."""
    if not dir_path.is_dir():
        raise ValueError(f"docs_dir not found: {dir_path}")
    docs: list[Document] = []
    for file in sorted(dir_path.rglob("*")):
        suffix = file.suffix.lower()
        if suffix in _TEXT_EXTS:
            text = file.read_text(encoding="utf-8").strip()
        elif suffix in _PDF_EXTS:
            text = _read_pdf(file)
        else:
            continue
        if not text:
            continue
        docs.append(Document(title=file.stem, source=str(file.relative_to(dir_path)), text=text))
    return docs


def parse_corpus(raw: dict[str, Any], source: Path | None = None) -> Corpus:
    documents = list(_documents(raw.get("documents")))

    docs_dir = raw.get("docs_dir")
    if docs_dir:
        base = source.parent if source is not None else Path(".")
        documents.extend(_load_docs_dir((base / docs_dir).resolve()))

    if not documents:
        raise ValueError(
            "no documents found: add 'documents:' entries in the YAML or point "
            "'docs_dir' at a folder with .md/.txt files"
        )

    chunk_words = int(raw.get("chunk_words", 30))
    if chunk_words < 1:
        raise ValueError("chunk_words must be >= 1")
    overlap = int(raw.get("overlap", 6))
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_words")

    return Corpus(
        name=str(raw.get("name") or "corpus"),
        documents=tuple(documents),
        queries=tuple(str(q) for q in (raw.get("queries") or [])),
        examples=tuple(str(t) for t in (raw.get("examples") or [])),
        chunk_words=chunk_words,
        overlap=overlap,
        top_k=int(raw.get("top_k", 2)),
        min_score=float(raw.get("min_score", 0.05)),
        stopwords=frozenset(str(w).lower() for w in (raw.get("stopwords") or [])),
        source=source,
    )


def _documents(rows: Any) -> list[Document]:
    out: list[Document] = []
    for i, row in enumerate(rows or []):
        if not isinstance(row, dict):
            raise ValueError(f"documents[{i}] must be a mapping with text")
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"documents[{i}] has no text")
        out.append(
            Document(
                title=str(row.get("title") or f"doc {i}"),
                source=str(row.get("source") or "unknown"),
                text=text,
            )
        )
    return out
