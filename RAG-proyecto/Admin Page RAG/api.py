#!/usr/bin/env python3
"""REST API: ingest documents into Chroma (Google embeddings) and answer with Gemini.

    uvicorn api:app --reload --port 8000

Docs at http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import chroma_store, google_ai
from rag.chunk import chunk_text
from rag.data import read_document

ROOT = Path(__file__).resolve().parent
with (ROOT / "config.yaml").open(encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh) or {}
DOCS_DIR = (ROOT / CFG.get("docs_dir", "docs")).resolve()
CHUNK_WORDS = int(CFG.get("chunk_words", 80))
OVERLAP = int(CFG.get("overlap", 15))
TOP_K = int(CFG.get("top_k", 3))
MIN_SCORE = float(CFG.get("api_min_score", 0.5))
chroma_store.CHROMA_DIR = (ROOT / CFG.get("chroma_dir", "chroma")).resolve()
MODEL = google_ai.EMBED_MODEL  # same model for documents and questions

app = FastAPI(title="Admin Page RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    id: str
    index: int
    source: str
    title: str
    text: str
    score: float


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool


def _google_error(exc: google_ai.GoogleAIError) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Google AI no disponible: {exc}")


def _mismatch_error(exc: chroma_store.ModelMismatchError) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


def _source_name(file: Path) -> str:
    try:
        return file.resolve().relative_to(DOCS_DIR).as_posix()
    except ValueError:
        return file.name


def _expand(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]


@app.get("/health")
def health() -> dict:
    try:
        chroma = {"ok": True, "chunks": chroma_store.collection(MODEL).count()}
    except Exception as exc:  # report it, /health itself must not fail
        chroma = {"ok": False, "error": str(exc)}
    return {"status": "ok", "embed_model": MODEL, "chroma": chroma}


@app.post("/ingest")
def ingest(
    files: list[UploadFile] | None = File(default=None),
    paths: list[str] | None = Form(default=None),
) -> dict:
    """Upload files (saved into docs/) and/or give server-side paths (files or folders)."""
    targets: list[Path] = []
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for up in files or []:
        dest = DOCS_DIR / Path(up.filename or "upload").name
        dest.write_bytes(up.file.read())
        targets.append(dest)
    for raw in paths or []:
        path = Path(raw)
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"no existe la ruta: {raw}")
        targets.extend(_expand(path))
    if not targets:
        raise HTTPException(status_code=400, detail="envía 'files' o 'paths'")

    documents = chunks = 0
    skipped: list[str] = []
    for file in targets:
        text = read_document(file)
        if not text:  # unsupported type or PDF without a text layer
            skipped.append(file.name)
            continue
        pieces = chunk_text(text, CHUNK_WORDS, OVERLAP)
        try:
            vectors = google_ai.embed_texts(pieces, "RETRIEVAL_DOCUMENT")
        except google_ai.GoogleAIError as exc:
            raise _google_error(exc) from exc
        try:
            chroma_store.replace_source(MODEL, _source_name(file), file.stem, pieces, vectors)
        except chroma_store.ModelMismatchError as exc:
            raise _mismatch_error(exc) from exc
        documents += 1
        chunks += len(pieces)
    return {"documents": documents, "chunks": chunks, "skipped": skipped}


def _abstain(reason: str) -> QueryOut:
    return QueryOut(answer=reason, citations=[], abstained=True)


@app.post("/query")
def query(body: QueryIn) -> QueryOut:
    try:
        [qvec] = google_ai.embed_texts([body.question], "RETRIEVAL_QUERY")
    except google_ai.GoogleAIError as exc:
        raise _google_error(exc) from exc

    try:
        hits = chroma_store.search(MODEL, qvec, body.top_k or TOP_K)
    except chroma_store.ModelMismatchError as exc:
        raise _mismatch_error(exc) from exc
    if not hits:
        return _abstain("No hay documentos indexados todavía. Usa POST /ingest primero.")
    if hits[0].score < MIN_SCORE:
        return _abstain("No tengo evidencia suficiente en los documentos para responder esa pregunta.")

    context = "\n\n".join(f"[{i}] ({h.source}) {h.text}" for i, h in enumerate(hits, 1))
    try:
        text = google_ai.generate(f"Contexto:\n{context}\n\nPregunta: {body.question}")
    except google_ai.GoogleAIError as exc:
        raise _google_error(exc) from exc
    if not text or google_ai.NO_EVIDENCE in text:
        return _abstain("No tengo evidencia suficiente en los documentos para responder esa pregunta.")

    citations = [
        Citation(
            id=h.id, index=h.index, source=h.source, title=h.title, text=h.text, score=round(h.score, 4)
        )
        for h in hits
    ]
    return QueryOut(answer=text, citations=citations, abstained=False)
