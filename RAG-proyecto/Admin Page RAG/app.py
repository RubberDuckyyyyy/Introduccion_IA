#!/usr/bin/env python3
"""Streamlit UI: upload documents into docs/ and ask questions over them.

Same retrieve -> answer pipeline as ask.py. Uploaded files are saved into the
docs_dir from config.yaml, so the CLI programs see them too.

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.cli import DEFAULT_DATA, settings  # noqa: E402
from rag.data import load_corpus  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.pipeline import build_index  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402

ALLOWED = ["md", "markdown", "txt", "pdf"]


def docs_dir() -> Path:
    with DEFAULT_DATA.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    path = (DEFAULT_DATA.parent / raw.get("docs_dir", "docs")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint(folder: Path) -> tuple[tuple[str, float, int], ...]:
    """Changes whenever a file is added, removed or modified -> cache key."""
    return tuple(
        (str(f.relative_to(folder)), f.stat().st_mtime, f.stat().st_size)
        for f in sorted(folder.rglob("*"))
        if f.is_file()
    )


@st.cache_resource(show_spinner="Indexando documentos…")
def load_index(_fp: tuple):
    corpus = load_corpus(DEFAULT_DATA)
    cfg = settings(SimpleNamespace(chunk_words=None, overlap=None, top_k=None), corpus)
    _, embedder, store = build_index(corpus, cfg.chunk_words, cfg.overlap)
    return corpus, cfg, embedder, store


st.set_page_config(page_title="Admin Page RAG", page_icon="📄")
st.title("Admin Page RAG")
folder = docs_dir()

with st.sidebar:
    st.header("Documentos")
    uploads = st.file_uploader(
        "Sube .md, .txt o .pdf", type=ALLOWED, accept_multiple_files=True
    )
    if uploads and st.button("Guardar e indexar", type="primary"):
        for up in uploads:
            (folder / Path(up.name).name).write_bytes(up.getbuffer())
        st.success(f"{len(uploads)} archivo(s) guardado(s) en {folder.name}/")
        st.rerun()

    st.divider()
    files = [f for f in sorted(folder.rglob("*")) if f.is_file()]
    if not files:
        st.info("Aún no hay documentos.")
    for f in files:
        col_name, col_del = st.columns([4, 1])
        col_name.write(f"`{f.relative_to(folder)}`")
        if col_del.button("🗑", key=f"del-{f}", help="Eliminar"):
            f.unlink()
            st.rerun()

try:
    corpus, cfg, embedder, store = load_index(fingerprint(folder))
except ValueError as exc:  # e.g. no documents yet
    st.warning(f"No se pudo construir el índice: {exc}")
    st.stop()

st.caption(f"{len(corpus.documents)} documentos · {len(store)} chunks indexados")

query = st.text_input("Pregunta", placeholder=corpus.queries[0] if corpus.queries else "")
if query:
    results = retrieve(query, embedder, store, cfg.top_k)
    ans = answer(query, results, embedder, corpus.min_score)

    if ans.grounded:
        st.success(ans.text)
    else:
        st.warning(ans.text)

    with st.expander("Chunks recuperados"):
        for r in results:
            st.markdown(f"**[{r.rank}]** `{r.chunk.source}` · cos = {r.score:.3f}")
            st.text(r.chunk.text)
