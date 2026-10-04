#!/usr/bin/env python3
"""Streamlit UI over the FastAPI service (api.py): upload documents and ask.

Start the API first, then:

    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
API_URL = os.getenv("API_URL", "http://localhost:8000")
ALLOWED = ["md", "markdown", "txt", "pdf"]


def api_error(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text


st.set_page_config(page_title="Admin Page RAG", page_icon="📄")
st.title("Admin Page RAG")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
except requests.RequestException:
    st.error(f"No se pudo conectar con la API en {API_URL}. ¿Está corriendo `uvicorn api:app`?")
    st.stop()

with st.sidebar:
    st.header("Documentos")
    chroma = health["chroma"]
    if chroma["ok"]:
        st.caption(f"API ok · {chroma['chunks']} chunks en Chroma")
    else:
        st.warning(f"Chroma no accesible: {chroma['error']}")

    uploads = st.file_uploader("Sube .md, .txt o .pdf", type=ALLOWED, accept_multiple_files=True)
    if uploads and st.button("Ingestar", type="primary"):
        files = [("files", (up.name, up.getvalue())) for up in uploads]
        with st.spinner("Chunkificando e incrustando…"):
            resp = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
        if resp.ok:
            out = resp.json()
            st.success(f"{out['documents']} documento(s), {out['chunks']} chunks indexados")
            if out["skipped"]:
                st.warning("Sin texto, ignorados: " + ", ".join(out["skipped"]))
        else:
            st.error(api_error(resp))

question = st.text_input("Pregunta", placeholder="¿cómo configuro el ambiente de admin page?")
top_k = st.slider("top_k", 1, 10, 3)
if question:
    with st.spinner("Buscando…"):
        resp = requests.post(
            f"{API_URL}/query", json={"question": question, "top_k": top_k}, timeout=120
        )
    if not resp.ok:
        st.error(api_error(resp))
        st.stop()
    out = resp.json()

    if out["abstained"]:
        st.warning(out["answer"])
    else:
        st.success(out["answer"])

    if out["citations"]:
        with st.expander("Citas"):
            for i, c in enumerate(out["citations"], 1):
                st.markdown(f"**[{i}]** {c['title']} (`{c['source']}`) · chunk {c['index']} · score = {c['score']:.3f}")
                st.text(c["text"])
