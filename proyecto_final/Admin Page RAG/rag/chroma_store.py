"""Persistent Chroma collection holding the chunks and their Google embeddings.

The collection records which embedding model built it: vectors from different
models are not comparable, so mixing them would make k-NN meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma"  # overridden from config.yaml by api.py
COLLECTION = "admin_page"


class ModelMismatchError(RuntimeError):
    """The index was built with a different embedding model than the current one."""


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    title: str
    index: int
    text: str
    score: float  # cosine similarity, 1 = identical


@lru_cache(maxsize=None)
def collection(embed_model: str):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Embeddings come from Google, so no embedding_function: we pass vectors.
    col = client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": embed_model},
        embedding_function=None,
    )
    built_with = (col.metadata or {}).get("embed_model")
    if built_with != embed_model:
        collection.cache_clear()
        raise ModelMismatchError(
            f"el índice se creó con '{built_with}' y ahora se usa '{embed_model}'; "
            f"borra la carpeta {CHROMA_DIR.name}/ y vuelve a ingestar"
        )
    return col


def replace_source(
    embed_model: str, source: str, title: str, chunks: list[str], vectors: list[list[float]]
) -> None:
    """Re-ingesting a file drops its old chunks first, so nothing goes stale."""
    col = collection(embed_model)
    col.delete(where={"source": source})
    col.add(
        ids=[f"{source}::{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": source, "title": title, "index": i} for i in range(len(chunks))],
    )


def search(embed_model: str, vector: list[float], k: int) -> list[Hit]:
    col = collection(embed_model)
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[vector], n_results=min(k, col.count()))
    return [
        Hit(
            id=id_,
            source=meta["source"],
            title=meta.get("title", ""),
            index=meta["index"],
            text=doc,
            score=1 - dist,
        )
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
