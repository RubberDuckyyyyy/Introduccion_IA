"""Plain-text formatting for the four programs (no tables library)."""

from __future__ import annotations

from rag.chunk import Chunk
from rag.data import Corpus
from rag.embed import Embedder
from rag.generate import Answer
from rag.retrieve import Retrieved
from rag.store import VectorStore
from rag.vectors import norm


def preview(text: str, width: int = 56) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def format_corpus(corpus: Corpus) -> str:
    lines = [f"Corpus: {corpus.name}  ({len(corpus.documents)} documentos)"]
    lines.append(f"{'source':<12} {'words':>5}  title")
    lines.append("-" * 58)
    for doc in corpus.documents:
        lines.append(f"{doc.source:<12} {len(doc.text.split()):>5}  {doc.title}")
    return "\n".join(lines)


def format_vector(embedder: Embedder, vec: list[float], limit: int = 8) -> str:
    pairs = embedder.nonzero(vec)
    shown = ", ".join(f"{tok}:{int(v)}" for tok, v in pairs[:limit])
    extra = "" if len(pairs) <= limit else f", (+{len(pairs) - limit} más)"
    body = shown + extra if shown else "(vacío: ninguna palabra está en el vocabulario)"
    return f"|v|={norm(vec):.3f}  ->  {body}"


def format_chunks(chunks: list[Chunk]) -> str:
    lines = [f"{'id':>2}  {'source':<12} {'#':>2} {'w':>3}  text"]
    lines.append("-" * 72)
    for c in chunks:
        lines.append(f"{c.id:>2}  {c.source:<12} {c.index:>2} {c.word_count:>3}  {preview(c.text)}")
    return "\n".join(lines)


def format_index(store: VectorStore, embedder: Embedder) -> str:
    lines = [f"Vector DB: {len(store)} vectores indexados"]
    lines.append(f"{'id':>2}  {'source':<12} {'|v|':>6}  top términos")
    lines.append("-" * 72)
    for entry in store.entries:
        pairs = embedder.nonzero(list(entry.vector))
        top = ", ".join(f"{tok}:{int(v)}" for tok, v in pairs[:5])
        lines.append(f"{entry.chunk.id:>2}  {entry.chunk.source:<12} {norm(entry.vector):>6.3f}  {top}")
    return "\n".join(lines)


def format_retrieved(retrieved: list[Retrieved]) -> str:
    if not retrieved:
        return "(sin resultados)"
    lines = [f"{'#':>2} {'cos':>6}  {'source':<12} text"]
    lines.append("-" * 72)
    for r in retrieved:
        lines.append(f"{r.rank:>2} {r.score:>6.3f}  {r.chunk.source:<12} {preview(r.chunk.text)}")
    return "\n".join(lines)


def format_answer(ans: Answer) -> str:
    tag = "respuesta" if ans.grounded else "abstención"
    cites = ", ".join(f"[{c}]" for c in ans.citations) if ans.citations else "—"
    return f"{tag} (citas {cites}):\n  {ans.text}"
