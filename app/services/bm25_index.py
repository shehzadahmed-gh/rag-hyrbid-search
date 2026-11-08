"""
BM25 sparse-retrieval stage, backed by `rank_bm25` (a real, pure-Python
Okapi BM25 implementation - not a stand-in).

rank_bm25 keeps its index purely in memory, so this module also owns a tiny
JSONL-backed corpus store: every ingested chunk is appended to
`corpus_path`, and the BM25 index (plus the chunk lookup table used by the
retrieval/generation routers) is rebuilt from that file on process startup.
That gives the demo real persistence across restarts without pulling in a
database just for chunk text.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]


class CorpusStore:
    """
    Owns the source of truth for chunk text/metadata (JSONL on disk) and the
    in-memory BM25 index built from it. Thread-safe for the simple
    single-process FastAPI deployment this project targets; for multi-worker
    deployments, back this with a real datastore (e.g. the same Postgres
    instance used in the agent-orchestration-system project) and rebuild the
    BM25 index per-worker on startup exactly as done here.
    """

    def __init__(self, corpus_path: str):
        self._path = corpus_path
        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._id_order: list[str] = []
        self._doc_tokens: dict[str, set[str]] = {}
        os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chunk = Chunk(**data)
                self._chunks[chunk.chunk_id] = chunk
        self._rebuild_index()

    def _append(self, chunk: Chunk) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(chunk)) + "\n")

    def _rebuild_index(self) -> None:
        self._id_order = list(self._chunks.keys())
        tokenized = [tokenize(self._chunks[cid].text) for cid in self._id_order]
        self._doc_tokens = {cid: set(toks) for cid, toks in zip(self._id_order, tokenized)}
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    # -- mutation ----------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                self._append(chunk)
            self._rebuild_index()

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
            for cid in to_delete:
                del self._chunks[cid]
            self._rewrite_all()
            self._rebuild_index()
            return len(to_delete)

    def _rewrite_all(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            for chunk in self._chunks.values():
                f.write(json.dumps(asdict(chunk)) + "\n")

    # -- reads ---------------------------------------------------------
    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def chunk_ids_for_document(self, document_id: str) -> list[str]:
        return [cid for cid, chunk in self._chunks.items() if chunk.document_id == document_id]

    def documents(self) -> dict[str, dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks.values():
            doc = docs.setdefault(
                chunk.document_id, {"document_id": chunk.document_id, "source": chunk.source, "chunk_count": 0, "metadata": {}}
            )
            doc["chunk_count"] += 1
        return docs

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted best-first."""
        with self._lock:
            if self._bm25 is None or not self._id_order:
                return []
            query_tokens = set(tokenize(query))
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(list(query_tokens))
            # Gate on token overlap rather than "score > 0": BM25's IDF term
            # can be zero or slightly negative on very small corpora (when a
            # term appears in most/all documents), which would otherwise
            # silently drop a genuine keyword match. Sharing at least one
            # token with the query is a more robust "is this worth returning"
            # signal than the raw score's sign, while still ranking by score.
            ranked = sorted(
                (
                    (cid, score)
                    for cid, score in zip(self._id_order, scores)
                    if self._doc_tokens.get(cid, set()) & query_tokens
                ),
                key=lambda x: -x[1],
            )
            return [(cid, float(score)) for cid, score in ranked[:top_k]]
