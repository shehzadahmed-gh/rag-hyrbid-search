"""BM25 retrieval (rank_bm25's Okapi BM25) plus the chunk store it reads from.

rank_bm25 only keeps its index in memory, so every chunk is also appended to a
JSONL file and the index is rebuilt from it on startup. Adding chunks only
marks the index stale; it is rebuilt once, on the next search, so ingesting N
documents costs one rebuild instead of N.
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
    """Chunk text and metadata (JSONL on disk) and the BM25 index built from it.

    A lock makes it safe across threads in one process. Several worker
    processes would each hold their own copy, so a multi-worker deployment
    should keep chunks in a database and rebuild the index per worker.
    """

    def __init__(self, corpus_path: str):
        self._path = corpus_path
        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._id_order: list[str] = []
        self._doc_tokens: dict[str, set[str]] = {}
        self._stale = False
        os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chunk = Chunk(**data)
                self._chunks[chunk.chunk_id] = chunk
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._id_order = list(self._chunks.keys())
        tokenized = [tokenize(self._chunks[cid].text) for cid in self._id_order]
        self._doc_tokens = {cid: set(toks) for cid, toks in zip(self._id_order, tokenized, strict=True)}
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    # -- mutation ----------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                for chunk in chunks:
                    self._chunks[chunk.chunk_id] = chunk
                    f.write(json.dumps(asdict(chunk)) + "\n")
            self._stale = True

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            remaining = {cid: c for cid, c in self._chunks.items() if c.document_id != document_id}
            deleted = len(self._chunks) - len(remaining)
            if deleted:
                # Update the file first; memory only changes if that succeeds.
                self._rewrite_all(remaining.values())
                self._chunks = remaining
                self._stale = True
            return deleted

    def _rewrite_all(self, chunks) -> None:
        # Write to a temporary file and swap it in, so a crash part-way
        # through leaves the old corpus intact instead of a truncated one.
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._path)

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
                chunk.document_id,
                {"document_id": chunk.document_id, "source": chunk.source, "chunk_count": 0, "metadata": {}},
            )
            doc["chunk_count"] += 1
        return docs

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted best-first."""
        with self._lock:
            if self._stale:
                self._rebuild_index()
                self._stale = False
            if self._bm25 is None or not self._id_order:
                return []
            query_tokens = set(tokenize(query))
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(list(query_tokens))
            # Keep chunks that share at least one query token, ranked by score.
            # Filtering on score > 0 would drop real matches in small corpora,
            # where a term in most documents gets a zero or negative IDF.
            ranked = sorted(
                (
                    (cid, score)
                    for cid, score in zip(self._id_order, scores, strict=True)
                    if self._doc_tokens.get(cid, set()) & query_tokens
                ),
                key=lambda x: -x[1],
            )
            return [(cid, float(score)) for cid, score in ranked[:top_k]]
