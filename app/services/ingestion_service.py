from __future__ import annotations

import uuid
from typing import Any

from app.services.bm25_index import Chunk, CorpusStore
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore


class IngestionService:
    def __init__(
        self,
        corpus: CorpusStore,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        chunk_size_words: int = 220,
        chunk_overlap_words: int = 40,
    ):
        self._corpus = corpus
        self._vector_store = vector_store
        self._embedder = embedder
        self._chunk_size = chunk_size_words
        self._overlap = chunk_overlap_words

    def ingest_text(self, source: str, text: str, metadata: dict[str, Any] | None = None) -> tuple[str, list[str]]:
        metadata = metadata or {}
        document_id = str(uuid.uuid4())
        spans = chunk_text(text, chunk_size_words=self._chunk_size, overlap_words=self._overlap)
        if not spans:
            raise ValueError("Document text is empty after normalization; nothing to ingest.")

        chunks = [
            Chunk(
                chunk_id=f"{document_id}:{span.index}",
                document_id=document_id,
                source=source,
                chunk_index=span.index,
                text=span.text,
                metadata=metadata,
            )
            for span in spans
        ]

        vectors = self._embedder.embed([c.text for c in chunks])
        payloads = [
            {"document_id": c.document_id, "source": c.source, "chunk_index": c.chunk_index, "text": c.text, **metadata}
            for c in chunks
        ]
        self._vector_store.upsert(ids=[c.chunk_id for c in chunks], vectors=vectors, payloads=payloads)
        self._corpus.add_chunks(chunks)

        return document_id, [c.chunk_id for c in chunks]

    def delete_document(self, document_id: str) -> int:
        chunk_ids = self._corpus.chunk_ids_for_document(document_id)
        if chunk_ids:
            self._vector_store.delete(chunk_ids)
        return self._corpus.delete_document(document_id)
