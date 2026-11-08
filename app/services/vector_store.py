"""Dense vector storage with two interchangeable backends, Qdrant and
ChromaDB (VECTOR_BACKEND). Each can talk to a server (*_URL, as in
docker-compose) or run embedded on disk (*_LOCAL_PATH) with no server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    chunk_id: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        dimension: int,
        url: str | None = None,
        api_key: str | None = None,
        local_path: str | None = None,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # Embedded mode: an on-disk (or ":memory:") engine, no server needed.
            self._client = QdrantClient(path=local_path or ":memory:")

        self._collection = collection_name
        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        # Qdrant point IDs must be ints or UUIDs, so the string chunk_id is
        # mapped to a UUID and also kept in the payload.
        points = [
            PointStruct(id=_to_qdrant_id(cid), vector=vec, payload={**payload, "chunk_id": cid})
            for cid, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        results = self._client.query_points(collection_name=self._collection, query=query_vector, limit=top_k)
        return [VectorSearchResult(chunk_id=p.payload["chunk_id"], score=float(p.score)) for p in results.points]

    def delete(self, ids: list[str]) -> None:
        self._client.delete(collection_name=self._collection, points_selector=[_to_qdrant_id(i) for i in ids])


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        url: str | None = None,
        local_path: str | None = None,
    ):
        import chromadb

        if url:
            host, _, port = url.replace("http://", "").replace("https://", "").partition(":")
            self._client = chromadb.HttpClient(host=host, port=int(port or 8000))
        else:
            self._client = chromadb.PersistentClient(path=local_path or "./data/chroma_local")

        # Embeddings are always passed in, so Chroma never needs to download
        # its default embedding model.
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        documents = [p.get("text", "") for p in payloads]
        self._collection.upsert(ids=ids, embeddings=vectors, metadatas=payloads, documents=documents)

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        # Chroma returns squared-L2 distance (lower is closer). Convert it so
        # higher is better, like Qdrant's cosine score.
        return [
            VectorSearchResult(chunk_id=cid, score=1.0 / (1.0 + dist)) for cid, dist in zip(ids, distances, strict=True)
        ]

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)


def _to_qdrant_id(chunk_id: str) -> str:
    import uuid

    # uuid5 is deterministic, so the same chunk_id always maps to the same point.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_vector_store(settings) -> VectorStore:
    if settings.vector_backend == "chroma":
        return ChromaVectorStore(
            collection_name=settings.collection_name,
            url=settings.chroma_url,
            local_path=settings.chroma_local_path,
        )
    return QdrantVectorStore(
        collection_name=settings.collection_name,
        dimension=settings.embedding_dimension,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
    )
