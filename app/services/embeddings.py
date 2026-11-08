"""Dense embedding providers.

LocalHashingEmbeddingProvider is the default. It hashes word unigrams and
bigrams into a fixed-size, L2-normalised vector with scikit-learn's
HashingVectorizer, so it needs no model download or network. It has no
learned meaning, so it behaves like a bag of words, and the retrieval
evaluation shows it adds little on top of BM25. Set EMBEDDING_BACKEND=openai
for real semantic embeddings; nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingProvider(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text, in order."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    name = "local-hashing"

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # alternate_sign=True makes hash collisions cancel out on average
        # instead of always adding up.
        self._vectorizer = HashingVectorizer(
            n_features=dimension,
            norm="l2",
            alternate_sign=True,
            ngram_range=(1, 2),
            lowercase=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        # response.data is returned in the same order as the input list
        return [item.embedding for item in response.data]


def build_embedding_provider(settings) -> EmbeddingProvider:
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_BACKEND=openai requires OPENAI_API_KEY to be set.")
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    return LocalHashingEmbeddingProvider(dimension=settings.embedding_dimension)
