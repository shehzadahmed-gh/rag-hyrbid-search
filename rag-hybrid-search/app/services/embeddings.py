"""
Pluggable dense-embedding providers.

`LocalHashingEmbeddingProvider` is the default. It uses scikit-learn's
feature-hashing trick to turn text into a fixed-dimensional, L2-normalized
dense vector with **no model download and no network call** - which is what
makes the rest of this service runnable end-to-end in an offline / CI / take
-home environment. It is a real, well-defined algorithm (not a fake), but it
has no learned semantics: it behaves like a smarter bag-of-words, not like a
trained embedding model.

For real semantic recall, switch EMBEDDING_BACKEND=openai (needs
OPENAI_API_KEY) or plug in a `sentence-transformers` model the same way -
`VectorStore` and `HybridSearchService` only depend on the `EmbeddingProvider`
interface, so nothing else in the pipeline needs to change.
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
        # alternate_sign=True approximates a random projection, which keeps
        # dot products/cosine similarity meaningful despite hash collisions.
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
        from openai import OpenAI  # local import: keeps `openai` an optional dependency

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
