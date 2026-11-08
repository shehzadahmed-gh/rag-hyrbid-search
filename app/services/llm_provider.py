"""
Pluggable generation backends (the "model backends" side of "Integrated
multiple model and storage backends, including GPT-4o, Claude Sonnet,
ChromaDB and Qdrant").

Every provider implements `generate_grounded_answer(query, sources)`, which
answers strictly from the supplied source chunks and cites them inline as
`[1]`, `[2]`, ... matching each source's 1-indexed position. The default
`LocalExtractiveProvider` needs no API key and does this deterministically
via lexical overlap, so `/generation/answer` works out of the box; GPT-4o and
Claude Sonnet are real, fully-implemented alternatives that activate as soon
as the corresponding API key is set.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.services.chunking import split_sentences

GROUNDED_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using ONLY "
    "the numbered source excerpts provided - do not use outside knowledge. "
    "Every factual sentence in your answer MUST end with a citation marker like [1] "
    "or [2] referencing the source excerpt(s) that support it. "
    "If the sources do not contain enough information to answer, say so plainly "
    "instead of guessing."
)


def build_context_prompt(query: str, sources: list) -> str:
    numbered = "\n\n".join(f"[{s.id}] (source: {s.source})\n{s.text}" for s in sources)
    return f"SOURCES:\n{numbered}\n\nQUESTION:\n{query}\n\nAnswer, citing sources as [n]:"


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def generate_grounded_answer(self, query: str, sources: list, max_tokens: int = 800) -> str:
        """Default path used by every *real* LLM provider: build a citation
        -instructed prompt and let the model do the reasoning. The offline
        stub below overrides this instead of `generate()`."""
        user_prompt = build_context_prompt(query, sources)
        return self.generate(system=GROUNDED_SYSTEM_PROMPT, user=user_prompt, max_tokens=max_tokens)


class OpenAIProvider(LLMProvider):
    """GPT-4o via the OpenAI Chat Completions API."""

    name = "openai-gpt-4o"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI  # optional dependency, imported lazily

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    """Claude Sonnet via the Anthropic Messages API."""

    name = "anthropic-claude-sonnet"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5"):
        import anthropic  # optional dependency, imported lazily

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class LocalExtractiveProvider(LLMProvider):
    """
    Deterministic, dependency-free stand-in generator. It does not reason
    about the question the way GPT-4o/Claude do - it scores each source
    sentence by token overlap with the query, keeps the best-matching
    sentence(s) per relevant source, and stitches them together with
    citation markers. This exists so the full ingestion -> retrieval ->
    generation -> citation-validation pipeline is exercisable end-to-end
    with zero API keys and zero network access; swap DEFAULT_LLM_PROVIDER to
    "openai" or "anthropic" for real generative answers.
    """

    name = "local-extractive-stub"
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
        "for", "and", "or", "but", "with", "as", "by", "at", "from", "that", "this", "it", "its",
        "what", "which", "who", "how", "why", "do", "does", "did", "can", "could", "should", "would",
    }

    def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        # Not used directly - generate_grounded_answer() is overridden below -
        # but implemented so LocalExtractiveProvider still satisfies the
        # LLMProvider interface for any generic (non-RAG) use.
        return "[local-extractive-stub] free-form generate() is not implemented; use generate_grounded_answer()."

    def _tokens(self, text: str) -> set[str]:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
        return {w for w in words if w not in self._STOPWORDS}

    def generate_grounded_answer(self, query: str, sources: list, max_tokens: int = 800) -> str:
        query_tokens = self._tokens(query)
        picked: list[tuple[float, int, str]] = []  # (score, source_id, sentence)

        for source in sources:
            best_sentence, best_score = None, 0.0
            for sentence in split_sentences(source.text):
                sent_tokens = self._tokens(sentence)
                if not sent_tokens:
                    continue
                overlap = len(query_tokens & sent_tokens) / len(query_tokens | sent_tokens)
                if overlap > best_score:
                    best_sentence, best_score = sentence, overlap
            if best_sentence and best_score > 0:
                picked.append((best_score, source.id, best_sentence))

        picked.sort(key=lambda x: -x[0])
        top_picks = picked[:4]

        if not top_picks:
            return (
                "The retrieved sources do not appear to contain information that directly "
                "answers this question, so I can't provide a grounded answer from them."
            )

        top_picks.sort(key=lambda x: x[1])  # restore citation order for readability
        sentences = [f"{sentence.rstrip('.')} [{source_id}]." for _, source_id, sentence in top_picks]
        return " ".join(sentences)


def build_llm_provider(settings, override: str | None = None) -> LLMProvider:
    provider_name = override or settings.default_llm_provider
    if provider_name == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("provider=openai requires OPENAI_API_KEY to be set.")
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    if provider_name == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("provider=anthropic requires ANTHROPIC_API_KEY to be set.")
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_chat_model)
    return LocalExtractiveProvider()
