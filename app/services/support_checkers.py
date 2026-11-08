"""Ways of deciding whether a source passage supports a claim.

Each checker returns a score in [0, 1] (higher means better supported) and has
a default threshold. Pick one with CITATION_CHECKER:

  lexical  fraction of the claim's content words that appear in the source.
           Free and instant, catches citations to unrelated passages, but a
           made-up sentence that reuses the source's vocabulary still passes.
  nli      probability of "entailment" from a cross-encoder NLI model
           (sentence-transformers). Catches claims that share words with the
           source but say something different. Downloads the model on first use.
  llm      asks the configured OpenAI/Anthropic model for a SUPPORTED /
           NOT_SUPPORTED verdict. Most flexible, costs one call per citation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "for", "and", "or", "but", "with", "as", "by", "at", "from", "that", "this", "it", "its",
}  # fmt: skip


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


class SupportChecker(ABC):
    name: str
    default_threshold: float

    @abstractmethod
    def score(self, claim: str, source_text: str) -> float:
        raise NotImplementedError

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.score(claim, source) for claim, source in pairs]


class LexicalSupportChecker(SupportChecker):
    name = "lexical"
    default_threshold = 0.15

    def score(self, claim: str, source_text: str) -> float:
        claim_tokens = content_tokens(claim)
        source_tokens = content_tokens(source_text)
        if not claim_tokens or not source_tokens:
            return 0.0
        return len(claim_tokens & source_tokens) / len(claim_tokens)


class NLISupportChecker(SupportChecker):
    """Entailment probability from an NLI cross-encoder, with the source as
    premise and the claim as hypothesis."""

    name = "nli"
    default_threshold = 0.5

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", model=None):
        if model is None:
            from sentence_transformers import CrossEncoder  # optional dependency

            model = CrossEncoder(model_name)
        self._model = model
        self._entailment_index = self._find_entailment_index(model)

    @staticmethod
    def _find_entailment_index(model) -> int:
        # Label order differs between NLI models, so read it from the config.
        config = getattr(model, "config", None) or model.model.config
        for index, label in config.id2label.items():
            if "entail" in label.lower():
                return int(index)
        raise ValueError(f"model has no entailment label: {config.id2label}")

    def score(self, claim: str, source_text: str) -> float:
        return self.score_many([(claim, source_text)])[0]

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        probabilities = self._model.predict([(source, claim) for claim, source in pairs], apply_softmax=True)
        return [float(row[self._entailment_index]) for row in probabilities]


class LLMSupportChecker(SupportChecker):
    name = "llm"
    default_threshold = 0.5

    _SYSTEM = (
        "You check citations. Given a source passage and a claim, answer SUPPORTED if the passage "
        "states or directly implies the claim, otherwise NOT_SUPPORTED. Reply with that one word."
    )

    def __init__(self, llm):
        self._llm = llm

    def score(self, claim: str, source_text: str) -> float:
        verdict = self._llm.generate(
            system=self._SYSTEM,
            user=f"SOURCE:\n{source_text}\n\nCLAIM:\n{claim}",
            max_tokens=5,
        )
        return 1.0 if verdict.strip().upper().startswith("SUPPORTED") else 0.0


def build_support_checker(settings, llm_factory=None) -> SupportChecker:
    if settings.citation_checker == "nli":
        return NLISupportChecker(model_name=settings.nli_model)
    if settings.citation_checker == "llm":
        if llm_factory is None or settings.default_llm_provider == "local":
            raise RuntimeError("CITATION_CHECKER=llm needs DEFAULT_LLM_PROVIDER set to openai or anthropic.")
        return LLMSupportChecker(llm_factory())
    return LexicalSupportChecker()
