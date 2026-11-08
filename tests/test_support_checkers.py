from types import SimpleNamespace

import numpy as np
import pytest

from app.services.support_checkers import (
    LexicalSupportChecker,
    LLMSupportChecker,
    NLISupportChecker,
    build_support_checker,
)


def test_lexical_score_is_fraction_of_claim_words_in_source():
    checker = LexicalSupportChecker()
    # content words: redis, celery, broker, fast -> 3 of 4 appear in the source
    assert checker.score("Redis is a fast Celery broker", "Redis is the Celery broker.") == pytest.approx(0.75)


class _FakeNLIModel:
    """Mimics sentence_transformers.CrossEncoder for an NLI model whose labels
    are not in the usual order, to check the entailment column is looked up."""

    config = SimpleNamespace(id2label={0: "neutral", 1: "contradiction", 2: "entailment"})

    def __init__(self):
        self.inputs = None

    def predict(self, pairs, apply_softmax=False):
        self.inputs = pairs
        assert apply_softmax
        return np.array([[0.1, 0.1, 0.8], [0.2, 0.7, 0.1]])


def test_nli_checker_reads_entailment_column_and_orders_premise_first():
    model = _FakeNLIModel()
    checker = NLISupportChecker(model=model)
    scores = checker.score_many([("claim a", "source a"), ("claim b", "source b")])
    assert scores == pytest.approx([0.8, 0.1])
    assert model.inputs == [("source a", "claim a"), ("source b", "claim b")]


def test_nli_checker_rejects_model_without_entailment_label():
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "positive", 1: "negative"}))
    with pytest.raises(ValueError):
        NLISupportChecker(model=model)


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, *, system, user, max_tokens=1024):
        assert "CLAIM:" in user
        return self.reply


@pytest.mark.parametrize(
    "reply, expected", [("SUPPORTED", 1.0), ("supported.", 1.0), ("NOT_SUPPORTED", 0.0), ("unsure", 0.0)]
)
def test_llm_checker_parses_verdict(reply, expected):
    assert LLMSupportChecker(_FakeLLM(reply)).score("claim", "source") == expected


def test_llm_checker_requires_a_real_provider():
    settings = SimpleNamespace(citation_checker="llm", default_llm_provider="local", nli_model="x")
    with pytest.raises(RuntimeError):
        build_support_checker(settings, llm_factory=lambda: None)


def test_default_checker_is_lexical():
    settings = SimpleNamespace(citation_checker="lexical", default_llm_provider="local", nli_model="x")
    assert isinstance(build_support_checker(settings), LexicalSupportChecker)
