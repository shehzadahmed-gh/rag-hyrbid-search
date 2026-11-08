from types import SimpleNamespace

from app.services.citation_validator import validate_citations


def _source(id_, text):
    return SimpleNamespace(id=id_, text=text)


def test_valid_well_supported_citation():
    sources = [_source(1, "Redis is used as the Celery message broker in this deployment.")]
    answer = "Redis is used as the Celery message broker in this deployment [1]."
    report = validate_citations(answer, sources, support_threshold=0.15)
    assert report.valid is True
    assert report.cited_sources == [1]
    assert report.issues == []


def test_out_of_range_citation_is_flagged():
    sources = [_source(1, "Some source text.")]
    answer = "This claim cites a source that does not exist [2]."
    report = validate_citations(answer, sources)
    assert report.valid is False
    assert any("do not match any retrieved source" in issue for issue in report.issues)


def test_missing_citations_entirely_is_flagged():
    sources = [_source(1, "Some source text.")]
    answer = "This answer has no citation markers at all."
    report = validate_citations(answer, sources)
    assert report.valid is False
    assert any("no [n] citation markers" in issue for issue in report.issues)


def test_low_overlap_citation_is_flagged_as_unsupported():
    sources = [_source(1, "Qdrant is a vector database written in Rust.")]
    answer = "The stock market closed higher today on strong earnings [1]."
    report = validate_citations(answer, sources, support_threshold=0.15)
    assert report.valid is False
    assert any("lexical checker" in issue for issue in report.issues)
    assert report.per_citation[0]["supported"] is False


def test_coverage_reflects_fraction_of_sources_cited():
    sources = [_source(1, "alpha text here"), _source(2, "beta text here")]
    answer = "Only the first source is cited [1]."
    report = validate_citations(answer, sources, support_threshold=0.0)
    assert report.coverage == 0.5


def test_each_citation_is_checked_against_its_own_clause():
    sources = [
        _source(1, "Redis is the Celery broker."),
        _source(2, "Qdrant is a vector database written in Rust."),
    ]
    answer = "Redis is the Celery broker [1] and Qdrant is written in Rust [2]."
    report = validate_citations(answer, sources, support_threshold=0.6)
    claims = {c["citation"]: c["claim"] for c in report.per_citation}
    assert claims[1] == "Redis is the Celery broker"
    assert claims[2] == "and Qdrant is written in Rust"
    assert report.valid is True


def test_adjacent_markers_share_a_claim():
    sources = [_source(1, "Redis is a broker."), _source(2, "Redis is an in-memory store.")]
    report = validate_citations("Redis is a broker [1][2].", sources, support_threshold=0.0)
    assert [(c["citation"], c["claim"]) for c in report.per_citation] == [
        (1, "Redis is a broker"),
        (2, "Redis is a broker"),
    ]


def test_lexical_checker_misses_fabrication_that_reuses_source_words():
    # Known limitation, documented in the README: overlap can't tell that
    # "exclusively on watercolour painting" contradicts the source.
    sources = [_source(1, "First-year Durham students study programming and SAT solving in their first year.")]
    report = validate_citations(
        "Durham students spend their first year exclusively on watercolour painting [1].", sources
    )
    assert report.valid is True


class _FakeChecker:
    name = "fake"
    default_threshold = 0.5

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    def score_many(self, pairs):
        self.calls.extend(pairs)
        return [self.verdicts[claim] for claim, _ in pairs]


def test_a_stricter_checker_flags_the_fabrication():
    sources = [_source(1, "First-year Durham students study programming and SAT solving.")]
    claim = "Durham students spend their first year exclusively on watercolour painting"
    checker = _FakeChecker({claim: 0.02})
    report = validate_citations(f"{claim} [1].", sources, checker=checker)
    assert report.valid is False
    assert checker.calls == [(claim, sources[0].text)]
    assert "fake checker" in report.issues[0]
