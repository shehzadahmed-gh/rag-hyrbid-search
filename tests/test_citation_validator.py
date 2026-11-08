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
    assert any("low lexical overlap" in issue for issue in report.issues)
    assert report.per_citation[0]["supported"] is False


def test_coverage_reflects_fraction_of_sources_cited():
    sources = [_source(1, "alpha text here"), _source(2, "beta text here")]
    answer = "Only the first source is cited [1]."
    report = validate_citations(answer, sources, support_threshold=0.0)
    assert report.coverage == 0.5
