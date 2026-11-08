"""End-to-end check against a running service: ingest three documents, run
hybrid search, generate cited answers, validate a bad answer, delete a
document.

Usage:
    uvicorn app.main:app --port 8000 &
    python scripts/smoke_test.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

DOCS = [
    {
        "source": "durham-cs-handbook.txt",
        "text": (
            "Durham University's Computer Science department offers a BSc and an MEng in Computer Science. "
            "First-year students take modules covering programming fundamentals, discrete mathematics, and "
            "computer systems. The department is based at the Lower Mountjoy site and shares facilities with "
            "the Department of Engineering. Formal methods, including propositional and predicate logic, SAT "
            "solving, and first-order resolution, are introduced in the first year as a foundation for later "
            "modules on algorithms and complexity. Students can join societies such as DUFS, the Durham "
            "University Finance Society, alongside their degree."
        ),
        "metadata": {"category": "university"},
    },
    {
        "source": "hybrid-search-notes.txt",
        "text": (
            "Hybrid search combines dense vector retrieval with sparse lexical retrieval such as BM25. Dense "
            "retrieval represents queries and documents as embeddings and ranks documents by cosine similarity, "
            "which is effective at matching paraphrases and synonyms. BM25 instead ranks documents using term "
            "frequency and inverse document frequency, and is very effective at matching exact keywords, "
            "acronyms, and identifiers that a dense embedding might blur together. Reciprocal Rank Fusion is a "
            "common technique for merging two ranked lists using only rank position, which avoids having to "
            "calibrate two incomparable scoring scales. A cross-encoder reranker is often applied afterward to "
            "the fused candidate set, since it can jointly attend to the query and each candidate for a more "
            "accurate final ranking than either retrieval stage alone."
        ),
        "metadata": {"category": "engineering"},
    },
    {
        "source": "celery-notes.txt",
        "text": (
            "Celery is a distributed task queue for Python that executes work asynchronously outside of the "
            "request-response cycle. A Celery worker process pulls tasks off a message broker, commonly Redis "
            "or RabbitMQ, and executes them independently of the web server. This separation lets a web "
            "application return a response immediately after enqueueing a long-running job, then let the "
            "client poll for the result or receive a callback once the worker finishes."
        ),
        "metadata": {"category": "engineering"},
    },
]

QUESTIONS = [
    "What formal methods topics do first-year Durham CS students study?",
    "Why combine BM25 with dense vector retrieval instead of using just one?",
    "What problem does Celery solve for a web application?",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=30.0)
    failures = 0

    print("== health ==")
    r = client.get("/health")
    assert r.status_code == 200, r.text
    print(r.json())

    print("\n== ingestion ==")
    document_ids = []
    for doc in DOCS:
        r = client.post("/ingestion/documents", json=doc)
        if r.status_code != 201:
            print("INGEST FAILED:", r.status_code, r.text)
            failures += 1
            continue
        body = r.json()
        document_ids.append(body["document_id"])
        print(f"  ingested {doc['source']!r} -> {body['chunk_count']} chunk(s)")

    r = client.get("/ingestion/documents")
    print(f"  {len(r.json())} document(s) now in the corpus")

    print("\n== retrieval (hybrid search) ==")
    for q in QUESTIONS:
        r = client.post("/retrieval/search", json={"query": q, "top_k": 3})
        if r.status_code != 200:
            print("SEARCH FAILED:", r.status_code, r.text)
            failures += 1
            continue
        body = r.json()
        print(f"  Q: {q}")
        print(f"    stages: {body['stages']}")
        for res in body["results"]:
            print(
                f"    #{res['rank']} [{res['source']}#{res['chunk_index']}] "
                f"dense={res['dense_score']} bm25={res['bm25_score']} "
                f"rrf={round(res['rrf_score'], 4) if res['rrf_score'] else None} "
                f"rerank={round(res['rerank_score'], 4) if res['rerank_score'] else None}"
            )
        if not body["results"]:
            print("    !! no results returned")
            failures += 1

    print("\n== generation (local extractive provider, no API key) ==")
    for q in QUESTIONS:
        r = client.post("/generation/answer", json={"query": q, "top_k": 3})
        if r.status_code != 200:
            print("GENERATE FAILED:", r.status_code, r.text)
            failures += 1
            continue
        body = r.json()
        print(f"  Q: {q}")
        print(f"  A: {body['answer']}")
        print(f"  provider_used={body['provider_used']}")
        report = body["citation_report"]
        print(
            f"  citation_report: valid={report['valid']} cited_sources={report['cited_sources']} "
            f"coverage={report['coverage']} issues={report['issues']}"
        )
        if not report["cited_sources"]:
            print("    !! answer had no citations")
            failures += 1
        print()

    print("== citation validation on a deliberately bad answer ==")
    r = client.post("/retrieval/search", json={"query": QUESTIONS[0], "top_k": 2})
    sources = [
        {
            "id": i + 1,
            **{k: v for k, v in res.items() if k in ("chunk_id", "document_id", "source", "chunk_index", "text")},
        }
        for i, res in enumerate(r.json()["results"])
    ]
    # [7] doesn't exist, and the [1] claim is invented but reuses words from source 1.
    bad_answer = (
        "Durham CS students spend their first year exclusively on watercolour painting [1] "
        "and also learn SAT solving [7]."
    )
    r = client.post("/citations/validate", json={"answer": bad_answer, "sources": sources})
    report = r.json()
    print(json.dumps(report, indent=2))
    if any("[7]" in issue for issue in report["issues"]):
        print("  out-of-range citation [7] flagged")
    else:
        print("    !! expected the invented citation [7] to be flagged")
        failures += 1
    first = next((c for c in report["per_citation"] if c["citation"] == 1), None)
    if first and first["supported"]:
        print(
            f"  NOTE: the made-up [1] claim passed (score {first['support_score']}). The default lexical "
            "checker can't catch this kind of fabrication; set CITATION_CHECKER=nli or llm."
        )
    elif first:
        print("  made-up [1] claim flagged as unsupported")

    print("\n== document deletion ==")
    if document_ids:
        r = client.delete(f"/ingestion/documents/{document_ids[0]}")
        print(" ", r.status_code, r.json())
        if r.status_code != 200:
            failures += 1

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
