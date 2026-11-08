"""Measure each retrieval stage on the Cranfield collection.

Cranfield is a classic IR test set: 1,400 aerodynamics abstracts, 225 queries
and human relevance judgments. The script downloads it once (about 1.8 MB,
checksum-verified), indexes it into fresh in-memory stores using the
embedding and reranker backends from your settings, and reports
Recall@10, MRR@10 and nDCG@10 for:

  BM25 only, dense only, hybrid (RRF) and hybrid + rerank

Usage:
    python scripts/evaluate_retrieval.py
    EMBEDDING_BACKEND=openai OPENAI_API_KEY=... python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k  # noqa: E402
from app.services.bm25_index import CorpusStore  # noqa: E402
from app.services.embeddings import build_embedding_provider  # noqa: E402
from app.services.hybrid_search import HybridSearchService  # noqa: E402
from app.services.ingestion_service import IngestionService  # noqa: E402
from app.services.reranker import build_reranker  # noqa: E402
from app.services.vector_store import QdrantVectorStore  # noqa: E402

BASE_URL = "https://raw.githubusercontent.com/oussbenk/cranfield-trec-dataset/main/"
FILES = {
    "cran.all.1400.xml": "369eb64e59d855b62463832f1338f471cede578571ee7fb33e3592a2a138ff47",
    "cran.qry.xml": "b609a59e980857ba59d098f33433822a5c200bcf6836a320babf2b1a5e7545eb",
    "cranqrel.trec.txt": "98a13b4913d61a02690725aee7ac4f6a1979c13fc9088ad9b4a81be58b1a6f11",
}
K = 10
CANDIDATES = 20  # chunks requested from each mode before collapsing to documents


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = data_dir / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {path}; delete it and re-run")


def load(data_dir: Path) -> tuple[dict[str, str], list[str], dict[int, set[str]]]:
    raw_docs = (data_dir / "cran.all.1400.xml").read_text(encoding="utf-8")
    docs = {}
    for block in re.findall(r"<doc>(.*?)</doc>", raw_docs, flags=re.S):
        docno = re.search(r"<docno>\s*(\d+)\s*</docno>", block).group(1)
        text = re.search(r"<text>(.*?)</text>", block, flags=re.S)
        docs[docno] = " ".join((text.group(1) if text else "").split())

    raw_queries = (data_dir / "cran.qry.xml").read_text(encoding="utf-8")
    queries = [" ".join(t.split()) for t in re.findall(r"<title>(.*?)</title>", raw_queries, flags=re.S)]

    # The qrels number queries 1..225 by position, not by the <num> in the query file.
    qrels: dict[int, set[str]] = {}
    for line in (data_dir / "cranqrel.trec.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and int(parts[3]) > 0:
            qrels.setdefault(int(parts[0]), set()).add(parts[2])
    return docs, queries, qrels


def build_index(docs: dict[str, str], workdir: str):
    settings = get_settings()
    embedder = build_embedding_provider(settings)
    corpus = CorpusStore(f"{workdir}/corpus.jsonl")
    vectors = QdrantVectorStore("cranfield", dimension=embedder.dimension, local_path=None)
    ingestion = IngestionService(
        corpus=corpus,
        vector_store=vectors,
        embedder=embedder,
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
    )
    for docno, text in docs.items():
        if text:
            ingestion.ingest_text(source=f"cranfield-{docno}", text=text, metadata={"docno": docno})
    search = HybridSearchService(
        corpus=corpus,
        vector_store=vectors,
        embedder=embedder,
        reranker=build_reranker(settings),
        rrf_k=settings.rrf_k,
        dense_k=CANDIDATES,
        bm25_k=CANDIDATES,
    )
    return settings, embedder, corpus, vectors, search


def to_documents(chunk_ids: list[str], corpus: CorpusStore) -> list[str]:
    """Collapse a ranked chunk list to a ranked list of unique document ids."""
    ranked: list[str] = []
    for cid in chunk_ids:
        chunk = corpus.get(cid)
        if chunk and chunk.metadata["docno"] not in ranked:
            ranked.append(chunk.metadata["docno"])
    return ranked[:K]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/eval/cranfield")
    parser.add_argument("--output", default="results/retrieval_eval.json")
    args = parser.parse_args()

    download(Path(args.data_dir))
    docs, queries, qrels = load(Path(args.data_dir))
    print(f"{len(docs)} documents, {len(queries)} queries, {sum(len(v) for v in qrels.values())} relevance judgments")

    with tempfile.TemporaryDirectory() as workdir:
        started = time.perf_counter()
        settings, embedder, corpus, vectors, search = build_index(docs, workdir)
        print(
            f"indexed in {time.perf_counter() - started:.1f}s with embeddings={settings.embedding_backend}, "
            f"reranker={settings.reranker_backend}"
        )

        modes = {
            "BM25 only": lambda q: [cid for cid, _ in corpus.search(q, top_k=CANDIDATES)],
            "Dense only": lambda q: [h.chunk_id for h in vectors.search(embedder.embed_one(q), top_k=CANDIDATES)],
            "Hybrid (RRF)": lambda q: [r.chunk_id for r in search.search(q, top_k=CANDIDATES, rerank=False)[0]],
            "Hybrid + rerank": lambda q: [r.chunk_id for r in search.search(q, top_k=CANDIDATES, rerank=True)[0]],
        }

        results = {}
        for name, run in modes.items():
            recalls, rrs, ndcgs, latencies = [], [], [], []
            for qid, query in enumerate(queries, start=1):
                relevant = qrels.get(qid)
                if not relevant:
                    continue
                t0 = time.perf_counter()
                ranked = to_documents(run(query), corpus)
                latencies.append((time.perf_counter() - t0) * 1000)
                recalls.append(recall_at_k(ranked, relevant, K))
                rrs.append(reciprocal_rank_at_k(ranked, relevant, K))
                ndcgs.append(ndcg_at_k(ranked, relevant, K))
            results[name] = {
                "recall@10": round(statistics.mean(recalls), 4),
                "mrr@10": round(statistics.mean(rrs), 4),
                "ndcg@10": round(statistics.mean(ndcgs), 4),
                "median_ms": round(statistics.median(latencies), 2),
                "queries": len(recalls),
            }

    print("\n| Mode | Recall@10 | MRR@10 | nDCG@10 | Median latency |\n|---|---|---|---|---|")
    for name, m in results.items():
        print(f"| {name} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {m['median_ms']:.1f} ms |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": "cranfield",
                "embedding_backend": settings.embedding_backend,
                "reranker_backend": settings.reranker_backend,
                "rrf_k": settings.rrf_k,
                "candidates_per_retriever": CANDIDATES,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
