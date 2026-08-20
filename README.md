# RAG Pipeline with Hybrid Search

A three-stage hybrid retrieval pipeline (dense vector search + BM25 + reranking) that generates grounded answers over internal documents, with automatic citation validation. Built as modular FastAPI services with pluggable, swappable backends.

```
Ingestion  ->  chunk -> embed -> write to vector store + BM25 corpus
Retrieval  ->  dense search  ⌐
               BM25 search   ⌐-> Reciprocal Rank Fusion -> rerank -> top-k
Generation ->  numbered sources -> LLM -> answer with [n] citations
Citations  ->  parse [n] markers -> range check + lexical support check
```

## Why it's built this way

Every external dependency sits behind a small interface (`EmbeddingProvider`, `VectorStore`, `Reranker`, `LLMProvider`), each with an **offline default** and a **real, API-backed implementation**:

| Stage | Offline default (no key, no network) | Production backend |
|---|---|---|
| Embeddings | scikit-learn `HashingVectorizer` (deterministic hashing trick) | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant or Chroma, **embedded/on-disk mode** | Qdrant or Chroma **server** (docker-compose) |
| Reranker | TF-IDF cosine similarity | `sentence-transformers` cross-encoder |
| Generation | deterministic extractive stub | GPT-4o or Claude Sonnet |

This means the whole pipeline — ingest, hybrid search, grounded generation, citation validation — runs and is testable with **zero API keys and zero external services**, while every production backend named in the spec (GPT-4o, Claude Sonnet, ChromaDB, Qdrant) is fully implemented and is a one-line `.env` change away from being live. The offline embedder/reranker are real algorithms, not mocks — they just don't carry learned semantics the way a trained model does, so answer quality with everything set to `local` is a correctness/plumbing demo, not a quality benchmark.

## Running it

**Docker (production-shaped, real Qdrant + Chroma servers):**
```bash
docker compose up --build
# service at http://localhost:8000, docs at /docs
```

**Locally, no Docker:**
```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

**End-to-end smoke test** (ingests 3 sample docs, runs hybrid search, generates cited answers, deliberately submits a hallucinated citation to confirm the validator catches it):
```bash
uvicorn app.main:app &
python scripts/smoke_test.py
```

**Unit tests** (chunking, BM25 + persistence, RRF math, citation validator — no server needed):
```bash
pytest tests/ -v
```

To use real GPT-4o / Claude Sonnet / OpenAI embeddings, set the relevant keys in `.env` (see `.env.example`) and pass `"provider": "openai"` or `"provider": "anthropic"` to `/generation/answer`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingestion/documents` | Chunk, embed, and index a document (`{source, text, metadata}`) |
| `GET /ingestion/documents` | List ingested documents |
| `DELETE /ingestion/documents/{id}` | Remove a document and its chunks from both stores |
| `POST /retrieval/search` | Run the 3-stage hybrid search directly; response includes per-stage scores (dense/bm25/rrf/rerank) for transparency |
| `POST /generation/answer` | Retrieve + generate a cited answer (`provider`: `local`/`openai`/`anthropic`) |
| `POST /citations/validate` | Validate an arbitrary `{answer, sources}` pair standalone |

Full interactive schema at `/docs` once running.

## Design notes / what a next iteration would add

- **"Modular services"** are implemented as decoupled routers + service classes in one FastAPI app rather than four physically separate deployable services — each module only talks to the others through the same Pydantic schemas, so splitting any one of them into its own process/repo later is a refactor, not a rewrite.
- **Citation validation is a heuristic screen, not entailment.** It reliably catches out-of-range/invented citation numbers and near-zero lexical overlap, but token overlap can't catch a fluent sentence that misrepresents a source it does share vocabulary with. A production system would add an NLI/entailment model or an LLM-as-judge pass on top of `validate_citations()`.
- **BM25 persistence** is a JSONL file rebuilt into memory on startup — fine for a single-process demo; a multi-worker deployment would move chunk storage to a real database (see the structured-state pattern in the agent-orchestration-system project) and rebuild the in-memory index per worker exactly the same way.
- Not included: auth/authz, rate limiting, file-upload parsing for PDF/DOCX (currently accepts raw text), request-level observability/tracing. All straightforward additions on top of the existing module boundaries.
