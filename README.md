# RAG Pipeline with Hybrid Search

[![CI](https://github.com/shehzadahmed-gh/rag-hybrid-search/actions/workflows/ci.yml/badge.svg)](https://github.com/shehzadahmed-gh/rag-hybrid-search/actions/workflows/ci.yml)

A retrieval-augmented generation service in FastAPI. Documents are chunked and indexed twice, as vectors (Qdrant or Chroma) and in a BM25 index. Queries search both, merge the two rankings with Reciprocal Rank Fusion, rerank, and pass the top chunks to a model that must cite them as `[n]`. A validator then checks every citation against the passage it points to.

I added the citation check after early versions gave confident answers that the documents didn't back up.

```
Ingestion   chunk -> embed -> vector store + BM25 corpus
Retrieval   dense search ┐
            BM25 search  ┴-> Reciprocal Rank Fusion -> rerank -> top-k chunks
Generation  numbered chunks -> LLM -> answer with [n] citations
Validation  each [n] in range? each cited claim supported by its source?
```

## Backends

Each stage sits behind a small interface with an offline default and an API-backed or model-backed option, chosen in `.env`.

| Stage | Offline default | Other options |
|---|---|---|
| Embeddings | scikit-learn `HashingVectorizer` | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant or Chroma, embedded on disk | Qdrant or Chroma server (docker-compose) |
| Reranker | TF-IDF cosine over the candidates | sentence-transformers cross-encoder |
| Generation | extractive (copies the best-matching source sentences) | GPT-4o, Claude |
| Citation check | lexical overlap | NLI cross-encoder, LLM judge |

With the defaults, everything runs with no API key and no network access. The offline embedder and generator are there to exercise the pipeline; they are not meant to produce good answers.

## Retrieval results

`scripts/evaluate_retrieval.py` runs every retrieval mode over the Cranfield collection (1,400 aerodynamics abstracts, 225 queries with human relevance judgments). It downloads the data on first run and checks it against stored SHA-256 hashes. CI re-runs it and fails if the numbers drift from `results/retrieval_eval.json`.

With the offline defaults (hashing embeddings, 384 dimensions, TF-IDF reranker):

| Mode | Recall@10 | MRR@10 | nDCG@10 | Median latency |
|---|---|---|---|---|
| BM25 only | 0.349 | 0.476 | 0.333 | 9 ms |
| Dense only | 0.153 | 0.283 | 0.157 | 2 ms |
| Hybrid (RRF) | 0.289 | 0.430 | 0.276 | 13 ms |
| Hybrid + rerank | 0.343 | 0.483 | 0.332 | 16 ms |

Hybrid search is worse than BM25 alone here. The hashing embedder has no learned meaning, so it ranks by roughly the same word overlap as BM25 but more noisily, and RRF gives its ranking equal weight. More hash buckets help the dense stage (nDCG@10 0.195 at 1,024 dimensions, 0.225 at 16,384) but hybrid still trails BM25 at every size I tried. Reranking pulls hybrid back to BM25's level.

So with these defaults I would ship BM25 plus reranking and leave the dense stage off. Hybrid search should pay off with embeddings that capture meaning, which is what the stage is for. To measure that, rerun the script with `EMBEDDING_BACKEND=openai` (or plug in any embedding model behind `EmbeddingProvider`).

## Citation validation

The validator splits the answer into sentences, then splits each sentence at its citation markers, so in "A is true [1] and B is true [2]" source 1 is checked against "A is true" and source 2 against "and B is true". Then it runs two checks:

1. Every `[n]` must refer to a retrieved source. A number outside the range is an invented citation.
2. Every cited claim must be supported by its source, as judged by the configured checker:

| `CITATION_CHECKER` | How it scores a claim | Default threshold | Catches |
|---|---|---|---|
| `lexical` (default) | Fraction of the claim's content words found in the source | 0.15 | Citations pointing at unrelated passages |
| `nli` | Entailment probability from an NLI cross-encoder (source as premise, claim as hypothesis) | 0.5 | Claims that reuse the source's words but say something else |
| `llm` | SUPPORTED / NOT_SUPPORTED verdict from the configured LLM | 0.5 | Same, at the cost of one call per citation |

The lexical checker has a real blind spot, and the smoke test demonstrates it. "Durham CS students spend their first year exclusively on watercolour painting [1]" is invented, but half its content words appear in source 1, so it passes. Only the invented `[7]` in the same answer is caught. The NLI and LLM checkers are there for this case. Both are unit-tested with stand-in models; they need `sentence-transformers` or an API key to run for real.

The extractive generator copies source sentences verbatim, so its citations always pass. The validator only gets a real test from LLM-written answers.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload          # API docs at http://localhost:8000/docs
```

With Qdrant and Chroma servers:

```bash
docker compose up --build
```

Checks:

```bash
pytest -q                                   # 37 unit tests
python scripts/smoke_test.py                # end to end against a running server
python scripts/evaluate_retrieval.py        # retrieval metrics on Cranfield
```

CI runs lint, the unit tests and the smoke test against both vector stores, plus the retrieval evaluation.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingestion/documents` | Chunk, embed and index `{source, text, metadata}` |
| `GET /ingestion/documents` | List documents |
| `DELETE /ingestion/documents/{id}` | Remove a document from both indexes |
| `POST /retrieval/search` | Hybrid search, with each result's dense, BM25, RRF and rerank scores |
| `POST /generation/answer` | Retrieve and answer with citations (`provider`: `local`, `openai` or `anthropic`) |
| `POST /citations/validate` | Validate any `{answer, sources}` pair |

## Implementation notes

- RRF merges rankings by position only (`1 / (k + rank)`, k = 60), which avoids comparing BM25 scores with cosine similarities.
- The BM25 index lives in memory, rebuilt from a JSONL chunk file at startup. Adding documents only marks the index stale, and it is rebuilt once on the next search. When every add rebuilt the whole index, ingesting the 1,400 Cranfield documents one by one took 90 s; it now takes 1.4 s.
- Deleting a document writes the new chunk file to a temporary path and renames it over the old one, and only updates memory once the rename succeeds. A crash mid-write can't truncate the corpus.
- Qdrant point IDs must be integers or UUIDs, so chunk IDs are mapped with `uuid5`, which keeps re-ingestion of the same chunk idempotent.

## Limitations

- Ingestion, retrieval, generation and validation are routers in one FastAPI app, not separate services. They only talk through Pydantic schemas, so splitting them out later is mostly packaging.
- The chunk store is a single JSONL file guarded by a thread lock, which is fine for one process. Several workers would each hold their own copy, so they would need a shared database.
- Re-ingesting the same text creates a new document. There is no deduplication.
- Only raw text is accepted; there is no PDF or DOCX parsing.
- No authentication or rate limiting.
