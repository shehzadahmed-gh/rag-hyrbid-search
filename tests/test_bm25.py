import os
import tempfile

from app.services.bm25_index import Chunk, CorpusStore


def _chunk(cid, doc_id, text, idx=0):
    return Chunk(chunk_id=cid, document_id=doc_id, source="test.txt", chunk_index=idx, text=text, metadata={})


def test_add_and_search_finds_relevant_chunk():
    with tempfile.TemporaryDirectory() as tmp:
        store = CorpusStore(os.path.join(tmp, "corpus.jsonl"))
        store.add_chunks(
            [
                _chunk("a", "doc1", "Redis is an in-memory data store used as a Celery broker."),
                _chunk("b", "doc1", "Qdrant is a vector database written in Rust."),
            ]
        )
        results = store.search("celery broker redis", top_k=5)
        assert results, "expected at least one BM25 hit"
        assert results[0][0] == "a"


def test_persistence_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "corpus.jsonl")
        store1 = CorpusStore(path)
        store1.add_chunks([_chunk("a", "doc1", "hello world")])

        store2 = CorpusStore(path)  # simulate process restart
        assert store2.get("a") is not None
        assert store2.get("a").text == "hello world"


def test_delete_document_removes_only_its_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        store = CorpusStore(os.path.join(tmp, "corpus.jsonl"))
        store.add_chunks(
            [
                _chunk("a1", "doc1", "alpha content"),
                _chunk("a2", "doc1", "alpha content two"),
                _chunk("b1", "doc2", "beta content"),
            ]
        )
        deleted = store.delete_document("doc1")
        assert deleted == 2
        assert store.get("a1") is None
        assert store.get("b1") is not None


def test_search_on_empty_corpus_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store = CorpusStore(os.path.join(tmp, "corpus.jsonl"))
        assert store.search("anything", top_k=5) == []


def test_index_is_rebuilt_once_after_many_adds(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        store = CorpusStore(os.path.join(tmp, "corpus.jsonl"))
        rebuilds = []
        original = store._rebuild_index
        monkeypatch.setattr(store, "_rebuild_index", lambda: (rebuilds.append(1), original())[1])
        for i in range(50):
            store.add_chunks([_chunk(f"c{i}", f"doc{i}", f"document number {i} about topic {i % 5}")])
        assert rebuilds == []
        assert store.search("topic", top_k=3)
        assert store.search("number", top_k=3)
        assert len(rebuilds) == 1


def test_delete_rewrites_file_atomically(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "corpus.jsonl")
        store = CorpusStore(path)
        store.add_chunks([_chunk("a", "doc1", "alpha"), _chunk("b", "doc2", "beta")])
        before = open(path).read()

        def crash(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", crash)
        try:
            store.delete_document("doc1")
        except OSError:
            pass
        # The original file is untouched and memory still matches it.
        assert open(path).read() == before
        assert store.get("a") is not None
        monkeypatch.undo()

        store.delete_document("doc1")
        assert CorpusStore(path).get("a") is None
        assert CorpusStore(path).get("b") is not None
