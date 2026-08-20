"""
Document chunking.

Uses a word-count sliding window with overlap rather than a token-accurate
splitter (e.g. tiktoken) so the service has zero external/network dependency
at chunk time - tiktoken's BPE files are fetched from a remote blob store on
first use, which is not guaranteed in every deployment environment (locked
down networks, air-gapped CI, etc.). If you want token-exact chunking for a
specific model, swap `_split_words` below for a tiktoken-based splitter; the
rest of the pipeline (embeddings, BM25, storage) is agnostic to how chunks
were produced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class ChunkSpan:
    index: int
    text: str
    start_word: int
    end_word: int


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40) -> list[ChunkSpan]:
    """
    Split `text` into overlapping chunks of ~`chunk_size_words` words.

    A sliding window over words (rather than raw characters) keeps chunks
    readable and roughly comparable in length, while overlap preserves
    context that would otherwise be severed at a chunk boundary - important
    for both BM25 (whole-term matches) and embedding quality.
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_size_words")

    words = normalize_whitespace(text).split(" ")
    words = [w for w in words if w]
    if not words:
        return []

    stride = chunk_size_words - overlap_words
    spans: list[ChunkSpan] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        spans.append(
            ChunkSpan(
                index=index,
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
            )
        )
        index += 1
        if end == len(words):
            break
        start += stride
    return spans
