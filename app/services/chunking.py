"""Word-window chunking with overlap.

Chunks are counted in words rather than model tokens, so chunking needs no
tokenizer download. For token-exact chunk sizes, replace the word split in
chunk_text with a tokenizer; the rest of the pipeline doesn't depend on how
chunks were made.
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
    """Split `text` into chunks of `chunk_size_words` words, each sharing
    `overlap_words` words with the previous one so a sentence cut at a
    boundary still appears whole in one of the two chunks."""
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
