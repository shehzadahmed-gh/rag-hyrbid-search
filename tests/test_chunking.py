from app.services.chunking import chunk_text, split_sentences


def test_chunk_text_single_chunk_when_short():
    spans = chunk_text("one two three four five", chunk_size_words=10, overlap_words=2)
    assert len(spans) == 1
    assert spans[0].text == "one two three four five"


def test_chunk_text_overlap_and_stride():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    spans = chunk_text(text, chunk_size_words=10, overlap_words=3)
    # stride = 7, so starts are 0, 7, 14, 21 -> 4 chunks covering all 25 words
    assert [s.start_word for s in spans] == [0, 7, 14, 21]
    assert spans[-1].end_word == 25
    # consecutive chunks should share exactly `overlap_words` words
    first_words = spans[0].text.split(" ")
    second_words = spans[1].text.split(" ")
    assert first_words[-3:] == second_words[:3]


def test_chunk_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size_words=5, overlap_words=5)


def test_chunk_text_empty_input():
    assert chunk_text("   ") == []


def test_split_sentences_basic():
    sentences = split_sentences("This is one. Is this two? Yes, three!")
    assert sentences == ["This is one.", "Is this two?", "Yes, three!"]
