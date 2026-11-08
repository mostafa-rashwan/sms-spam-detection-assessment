"""Tests for the academic preprocessing pipeline (hypothetical strings are used only for unit tests)."""

import pytest

from course_assessment.examples.selected_sentences import (
    clean_records,
    read_raw_records,
    split_records,
)
from course_assessment.preprocessing import text_preprocessor as tp


def test_lowercasing():
    assert tp.lowercase_text("FREE Entry NOW") == "free entry now"


def test_segmentation_one_and_many_sentences():
    assert tp.segment_sentences("ok see you at six") == ["ok see you at six"]
    assert tp.segment_sentences("call me. i am late! ok?") == ["call me.", "i am late!", "ok?"]


def test_word_tokenization_keeps_punctuation_tokens():
    assert tp.tokenize_words("hello, world!") == ["hello", ",", "world", "!"]


def test_punctuation_removal():
    assert tp.remove_punctuation(["hi", "!", "...", "``", "£", "u"]) == ["hi", "u"]


def test_number_removal_removes_digit_containing_tokens():
    tokens = ["call", "09061701461", "150p", "2", "m8s", "£1000", "now"]
    assert tp.remove_numbers(tokens) == ["call", "now"]


def test_stop_word_removal():
    assert tp.remove_stop_words(["you", "are", "winning", "the", "prize"]) == ["winning", "prize"]


def test_porter_stemming_matches_lecture_examples():
    assert tp.stem_tokens(["natural", "language", "processing", "tasks"]) == [
        "natur", "languag", "process", "task"]


def test_full_pipeline_output_type_and_value():
    result = tp.normalize_message("WIN a FREE prize now! Call 09061701461. Thanks.")
    assert isinstance(result, list) and all(isinstance(s, list) for s in result)
    assert result == [["win", "free", "prize"], ["call", "thank"]]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input(text):
    assert tp.normalize_message(text) == []
    assert tp.segment_sentences(text) == []
    assert tp.tokenize_words(text) == []


def test_non_string_rejected():
    with pytest.raises(TypeError):
        tp.normalize_message(None)


def test_sentence_with_only_punctuation_becomes_empty():
    result = tp.normalize_message("!!!")
    assert result and all(sentence == [] for sentence in result)


def test_trace_matches_pipeline_and_is_inspectable():
    text = "Hello THERE. Call 12345 now!"
    trace = tp.trace_message(text)
    assert trace["original_message"] == text
    assert [s["final_tokens"] for s in trace["sentence_traces"]] == tp.normalize_message(text)
    assert trace["number_of_sentences"] == len(trace["sentence_traces"])


def test_original_dataset_is_not_modified():
    records = [{"row_id": 0, "label": "ham", "text": "Hello THERE"}]
    snapshot = [dict(r) for r in records]
    for r in records:
        tp.normalize_message(r["text"])
        tp.trace_message(r["text"])
    assert records == snapshot


def test_loader_cleaning_and_split(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    rows = [("ham", f"message number {i} here") for i in range(40)] + \
           [("spam", f"win prize {i} now") for i in range(10)] + [("ham", "message number 0 here")]
    path.write_text("\n".join(f"{label}\t{text}" for label, text in rows), encoding="utf-8")
    records, stats = clean_records(read_raw_records(path))
    assert stats["duplicates_dropped"] == 1 and len(records) == 50
    train_a, test_a = split_records(records, 0.2, 7)
    train_b, test_b = split_records(records, 0.2, 7)
    assert test_a == test_b and len(test_a) == 10
    assert {r["row_id"] for r in train_a}.isdisjoint({r["row_id"] for r in test_a})
    assert sum(r["label"] == "spam" for r in test_a) == 2


def test_loader_rejects_label_conflict(tmp_path):
    with pytest.raises(ValueError, match="conflicting"):
        clean_records([("ham", "same"), ("spam", "same")])
