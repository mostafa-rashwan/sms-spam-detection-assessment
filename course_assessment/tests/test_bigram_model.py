"""Tests for the bigram language model and for the generated assessment results."""

import json
import math
from pathlib import Path

import pytest

from course_assessment.language_model.bigram_model import (
    END_TOKEN,
    START_TOKEN,
    UNKNOWN_TOKEN,
    BigramLanguageModel,
    format_sentence_report,
)

LECTURE_SENTENCES = [
    ["i", "love", "machine", "learning"],
    ["i", "love", "natural", "language", "processing"],
    ["i", "study", "machine", "learning"],
    ["machine", "learning", "is", "interesting"],
    ["natural", "language", "processing", "is", "useful"],
]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def lecture_model():
    return BigramLanguageModel(use_unknown_token=False).fit(LECTURE_SENTENCES)


def test_unigram_and_context_counts(lecture_model):
    assert lecture_model.word_counts["machine"] == 3
    assert lecture_model.context_counts["i"] == 3
    assert lecture_model.context_counts[START_TOKEN] == 5
    assert lecture_model.context_counts[END_TOKEN] == 0


def test_bigram_counts_and_sentence_boundaries(lecture_model):
    assert lecture_model.bigram_counts[(START_TOKEN, "i")] == 3
    assert lecture_model.bigram_counts[("i", "love")] == 2
    assert lecture_model.bigram_counts[("learning", END_TOKEN)] == 2
    assert lecture_model.bigram_counts[("love", "study")] == 0


def test_vocabulary_size_matches_lecture(lecture_model):
    assert lecture_model.vocabulary_size == 12


def test_laplace_smoothing_formula(lecture_model):
    step = lecture_model.conditional_probability(START_TOKEN, "i")
    assert (step["bigram_count"], step["previous_word_count"]) == (3, 5)
    assert step["numerator"] == 4 and step["denominator"] == 17
    assert step["probability"] == pytest.approx(4 / 17)
    unseen = lecture_model.conditional_probability("love", "study")
    assert unseen["probability"] == pytest.approx(1 / (2 + 12))


@pytest.mark.parametrize(
    "sentence_index,expected", [(0, 0.0003585434), (1, 0.0000440993), (2, 0.0002574158)])
def test_reproduces_lecture_sentence_probabilities(lecture_model, sentence_index, expected):
    result = lecture_model.sentence_probability(LECTURE_SENTENCES[sentence_index])
    assert result["probability"] == pytest.approx(expected, rel=1e-6)


def test_probability_is_product_of_conditionals_and_log_sum(lecture_model):
    result = lecture_model.sentence_probability(LECTURE_SENTENCES[0])
    product = math.prod(step["probability"] for step in result["steps"])
    assert result["probability"] == pytest.approx(product, rel=1e-12)
    assert result["probability_from_log"] == pytest.approx(product, rel=1e-9)
    assert result["log_probability"] == pytest.approx(math.log(product))
    assert len(result["steps"]) == len(LECTURE_SENTENCES[0]) + 1


def test_probabilities_between_zero_and_one(lecture_model):
    for sentence in [*LECTURE_SENTENCES, ["love", "love"], []]:
        result = lecture_model.sentence_probability(sentence)
        assert 0 < result["probability"] < 1
        assert all(0 < s["probability"] < 1 for s in result["steps"])


def test_conditional_distribution_sums_to_one():
    model = BigramLanguageModel().fit(LECTURE_SENTENCES)
    candidates = model.vocabulary
    for context in (START_TOKEN, "machine", "learning", "is"):
        total = sum(model.conditional_probability(context, w)["probability"] for w in candidates)
        assert total == pytest.approx(1.0)


def test_unseen_word_handling():
    model = BigramLanguageModel().fit(LECTURE_SENTENCES)
    assert model.vocabulary_size == 13 and UNKNOWN_TOKEN in model.vocabulary
    result = model.sentence_probability(["i", "zzzz"])
    assert result["unseen_words_replaced"] == ["zzzz"]
    assert result["padded_sentence"] == [START_TOKEN, "i", UNKNOWN_TOKEN, END_TOKEN]
    assert result["steps"][-1]["probability"] == pytest.approx(1 / 13)


def test_empty_sentence_is_start_followed_by_end(lecture_model):
    result = lecture_model.sentence_probability([])
    assert result["padded_sentence"] == [START_TOKEN, END_TOKEN]
    assert result["probability"] == pytest.approx(1 / (5 + 12))


def test_empty_training_data_rejected_and_empty_sentences_skipped():
    with pytest.raises(ValueError):
        BigramLanguageModel().fit([[], []])
    assert BigramLanguageModel().fit([[], ["a"]]).number_of_training_sentences == 1


def test_invalid_input_rejected(lecture_model):
    with pytest.raises(TypeError):
        lecture_model.sentence_probability("not a list")


def test_reproducible(lecture_model):
    again = BigramLanguageModel(use_unknown_token=False).fit(LECTURE_SENTENCES)
    first = lecture_model.sentence_probability(LECTURE_SENTENCES[1])
    assert first == again.sentence_probability(LECTURE_SENTENCES[1])


def test_report_text_contains_every_displayed_number(lecture_model):
    result = lecture_model.sentence_probability(LECTURE_SENTENCES[0])
    text = format_sentence_report(result)
    for step in result["steps"]:
        assert f"{step['numerator']}/{step['denominator']}" in text
        assert f"{step['probability']:.6f}" in text
    assert f"{result['probability']:.10e}" in text


RESULT_FILE = ROOT / "outputs" / "sentence_probability_results.json"


def _dataset_available() -> bool:
    return (ROOT.parent / "data" / "raw" / "SMSSpamCollection").exists()


@pytest.mark.skipif(not (_dataset_available() and RESULT_FILE.exists()), reason="dataset/results missing")
def test_selected_sentences_are_actual_dataset_records_and_results_match_a_fresh_run():
    from course_assessment.examples.selected_sentences import load_sms_split
    from course_assessment.run_assessment import run_language_model

    train, heldout, info = load_sms_split()
    fresh = run_language_model(train, heldout, info["seed"], info)
    saved = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    assert fresh["model"] == saved["model"]
    assert len(saved["sentences"]) == 3
    heldout_by_row = {r["row_id"]: r for r in heldout}
    train_texts = {r["text"] for r in train}
    for fresh_item, saved_item in zip(fresh["sentences"], saved["sentences"], strict=True):
        record = heldout_by_row[saved_item["row_id"]]
        assert record["text"] == saved_item["original_message"]
        assert record["label"] == saved_item["label"]
        assert saved_item["sentence"] in saved_item["original_message"].lower()
        assert saved_item["original_message"] not in train_texts
        assert fresh_item["calculation"]["probability"] == saved_item["calculation"]["probability"]
        steps = saved_item["calculation"]["steps"]
        assert all(0 < s["probability"] < 1 for s in steps)
        assert math.prod(s["probability"] for s in steps) == pytest.approx(
            saved_item["calculation"]["probability"], rel=1e-9)


@pytest.mark.skipif(not RESULT_FILE.exists(), reason="results missing")
def test_documentation_contains_the_executed_probabilities():
    doc = ROOT / "ACADEMIC_EVALUATION.md"
    if not doc.exists():
        pytest.skip("documentation not written yet")
    text = doc.read_text(encoding="utf-8")
    saved = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    assert f"{saved['model']['vocabulary_size_V']}" in text
    for item in saved["sentences"]:
        assert f"{item['calculation']['probability']:.10e}" in text
