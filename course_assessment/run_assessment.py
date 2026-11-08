"""Run the preprocessing demonstration and the bigram model and write the JSON outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nltk

from course_assessment.examples.selected_sentences import (
    load_sms_split,
    select_demo_messages,
    select_heldout_sentences,
)
from course_assessment.language_model.bigram_model import (
    END_TOKEN,
    START_TOKEN,
    BigramLanguageModel,
    extract_ngrams,
    format_sentence_report,
)
from course_assessment.preprocessing.text_preprocessor import (
    STAGE_DESCRIPTIONS,
    compare_stemming_and_lemmatization,
    ensure_nltk_resources,
    normalize_message,
    trace_message,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PREPROCESSING_OUTPUT = OUTPUT_DIR / "preprocessing_examples.json"
PROBABILITY_OUTPUT = OUTPUT_DIR / "sentence_probability_results.json"


def build_training_sentences(train_records) -> tuple[list[list[str]], dict]:
    """Normalize every training message and return the non-empty sentences with statistics."""
    sentences, empty_sentences, segmented_total = [], 0, 0
    for record in train_records:
        text = record["text"]
        for tokens in normalize_message(text):
            segmented_total += 1
            if tokens:
                sentences.append(tokens)
            else:
                empty_sentences += 1
    return sentences, {
        "training_messages": len(train_records),
        "sentences_after_segmentation": segmented_total,
        "sentences_empty_after_normalization": empty_sentences,
        "sentences_used_for_training": len(sentences),
    }


def run_preprocessing_demo(train_records, seed: int, info: dict) -> dict:
    """Trace three actual training messages and summarize the normalized training corpus."""
    traces = []
    for message in select_demo_messages(train_records, seed):
        trace = trace_message(message["text"])
        trace.update({"row_id": message["row_id"], "label": message["label"],
                      "selection_rule": message["selection_rule"]})
        traces.append(trace)

    sentences, stats = build_training_sentences(train_records)
    frequency = nltk.FreqDist(token for sentence in sentences for token in sentence)
    stemming_input = list(dict.fromkeys(
        token for trace in traces for sentence in trace["sentence_traces"]
        for token in sentence["stop_word_removal"]))
    return {
        "dataset": info,
        "note": "All examples are ACTUAL records of the SMS Spam Collection (training split).",
        "stage_descriptions": STAGE_DESCRIPTIONS,
        "demonstration_messages": traces,
        "training_corpus_statistics": {
            **stats,
            "training_tokens": frequency.N(),
            "training_word_types": frequency.B(),
            "most_common_20": frequency.most_common(20),
        },
        "stemming_vs_lemmatization_illustration": compare_stemming_and_lemmatization(stemming_input),
    }


def run_language_model(train_records, heldout_records, seed: int, info: dict) -> dict:
    """Fit the bigram model and score three held-out sentences."""
    training_sentences, stats = build_training_sentences(train_records)
    model = BigramLanguageModel().fit(training_sentences)
    results = []
    for selected in select_heldout_sentences(heldout_records, seed):
        calc = model.sentence_probability(selected["normalized_tokens"])
        results.append({**selected, "ngrams": extract_ngrams(calc["padded_sentence"]), "calculation": calc})
    return {
        "dataset": info,
        "training_statistics": stats,
        "model": {
            "vocabulary_size_V": model.vocabulary_size,
            "distinct_training_words": len(model.word_counts),
            "training_sentences": model.number_of_training_sentences,
            "count_of_start_symbol": model.context_counts[START_TOKEN],
            "number_of_distinct_bigrams": len(model.bigram_counts),
            "start_symbol": START_TOKEN,
            "end_symbol": END_TOKEN,
        },
        "sentences": results,
    }


def main() -> None:
    """Run both parts, print the report and write the JSON outputs."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        ensure_nltk_resources()
    except RuntimeError as error:
        sys.exit(f"ERROR: {error}")
    try:
        train_records, heldout_records, info = load_sms_split()
    except FileNotFoundError as error:
        sys.exit(f"ERROR: {error}")
    seed = info["seed"]

    print(f"Dataset: {info['parsed_rows']} parsed rows, {info['cleaning']['rows_out']} unique messages "
          f"-> {info['train_messages']} training / {info['heldout_messages']} held-out (seed {seed})")

    preprocessing = run_preprocessing_demo(train_records, seed, info)
    print("\n===== PART A: PREPROCESSING (actual dataset messages) =====")
    for trace in preprocessing["demonstration_messages"]:
        print(f"\n[{trace['label']}] row {trace['row_id']}: {trace['original_message']!r}")
        print(f"  lowercased        : {trace['lowercasing']!r}")
        print(f"  sentences ({trace['number_of_sentences']})     : {trace['sentence_segmentation']}")
        for s in trace["sentence_traces"]:
            for stage in ("word_tokenization", "punctuation_removal", "number_removal",
                          "stop_word_removal", "porter_stemming"):
                print(f"    {stage:<20}: {s[stage]}")
    corpus = preprocessing["training_corpus_statistics"]
    print(f"\nTraining corpus after normalization: {corpus['training_tokens']} tokens, "
          f"{corpus['training_word_types']} word types")

    language_model = run_language_model(train_records, heldout_records, seed, info)
    model_info = language_model["model"]
    print("\n===== PART B: BIGRAM LANGUAGE MODEL =====")
    print(f"Training sentences = {model_info['training_sentences']}, vocabulary size V = "
          f"{model_info['vocabulary_size_V']} ({model_info['distinct_training_words']} words + "
          f"{END_TOKEN} + <UNK>), C({START_TOKEN}) = {model_info['count_of_start_symbol']}")
    for number, item in enumerate(language_model["sentences"], start=1):
        calc = item["calculation"]
        print(f"\n--- Sentence {number} [{item['label']}] (row {item['row_id']}) ---")
        print(f"Original message : {item['original_message']!r}")
        print(f"Sentence used    : {item['sentence']!r}")
        print(f"Normalized tokens: {item['normalized_tokens']}  (unseen -> <UNK>: "
              f"{calc['unseen_words_replaced']})")
        print(f"Bigrams          : {item['ngrams']['bigrams']}")
        print(f"Trigrams         : {item['ngrams']['trigrams']}")
        print(format_sentence_report(calc))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in ((PREPROCESSING_OUTPUT, preprocessing), (PROBABILITY_OUTPUT, language_model)):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {PREPROCESSING_OUTPUT.name} and {PROBABILITY_OUTPUT.name} to course_assessment/outputs/")


if __name__ == "__main__":
    main()
