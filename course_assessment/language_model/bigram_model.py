"""Bigram language model with Laplace smoothing, boundary markers and an <UNK> extension."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import nltk

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<UNK>"


class BigramLanguageModel:
    """Bigram language model trained on a list of tokenized sentences."""

    def __init__(self, use_unknown_token: bool = True) -> None:
        """Create an untrained model (``use_unknown_token=False`` reproduces the lecture exactly)."""
        self.use_unknown_token = use_unknown_token
        self.word_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.vocabulary: set[str] = set()
        self.number_of_training_sentences = 0

    def fit(self, training_sentences: list[list[str]]) -> BigramLanguageModel:
        """Count words and bigrams of the non-empty training sentences and build the vocabulary."""
        self.word_counts = Counter()
        self.context_counts = Counter()
        self.bigram_counts = Counter()
        self.number_of_training_sentences = 0
        for tokens in training_sentences:
            if not tokens:
                continue
            padded = [START_TOKEN, *tokens, END_TOKEN]
            self.word_counts.update(tokens)
            self.context_counts.update(padded[:-1])
            self.bigram_counts.update(nltk.bigrams(padded))
            self.number_of_training_sentences += 1
        if self.number_of_training_sentences == 0:
            raise ValueError("Cannot train on an empty set of sentences")
        self.vocabulary = set(self.word_counts) | {END_TOKEN}
        if self.use_unknown_token:
            self.vocabulary.add(UNKNOWN_TOKEN)
        return self

    @property
    def vocabulary_size(self) -> int:
        """Return V, the number of distinct training words plus the end and unknown markers."""
        return len(self.vocabulary)

    def replace_unseen_words(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        """Replace words outside the vocabulary by <UNK> and return (mapped tokens, replaced words)."""
        if not self.use_unknown_token:
            return list(tokens), []
        mapped, replaced = [], []
        for token in tokens:
            if token in self.vocabulary:
                mapped.append(token)
            else:
                mapped.append(UNKNOWN_TOKEN)
                replaced.append(token)
        return mapped, replaced

    def conditional_probability(self, previous_word: str, current_word: str) -> dict[str, Any]:
        """Return the Laplace-smoothed P(current_word | previous_word) with all its ingredients."""
        bigram_count = self.bigram_counts[(previous_word, current_word)]
        previous_word_count = self.context_counts[previous_word]
        numerator = bigram_count + 1
        denominator = previous_word_count + self.vocabulary_size
        return {
            "previous_word": previous_word,
            "current_word": current_word,
            "bigram_count": bigram_count,
            "previous_word_count": previous_word_count,
            "numerator": numerator,
            "denominator": denominator,
            "vocabulary_size": self.vocabulary_size,
            "probability": numerator / denominator,
        }

    def sentence_probability(self, tokens: list[str]) -> dict[str, Any]:
        """Return the probability, log probability and every conditional step of one sentence."""
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise TypeError("tokens must be a list of strings")
        mapped, replaced = self.replace_unseen_words(tokens)
        padded = [START_TOKEN, *mapped, END_TOKEN]

        steps = []
        running_product = 1.0
        log_probability = 0.0
        for previous_word, current_word in nltk.bigrams(padded):
            step = self.conditional_probability(previous_word, current_word)
            running_product *= step["probability"]
            log_probability += math.log(step["probability"])
            step["running_product"] = running_product
            steps.append(step)
        return {
            "tokens": tokens,
            "tokens_after_unk_replacement": mapped,
            "unseen_words_replaced": replaced,
            "padded_sentence": padded,
            "steps": steps,
            "probability": running_product,
            "log_probability": log_probability,
            "probability_from_log": math.exp(log_probability),
        }


def extract_ngrams(padded_sentence: list[str]) -> dict[str, list[str]]:
    """Return the unigrams, bigrams and trigrams of a padded sentence as strings."""
    return {
        "unigrams": list(padded_sentence),
        "bigrams": [" ".join(pair) for pair in nltk.bigrams(padded_sentence)],
        "trigrams": [" ".join(triple) for triple in nltk.trigrams(padded_sentence)],
    }


def format_sentence_report(result: dict[str, Any]) -> str:
    """Format the step-by-step calculation of one sentence as console text."""
    lines = []
    for step in result["steps"]:
        lines.append(
            f"P({step['current_word']} | {step['previous_word']}) = "
            f"({step['bigram_count']} + 1) / ({step['previous_word_count']} + "
            f"{step['vocabulary_size']}) = {step['numerator']}/{step['denominator']} = "
            f"{step['probability']:.6f}   running product = {step['running_product']:.10e}"
        )
    lines.append(f"P(Sentence) = {result['probability']:.10e}")
    lines.append(f"ln P(Sentence) = {result['log_probability']:.6f}")
    return "\n".join(lines)
