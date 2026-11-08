"""Bigram language model with the Markov assumption and Laplace smoothing (Section 5).

Lecture formula (Laplace / add-one smoothing):

    P(w_i | w_{i-1}) = (C(w_{i-1}, w_i) + 1) / (C(w_{i-1}) + V)

    P(S) = product over i of P(w_i | w_{i-1})

Counting strategy (identical to the lecture's ``calculate_probability``):
  * every sentence is padded:  ['<s>'] + tokens + ['</s>']
  * context_counts   C(w_{i-1})      <- Counter.update(padded[:-1])  (all words except </s>)
  * bigram_counts    C(w_{i-1}, w_i) <- Counter.update(nltk.bigrams(padded))
  * vocabulary V = distinct training words + '</s>' (the lecture's definition; '<s>' is not
    counted because it is never predicted) + '<UNK>' (EXTENSION, see below).

EXTENSION (not in the lecture): unseen words. The lecture builds and evaluates on the same
sentences, so it never meets an unseen word. When the evaluated sentence is held out, a word
may be missing from the training vocabulary. We replace it by '<UNK>' and count '<UNK>' once in V.
'<UNK>' is never seen in training, so C(<UNK>) = 0 and its probabilities are 1 / V.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import nltk

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<UNK>"


class BigramLanguageModel:
    """Bigram language model trained on a list of tokenized sentences.

    Typical use::

        model = BigramLanguageModel().fit(training_sentences)
        result = model.sentence_probability(["free", "entri"])
    """

    def __init__(self, use_unknown_token: bool = True) -> None:
        """Create an untrained model.

        Parameters:
            use_unknown_token: True (default) adds the '<UNK>' extension for unseen words.
                False reproduces the lecture exactly (V = distinct words + '</s>').
        """
        self.use_unknown_token = use_unknown_token
        self.word_counts: Counter = Counter()      # how often each word occurs (unigram counts)
        self.context_counts: Counter = Counter()   # C(w_{i-1}): word as the FIRST item of a bigram
        self.bigram_counts: Counter = Counter()    # C(w_{i-1}, w_i)
        self.vocabulary: set[str] = set()
        self.number_of_training_sentences = 0

    # ------------------------------------------------------------------ training
    def fit(self, training_sentences: list[list[str]]) -> BigramLanguageModel:
        """Count words and bigrams in the training sentences and build the vocabulary.

        Parameters:
            training_sentences: list of sentences, each a list of string tokens.
                Empty sentences (nothing left after normalization) are skipped because
                they would only add a meaningless '<s> </s>' pair.

        Returns:
            ``self`` so calls can be chained.
        """
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
        """V: number of distinct training words plus '</s>' and '<UNK>'."""
        return len(self.vocabulary)

    # ------------------------------------------------------------------ inference
    def replace_unseen_words(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        """Replace every word outside the training vocabulary by '<UNK>'.

        Returns:
            (mapped_tokens, list_of_replaced_original_words)
        """
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
        """Laplace-smoothed P(current_word | previous_word) with all its ingredients.

        Returns a dict with ``bigram_count``, ``previous_word_count``, ``numerator``,
        ``denominator``, ``vocabulary_size`` and ``probability``.
        """
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
        """Probability of one sentence under the model (Markov assumption).

        Parameters:
            tokens: normalized tokens WITHOUT boundary markers. An empty list is allowed
                and gives P(</s> | <s>), the probability of an empty sentence.

        Returns:
            dict with the padded sentence, one entry per conditional probability
            (``steps``, each with its running product), the final ``probability``, and
            ``log_probability`` (natural log) computed as a sum of logs.
        """
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
    """Unigrams, bigrams and trigrams of a padded sentence (``nltk.bigrams`` / ``nltk.trigrams``).

    Bigrams and trigrams are returned as space-joined strings, as in the lecture's ``gram()``.
    Trigrams are only demonstrated here; the probability model itself is a BIGRAM model.
    """
    return {
        "unigrams": list(padded_sentence),
        "bigrams": [" ".join(pair) for pair in nltk.bigrams(padded_sentence)],
        "trigrams": [" ".join(triple) for triple in nltk.trigrams(padded_sentence)],
    }


def format_sentence_report(result: dict[str, Any]) -> str:
    """Human-readable calculation of one sentence (used for console output)."""
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
