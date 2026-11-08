"""Academic text-normalization pipeline (Section 4 of the course).

IMPORTANT: this module is deliberately separate from ``spamdetect.preprocessing``.
The production spam classifier KEEPS numbers, currency, URLs and phone numbers (as
placeholder tokens) because they are strong spam signals. The university assessment
asks for the opposite: remove punctuation and numbers, remove stop words and stem.
Nothing here is imported by, or changes, the spam classifier.

Pipeline order (one SMS record in, a list of normalized sentences out):

    original message
      -> 1. lowercase_text()            lowercasing
      -> 2. segment_sentences()         sentence segmentation  (nltk.sent_tokenize)
      -> 3. tokenize_words()            word tokenization      (nltk.word_tokenize)
      -> 4. remove_punctuation()        punctuation removal    (string.punctuation)
      -> 5. remove_numbers()            number removal         (any digit in token)
      -> 6. remove_stop_words()         stop word removal      (nltk stopwords)
      -> 7. stem_tokens()               Porter stemming        (nltk PorterStemmer)
"""

from __future__ import annotations

import string
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# (resource name, path used by nltk.data.find) for everything the pipeline needs.
REQUIRED_NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}

# The lecture removes tokens found in string.punctuation. word_tokenize() also produces
# tokens such as "..." or "``", and SMS text contains currency/typographic symbols that
# are not in string.punctuation. This extra set is a small, clearly labelled EXTENSION.
EXTRA_SYMBOLS = "£€¥₹‘’“”–—…"
PUNCTUATION_CHARACTERS = set(string.punctuation) | set(EXTRA_SYMBOLS)

# Documentation of each stage: used by the trace so the explanation and the code agree.
STAGE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "lowercasing": {
        "function": "lowercase_text",
        "why": "Makes 'Free', 'FREE' and 'free' the same word so counts are not split.",
    },
    "sentence_segmentation": {
        "function": "segment_sentences",
        "why": "Splits a message into sentences; the language model works sentence by sentence.",
    },
    "word_tokenization": {
        "function": "tokenize_words",
        "why": "Splits a sentence into word and punctuation tokens that can be filtered/counted.",
    },
    "punctuation_removal": {
        "function": "remove_punctuation",
        "why": "Punctuation marks carry little lexical meaning for a word-level model.",
    },
    "number_removal": {
        "function": "remove_numbers",
        "why": "Numbers (prices, phone numbers, short codes) are mostly unique and sparse tokens.",
    },
    "stop_word_removal": {
        "function": "remove_stop_words",
        "why": "Very frequent function words ('the', 'to', 'you') add little content information.",
    },
    "porter_stemming": {
        "function": "stem_tokens",
        "why": "Reduces inflected forms ('winning', 'wins') to a common stem to reduce sparsity.",
    },
}


def ensure_nltk_resources(download: bool = False) -> None:
    """Check that the required NLTK resources exist, optionally downloading them.

    Parameters:
        download: if True, try ``nltk.download`` for every missing resource first.

    Raises:
        RuntimeError: with installation instructions when a resource is still missing.
    """
    missing = []
    for name, path in REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            if download:
                nltk.download(name, quiet=True)
                try:
                    nltk.data.find(path)
                    continue
                except LookupError:
                    pass
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"Missing NLTK resources: {', '.join(missing)}. Install them with:\n"
            f"    python -c \"import nltk; {'; '.join(f'nltk.download({m!r})' for m in missing)}\""
        )


def _require_text(text: Any) -> str:
    """Validate that ``text`` is a string (empty strings are allowed and handled later)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    return text


def lowercase_text(text: str) -> str:
    """Stage 1 - lowercasing. Returns ``text.lower()``."""
    return _require_text(text).lower()


def segment_sentences(text: str) -> list[str]:
    """Stage 2 - sentence segmentation with ``nltk.sent_tokenize``.

    An SMS record is one message, not necessarily a grammatical sentence, so the result
    is often a single sentence; messages with several '.', '!' or '?' split into more.
    Returns an empty list for empty / whitespace-only text.
    """
    text = _require_text(text).strip()
    if not text:
        return []
    return sent_tokenize(text)


def tokenize_words(sentence: str) -> list[str]:
    """Stage 3 - word tokenization with ``nltk.word_tokenize`` (keeps punctuation tokens)."""
    sentence = _require_text(sentence).strip()
    if not sentence:
        return []
    return word_tokenize(sentence)


def is_punctuation_token(token: str) -> bool:
    """True if every character of ``token`` is punctuation (e.g. '!', '...', '``')."""
    return len(token) > 0 and all(char in PUNCTUATION_CHARACTERS for char in token)


def remove_punctuation(tokens: list[str]) -> list[str]:
    """Stage 4 - drop tokens made only of punctuation characters."""
    return [token for token in tokens if not is_punctuation_token(token)]


def contains_digit(token: str) -> bool:
    """True if the token contains at least one digit, e.g. '2', '150p', '09061701461'."""
    return any(char.isdigit() for char in token)


def remove_numbers(tokens: list[str]) -> list[str]:
    """Stage 5 - drop every token that contains a digit (prices, phone numbers, short codes)."""
    return [token for token in tokens if not contains_digit(token)]


def get_stop_words() -> set[str]:
    """Return NLTK's English stop-word list as a set (fast membership tests)."""
    return set(stopwords.words("english"))


def remove_stop_words(tokens: list[str], stop_words: set[str] | None = None) -> list[str]:
    """Stage 6 - drop tokens that are in NLTK's English stop-word list.

    Parameters:
        tokens: already lowercased tokens.
        stop_words: optional custom set (defaults to ``get_stop_words()``).
    """
    stop_words = get_stop_words() if stop_words is None else stop_words
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens: list[str], stemmer: PorterStemmer | None = None) -> list[str]:
    """Stage 7 - apply the Porter stemmer to every token (``nltk.stem.PorterStemmer``)."""
    stemmer = PorterStemmer() if stemmer is None else stemmer
    return [stemmer.stem(token) for token in tokens]


def normalize_sentence(sentence: str) -> list[str]:
    """Run stages 3-7 on ONE (already lowercased) sentence and return the final tokens."""
    tokens = tokenize_words(sentence)
    tokens = remove_punctuation(tokens)
    tokens = remove_numbers(tokens)
    tokens = remove_stop_words(tokens)
    return stem_tokens(tokens)


def normalize_message(text: str) -> list[list[str]]:
    """Run the full pipeline on one SMS record.

    Returns:
        One list of final tokens per sentence. Sentences that become empty after
        normalization (e.g. '!!!') are kept as empty lists; callers decide what to do.
    """
    lowered = lowercase_text(text)
    return [normalize_sentence(sentence) for sentence in segment_sentences(lowered)]


def trace_message(text: str) -> dict[str, Any]:
    """Return every intermediate result of the pipeline for one message (for inspection).

    The result is generated by running the real functions above, never hand-written.
    """
    lowered = lowercase_text(text)
    sentences = segment_sentences(lowered)
    sentence_traces = []
    for sentence in sentences:
        tokens = tokenize_words(sentence)
        no_punctuation = remove_punctuation(tokens)
        no_numbers = remove_numbers(no_punctuation)
        no_stop_words = remove_stop_words(no_numbers)
        stemmed = stem_tokens(no_stop_words)
        sentence_traces.append({
            "sentence": sentence,
            "word_tokenization": tokens,
            "punctuation_removal": no_punctuation,
            "number_removal": no_numbers,
            "stop_word_removal": no_stop_words,
            "porter_stemming": stemmed,
            "removed_as_punctuation": [t for t in tokens if t not in no_punctuation],
            "removed_as_numbers": [t for t in no_punctuation if t not in no_numbers],
            "removed_as_stop_words": [t for t in no_numbers if t not in no_stop_words],
            "stem_changes": {a: b for a, b in zip(no_stop_words, stemmed, strict=True) if a != b},
            "final_tokens": stemmed,
        })
    return {
        "original_message": text,
        "lowercasing": lowered,
        "sentence_segmentation": sentences,
        "number_of_sentences": len(sentences),
        "sentence_traces": sentence_traces,
    }


def compare_stemming_and_lemmatization(tokens: list[str]) -> dict[str, Any]:
    """ILLUSTRATION ONLY (not part of the pipeline): Porter stems vs WordNet lemmas.

    The assessment requires stemming OR lemmatization; the pipeline uses stemming. This
    helper only shows the difference. Returns ``{"available": False, ...}`` when the
    WordNet data is not installed instead of failing.
    """
    stemmer = PorterStemmer()
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # triggers the resource lookup
    except LookupError:
        return {"available": False,
                "note": "WordNet not installed: python -c \"import nltk; nltk.download('wordnet')\""}
    return {
        "available": True,
        "rows": [{"token": t,
                  "porter_stem": stemmer.stem(t),
                  "lemma_as_noun": lemmatizer.lemmatize(t),
                  "lemma_as_verb": lemmatizer.lemmatize(t, pos="v")} for t in tokens],
    }
