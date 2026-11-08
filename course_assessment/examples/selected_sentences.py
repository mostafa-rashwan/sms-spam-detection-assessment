"""Load the project's SMS dataset and select demonstration messages / sentences.

Dataset handling mirrors ``spamdetect.data`` (tab-separated parse without quote handling,
strip, drop empty texts, reject label conflicts, drop exact duplicate texts, stratified split
with ``sklearn.model_selection.train_test_split`` using the seed and test_size of the project's
``config.yaml``), so the training / held-out split is the one used by the spam classifier.

It is re-implemented here WITHOUT pandas on purpose: the assessment module stays independent
from the production code and runs even where pandas cannot be imported. Nothing is written to
disk and nothing in the production pipeline is modified.

A "record" is a plain dict: ``{"row_id": int, "label": "ham" | "spam", "text": str}``.
``row_id`` is the position in the cleaned (de-duplicated) dataset, identical to the row id
used by ``spamdetect.data``.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

from course_assessment.preprocessing.text_preprocessor import (
    lowercase_text,
    normalize_sentence,
    segment_sentences,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Rule used to pick the three held-out sentences (documented in ACADEMIC_EVALUATION.md).
MIN_SENTENCE_TOKENS = 4
MAX_SENTENCE_TOKENS = 6
SENTENCE_LABEL_PLAN = ("ham", "ham", "spam")

Record = dict


def read_raw_records(path: str | Path) -> list[tuple[str, str]]:
    """Read the tab-separated corpus as ``(label, text)`` pairs (UTF-8, latin-1 fallback).

    Raises:
        ValueError: for a row that does not have exactly two fields or has an unknown label.
    """
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    pairs = []
    # QUOTE_NONE: quotation marks inside SMS text are ordinary characters.
    for line_number, row in enumerate(csv.reader(content.splitlines(), delimiter="\t",
                                                 quoting=csv.QUOTE_NONE), start=1):
        if len(row) != 2:
            raise ValueError(f"Line {line_number}: expected 2 tab-separated fields, found {len(row)}")
        label, text = row
        if label not in ("ham", "spam"):
            raise ValueError(f"Line {line_number}: unexpected label {label!r}")
        pairs.append((label, text))
    return pairs


def clean_records(pairs: list[tuple[str, str]]) -> tuple[list[Record], dict]:
    """Strip texts, drop empty ones, reject label conflicts, drop exact duplicate texts."""
    stripped = [(label, text.strip()) for label, text in pairs]
    non_empty = [(label, text) for label, text in stripped if text]
    labels_by_text: dict[str, set[str]] = {}
    for label, text in non_empty:
        labels_by_text.setdefault(text, set()).add(label)
    conflicts = [text for text, labels in labels_by_text.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(f"{len(conflicts)} texts appear with conflicting labels")
    seen: set[str] = set()
    records: list[Record] = []
    for label, text in non_empty:
        if text in seen:
            continue
        seen.add(text)
        records.append({"row_id": len(records), "label": label, "text": text})
    stats = {
        "rows_in": len(pairs),
        "empty_dropped": len(stripped) - len(non_empty),
        "duplicates_dropped": len(non_empty) - len(records),
        "rows_out": len(records),
        "spam": sum(r["label"] == "spam" for r in records),
        "ham": sum(r["label"] == "ham" for r in records),
    }
    return records, stats


def split_records(records: list[Record], test_size: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Stratified, reproducible split (same call as the production code: indices + stratify)."""
    train_ids, test_ids = train_test_split(
        list(range(len(records))), test_size=test_size,
        stratify=[r["label"] == "spam" for r in records], random_state=seed)
    return [records[i] for i in sorted(train_ids)], [records[i] for i in sorted(test_ids)]


def load_sms_split(config_path: str | Path = CONFIG_PATH) -> tuple[list[Record], list[Record], dict]:
    """Load the SMS Spam Collection and return ``(train_records, heldout_records, info)``.

    Raises:
        FileNotFoundError: with download instructions if the raw dataset is missing.
    """
    from spamdetect.config import load_config  # existing project configuration (seed, paths)

    cfg = load_config(config_path)
    raw_file = Path(cfg["paths"]["raw_dir"]) / cfg["dataset"]["member"]
    if not raw_file.exists():
        raise FileNotFoundError(
            f"SMS dataset not found at {raw_file}. Download it with:\n"
            "    python -m spamdetect.data\n"
            "or place the UCI 'SMSSpamCollection' file in data/raw/."
        )
    pairs = read_raw_records(raw_file)
    records, cleaning_stats = clean_records(pairs)
    train, heldout = split_records(records, cfg["test_size"], cfg["seed"])
    info = {
        "seed": cfg["seed"],
        "test_size": cfg["test_size"],
        "parsed_rows": len(pairs),
        "cleaning": cleaning_stats,
        "train_messages": len(train),
        "heldout_messages": len(heldout),
    }
    return train, heldout, info


def _shuffled(records: list[Record], seed: int) -> list[Record]:
    """Deterministic shuffled copy (seeded ``random.Random``)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def select_demo_messages(records: list[Record], seed: int) -> list[Record]:
    """Pick three ACTUAL messages for the preprocessing demonstration (deterministic).

    Rules, tried over a seeded shuffle of the records:
      1. a ham message with exactly one sentence and 8-15 words;
      2. a spam message with at least two sentences and at least one digit;
      3. a ham message with at least two sentences.
    """
    rules = [
        ("ham, one sentence, 8-15 words",
         lambda label, text, n: label == "ham" and n == 1 and 8 <= len(text.split()) <= 15),
        ("spam, two or more sentences, contains a digit",
         lambda label, text, n: label == "spam" and n >= 2 and any(c.isdigit() for c in text)),
        ("ham, two or more sentences",
         lambda label, text, n: label == "ham" and n >= 2),
    ]
    ordered = _shuffled(records, seed)
    selected: list[Record] = []
    for rule_name, predicate in rules:
        for record in ordered:
            sentence_count = len(segment_sentences(lowercase_text(record["text"])))
            if record not in [s["record"] for s in selected] and predicate(
                    record["label"], record["text"], sentence_count):
                selected.append({"record": record, "selection_rule": rule_name})
                break
    return [{**s["record"], "selection_rule": s["selection_rule"]} for s in selected]


def _find_sentence(ordered: list[Record], wanted_label: str, used_rows: set[int]) -> dict | None:
    """First suitable sentence (in the given message order) with the wanted label, or None."""
    for record in ordered:
        if record["row_id"] in used_rows or record["label"] != wanted_label:
            continue
        for sentence in segment_sentences(lowercase_text(record["text"])):
            tokens = normalize_sentence(sentence)
            if MIN_SENTENCE_TOKENS <= len(tokens) <= MAX_SENTENCE_TOKENS:
                return {"row_id": record["row_id"], "label": wanted_label,
                        "original_message": record["text"], "sentence": sentence,
                        "normalized_tokens": tokens}
    return None


def select_heldout_sentences(heldout: list[Record], seed: int) -> list[dict]:
    """Pick THREE actual sentences (labels ham, ham, spam) from the held-out messages.

    A candidate is a sentence of a held-out message whose normalized form has between
    ``MIN_SENTENCE_TOKENS`` and ``MAX_SENTENCE_TOKENS`` tokens. Messages are visited in a
    seeded shuffle, so the choice is deterministic. Each sentence comes from a different message.

    Raises:
        ValueError: if the held-out data does not contain enough suitable sentences.
    """
    ordered = _shuffled(heldout, seed)
    selected: list[dict] = []
    for wanted_label in SENTENCE_LABEL_PLAN:
        found = _find_sentence(ordered, wanted_label, {s["row_id"] for s in selected})
        if found is None:
            raise ValueError(f"No suitable held-out {wanted_label} sentence found")
        selected.append(found)
    return selected
