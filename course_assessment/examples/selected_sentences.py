"""Load the SMS dataset without pandas and select demonstration messages and sentences."""

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

MIN_SENTENCE_TOKENS = 4
MAX_SENTENCE_TOKENS = 6
SENTENCE_LABEL_PLAN = ("ham", "ham", "spam")

Record = dict


def read_raw_records(path: str | Path) -> list[tuple[str, str]]:
    """Read the tab-separated corpus as (label, text) pairs, raising ValueError on malformed rows."""
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    pairs = []
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
    """Strip texts, drop empty ones, reject label conflicts and drop exact duplicate texts."""
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
    """Split records into a stratified, reproducible training and held-out set."""
    train_ids, test_ids = train_test_split(
        list(range(len(records))), test_size=test_size,
        stratify=[r["label"] == "spam" for r in records], random_state=seed)
    return [records[i] for i in sorted(train_ids)], [records[i] for i in sorted(test_ids)]


def load_sms_split(config_path: str | Path = CONFIG_PATH) -> tuple[list[Record], list[Record], dict]:
    """Load the SMS Spam Collection and return (train_records, heldout_records, info)."""
    from spamdetect.config import load_config

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
    """Return a deterministic shuffled copy of the records."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def select_demo_messages(records: list[Record], seed: int) -> list[Record]:
    """Pick three actual messages for the preprocessing demonstration."""
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
    """Return the first suitable sentence with the wanted label, or None."""
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
    """Pick three actual held-out sentences (ham, ham, spam), each from a different message."""
    ordered = _shuffled(heldout, seed)
    selected: list[dict] = []
    for wanted_label in SENTENCE_LABEL_PLAN:
        found = _find_sentence(ordered, wanted_label, {s["row_id"] for s in selected})
        if found is None:
            raise ValueError(f"No suitable held-out {wanted_label} sentence found")
        selected.append(found)
    return selected
