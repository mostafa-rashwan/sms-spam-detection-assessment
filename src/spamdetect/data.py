"""Dataset acquisition, parsing, validation, de-duplication and splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

LABEL_MAP = {"ham": 0, "spam": 1}


def sha256_of(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: str | Path, url: str, member: str = "SMSSpamCollection") -> Path:
    """Download the UCI zip, extract ``member`` into ``raw_dir`` and write its SHA-256 to CHECKSUM."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / member
    zip_path = raw_dir / "sms_spam_collection.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "spamdetect/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as out:
            out.write(resp.read())
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the dataset ({exc}). Download {url} manually and place the "
            f"'{member}' file in {raw_dir}."
        ) from exc
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, raw_dir)
    digest = sha256_of(target)
    (raw_dir / "CHECKSUM").write_text(f"{digest}  {member}\n", encoding="utf-8")
    log.info("Downloaded %s (sha256=%s)", target, digest)
    return target


def parse(path: str | Path) -> pd.DataFrame:
    """Parse the tab-separated corpus into a DataFrame with string ``label`` and ``text``."""
    kwargs: dict[str, Any] = dict(
        sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE, dtype=str
    )
    try:
        df = pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        log.warning("UTF-8 decoding failed for %s; retrying with latin-1", path)
        df = pd.read_csv(path, encoding="latin-1", **kwargs)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed corpus file (expected 2 tab-separated columns): {exc}") from exc
    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns, found {df.shape[1]}")
    bad = set(df["label"].dropna().unique()) - set(LABEL_MAP)
    if bad or df["label"].isna().any():
        raise ValueError(f"Unexpected labels: {sorted(map(str, bad)) or 'missing'}")
    log.info("Parsed %d rows from %s", len(df), path)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip, drop empties, map labels, reject label conflicts, drop exact duplicate texts."""
    stats: dict[str, int] = {"rows_in": len(df)}
    out = df.copy()
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out = out[out["text"] != ""]
    stats["empty_dropped"] = stats["rows_in"] - len(out)
    out["label"] = out["label"].map(LABEL_MAP)
    if out["label"].isna().any():
        raise ValueError("Unmapped labels after cleaning")
    out["label"] = out["label"].astype(int)
    conflicts = out.groupby("text")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{int((conflicts > 1).sum())} texts appear with conflicting labels")
    before = len(out)
    out = out.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    stats["duplicates_dropped"] = before - len(out)
    stats["rows_out"] = len(out)
    stats["spam"] = int(out["label"].sum())
    stats["ham"] = int((out["label"] == 0).sum())
    log.info("Cleaning: %s", stats)
    return out, stats


def split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified, reproducible train/test split. Indices refer to the cleaned frame."""
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(), test_size=test_size, stratify=df["label"], random_state=seed
    )
    return df.loc[sorted(train_idx)].copy(), df.loc[sorted(test_idx)].copy()


def prepare_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse + clean + split the raw corpus, write indices and manifest, return (train, test, manifest)."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    member = cfg["dataset"]["member"]
    path = raw_dir / member
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m spamdetect.data` to download it, or place the "
            "file there manually."
        )
    raw = parse(path)
    expected = cfg["dataset"].get("expected_rows")
    if expected and len(raw) != expected:
        log.warning("Parsed %d rows but UCI documents %d", len(raw), expected)
    df, stats = clean(raw)
    train, test = split(df, cfg["test_size"], cfg["seed"])
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc / "clean.csv", index_label="row_id")
    pd.Series(train.index, name="row_id").to_csv(proc / "train_idx.csv", index=False)
    pd.Series(test.index, name="row_id").to_csv(proc / "test_idx.csv", index=False)
    manifest = {
        "checksum_sha256": sha256_of(path),
        "seed": cfg["seed"],
        "test_size": cfg["test_size"],
        "parsed_rows": len(raw),
        "cleaning": stats,
        "train": {"n": len(train), "spam": int(train["label"].sum()),
                  "ham": int((train["label"] == 0).sum())},
        "test": {"n": len(test), "spam": int(test["label"].sum()),
                 "ham": int((test["label"] == 0).sum())},
    }
    (proc / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train.reset_index(drop=True), test.reset_index(drop=True), manifest


def main() -> None:
    """CLI: download the dataset and build the split."""
    from .config import load_config
    from .logging_utils import setup_logging

    setup_logging()
    cfg = load_config("config.yaml")
    download(cfg["paths"]["raw_dir"], cfg["dataset"]["url"], cfg["dataset"]["member"])
    _, _, manifest = prepare_data(cfg)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
