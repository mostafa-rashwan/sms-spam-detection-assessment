import json
import time
from pathlib import Path

from spamdetect import data, train


def test_end_to_end_smoke(tiny_cfg, raw_df):
    df, _ = data.clean(raw_df)
    tr, te = data.split(df, tiny_cfg["test_size"], tiny_cfg["seed"])
    t0 = time.time()
    m = train.run_training(tiny_cfg, tr.reset_index(drop=True), te.reset_index(drop=True), "abc")
    assert time.time() - t0 < 30
    models, reports = Path(tiny_cfg["paths"]["models_dir"]), Path(tiny_cfg["paths"]["reports_dir"])
    assert (models / "spam_pipeline.joblib").exists()
    meta = json.loads((models / "metadata.json").read_text())
    assert meta["threshold"] == m["threshold"] and meta["data_checksum_sha256"] == "abc"
    assert (reports / "metrics.json").exists() and (reports / "errors.csv").exists()
    assert m["cv"]["pooled"]["ham_fpr"] <= tiny_cfg["max_ham_fpr"]
    assert 0 <= m["test"]["recall"] <= 1


def test_deterministic(tiny_cfg, raw_df):
    df, _ = data.clean(raw_df)
    tr, te = data.split(df, tiny_cfg["test_size"], tiny_cfg["seed"])
    a = train.run_training(tiny_cfg, tr.reset_index(drop=True), te.reset_index(drop=True))
    b = train.run_training(tiny_cfg, tr.reset_index(drop=True), te.reset_index(drop=True))
    assert round(a["test"]["recall"], 4) == round(b["test"]["recall"], 4)
    assert round(a["threshold"], 4) == round(b["threshold"], 4)
