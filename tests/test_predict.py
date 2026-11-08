import json
import warnings
from pathlib import Path

import pytest

from spamdetect.predict import SpamClassifier


def _clf(cfg):
    return SpamClassifier.load(cfg["paths"]["models_dir"])


def test_schema_and_obvious_examples(trained):
    clf = _clf(trained[0])
    spam = clf.predict("WINNER!! Claim your £1000 prize, call 09061701461 now")
    ham = clf.predict("Ok see you at the station, bring the umbrella")
    assert set(spam) == {"label", "spam_probability", "threshold"}
    assert spam["label"] == "spam" and ham["label"] == "ham"
    assert 0 <= ham["spam_probability"] < spam["spam_probability"] <= 1
    assert len(clf.predict_many(["hello there", "free prize now"])) == 2


@pytest.mark.parametrize("bad", ["", "   ", None, 42, "x" * 5000])
def test_invalid_inputs_raise(trained, bad):
    with pytest.raises(ValueError):
        _clf(trained[0]).predict(bad)


def test_missing_model_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train first"):
        SpamClassifier.load(tmp_path)


def test_version_mismatch_warns(trained):
    models = Path(trained[0]["paths"]["models_dir"])
    meta = json.loads((models / "metadata.json").read_text())
    meta["sklearn_version"] = "0.0.1"
    (models / "metadata.json").write_text(json.dumps(meta))
    with pytest.warns(RuntimeWarning, match="scikit-learn"):
        SpamClassifier.load(models)


def test_no_warning_when_versions_match(trained):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _clf(trained[0])


def test_explain_returns_ngrams(trained):
    clf = _clf(trained[0])
    top = clf.explain("WINNER!! Claim your £1000 prize now", top_k=3)
    assert 0 < len(top) <= 3 and all(isinstance(n, str) for n, _ in top)
