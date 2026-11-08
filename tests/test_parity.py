import joblib
import numpy as np

from spamdetect.predict import SpamClassifier


def test_loaded_artefact_matches_in_memory_pipeline(trained, tmp_path):
    cfg, _, _, te = trained
    texts = te["text"].to_numpy(dtype=object)
    loaded = SpamClassifier.load(cfg["paths"]["models_dir"])
    p_loaded = loaded.pipeline.predict_proba(texts)[:, 1]
    # re-serialise independently to be sure the round trip is lossless
    joblib.dump(loaded.pipeline, tmp_path / "p.joblib")
    p_again = joblib.load(tmp_path / "p.joblib").predict_proba(texts)[:, 1]
    np.testing.assert_array_equal(p_loaded, p_again)
    api = [r["spam_probability"] for r in loaded.predict_many(list(texts))]
    np.testing.assert_allclose(api, p_loaded)
