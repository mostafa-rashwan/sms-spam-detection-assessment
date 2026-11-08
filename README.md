# spamdetect — SMS spam classifier

Leakage-free classifier for **SMS** (not email) spam, trained on the UCI SMS Spam Collection.
It picks the model and decision threshold that **maximise spam recall subject to a ham false-positive
rate (FPR) ≤ 0.5 %**, because blocking a legitimate message costs more than letting some spam through.

## Dataset
UCI SMS Spam Collection v.1 (id 228), Almeida & Gómez Hidalgo (2011), DOI 10.24432/C5CC84, CC BY 4.0.
Downloaded at run time; not redistributed. See `NOTICE.md`.

## Install & run
```bash
pip install -e ".[dev,app]"            # plus: pip install nltk  (baseline / stemming ablation only)
python -m spamdetect.data              # download + checksum + split
python -m spamdetect.train --config config.yaml
python -m spamdetect.predict "Free prize! Call 09061701461 now"
python -m spamdetect.predict --file messages.csv --model-dir models
streamlit run app/streamlit_app.py
pytest && ruff check .
```

## Layout
`src/spamdetect`: `data` (download/parse/dedup/split), `preprocessing` (the single `preprocess`),
`features` (TF-IDF word+char FeatureUnion, optional meta features), `models` (candidates+grids),
`train` (tuning, threshold, selection, one test evaluation), `evaluate`, `predict` (API+CLI+explain).
`app/streamlit_app.py` is a thin client over `spamdetect.predict`.

## Methodology
- Parsed 5,574 rows (matches UCI; the "5,572" seen elsewhere comes from pandas quote handling, avoided with `QUOTE_NONE`).
  Whitespace-stripped exact duplicates (414) are dropped before splitting → 5,160 unique (642 spam / 4,518 ham). No label conflicts.
- Stratified 80/20 split (seed 42): train 4,128 (514 spam), test 1,032 (128 spam).
- Preprocessing keeps digit/currency signal through placeholder tokens (url/email/phone/currency/number).
- Word 1–2-gram + char_wb 2–5-gram TF-IDF, all fitted inside the `Pipeline` on training data only.
- 5-fold stratified CV on the training split only; hyper-parameters picked by PR-AUC; out-of-fold scores give the
  threshold (max recall s.t. ham FPR ≤ cap). Near-ties (≤0.5 pp recall) go to the simpler model.
- Test split scored once, at the OOF-chosen threshold.

## Results (from `reports/metrics.json`)
Selected: calibrated LinearSVC (C=3, word min_df=1), threshold 0.3463. Ablations (stop-words, stemming, no-char, meta-features) gave no gain > 0.5 pp, so none were kept.

| | Precision | Recall | F1 | PR-AUC | Ham FPR | FP / FN |
|---|---|---|---|---|---|---|
| **Selected, CV (pooled OOF, train)** | 0.971 | 0.963 | 0.967 | 0.985 | 0.0042 | 15 / 19 |
| **Selected, test** | 0.969 | 0.977 | 0.973 | 0.996 | 0.0044 | 4 / 3 |
| Baseline (Repo-B-style NB, thr 0.5), test | 0.990 | 0.758 | 0.858 | 0.945 | 0.0011 | 1 / 31 |
| Baseline, FPR-cap threshold from OOF, test | 0.940 | 0.852 | 0.893 | 0.945 | 0.0077 | 7 / 19 |

Other candidates' pooled-OOF recall: MultinomialNB 0.946, ComplementNB 0.946, LogReg 0.951 (FPR 0.0050), RandomForest 0.928.
The baseline's stop-word list is scikit-learn's (not NLTK's) to avoid corpus downloads. Test numbers rest on only 128 spam
messages (3 misses), so expect roughly ±1–2 pp of noise. The baseline's OOF-derived threshold exceeded the cap on test (0.77 %),
a reminder that the cap holds in expectation, not per sample.

## Limitations
Dataset is old (2011) and from a UK/Singapore context; spam class is small; near-duplicates (not exact) may remain across
the split, which can flatter scores; hyper-parameter search and threshold share the same CV folds (mild optimism).
Legitimate transactional messages (e.g. bank one-time codes) can be flagged — try the example button in the app.
Over-long messages (> 1,000 chars) are rejected with `ValueError`.

## Attribution
See `NOTICE.md`.

## Course assessment (NLP practical)
A separate NLP assessment module (preprocessing + bigram language model) lives in [`course_assessment/`](course_assessment/README.md); it does not affect the classifier.
