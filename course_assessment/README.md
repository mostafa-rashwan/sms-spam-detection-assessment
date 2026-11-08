# course_assessment — NLP practical assessment

Self-contained module (does **not** touch the spam classifier) implementing:

* **Part A** – text preprocessing of the SMS Spam Collection: lowercasing, sentence segmentation, word tokenization,
  punctuation removal, number removal, stop-word removal, Porter stemming (`preprocessing/text_preprocessor.py`).
* **Part B** – a Laplace-smoothed **bigram language model** with `<s>` / `</s>` and the Markov assumption, scoring three
  actual held-out sentences (`language_model/bigram_model.py`).

Study material: [`ACADEMIC_EVALUATION.md`](ACADEMIC_EVALUATION.md). PDF-generation prompt: [`CLAUDE_PDF_PROMPT.md`](CLAUDE_PDF_PROMPT.md).

## Run (Windows PowerShell, from the repository root)

```powershell
.\.venv\Scripts\Activate.ps1                         # reuse the project's virtual environment
pip install -r course_assessment\requirements.txt    # adds nltk (rest is already installed)
python -c "import nltk; [nltk.download(r) for r in ('punkt_tab','stopwords','wordnet')]"
python -m spamdetect.data                            # only if data\raw\SMSSpamCollection is missing
python -m course_assessment.run_assessment
python -m pytest course_assessment\tests
```

`wordnet` is optional (only for the stemming-vs-lemmatization illustration).

Outputs are written to `course_assessment/outputs/` (`preprocessing_examples.json`, `sentence_probability_results.json`).
The raw dataset is not committed. Everything is deterministic (seed 42 from the project's `config.yaml`).
