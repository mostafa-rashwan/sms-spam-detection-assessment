# Notice

## Dataset

This project trains on the **SMS Spam Collection v.1** (UCI Machine Learning Repository, id 228).

- Almeida, T. & Gómez Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
- Licensed under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).
- The data is downloaded at run time into `data/raw/` and is **not** redistributed in this repository.

## Provenance / inspiration

The design was informed by an analysis of two public repositories that use the same corpus.
Neither carries a license, so **no code, notebooks, images or model files were copied**; only
high-level ideas (and their published pitfalls) were used and everything was re-implemented from scratch.
Both are credited as inspiration:

- SMS spam classifier repo A (TF-IDF + oversampling approach) — GitHub, author unspecified in this project's brief.
- SMS spam classifier repo B (TF-IDF + MultinomialNB with Streamlit app) — GitHub, author unspecified in this project's brief.

(The exact repository URLs were not supplied in the implementation brief; add them here.)
