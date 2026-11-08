# Academic Evaluation Reference — Text Preprocessing and Bigram Language Model on the SMS Spam Collection

> Every number in this document was produced by running `python -m course_assessment.run_assessment`
> (results saved in `outputs/preprocessing_examples.json` and `outputs/sentence_probability_results.json`).
> Examples labelled **ACTUAL** are records of the SMS Spam Collection. Examples labelled **HYPOTHETICAL**
> or **LECTURE** are not from the dataset and are used only to explain an idea.

---

## 1. Assessment Overview

The assessment has two parts.

| Part | Requirement | Where implemented |
|---|---|---|
| A | Select a dataset and preprocess it: sentence segmentation, lowercasing, word tokenization, punctuation removal, number removal, stop-word removal, Porter stemming | `preprocessing/text_preprocessor.py` |
| B | Build a Bigram Language Model (Markov assumption, Laplace smoothing, `<s>` / `</s>`), and compute the probability of **three** actual sentences | `language_model/bigram_model.py` |

The orchestration is in `run_assessment.py`. The implementation follows the four lecture notebooks:
`NLP_Section_1_2` (strings, `split`/`join`, lists, sets, comprehensions, regular expressions),
`NLP_S3_2025` (corpus, `FreqDist`, vocabulary), `Text_Normalization_S4_2025` (segmentation, tokenization,
stop words, punctuation, stemming, lemmatization) and `N_Grams_S5_2025_updated` (n-grams, bigram model,
Laplace smoothing).

## 2. Project Context

The repository contains `spamdetect`, a production SMS spam classifier. That classifier **deliberately keeps**
numbers, currency, URLs and phone numbers (as placeholder tokens) because they are strong spam signals.
The university assessment requires the **opposite** (remove numbers and punctuation). Therefore
`course_assessment/` is a **separate** module:

* it never imports or modifies `spamdetect.preprocessing`, the training code or the model;
* it re-uses only the project's `config.yaml` (seed, test size, dataset path) and the same dataset file.

## 3. Dataset Selection and Justification

**Dataset:** UCI SMS Spam Collection v.1 (Almeida & Gómez Hidalgo, 2011; DOI 10.24432/C5CC84; licence CC BY 4.0).

Why it fits the topic:

1. It is the dataset of the project itself (SMS spam detection), so the NLP techniques are applied to the
   real problem, not an unrelated toy corpus.
2. Messages are short, informal, noisy text (abbreviations such as `u`, `ur`, `txt`, digits, symbols), which makes
   every preprocessing step visible and meaningful.
3. It is a labelled **corpus** (ham/spam), so the lecture's corpus concepts (vocabulary, frequency) apply directly.
4. It is small enough that every count in this document can be reproduced in seconds.

## 4. Corpus Description

**Corpus** (plural *corpora*): a large, structured set of texts used for statistical analysis of language, for checking
word occurrences, or for validating linguistic rules (Section 3 lecture). The SMS Spam Collection is a
*specific* (SMS genre), *monolingual* (English) corpus with *annotation* (the ham/spam label per message).

Numbers produced by the program (`run_assessment`):

| Quantity | Value |
|---|---|
| Rows parsed from the raw file | 5,574 |
| Unique messages after trimming and exact-duplicate removal | 5,160 (414 duplicates removed, 0 empty, 0 label conflicts) |
| Training messages / held-out messages | 4,128 / 1,032 |
| Split | stratified, `random_state = 42`, `test_size = 0.2` (same as `config.yaml`) |

Vocabulary and frequency analysis (Section 3 idea, `nltk.FreqDist`) on the **normalized training sentences**:
36,324 tokens and 5,816 distinct word types. The ten most frequent normalized tokens are
`u` (797), `call` (431), `'s` (377), `go` (353), `get` (347), `'m` (309), `n't` (286), `ur` (254), `come` (222), `gt` (219).
(`gt` and `lt` are the remains of the HTML entities `&gt;` / `&lt;` present in the raw data.)

## 5. Dataset Structure and Labels

The raw file `data/raw/SMSSpamCollection` has one message per line: `label<TAB>text`, with label `ham`
(legitimate) or `spam`. Quotation marks inside messages are ordinary characters, so the file is read with
`csv.QUOTE_NONE` (this is why exactly 5,574 rows are parsed). The loader (`examples/selected_sentences.py`)
strips whitespace, drops empty texts, raises an error if the same text has two labels, and drops exact duplicate texts.
Each record is a dictionary `{"row_id", "label", "text"}`; `row_id` is the position in the cleaned dataset.

## 6. Text Preprocessing Overview

**Text normalization** (Section 4) converts raw text to a consistent form that a statistical model can count. The pipeline:

```
Original SMS -> Lowercasing -> Sentence Segmentation -> Word Tokenization
  -> Punctuation Removal -> Number Removal -> Stop Word Removal -> Porter Stemming -> Final Normalized Tokens
```

Each stage is a separate, testable function in `text_preprocessor.py`; `trace_message(text)` records the output of every
stage so that each can be inspected independently. `normalize_message(text)` runs the whole pipeline and returns one token
list per sentence.

### Actual messages used for the trace (selected deterministically)

| # | Row id | Label | Selection rule | Original message |
|---|---|---|---|---|
| M1 | 2972 | ham | one sentence, 8–15 words | `You're right I have now that I think about it` |
| M2 | 4719 | spam | ≥2 sentences, contains a digit | `5p 4 alfie Moon's Children in need song on ur mob. Tell ur m8s. Txt Tone charity to 8007 for Nokias or Poly charity for polys: zed 08701417012 profit 2 charity.` |
| M3 | 3884 | ham | ≥2 sentences | `Infact happy new year. How are you where are you when are we seeing` |

The rule is applied to a shuffle made with `random.Random(42)`, so the choice is reproducible (`select_demo_messages`).

## 7. Sentence Segmentation

**Definition.** Sentence segmentation (a.k.a. sentence boundary detection) divides a text into sentences.
**Function:** `segment_sentences(text)` → `nltk.tokenize.sent_tokenize` (the Punkt model; resource `punkt_tab`).
**Why:** the language model is built over sentences with start/end symbols.

An SMS record is *one message*, **not necessarily one grammatical sentence**. Segmentation produces **one** sentence when there is no
sentence-ending punctuation followed by a new sentence (M1), and **several** when `.`, `!` or `?` end a sentence (M2 → 3 sentences, M3 → 2).

| Message | Sentences produced (lowercased text) |
|---|---|
| M1 | `you're right i have now that i think about it` |
| M2 | `5p 4 alfie moon's children in need song on ur mob.` / `tell ur m8s.` / `txt tone charity to 8007 for nokias or poly charity for polys: zed 08701417012 profit 2 charity.` |
| M3 | `infact happy new year.` / `how are you where are you when are we seeing` |

*Observed side-effect (important):* because the assessment order lowercases **before** segmenting, Punkt loses the capital letter that
usually signals a new sentence. E.g. the **HYPOTHETICAL** text `win a prize now! call 09061701461. thanks.` is split after `now!`
but not after `09061701461.` (unit test `test_full_pipeline_output_type_and_value`). This is a known limitation of the required order.

## 8. Lowercasing

**Definition.** Converting all letters to lower case. **Function:** `lowercase_text` = `text.lower()`.
**Why:** `Free`, `FREE`, `free` must be the same type, otherwise counts are split and the vocabulary grows.
Example (M2): `5p 4 alfie Moon's Children ...` → `5p 4 alfie moon's children ...`.

## 9. Word Tokenization

**Definition.** Splitting a sentence into tokens (words and punctuation). **Function:** `tokenize_words` → `nltk.word_tokenize`.
Unlike `str.split()` (Section 1–2), it separates punctuation and clitics: `you're` → `you`, `'re`; `moon's` → `moon`, `'s`; `mob.` → `mob`, `.`.

| Sentence (M2, first) | Tokens |
|---|---|
| `5p 4 alfie moon's children in need song on ur mob.` | `5p, 4, alfie, moon, 's, children, in, need, song, on, ur, mob, .` |

## 10. Punctuation Removal

**Definition.** Removing tokens that are punctuation marks. **Function:** `remove_punctuation`.
Lecture method: keep a token only if it is not in `string.punctuation`. This implementation keeps that idea and removes a token
when **all its characters** are in `string.punctuation` (so `...` and `` `` `` also disappear) plus a small, labelled **extension**
`EXTRA_SYMBOLS = "£€¥₹‘’“”–—…"`. Tokens such as `'s` (apostrophe + letter) are **not** removed (they are not purely punctuation); this is a limitation (§32).
Example (M2, first sentence): removed `['.']`. Example (M2, third sentence): removed `[':', '.']`.

## 11. Number Removal

**Definition.** Removing tokens that represent numbers. **Function:** `remove_numbers`: a token is removed if it **contains any digit**
(`contains_digit`). This catches `4`, `2`, `8007`, `08701417012`, and also mixed tokens like `5p` and `m8s`.
**Why:** prices, phone numbers and short codes are almost unique strings; they inflate the vocabulary without helping a word-level model.
(The production spam classifier does the opposite on purpose — see §2.)
Actual results: M2 sentence 1 removed `['5p', '4']`; sentence 2 removed `['m8s']`; sentence 3 removed `['8007', '08701417012', '2']`.

## 12. Stop Word Removal

**Definition.** Stop words are very frequent function words (`the`, `is`, `you`, `to`) that carry little content. **Function:** `remove_stop_words`
with the set from `nltk.corpus.stopwords.words('english')` (a `set` gives fast membership tests; list comprehension as in Section 1–2).
Actual results: M1 removed `['you','i','have','now','that','i','about','it']` leaving `["'re", 'right', 'think']`; M3 sentence 2 removed
`['how','are','you','where','are','you','when','are','we']` leaving `['seeing']`.
Caution: NLTK's list also contains negations (`no`, `not`) — removing them can change meaning.

## 13. Porter Stemming

**Definition.** Stemming reduces a word to a *stem* by applying suffix-stripping rules; the stem need not be a real word.
**Function:** `stem_tokens` with `nltk.stem.PorterStemmer` (the lecture's primary stemmer).
Actual stem changes in the demonstration: `alfie→alfi`, `charity→chariti`, `nokias→nokia`, `poly→poli`, `polys→poli`, `happy→happi`, `seeing→see`.
Note `poly` and `polys` merge into the same stem `poli` — the goal of stemming (fewer distinct types).

## 14. Stemming vs. Lemmatization

| | Stemming (Porter) | Lemmatization (WordNet) |
|---|---|---|
| Method | rule-based suffix stripping | dictionary lookup using a lexicon |
| Output | may not be a real word (`chariti`) | a real word, the *lemma* (`charity`) |
| Speed | faster | slower, needs WordNet data and ideally a part-of-speech |
| Used in this assessment | **yes (pipeline stage 7)** | illustration only |

The assignment requires stemming **or** lemmatization; the pipeline uses stemming. The function `compare_stemming_and_lemmatization`
(not part of the pipeline) runs both on the tokens left in the demonstration. Actual rows:

| Token | Porter stem | Lemma (as noun) | Lemma (as verb) |
|---|---|---|---|
| charity | chariti | charity | charity |
| children | children | child | children |
| seeing | see | seeing | see |
| happy | happi | happy | happy |

The lemmatizer needs the right part of speech (`seeing` → `see` only as a verb), as the lecture shows with `pos='v'`.

## 15. Complete Preprocessing Example (ACTUAL — message M2, row 4719, spam)

| Stage | Function | Output |
|---|---|---|
| Original | – | `5p 4 alfie Moon's Children in need song on ur mob. Tell ur m8s. Txt Tone charity to 8007 for Nokias or Poly charity for polys: zed 08701417012 profit 2 charity.` |
| Lowercasing | `lowercase_text` | `5p 4 alfie moon's children in need song on ur mob. tell ur m8s. txt tone charity to 8007 for nokias or poly charity for polys: zed 08701417012 profit 2 charity.` |
| Segmentation | `segment_sentences` | 3 sentences |
| Tokenization (sentence 3) | `tokenize_words` | `txt, tone, charity, to, 8007, for, nokias, or, poly, charity, for, polys, :, zed, 08701417012, profit, 2, charity, .` |
| Punctuation removal | `remove_punctuation` | `txt, tone, charity, to, 8007, for, nokias, or, poly, charity, for, polys, zed, 08701417012, profit, 2, charity` (removed `:` and `.`) |
| Number removal | `remove_numbers` | `txt, tone, charity, to, for, nokias, or, poly, charity, for, polys, zed, profit, charity` (removed `8007, 08701417012, 2`) |
| Stop-word removal | `remove_stop_words` | `txt, tone, charity, nokias, poly, charity, polys, zed, profit, charity` (removed `to, for, or, for`) |
| Porter stemming | `stem_tokens` | `txt, tone, chariti, nokia, poli, chariti, poli, zed, profit, chariti` |

Sentence 1 of M2 ends as `alfi, moon, 's, children, need, song, ur, mob`; sentence 2 (`tell ur m8s.`) ends as `tell, ur`.
The complete traces of all three messages are in `outputs/preprocessing_examples.json`.

## 16. N-Gram Language Models

**N-gram:** a contiguous sequence of *n* items (here words) from a text. An **n-gram language model** estimates the probability of a word
(or sentence) from the counts of n-grams in a corpus. In this assessment n = 2 (a bigram model).

## 17. Unigrams
*n = 1*: single words, e.g. `money`, `sha`. Counts of unigrams (Counter `word_counts`) describe word frequency.

## 18. Bigrams
*n = 2*: pairs of consecutive words. Extracted with `nltk.bigrams`. ACTUAL, sentence 2 padded: `<s> saturday`, `saturday sunday`, `sunday holiday`, `holiday difficult`, `difficult </s>`.

## 19. Trigrams
*n = 3*: triples of consecutive words, `nltk.trigrams`. ACTUAL, sentence 2: `<s> saturday sunday`, `saturday sunday holiday`, `sunday holiday difficult`, `holiday difficult </s>`.
A **trigram model** would condition each word on the previous *two* words. It is the next higher-order n-gram. The lecture's probability
code is a **bigram** model, so that is what is implemented; trigrams are only *extracted* (`extract_ngrams`), not used for probabilities.

## 20. Markov Assumption

The probability of a word depends only on a limited history. For a bigram model (first-order Markov): only the **previous word** matters.

$$
P(w_i \mid w_1, w_2, \dots, w_{i-1}) \approx P(w_i \mid w_{i-1})
$$

| Symbol | Meaning |
|---|---|
| $P(\cdot)$ | probability |
| $w_i$ | the word at position $i$ in the sentence |
| $w_1,\dots,w_{i-1}$ | all words before position $i$ (the full history) |
| $w_{i-1}$ | the single word immediately before $w_i$ |
| $\mid$ | "given" |
| $\approx$ | approximately equal (the assumption simplifies the history) |

*Why we need it:* the exact history of a long sentence almost never occurs in any corpus, so its probability cannot be counted. Using only the previous word makes the counts reliable.

## 21. Bigram Probability Formula

Maximum-likelihood estimate (before smoothing):

$$
P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

## 22. Laplace Smoothing

**Definition.** Laplace (add-one) smoothing adds 1 to every bigram count so that no bigram has probability zero. To keep the probabilities summing to 1, the denominator grows by the vocabulary size. This is the lecture's formula and the code's formula:

$$
P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}
$$

| Symbol | Meaning | Code |
|---|---|---|
| $P(w_i \mid w_{i-1})$ | smoothed probability of word $w_i$ following $w_{i-1}$ | `step["probability"]` |
| $w_i$ | current word | `current_word` |
| $w_{i-1}$ | previous word (may be `<s>`) | `previous_word` |
| $C(w_{i-1}, w_i)$ | number of times the pair occurs in the training sentences | `bigram_counts[(previous, current)]` |
| $C(w_{i-1})$ | number of times $w_{i-1}$ occurs as the first word of a bigram | `context_counts[previous]` |
| $V$ | vocabulary size | `vocabulary_size` |
| $+1$ | the Laplace "add-one" | `numerator = bigram_count + 1` |

Relationship: the numerator adds one pseudo-count to the observed pair; the denominator adds one pseudo-count for each of the $V$ words that could follow, so $\sum_{w} P(w \mid w_{i-1}) = 1$ (the unit test `test_conditional_distribution_sums_to_one` checks this).

**Worked example with real numbers** (sentence 3, $P(\text{call} \mid \text{free})$): $C(\text{free},\text{call}) = 9$, $C(\text{free}) = 192$, $V = 5818$:
$(9+1)/(192+5818) = 10/6010 = 0.001664$.

## 23. Vocabulary and Frequency Counting

Counting strategy (identical to the lecture's `calculate_probability`; implemented in `BigramLanguageModel.fit`):

1. Pad every training sentence: `['<s>'] + tokens + ['</s>']`.
2. `context_counts.update(padded[:-1])` — counts $C(w_{i-1})$ for every word except `</s>` (including `<s>`).
3. `bigram_counts.update(nltk.bigrams(padded))` — counts $C(w_{i-1}, w_i)$.
4. $V$ = number of distinct training words + `</s>` (the lecture does not count `<s>`, which is never predicted) + `<UNK>` (extension).

Actual values from the training run:

| Quantity | Value |
|---|---|
| Training messages | 4,128 |
| Sentences after segmentation | 7,782 |
| Sentences empty after normalization (e.g. only numbers/stop words), skipped | 316 |
| **Sentences used for training** | **7,466** (so $C(\texttt{<s>}) = 7466$) |
| Distinct training words | 5,816 |
| **Vocabulary size $V$** | **5,818** = 5,816 + `</s>` + `<UNK>` |
| Distinct bigrams observed | 26,408 |

**Unseen words (`<UNK>`) — EXTENSION, not in the lecture.** The lecture scores sentences it was trained on. Here the sentences come from the held-out set, so a word may be missing from the training vocabulary. Such words are replaced by `<UNK>` (`replace_unseen_words`). `<UNK>` is counted once in $V$ and never occurs in training, so $C(\texttt{<UNK>})=0$ and any probability involving it is $1/V$-based. With `BigramLanguageModel(use_unknown_token=False)` the model reproduces the lecture exactly ($V$ = 12 on the lecture's toy corpus; unit tests reproduce the lecture's printed probabilities 0.0003585434, 0.0000440993 and 0.0002574158). In the three selected sentences **no word was unseen**.

## 24. Sentence Probability Calculation

$$
P(S) = P(w_1, w_2, \dots, w_n) \approx \prod_{i=1}^{n+1} P(w_i \mid w_{i-1})
$$

where $w_0 = \texttt{<s>}$ and $w_{n+1} = \texttt{</s>}$.

| Symbol | Meaning |
|---|---|
| $S$ | the sentence |
| $P(S)$ | probability of the whole sentence |
| $n$ | number of words in the sentence (without boundary symbols) |
| $\prod_{i=1}^{n+1}$ | product over all $n+1$ transitions (including the final one to `</s>`) |
| $w_0,\ w_{n+1}$ | the start symbol `<s>` and end symbol `</s>` |

**Why multiply?** By the chain rule, $P(w_1,\dots,w_n) = \prod_i P(w_i \mid w_1,\dots,w_{i-1})$ (the probability of a joint event is the probability of the first times the conditional probability of each next event given what came before). The Markov assumption replaces each long history by only the previous word, giving the product of bigram probabilities.

**Log probabilities.** Products of many small numbers underflow, so the code also keeps

$$
\ln P(S) = \sum_{i=1}^{n+1} \ln P(w_i \mid w_{i-1}), \qquad P(S) = e^{\ln P(S)}
$$

($\ln$ = natural logarithm; because $\ln(ab)=\ln a+\ln b$). The program prints the ordinary product (computed step by step) and $\ln P(S)$; tests verify they agree (`probability_from_log`).

## 25. Detailed Calculations for the Three Actual Sentences

Model: trained on the 7,466 normalized training sentences; $V = 5818$; $C(\texttt{<s>}) = 7466$. The sentences come from the **held-out** messages (not used for training). Selection rule (`select_heldout_sentences`): visit held-out messages in a `random.Random(42)` shuffle; take the first sentence whose normalized form has 4–6 tokens, with labels ham, ham, spam (each from a different message).

### Sentence 1 — ham (row 1444)

* Original message: `How are you with moneY...as in to you...money aint a thing....how are you sha!`
* Sentence used (lowercased): `how are you with money...as in to you...money aint a thing....how are you sha!` (Punkt did not split inside it)
* Normalized tokens: `['money', 'money', 'aint', 'thing', 'sha']` — unseen words: none
* Padded bigrams: `<s> money`, `money money`, `money aint`, `aint thing`, `thing sha`, `sha </s>`

| $i$ | $w_{i-1}$ | $w_i$ | $C(w_{i-1},w_i)$ | $C(w_{i-1})$ | Numerator | Denominator | $P(w_i\mid w_{i-1})$ | Running product |
|---|---|---|---|---|---|---|---|---|
| 1 | `<s>` | money | 6 | 7466 | 7 | 13284 | 0.000527 | 5.2694971394e-04 |
| 2 | money | money | 1 | 46 | 2 | 5864 | 0.000341 | 1.7972364050e-07 |
| 3 | money | aint | 0 | 46 | 1 | 5864 | 0.000171 | 3.0648642650e-11 |
| 4 | aint | thing | 0 | 4 | 1 | 5822 | 0.000172 | 5.2642807712e-15 |
| 5 | thing | sha | 0 | 95 | 1 | 5913 | 0.000169 | 8.9028932372e-19 |
| 6 | sha | `</s>` | 0 | 2 | 1 | 5820 | 0.000172 | 1.5297067418e-22 |

**$P(S_1) = 1.5297067418\times10^{-22}$**, $\ln P(S_1) = -50.231796$. Observed (non-zero) bigrams: `<s> money` (6) and `money money` (1).

### Sentence 2 — ham (row 4322)

* Original message: `Only saturday and sunday holiday so its very difficult:)`
* Sentence used: `only saturday and sunday holiday so its very difficult:)`
* Normalized tokens: `['saturday', 'sunday', 'holiday', 'difficult']` — unseen words: none

| $i$ | $w_{i-1}$ | $w_i$ | $C(w_{i-1},w_i)$ | $C(w_{i-1})$ | Numerator | Denominator | $P(w_i\mid w_{i-1})$ | Running product |
|---|---|---|---|---|---|---|---|---|
| 1 | `<s>` | saturday | 0 | 7466 | 1 | 13284 | 0.000075 | 7.5278530563e-05 |
| 2 | saturday | sunday | 1 | 11 | 2 | 5829 | 0.000343 | 2.5828969142e-08 |
| 3 | sunday | holiday | 0 | 4 | 1 | 5822 | 0.000172 | 4.4364426557e-12 |
| 4 | holiday | difficult | 0 | 33 | 1 | 5851 | 0.000171 | 7.5823665283e-16 |
| 5 | difficult | `</s>` | 2 | 7 | 3 | 5825 | 0.000515 | 3.9050814738e-19 |

**$P(S_2) = 3.9050814738\times10^{-19}$**, $\ln P(S_2) = -42.386838$. Observed bigrams: `saturday sunday` (1) and `difficult </s>` (2).

### Sentence 3 — spam (row 3229)

* Original message: `Freemsg: 1-month unlimited free calls! Activate SmartCall Txt: CALL to No: 68866. Subscriptn3gbp/wk unlimited calls Help: 08448714184 Stop?txt stop landlineonly`
* Sentence used (first sentence of the message): `freemsg: 1-month unlimited free calls!`
* Normalized tokens: `['freemsg', 'unlimit', 'free', 'call']` — unseen words: none

| $i$ | $w_{i-1}$ | $w_i$ | $C(w_{i-1},w_i)$ | $C(w_{i-1})$ | Numerator | Denominator | $P(w_i\mid w_{i-1})$ | Running product |
|---|---|---|---|---|---|---|---|---|
| 1 | `<s>` | freemsg | 11 | 7466 | 12 | 13284 | 0.000903 | 9.0334236676e-04 |
| 2 | freemsg | unlimit | 0 | 11 | 1 | 5829 | 0.000172 | 1.5497381485e-07 |
| 3 | unlimit | free | 0 | 9 | 1 | 5827 | 0.000172 | 2.6595815145e-11 |
| 4 | free | call | 9 | 192 | 10 | 6010 | 0.001664 | 4.4252604234e-14 |
| 5 | call | `</s>` | 112 | 431 | 113 | 6249 | 0.018083 | 8.0021511898e-16 |

**$P(S_3) = 8.0021511898\times10^{-16}$**, $\ln P(S_3) = -34.761651$. Observed bigrams: `<s> freemsg` (11), `free call` (9), `call </s>` (112).

All probabilities are tiny (a vocabulary of 5,818 words); what is meaningful is the comparison between sentences with the **same length**. The numbers are not directly comparable across lengths because every extra word multiplies by another number below 1. The spam sentence has the highest probability mainly because it is the shortest of the three and its bigrams `free call` (9) and `call </s>` (112) have high training counts; this says nothing about its being spam.

## 26. Complete Code Walkthrough

```
course_assessment/
  preprocessing/text_preprocessor.py   Part A functions
  language_model/bigram_model.py       Part B model
  examples/selected_sentences.py       dataset loading + selecting actual messages/sentences
  run_assessment.py                    runs everything, prints the report, writes outputs/*.json
  tests/                               pytest tests
```

### 26.1 `preprocessing/text_preprocessor.py`
| Function | Input → output | Role |
|---|---|---|
| `ensure_nltk_resources(download=False)` | – → None / `RuntimeError` | checks `punkt_tab` and `stopwords`; the error message contains the exact `nltk.download` command |
| `lowercase_text(text)` | str → str | stage 1 (`str.lower`) |
| `segment_sentences(text)` | str → list[str] | stage 2 (`sent_tokenize`); empty text → `[]` |
| `tokenize_words(sentence)` | str → list[str] | stage 3 (`word_tokenize`) |
| `is_punctuation_token`, `remove_punctuation(tokens)` | list → list | stage 4; token removed if all characters are in `PUNCTUATION_CHARACTERS` |
| `contains_digit`, `remove_numbers(tokens)` | list → list | stage 5; token removed if it contains a digit |
| `get_stop_words`, `remove_stop_words(tokens)` | list → list | stage 6 (NLTK English list as a `set`) |
| `stem_tokens(tokens)` | list → list | stage 7 (`PorterStemmer().stem`) |
| `normalize_sentence(sentence)` | str → list[str] | stages 3–7 on one sentence |
| `normalize_message(text)` | str → list[list[str]] | stages 1–7 on a message; one list per sentence |
| `trace_message(text)` | str → dict | runs the same functions and stores every intermediate result plus removed tokens |
| `compare_stemming_and_lemmatization(tokens)` | list → dict | illustration only |

Key design points: functions raise `TypeError` for non-strings and return empty lists for empty text; nothing mutates its input; the constant `STAGE_DESCRIPTIONS` holds the function name and the reason for each stage (used in the JSON output).

### 26.2 `language_model/bigram_model.py`
* `START_TOKEN='<s>'`, `END_TOKEN='</s>'`, `UNKNOWN_TOKEN='<UNK>'`.
* `BigramLanguageModel.__init__(use_unknown_token=True)` — creates the three `Counter`s (`word_counts`, `context_counts`, `bigram_counts`) and an empty `vocabulary`.
* `fit(training_sentences)` — pads each non-empty sentence, updates the counters (`padded[:-1]` for contexts, `nltk.bigrams(padded)` for pairs), builds the vocabulary; raises `ValueError` if there is no training sentence.
* `vocabulary_size` — $V$.
* `replace_unseen_words(tokens)` — maps out-of-vocabulary words to `<UNK>` and returns the replaced words.
* `conditional_probability(previous, current)` — returns bigram count, context count, numerator, denominator, $V$ and the probability.
* `sentence_probability(tokens)` — pads, loops over `nltk.bigrams(padded)`, multiplies step by step (`running_product`), sums `math.log` for `log_probability`, returns every step.
* `extract_ngrams(padded)` — unigrams, bigrams, trigrams for demonstration.
* `format_sentence_report(result)` — the text printed by the runner; a test checks that every displayed number matches the result dictionary.

### 26.3 `examples/selected_sentences.py`
`read_raw_records` (parse), `clean_records` (strip, drop empties, conflict check, de-duplicate), `split_records` (stratified `train_test_split`, same seed/size as the project), `load_sms_split` (reads `config.yaml`, raises `FileNotFoundError` with download instructions), `select_demo_messages`, `select_heldout_sentences` (§25 rule).

### 26.4 `run_assessment.py`
`build_training_sentences` normalizes all training messages and drops sentences that became empty; `run_preprocessing_demo` builds the traces and corpus statistics (`nltk.FreqDist`); `run_language_model` trains the model and scores the three sentences; `main` prints and writes `outputs/*.json`.

## 27. Explanation of Important Functions and Variables

| Name | Meaning |
|---|---|
| `padded` / `padded_sentence` | `['<s>', *tokens, '</s>']` |
| `context_counts[w]` | $C(w)$, the count of $w$ as the first element of a bigram (the lecture's `unigram_count`) |
| `bigram_counts[(a, b)]` | $C(a,b)$ |
| `word_counts[w]` | frequency of $w$ as a token (FreqDist-style count) |
| `vocabulary_size` | $V$ |
| `running_product` | product of the conditional probabilities up to this step |
| `log_probability` | $\ln P(S)$ |
| `normalized_tokens` | final tokens of the sentence after stage 7 |
| `row_id` | position in the cleaned dataset, links a result to its message |
| `SENTENCE_LABEL_PLAN` | `("ham","ham","spam")`, the labels of the three sentences |

## 28. Execution Instructions (Windows PowerShell)

```powershell
cd <path-to-the-cloned-repository>
.\.venv\Scripts\Activate.ps1                 # if a virtual environment already exists
# otherwise: python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
pip install -r course_assessment\requirements.txt
python -c "import nltk; [nltk.download(r) for r in ('punkt_tab','stopwords','wordnet')]"
python -m spamdetect.data                    # only if data\raw\SMSSpamCollection does not exist
python -m course_assessment.run_assessment
python -m pytest course_assessment\tests
```
`wordnet` is only needed for the optional stemming-vs-lemmatization table (the script continues without it).

## 29. Expected Outputs

* Console report of Part A and Part B (sections 15 and 25 of this document).
* `course_assessment/outputs/preprocessing_examples.json` — stage-by-stage traces, corpus statistics, lemmatization illustration.
* `course_assessment/outputs/sentence_probability_results.json` — model statistics, three sentences, n-grams, every conditional probability, final probabilities.

## 30. Testing and Validation

Tests live in `course_assessment/tests/`: 35 tests (the two result-dependent tests are skipped automatically if the dataset or output files are missing; with the dataset present all 35 run). They cover lowercasing, segmentation (one and many sentences), tokenization, punctuation, number, stop-word removal, Porter stemming, empty input, output types, non-modification of the dataset; unigram/bigram counts, vocabulary size, boundary handling, Laplace formula, **reproduction of the lecture's printed probabilities**, conditional distributions summing to 1, unseen words, empty sentence, reproducibility, probabilities in $(0,1)$, and that the three saved sentences are real held-out records whose saved results equal a fresh run. One test checks that this document contains the executed probabilities.

## 31. Methodological Decisions

1. **Training vs held-out.** The model is built on the training split and the three sentences come from the held-out split, so the evaluated text is never used for counting. The lecture trains and evaluates on the same sentences; the *formula is unchanged* — only the data separation is an improvement.
2. **Bigram, not trigram.** The lecture's probability code is a Bigram model; it is followed exactly.
3. **`<UNK>`** is an extension needed only because sentences are held out (§23).
4. **Preprocessing is a separate module** from the production classifier (§2).
5. **Pandas-free loader.** In the environment used to create this module, Windows Application Control blocked pandas' DLL, so the loader re-implements `spamdetect.data`'s parse/clean/split logic with the `csv` module and the same `train_test_split` call (same seed and test size). The module also stays independent from the production code. If you want to confirm the split equals the project's, compare `row_id` lists from both loaders on a machine where pandas works.
6. **Selection rule** (lengths 4–6 tokens, labels ham/ham/spam) was fixed in advance and uses a seeded shuffle.

## 32. Limitations

* The required order (lowercase before segmentation) weakens Punkt's sentence detection (§7).
* Contraction fragments (`'s`, `'re`, `'m`, `n't`) survive punctuation and stop-word removal and appear among the most frequent tokens; the lecture's method has the same behaviour.
* NLTK's stop-word list removes negations; Porter stems are not always words (`chariti`).
* Short SMS slang (`u`, `ur`, `txt`) is not standard English and is not normalized.
* Raw HTML entities remain as tokens (`gt`, `lt`).
* A bigram model sees only one previous word and 5,818 vocabulary entries with add-one smoothing give very flat distributions; absolute probabilities are tiny and only comparable between sentences of similar length.
* Only three sentences are scored; no conclusion about spam detection should be drawn from them. The spam classifier of the repository uses different, supervised methods.
* The raw dataset is not committed; it must be downloaded.

## 33. References

* Course notebooks: `NLP_Section_1_2.ipynb`, `NLP_S3_2025.ipynb`, `Text_Normalization_S4_2025.ipynb`, `N_Grams_S5_2025_updated.ipynb`.
* Almeida, T. & Gómez Hidalgo, J. (2011). SMS Spam Collection v.1. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84 (CC BY 4.0).
* NLTK documentation: https://www.nltk.org (`sent_tokenize`, `word_tokenize`, `stopwords`, `PorterStemmer`, `WordNetLemmatizer`, `bigrams`, `trigrams`, `FreqDist`).
* Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.
* Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, chapter on N-gram language models (background for the formulas; the formulas themselves follow the lecture notebook).
