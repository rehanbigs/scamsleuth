# Data Card

## Source

| | |
|---|---|
| **Dataset** | SMS Phishing Dataset for Machine Learning and Pattern Recognition |
| **Authors** | Sandhya Mishra, Devpriya Soni (Jaypee Institute of Information Technology), 2022 |
| **DOI** | [10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **File** | `Dataset_5971.zip` → `Dataset_5971.csv`, SHA-256 `9bbf3188…233cc3` (verified on download) |

The raw file is downloaded at build time and is **not** stored in this repository.

## Raw contents

5,971 English SMS messages with a class label and three yes/no annotations (`URL`, `EMAIL`,
`PHONE`).

| Class | Messages | Share |
|---|---:|---:|
| ham | 4,844 | 81.1% |
| spam | 489 | 8.2% |
| smishing | 638 | 10.7% |

## Processing

Run with `uv run python -m scamsleuth.data.prepare`. The pipeline is deterministic (seed 42).

1. **Validation.** Required columns are checked. Labels are normalised, since the source mixes
   `Smishing`/`smishing` and `Spam`/`spam`. Annotation flags are converted to booleans, and
   surrounding whitespace (including leading tabs) is stripped.
2. **Exact de-duplication.** Messages that are identical after lower-casing and removing
   punctuation, spacing and encoding-error characters are collapsed to one row.
   - 139 duplicate groups were found, and **146 rows were removed** (5,971 → 5,825).
   - **35 groups had conflicting labels**, all `spam` vs `smishing`. The majority label
     wins, and ties go to the more severe class (`smishing` > `spam` > `ham`).
3. **Near-duplicate grouping.** Messages sent from the same template with small edits, such as
   a different phone number or prize amount, are grouped with MinHash LSH (character 5-grams,
   estimated Jaccard ≥ 0.8). This found **164 groups covering 424 messages**.
4. **Group-aware stratified split.** `StratifiedGroupKFold` (7 folds): one fold is test, one
   is validation and the rest is training. Every near-duplicate group lies entirely within one
   split, and class proportions are preserved.

## Final splits

| Split | ham | spam | smishing | Total | Smishing share |
|---|---:|---:|---:|---:|---:|
| train | 3,449 | 310 | 402 | 4,161 | 9.7% |
| val | 690 | 62 | 80 | 832 | 9.6% |
| test | 690 | 62 | 80 | 832 | 9.6% |

Leakage checks: **0** near-duplicate groups and **0** normalised texts appear in more than one
split.

## Known issues and limitations

- **Spam vs smishing is a fuzzy boundary.** All 35 label conflicts are between these two
  classes, so annotators did not always agree. Expect most model confusion here as well.
- **Encoding damage.** 145 messages contain the Unicode replacement character `�`, most often in
  place of a currency symbol (`£`, `€`), sometimes an ellipsis or apostrophe.
- **Overlap with the UCI SMS Spam Collection.** A large share of the `ham` and `spam` messages
  also appears in the older UCI corpus
  ([systematic review, 2026](https://arxiv.org/abs/2604.11429)). UCI therefore cannot serve as an
  independent test set, and pretrained models may have seen these messages.
- **Age and coverage.** The data is from 2022 and mostly English, with UK and Indian phrasing.
  Recent scam types (unpaid tolls, task scams, delivery scams with modern URL shorteners) are
  under-represented, so results on recent messages should be reported as a separate slice.
- **Annotation flags are coarse.** The `URL` flag misses some bare domains (for example
  `smsg.io/...`), so link detection uses this project's own extractor.

## Ethical notes

The messages are a public research corpus. Messages may contain phone numbers and
short codes. Personal data is never added to the dataset from user submissions without consent.
