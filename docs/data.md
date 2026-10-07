# Data Card

ScamSleuth combines a labelled research corpus with real smishing reports from 2017–2024,
and adds two evaluation-only sets. Every source is pinned to an exact version and verified
by SHA-256 on download. No raw data is stored in this repository.

## Sources

| Source | Content | License | Used for |
|---|---|---|---|
| **Mendeley SMS Phishing Dataset**: Mishra & Soni (2022), [DOI 10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1) | 5,971 English SMS labelled `ham`, `spam` or `smishing` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Train / val / test |
| **IMC 2025 smishing reports**: Agarwal, Papasavva, Suarez-Tangil & Vasek, *Fishing for Smishing*, IMC 2025, [DOI 10.1145/3730567.3764431](https://doi.org/10.1145/3730567.3764431), [repository](https://github.com/reportsmishing/Smishing-Dataset-IMC25) | 33,869 real smishing messages reported by users on public forums (2017–2024), 66 languages, labelled by scam type | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Train / val / test (English); `multilingual` eval set (other languages) |
| **Indian Scam SMS, synthetic and audited**: [Ridham115 on Hugging Face](https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited) | 1,580 LLM-written scams and genuine look-alikes (bank OTPs, courier and bill notices) in English, Hinglish, Hindi and Roman-script code-mixed styles | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `hardneg` eval set only |

### Why three sources

The Mendeley corpus alone is too easy and too old. 41% of its smishing messages are
premium-rate prize texts from the feature-phone era ("txt WIN to 81010, 150p/msg"), and only
0.2% are job, crypto or "Hey Mum" scams. A simple TF-IDF model already reaches 0.97 PR-AUC on
it. The IMC 2025 reports bring today's scams: bank impersonation, parcel delivery, tax
refunds, telecom account suspensions, "wrong number" and "Hey Mum" lures.

No public dataset contains modern *legitimate* transactional messages (real bank alerts,
courier updates), because those are private. The synthetic `hardneg` set is the closest
available stand-in. It is used only to measure false alarms on look-alikes, never to estimate
real-world accuracy.

## Processing

Run with `uv run python -m scamsleuth.data.prepare`. The pipeline is deterministic (seed 42).

### Mendeley

1. **Validation.** Required columns are checked. Labels are normalised, since the source mixes
   `Smishing`/`smishing` and `Spam`/`spam`. Surrounding whitespace (including leading tabs)
   is stripped.

### IMC 2025

The source anonymised entities with Presidio-style tags such as `<URL>`, `<NAMED_ENTITY>` and
`<DATE_TIME>`.

1. **Tag handling.** `<URL>`, `<IP_ADDRESS>`, `<PHONE_NUMBER>` and `<EMAIL_ADDRESS>` become the
   same placeholder tokens (`xxurl`, `xxphone`, `xxemail`) that the model's preprocessing
   produces for real links and numbers in every other source. All other tags (names, dates,
   amounts, ID numbers) are removed. They occur only in this source, so keeping them would
   let a model recognise the *source* rather than the scam.
2. **Category filter.** The `others` category (6,922 rows) is excluded. It mixes real scams
   with genuine notifications (receipts, appointment reminders, password-reset codes) that
   users reported out of context, so its labels are unreliable. The `spam` category maps to
   `spam`; banking, delivery, government, telecom, wrong number and "Hey Mum" map to `smishing`.
3. **OTP notices.** 96 one-time-code notifications with no link or phone number are removed.
   These are genuine messages that users did not expect; the message itself is not the attack.
4. **Length.** 225 messages shorter than 20 characters after cleaning (for example a bare
   link) are removed.

This leaves 17,306 English and 9,239 non-English messages.

### Combined English set

1. **Exact de-duplication** across both sources, after lower-casing and removing punctuation,
   spacing and encoding-error characters: **2,250 duplicate groups, 6,172 rows removed**
   (23,277 → 17,105). Users often report the same campaign many times. **39 groups had
   conflicting labels**, all `spam` vs `smishing`. The majority label wins, and ties go to the
   more severe class.
2. **Near-duplicate grouping.** MinHash LSH (character 5-grams, estimated Jaccard ≥ 0.8)
   found **1,134 template groups covering 4,158 messages**, for example the same delivery
   lure with different tracking numbers.
3. **Group-aware stratified split.** `StratifiedGroupKFold` (7 folds) stratified on
   source × label: one fold is test, one is validation and the rest is training. Every
   near-duplicate group lies entirely within one split.

## Final splits

| Split | Mendeley ham | Mendeley spam | Mendeley smishing | IMC spam | IMC smishing | Total |
|---|---:|---:|---:|---:|---:|---:|
| train | 3,449 | 310 | 401 | 823 | 7,233 | 12,216 |
| val | 690 | 62 | 81 | 165 | 1,447 | 2,445 |
| test | 690 | 62 | 80 | 165 | 1,447 | 2,444 |

IMC scam types across all splits: banking 6,070, telecom 1,325, government 1,274,
delivery 1,201, spam 1,153, wrong number 203 and "Hey Mum" 54.

Leakage checks: **0** near-duplicate groups and **0** normalised texts appear in more than one
split.

### Evaluation-only sets

| Set | Rows | Content |
|---|---:|---|
| `multilingual` | 6,805 | Non-English IMC reports: smishing 6,528, spam 277, in 58 languages (Spanish 2,605, Dutch 1,378, French 631, … Hindi 102, Urdu 36) |
| `hardneg` | 1,580 | Synthetic: 783 genuine look-alikes (`ham`), 797 scams (`smishing`) |

## Known issues and limitations

- **All `ham` is from 2022 or earlier and is mostly personal chat.** Real modern
  legitimate transactional messages are not publicly available. A model may therefore learn
  that "anything official-looking is a scam". The `hardneg` false-alarm rate measures this
  risk directly.
- **Source and label are partly confounded.** IMC contributes only `spam` and `smishing`, so
  every `ham` message comes from Mendeley. Results are always reported per source.
- **IMC labels come from user reports**, classified by the IMC authors with GPT-4o. Some
  `spam`/`smishing` boundaries are fuzzy (39 conflicting duplicates).
- **Anonymisation removes detail.** IMC URLs are replaced by `xxurl`, so URL-level features
  (domain, TLD, shortener) cannot be learned from this source.
- **Synthetic stress set.** `hardneg` was written by LLMs and audited only in part. It shows
  *relative* false-alarm behaviour, not real-world rates.
- **Encoding damage in Mendeley.** 145 messages contain the replacement character `�`, most
  often in place of a currency symbol (`£`, `€`), sometimes an ellipsis or apostrophe.
- **Overlap with the UCI SMS Spam Collection.** Much of the Mendeley `ham` and `spam` also
  appears in the older UCI corpus ([systematic review, 2026](https://arxiv.org/abs/2604.11429)),
  so UCI cannot serve as an independent test set.

## Ethical notes

All sources are public research datasets released under CC BY 4.0. The IMC authors
pseudo-anonymised personal data before release. Personal data is never added from user
submissions without consent.
