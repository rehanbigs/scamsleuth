# ScamSleuth

**Forensic analysis for suspicious SMS, WhatsApp and email messages.**

[![CI](https://github.com/rehanbigs/scamsleuth/actions/workflows/ci.yml/badge.svg)](https://github.com/rehanbigs/scamsleuth/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

People lost **$470 million** in 2024 to scams that started with a text message, more than five
times the 2020 figure ([FTC](https://www.ftc.gov/news-events/data-visualizations/data-spotlight/2025/04/top-text-scams-2024)).
ScamSleuth examines a suspicious message and explains whether it is a scam, what gives it
away, and what to do next.

## Highlights

- **Leakage-safe dataset pipeline.** Downloads a public SMS phishing corpus, verifies its
  checksum, removes 146 duplicates, resolves 35 conflicting labels, and groups near-duplicate
  templates with MinHash LSH so that no variant leaks from training into test.
- **Adversarial-aware entity extraction.** Finds URLs, emails, phone numbers and premium SMS
  short codes, even when scammers hide them behind zero-width characters, full-width dots,
  broken schemes (`http:/bit.do/...`) or defanged text, and recovers 99% of annotated links in
  the corpus.
- **Safe by construction.** Suspicious links are parsed as text only. They are never fetched
  or rendered, and they are always shown defanged (`hxxp://evil[.]com`).
- **Measured, not guessed.** Success is defined up front as smishing recall at a fixed
  false-alarm rate, because accuracy is misleading on data that is 81% legitimate.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rehanbigs/scamsleuth.git
cd scamsleuth
uv sync                                   # create the environment
uv run python -m scamsleuth.data.prepare  # download, clean and split the dataset
uv run pytest                             # run the test suite
```

### Extract indicators from a message

```python
>>> from scamsleuth.features.entities import extract_entities
>>> msg = "URGENT: your parcel is held. Pay the fee at http:/parcel-fee[.]top/pay or call 0844 861 85 85"
>>> extract_entities(msg).defanged()
Entities(urls=('hxxp://parcel-fee[.]top/pay',), emails=(), phones=('08448618585',), short_codes=())
```

## Dataset

[SMS Phishing Dataset](https://doi.org/10.17632/f45bkkt8pr.1) by Mishra & Soni (2022),
CC BY 4.0: 5,971 messages labelled `ham`, `spam` or `smishing`.

| Split | ham | spam | smishing | Total |
|---|---:|---:|---:|---:|
| train | 3,449 | 310 | 402 | 4,161 |
| val | 690 | 62 | 80 | 832 |
| test | 690 | 62 | 80 | 832 |

Splits are stratified by class and grouped by near-duplicate template. The full processing
steps, quality issues and limitations are in the [data card](docs/data.md), and the analysis
is in the [EDA notebook](notebooks/01_eda.ipynb).

## Project layout

```
src/scamsleuth/
├── data/        # download, validation, de-duplication, splitting
└── features/    # entity extraction and defanging
tests/           # unit tests (pytest)
notebooks/       # exploratory analysis
docs/            # problem statement and data card
```

## Documentation

- [Problem statement and success metrics](docs/problem.md)
- [Data card](docs/data.md)

## Development

```bash
uv run pre-commit install   # ruff + mypy run on every commit
uv run ruff check . && uv run mypy src tests && uv run pytest
```

CI runs linting, strict type checking and the test suite on Python 3.11, 3.12 and 3.13 for
every push and pull request.

## License

Code is released under the [MIT License](LICENSE). The dataset is © its authors and used
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

ScamSleuth gives advice, not certainty. Always verify through an independent channel, for
example by calling the number printed on your bank card.
