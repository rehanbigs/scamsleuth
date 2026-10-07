# Problem Statement

## The problem

Text messages are now the most common way scammers first reach people. In its April 2025
Data Spotlight *Top text scams of 2024*, the FTC reported that people lost **$470 million**
to scams that started with a text message in 2024, more than five times the 2020 figure.
The most common lures were fake package-delivery notices, phony job offers, fake fraud
alerts, unpaid-toll demands and "wrong number" openers.
([FTC, 2025](https://www.ftc.gov/news-events/data-visualizations/data-spotlight/2025/04/top-text-scams-2024))

Most people who receive these messages are not security experts. They need a quick,
trustworthy answer to one question: *is this message safe, and what should I do?*

## Who it is for

- **Non-technical people** (for example, older relatives) who forward suspicious messages to
  someone they trust before acting on them.
- **Students and job seekers**, who are heavily targeted by fake job and "task" scams.
- **Community groups** (WhatsApp groups, university channels) where one scam message reaches
  many people at once.

## What ScamSleuth decides

Each message is assigned one of three classes:

| Class | Meaning | Example |
|---|---|---|
| `ham` | Legitimate personal or service message | "Running late, see you at 6" |
| `spam` | Unwanted marketing; annoying but not trying to defraud | "50% off all pizzas this weekend" |
| `smishing` | SMS phishing: tries to steal money, credentials or personal data | "Your parcel is on hold. Pay the fee: hxxp://parcel-fee[.]top" |

## Why accuracy is the wrong metric

The data is imbalanced: about 81% of messages are `ham` and about 11% are `smishing`. A model
that labels every message `ham` would be 81% accurate while catching no scams at all.

The two kinds of error also cost very different amounts:

- **A missed scam** (false negative) can cost someone their savings.
- **A false alarm** (false positive) costs a moment of doubt. Too many of them, however,
  teach people to ignore warnings, or to distrust real messages from their bank.

So ScamSleuth is measured on how many scams it catches *while keeping false alarms low*,
not on overall accuracy.

## Success metrics

All metrics are reported on a held-out test set that never influences training or model
selection.

| Metric | Definition | Target |
|---|---|---|
| **Smishing recall** (primary) | Share of real smishing messages flagged as `smishing` | **≥ 90%** |
| **False-alarm rate** (primary constraint) | Share of legitimate (`ham`) messages flagged as `smishing` | **≤ 2%** |
| Macro F1 | Unweighted mean F1 across the three classes | Reported |
| Smishing PR-AUC | Area under the precision–recall curve for `smishing` | Reported |
| Calibration (ECE) | Gap between predicted probability and observed frequency | Reported |

The decision threshold is chosen on the validation set to satisfy the false-alarm
constraint, and recall is then reported at that threshold on the test set.

## Safety rules

These are hard rules, enforced in code rather than left to a model's judgement:

1. **Never fetch, open or render a suspicious URL.** Links are only analysed as text.
2. **Always display links defanged** (`hxxp://example[.]com`) so nothing is clickable.
3. **Treat message content as untrusted data**, never as instructions. Scam text is written
   by an adversary and may try to manipulate an AI system that reads it.
4. **Present results as advice, not certainty**, and always point people to an independent
   way to verify (for example, calling the number printed on their bank card).
5. **Do not store or share personal data** from submitted messages without consent.

## Non-goals

- Replacing a carrier-level or email-gateway spam filter.
- Guaranteeing that a message is safe. A `ham` verdict means "no scam indicators found".
- Identifying or tracking the people behind scam campaigns.
- Generating new scam content, beyond harmless perturbations used to test robustness.

## Known risks and limitations

- **Dataset age and coverage.** The classic labelled corpus is from 2022 and is dominated by
  feature-phone-era prize scams. Training therefore adds real 2017–2024 smishing reports, and
  results are reported separately per source and per scam type.
- **No modern legitimate notifications.** Real bank and courier messages are private, so all
  legitimate training messages are personal chat. False alarms on official-looking genuine
  messages are measured on a synthetic look-alike set.
- **Label noise.** Some identical messages carry different labels in the source data (see
  [data.md](data.md)).
- **Adversarial drift.** Scammers change wording to evade filters; robustness to obfuscation
  (look-alike characters, spacing, leetspeak) is tested explicitly.
