# Kaggle Competition Forms Draft

This document is a copy-paste pack for the Kaggle competition setup based on the current package in `artifacts/kaggle/full_masked_competition/`.

It is written to match the standard Kaggle competition structure:
- participant files: `train.csv`, `test.csv`, `sample_submission.csv`
- hidden labels: `public_labels.csv`, `private_labels.csv`
- opaque participant IDs: `row_000001` style
- participant-visible columns:
  - `train.csv`: `id`, `original_text`, `masked_text`
  - `test.csv`: `id`, `original_text`
  - `sample_submission.csv`: `id`, `masked_text`

Current package summary:
- training rows: 5,510
- test rows: 534
- public leaderboard rows: 267
- private leaderboard rows: 267
- excluded unsupported-label rows: 4

Recommended evaluation description:
- metric: span-level micro-F1
- scoring unit: exact `(row id, start, end, mask_type)` match
- predictions are reconstructed from submitted `masked_text`

---

## Competition Title

Arabic Entity Mask Reconstruction Challenge

## Competition Subtitle

Recover masked named entities in Arabic online commentary using typed placeholders.

## Short Summary

Build a system that reconstructs masked entity spans in Arabic text. Given an input comment, participants must submit a `masked_text` version that replaces entity mentions with typed placeholders such as `[PERSON_1]`, `[LOC_1]`, and `[ORG_1]`.

---

## Overview

Arabic user-generated text often mixes dialectal variation, informal spelling, sparse punctuation, and entity mentions that are difficult to detect reliably. This competition focuses on structured entity masking in Arabic online commentary.

Participants are given raw Arabic comments and must produce a masked version of each comment by replacing entity mentions with typed placeholders. The task is designed to evaluate robust entity recognition and span reconstruction under noisy real-world conditions.

The competition includes Modern Standard Arabic and multiple Arabic dialect varieties. However, dialect labels are not exposed in participant-facing files. Systems must generalize from the released training data and infer masking behavior directly from text.

This challenge is intended for teams working on Arabic NLP, named entity recognition, span prediction, privacy-oriented text transformation, and robust structured generation.

---

## Description

In this competition, each input row contains an Arabic comment in the `original_text` field. Your task is to generate `masked_text`, a version of the same comment where eligible entity spans are replaced with typed placeholders.

Examples of valid placeholders include:
- `[PERSON_1]`
- `[LOC_1]`
- `[ORG_1]`
- `[DATE_1]`
- `[NUM_1]`

The numbering restarts within each row and must remain contiguous within each entity type. If the same entity appears again in the same row, the same placeholder should be reused.

Your system must preserve all non-masked text exactly while replacing only the relevant entity spans. This is a span-sensitive task: both the entity type and the exact character boundaries matter.

The released participant files use opaque IDs and do not expose dialect labels. Your model should rely only on the training examples and the raw text itself.

### Supported entity types

- `PERSON`
- `LOC`
- `ORG`
- `DATE`
- `TIME`
- `NUM`
- `HANDLE`
- `URL`
- `EMAIL`
- `PHONE`
- `ID`

### What to preserve

- Keep all unmasked text exactly as written.
- Do not rewrite, normalize, summarize, or translate the input.
- Do not add or remove rows.
- Return one `masked_text` string for every row in the test set.

---

## Evaluation

Submissions are evaluated with **span-level micro-F1**.

For each submitted row, the system reconstructs predicted masked spans from the submitted `masked_text` using the original input text. Each predicted span is compared against the hidden ground truth using:
- exact start offset
- exact end offset
- exact `mask_type`

A predicted span is counted as correct only if all three match the hidden label exactly.

The final score is the micro-F1 over all spans in the evaluation set:

`F1 = 2 * TP / (2 * TP + FP + FN)`

Where:
- `TP` = correctly predicted spans
- `FP` = predicted spans not present in the ground truth
- `FN` = ground-truth spans that were missed

If a submitted `masked_text` cannot be aligned back to the original text, that row receives zero predicted spans for scoring.

### Why this metric

This metric rewards:
- correct entity typing
- exact span boundaries
- correct handling of rows with multiple masked entities

It is stricter and more faithful to the task than full-string exact match, while remaining robust across different rows and entity counts.

---

## Data

The competition package includes:

### `train.csv`

Columns:
- `id`
- `original_text`
- `masked_text`

This file contains released training examples. Some rows contain one or more masked entities, while others are valid zero-mask rows that teach the model when no masking is needed.

### `test.csv`

Columns:
- `id`
- `original_text`

This file contains the evaluation rows. Participants must generate `masked_text` for each row.

### `sample_submission.csv`

Columns:
- `id`
- `masked_text`

This file shows the exact format expected for submission.

### Notes

- IDs are opaque and do not reveal dialect or internal source identifiers.
- Dialect labels are not part of participant-facing files.
- Public and private leaderboard labels are hidden from participants.

---

## Submission Format

Your submission file must be a CSV with exactly two columns:

- `id`
- `masked_text`

Requirements:
- include every `id` from `test.csv`
- keep the same row coverage
- provide exactly one prediction per row
- use valid typed placeholders such as `[PERSON_1]`, `[LOC_1]`, `[ORG_1]`

Example:

```csv
id,masked_text
row_000001,زار [PERSON_1] [PERSON_2] في [LOC_1] يوم [DATE_1]
row_000002,تواصل [PERSON_1] مع [ORG_1] عبر [HANDLE_1]
```

---

## Rules

1. Use only the data released through the competition unless external data is explicitly allowed in the final competition settings.
2. Do not manually inspect or attempt to reconstruct hidden public or private labels.
3. Submissions must be generated programmatically and must follow the required CSV format exactly.
4. Teams are responsible for ensuring that their submissions preserve non-masked text and use valid placeholder syntax.
5. Organizers may disqualify submissions that exploit leakage, reverse-engineer hidden labels, or violate platform rules.

### Recommended final policy choice

If you want a cleaner first launch, set external data to **not allowed** unless you explicitly want an open-resource track.

---

## Prizes

This section depends on whether the competition is:
- research-only
- academic
- internal
- sponsored
- prize-based

If there are no monetary awards, use:

This is a research competition intended to benchmark Arabic entity masking systems on noisy online commentary. The main goal is reproducible model comparison and stronger Arabic NLP baselines rather than cash awards.

If there are awards, replace this section with your actual prize breakdown.

---

## Timeline

Fill these in once the competition is created:

- Competition launch:
- Entry deadline:
- Team merger deadline:
- Final submission deadline:

Recommended note:

All deadlines follow the times shown in the Kaggle competition interface. Late submissions will not be accepted after the final deadline.

---

## Getting Started

1. Download `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Train a system that maps `original_text` to `masked_text`.
3. Generate one prediction for each row in `test.csv`.
4. Save predictions in the exact `sample_submission.csv` format.
5. Upload your submission to receive a leaderboard score.

---

## FAQ

### Are dialect labels provided?

No. Participant-facing files do not include dialect labels, and IDs are opaque.

### Are all test rows guaranteed to contain at least one masked span?

Yes. The evaluation pool is drawn from supported masked rows only.

### Can a training row contain no masked entities?

Yes. The training file includes valid zero-mask examples.

### Do exact character spans matter?

Yes. Evaluation is based on exact span boundaries and exact entity type.

### What happens if my masking cannot be aligned back to the original text?

That row receives zero predicted spans during scoring.

---

## Citation / Acknowledgment

If you use this competition data, please cite the original AOC-related resources and any competition paper or release note that accompanies this benchmark.

If you want a short acknowledgment field:

This competition is built from an Arabic online commentary resource and a manually reviewed masking pipeline designed for structured entity reconstruction.

---

## Host Settings Checklist

These are not long-form text fields, but they are the settings you likely want when filling the competition:

- task type: prediction competition
- train file: `train.csv`
- test file: `test.csv`
- sample submission: `sample_submission.csv`
- leaderboard split: public/private hidden labels
- participant-visible ID: opaque sequential ID
- participant-visible dialect label: no
- primary metric label in description: span-level micro-F1
- submission column to predict: `masked_text`

---

## Optional Short Version For Kaggle Landing Text

Recover masked entity spans in Arabic online commentary by generating typed placeholder versions of each input text. Systems are evaluated with span-level micro-F1 over exact entity type and exact character boundaries.
