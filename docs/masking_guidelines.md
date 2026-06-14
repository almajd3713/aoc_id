# Masking Guidelines

This dataset is intended for a competition where systems reconstruct masked entity spans in Arabic text.

## Annotation Goal

For each input row:
- keep `original_text` unchanged
- produce `masked_text` by replacing entity spans with typed placeholders
- fill `mask_spans_json` with exact offsets over `original_text`
- set `mask_count` to the number of masked spans
- set `annotation_status` to `approved` only after review

## What To Mask

Mask contiguous spans for:
- `PERSON`: personal names, nicknames, kunyah-style references when they identify a person
- `LOC`: countries, cities, neighborhoods, streets, landmarks
- `ORG`: companies, teams, schools, ministries, newspapers, parties
- `DATE`: full dates, month-year references, holiday dates
- `TIME`: times of day and time expressions when specific
- `NUM`: money amounts, ages, counts, rankings, jersey numbers when identifying
- `HANDLE`: usernames, social handles, tagged accounts
- `URL`: web links
- `EMAIL`: email addresses
- `PHONE`: phone numbers
- `ID`: explicit identifiers such as account numbers or document numbers

## What Not To Mask

Do not mask:
- dialect markers by themselves
- general sentiment words
- ordinary nouns and verbs
- topical keywords unless they are part of an entity span
- pronouns unless they are embedded in a named expression being masked

## Placeholder Format

Use typed placeholders in the masked text:
- `[PERSON_1]`
- `[LOC_1]`
- `[ORG_1]`
- `[DATE_1]`

Rules:
- numbering restarts for each row
- numbering must be contiguous within each type in a row: `[PERSON_1]`, `[PERSON_2]`, not `[PERSON_1]`, `[PERSON_3]`
- if the same entity appears again in the same row, reuse the same placeholder
- preserve punctuation and all unmasked text exactly
- do not nest or overlap spans

Example:

```text
original: محمد قابل أحمد في الرباط ثم اتصل محمد بأحمد
masked: [PERSON_1] قابل [PERSON_2] في [LOC_1] ثم اتصل [PERSON_1] ب[PERSON_2]
```

## `mask_spans_json` Format

Store a JSON list. Each span object should look like:

```json
[
  {
    "start": 15,
    "end": 27,
    "placeholder": "[PERSON_1]",
    "surface_form": "محمد صلاح",
    "mask_type": "PERSON"
  }
]
```

Rules:
- `start` is inclusive, `end` is exclusive
- offsets must refer to `original_text`, not `normalized_text`
- `surface_form` must exactly match `original_text[start:end]`

## Review Checklist

Before approving a row:
- every placeholder in `masked_text` appears in `mask_spans_json`
- `mask_count` equals the number of span objects
- all offsets are valid
- all masked spans are entity-like and competition-relevant
- repeated mentions use consistent placeholder IDs
