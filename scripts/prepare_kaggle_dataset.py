#!/usr/bin/env python3
"""Prepare the merged masked dataset for a Kaggle competition package."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from scripts.kaggle_metric import write_metric_artifact
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.kaggle_metric import write_metric_artifact


SUPPORTED_MASK_TYPES = {
    "PERSON",
    "LOC",
    "ORG",
    "DATE",
    "TIME",
    "NUM",
    "HANDLE",
    "URL",
    "EMAIL",
    "PHONE",
    "ID",
}

MASK_COUNT_BUCKETS = (1, 2, 3, 4)
TRAIN_COLUMNS = ["id", "original_text", "masked_text"]
TEST_COLUMNS = ["id", "original_text"]
SUBMISSION_COLUMNS = ["id", "masked_text"]
LABEL_COLUMNS = ["id", "original_text", "masked_text", "mask_spans_json"]
SOLUTION_COLUMNS = ["id", "Usage", "original_text", "masked_text", "mask_spans_json"]
ID_MAPPING_COLUMNS = ["id", "example_id", "dialect", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("app_state/projects/full/merged/masked_dataset_merged.csv"),
        help="Merged approved dataset to package for Kaggle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kaggle/full_masked_competition"),
        help="Directory for Kaggle package outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--public-fraction",
        type=float,
        default=0.1,
        help="Fraction of supported masked rows assigned to the public test set.",
    )
    parser.add_argument(
        "--private-fraction",
        type=float,
        default=0.1,
        help="Fraction of supported masked rows assigned to the private test set.",
    )
    return parser.parse_args()


def mask_count_bucket(mask_count: int) -> str:
    if mask_count <= 0:
        return "0"
    if mask_count in MASK_COUNT_BUCKETS:
        return str(mask_count)
    return "5+"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_mask_spans(row: Dict[str, str]) -> List[Dict[str, object]]:
    spans = json.loads(row.get("mask_spans_json") or "[]")
    if not isinstance(spans, list):
        raise ValueError(f"{row.get('example_id', '<missing_example_id>')}: mask_spans_json must be a list")
    return spans


def classify_rows(
    rows: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    supported_masked_rows: List[Dict[str, str]] = []
    unmasked_rows: List[Dict[str, str]] = []
    excluded_rows: List[Dict[str, str]] = []

    for row in rows:
        if (row.get("annotation_status") or "").strip() != "approved":
            continue

        mask_count = int(row.get("mask_count") or "0")
        spans = _parse_mask_spans(row)
        unsupported_types = sorted(
            {
                str(span.get("mask_type") or "")
                for span in spans
                if str(span.get("mask_type") or "") not in SUPPORTED_MASK_TYPES
            }
        )

        if unsupported_types:
            excluded_rows.append(
                {
                    **row,
                    "exclusion_reason": f"unsupported_mask_types:{','.join(unsupported_types)}",
                }
            )
            continue

        if mask_count > 0:
            supported_masked_rows.append(row)
        else:
            unmasked_rows.append(row)

    return supported_masked_rows, unmasked_rows, excluded_rows


def allocate_bucket_counts(size: int, public_fraction: float, private_fraction: float) -> Tuple[int, int]:
    public_count = int(size * public_fraction)
    private_count = int(size * private_fraction)

    if size >= 3 and public_count == 0:
        public_count = 1
    if size >= 3 and private_count == 0:
        private_count = 1

    while public_count + private_count >= size and size > 1:
        if public_count >= private_count and public_count > 0:
            public_count -= 1
        elif private_count > 0:
            private_count -= 1
        else:
            break

    return public_count, private_count


def split_masked_rows(
    rows: Sequence[Dict[str, str]],
    seed: int,
    public_fraction: float,
    private_fraction: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dialect"], mask_count_bucket(int(row["mask_count"] or "0")))].append(row)

    rng = random.Random(seed)
    train_rows: List[Dict[str, str]] = []
    public_rows: List[Dict[str, str]] = []
    private_rows: List[Dict[str, str]] = []

    for key in sorted(grouped):
        bucket_rows = list(grouped[key])
        rng.shuffle(bucket_rows)
        public_count, private_count = allocate_bucket_counts(
            len(bucket_rows), public_fraction=public_fraction, private_fraction=private_fraction
        )
        public_rows.extend(bucket_rows[:public_count])
        private_rows.extend(bucket_rows[public_count : public_count + private_count])
        train_rows.extend(bucket_rows[public_count + private_count :])

    return sort_rows(train_rows), sort_rows(public_rows), sort_rows(private_rows)


def sort_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda row: row["example_id"])


def shuffle_rows(rows: Sequence[Dict[str, str]], seed: int) -> List[Dict[str, str]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def assign_split_ids(rows: Sequence[Dict[str, str]], prefix: str, split_name: str) -> List[Dict[str, str]]:
    width = max(6, len(str(len(rows))))
    assigned_rows: List[Dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        assigned_rows.append(
            {
                **row,
                "id": f"{prefix}_{index:0{width}d}",
                "split": split_name,
            }
        )
    return assigned_rows


def build_train_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "id": row["id"],
            "original_text": row["original_text"],
            "masked_text": row["masked_text"],
        }
        for row in rows
    ]


def build_test_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"id": row["id"], "original_text": row["original_text"]} for row in rows]


def build_label_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "id": row["id"],
            "original_text": row["original_text"],
            "masked_text": row["masked_text"],
            "mask_spans_json": row["mask_spans_json"],
        }
        for row in rows
    ]


def build_solution_rows(
    public_rows: Sequence[Dict[str, str]],
    private_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for usage, split_rows in (("Public", public_rows), ("Private", private_rows)):
        for row in split_rows:
            rows.append(
                {
                    "id": row["id"],
                    "Usage": usage,
                    "original_text": row["original_text"],
                    "masked_text": row["masked_text"],
                    "mask_spans_json": row["mask_spans_json"],
                }
            )
    return rows


def build_id_mapping_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "id": row["id"],
            "example_id": row["example_id"],
            "dialect": row["dialect"],
            "split": row["split"],
        }
        for row in rows
    ]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    dialect_counts = Counter(row["dialect"] for row in rows)
    bucket_counts = Counter(mask_count_bucket(int(row.get("mask_count") or "0")) for row in rows)
    return {
        "row_count": len(rows),
        "dialect_counts": dict(sorted(dialect_counts.items())),
        "mask_count_buckets": dict(sorted(bucket_counts.items())),
    }


def prepare_kaggle_dataset(
    input_csv: Path,
    output_dir: Path,
    seed: int = 17,
    public_fraction: float = 0.1,
    private_fraction: float = 0.1,
) -> Dict[str, object]:
    rows = load_rows(input_csv)
    supported_masked_rows, unmasked_rows, excluded_rows = classify_rows(rows)
    masked_train_rows, public_rows, private_rows = split_masked_rows(
        supported_masked_rows,
        seed=seed,
        public_fraction=public_fraction,
        private_fraction=private_fraction,
    )

    shuffled_train_source_rows = shuffle_rows([*masked_train_rows, *unmasked_rows], seed=seed + 101)
    shuffled_test_source_rows = shuffle_rows([*public_rows, *private_rows], seed=seed + 202)

    train_example_ids = {row["example_id"] for row in masked_train_rows}
    public_example_ids = {row["example_id"] for row in public_rows}
    private_example_ids = {row["example_id"] for row in private_rows}

    participant_train_rows_with_ids = assign_split_ids(
        shuffled_train_source_rows,
        prefix="train",
        split_name="train",
    )
    combined_test_rows = assign_split_ids(
        shuffled_test_source_rows,
        prefix="test",
        split_name="test",
    )

    masked_train_rows = [
        row for row in participant_train_rows_with_ids if row["example_id"] in train_example_ids and int(row["mask_count"]) > 0
    ]
    unmasked_rows = [
        row for row in participant_train_rows_with_ids if int(row["mask_count"]) == 0
    ]
    public_rows = [row for row in combined_test_rows if row["example_id"] in public_example_ids]
    private_rows = [row for row in combined_test_rows if row["example_id"] in private_example_ids]

    participant_train_rows = sorted(
        build_train_rows([*masked_train_rows, *unmasked_rows]),
        key=lambda row: row["id"],
    )
    participant_test_rows = build_test_rows(combined_test_rows)
    sample_submission_rows = [{"id": row["id"], "masked_text": ""} for row in participant_test_rows]
    public_label_rows = build_label_rows(public_rows)
    private_label_rows = build_label_rows(private_rows)
    solution_rows = build_solution_rows(public_rows, private_rows)
    id_mapping_rows = build_id_mapping_rows([*masked_train_rows, *unmasked_rows, *public_rows, *private_rows])

    write_csv(output_dir / "train.csv", TRAIN_COLUMNS, participant_train_rows)
    write_csv(output_dir / "test.csv", TEST_COLUMNS, participant_test_rows)
    write_csv(output_dir / "sample_submission.csv", SUBMISSION_COLUMNS, sample_submission_rows)
    write_csv(output_dir / "public_labels.csv", LABEL_COLUMNS, public_label_rows)
    write_csv(output_dir / "private_labels.csv", LABEL_COLUMNS, private_label_rows)
    write_csv(output_dir / "solution.csv", SOLUTION_COLUMNS, solution_rows)
    write_csv(output_dir / "id_mapping.csv", ID_MAPPING_COLUMNS, id_mapping_rows)
    write_metric_artifact(output_dir / "metric.py")

    excluded_fieldnames = list(rows[0].keys()) + ["exclusion_reason"] if rows else ["exclusion_reason"]
    write_csv(output_dir / "excluded_rows.csv", excluded_fieldnames, excluded_rows)

    summary = {
        "seed": seed,
        "input_csv": str(input_csv),
        "public_fraction": public_fraction,
        "private_fraction": private_fraction,
        "approved_rows_total": sum(1 for row in rows if (row.get("annotation_status") or "").strip() == "approved"),
        "participant_train_columns": TRAIN_COLUMNS,
        "participant_test_columns": TEST_COLUMNS,
        "sample_submission_columns": SUBMISSION_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "solution_columns": SOLUTION_COLUMNS,
        "solution_split_column": "Usage",
        "metric_artifact": "metric.py",
        "train_id_format": "train_XXXXXX",
        "test_id_format": "test_XXXXXX",
        "participant_files_hide_dialect": True,
        "supported_masked_summary": summarize_rows(supported_masked_rows),
        "unmasked_train_only_summary": summarize_rows(unmasked_rows),
        "masked_train_summary": summarize_rows(masked_train_rows),
        "public_test_summary": summarize_rows(public_rows),
        "private_test_summary": summarize_rows(private_rows),
        "combined_test_row_count": len(combined_test_rows),
        "train_row_count": len(participant_train_rows),
        "excluded_row_count": len(excluded_rows),
        "excluded_reasons": dict(
            sorted(Counter(row["exclusion_reason"] for row in excluded_rows).items())
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    prepare_kaggle_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        public_fraction=args.public_fraction,
        private_fraction=args.private_fraction,
    )


if __name__ == "__main__":
    main()
