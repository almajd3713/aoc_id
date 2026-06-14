#!/usr/bin/env python3
"""Build an annotation-ready masked Arabic dataset from MultiTrain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
NON_WORD_SPACE_RE = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
TATWEEL = "\u0640"

OUTPUT_COLUMNS = [
    "example_id",
    "source_split",
    "source_row_id",
    "dialect",
    "original_text",
    "normalized_text",
    "masked_text",
    "mask_spans_json",
    "mask_count",
    "annotation_status",
    "annotator_model",
    "notes",
]

MASK_TYPES = [
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
]


@dataclass(frozen=True)
class Candidate:
    source_row_id: str
    dialect: str
    original_text: str
    normalized_text: str
    example_id: str
    word_count: int
    char_count: int
    near_duplicate_key: str


def normalize_for_comparison(text: str) -> str:
    """Normalize Arabic text conservatively for dedupe and IDs."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace(TATWEEL, "")
    normalized = ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    normalized = normalized.replace("ؤ", "و").replace("ئ", "ي").replace("ى", "ي")
    normalized = normalized.replace("ة", "ه")
    normalized = NON_WORD_SPACE_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def tokenize(text: str) -> List[str]:
    return [token for token in text.split(" ") if token]


def build_near_duplicate_key(normalized_text: str) -> str:
    """Conservative key for obvious near-duplicates only."""
    tokens = tokenize(normalized_text)
    if not tokens:
        return ""

    anchor = tokens[:3] + tokens[-3:]
    deduped_tokens: List[str] = []
    previous = None
    for token in tokens:
        if token != previous:
            deduped_tokens.append(token)
        previous = token

    length_bucket = len(tokens) // 3
    key_tokens = deduped_tokens[:8]
    return "|".join(anchor + [str(length_bucket)] + key_tokens)


def build_example_id(dialect: str, normalized_text: str) -> str:
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:8]
    return f"mt_{dialect.lower()}_{digest}"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def is_usable_text(text: str, min_words: int) -> Tuple[bool, str]:
    normalized = normalize_for_comparison(text)
    if not normalized:
        return False, "empty_after_normalization"

    tokens = tokenize(normalized)
    if len(tokens) < min_words:
        return False, "too_short"

    arabic_chars = len(ARABIC_LETTER_RE.findall(normalized))
    if arabic_chars < max(8, len(normalized) // 4):
        return False, "too_little_arabic"

    if len(normalized) < 15:
        return False, "too_short_chars"

    return True, ""


def load_candidates(input_csv: Path, min_words: int) -> Tuple[List[Candidate], List[Dict[str, str]]]:
    accepted: List[Candidate] = []
    rejected: List[Dict[str, str]] = []
    seen_exact: Dict[str, Candidate] = {}
    seen_near: Dict[Tuple[str, str], Candidate] = {}

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_text = (row.get("text") or "").strip()
            dialect = (row.get("label") or "").strip()
            source_row_id = str(row.get("") or row.get("index") or row.get("id") or "")

            usable, reason = is_usable_text(raw_text, min_words=min_words)
            normalized_text = normalize_for_comparison(raw_text)

            if not dialect:
                rejected.append(
                    {
                        "source_row_id": source_row_id,
                        "dialect": dialect,
                        "original_text": raw_text,
                        "normalized_text": normalized_text,
                        "reason": "missing_dialect",
                    }
                )
                continue

            if not usable:
                rejected.append(
                    {
                        "source_row_id": source_row_id,
                        "dialect": dialect,
                        "original_text": raw_text,
                        "normalized_text": normalized_text,
                        "reason": reason,
                    }
                )
                continue

            exact_key = normalized_text
            if exact_key in seen_exact:
                rejected.append(
                    {
                        "source_row_id": source_row_id,
                        "dialect": dialect,
                        "original_text": raw_text,
                        "normalized_text": normalized_text,
                        "reason": "exact_duplicate",
                    }
                )
                continue

            near_key = build_near_duplicate_key(normalized_text)
            if near_key and (dialect, near_key) in seen_near:
                rejected.append(
                    {
                        "source_row_id": source_row_id,
                        "dialect": dialect,
                        "original_text": raw_text,
                        "normalized_text": normalized_text,
                        "reason": "near_duplicate",
                    }
                )
                continue

            candidate = Candidate(
                source_row_id=source_row_id,
                dialect=dialect,
                original_text=raw_text,
                normalized_text=normalized_text,
                example_id=build_example_id(dialect, normalized_text),
                word_count=len(tokenize(normalized_text)),
                char_count=len(normalized_text),
                near_duplicate_key=near_key,
            )
            accepted.append(candidate)
            seen_exact[exact_key] = candidate
            if near_key:
                seen_near[(dialect, near_key)] = candidate

    return accepted, rejected


def group_by_dialect(candidates: Iterable[Candidate]) -> Dict[str, List[Candidate]]:
    grouped: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.dialect].append(candidate)
    return grouped


def length_bucket(word_count: int) -> int:
    if word_count <= 8:
        return 0
    if word_count <= 16:
        return 1
    if word_count <= 28:
        return 2
    return 3


def stratified_sample(
    candidates: Sequence[Candidate],
    per_dialect_target: int,
    seed: int,
) -> List[Candidate]:
    grouped = group_by_dialect(candidates)
    if not grouped:
        return []

    feasible_target = min(per_dialect_target, min(len(rows) for rows in grouped.values()))
    rng = random.Random(seed)
    sampled: List[Candidate] = []

    for dialect, rows in sorted(grouped.items()):
        rows = list(rows)
        rng.shuffle(rows)

        buckets: Dict[int, List[Candidate]] = defaultdict(list)
        for row in rows:
            buckets[length_bucket(row.word_count)].append(row)

        ordered_buckets = sorted(buckets.items())
        bucket_targets = {bucket: int(feasible_target * len(items) / len(rows)) for bucket, items in ordered_buckets}
        assigned = sum(bucket_targets.values())
        remainder = feasible_target - assigned

        ranked_buckets = sorted(
            ordered_buckets,
            key=lambda item: (len(item[1]) - bucket_targets[item[0]], len(item[1])),
            reverse=True,
        )
        for bucket, items in ranked_buckets:
            if remainder <= 0:
                break
            if bucket_targets[bucket] < len(items):
                bucket_targets[bucket] += 1
                remainder -= 1

        dialect_sample: List[Candidate] = []
        for bucket, items in ordered_buckets:
            dialect_sample.extend(items[: bucket_targets[bucket]])

        if len(dialect_sample) < feasible_target:
            chosen_ids = {row.example_id for row in dialect_sample}
            leftovers = [row for row in rows if row.example_id not in chosen_ids]
            dialect_sample.extend(leftovers[: feasible_target - len(dialect_sample)])

        rng.shuffle(dialect_sample)
        sampled.extend(dialect_sample[:feasible_target])

    return sampled


def interleave_by_dialect(candidates: Sequence[Candidate], seed: int) -> List[Candidate]:
    grouped = group_by_dialect(candidates)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)

    ordered: List[Candidate] = []
    dialects = sorted(grouped)
    while any(grouped[dialect] for dialect in dialects):
        for dialect in dialects:
            if grouped[dialect]:
                ordered.append(grouped[dialect].pop())
    return ordered


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def candidate_row(candidate: Candidate) -> Dict[str, object]:
    return {
        "example_id": candidate.example_id,
        "source_split": "train",
        "source_row_id": candidate.source_row_id,
        "dialect": candidate.dialect,
        "original_text": candidate.original_text,
        "normalized_text": candidate.normalized_text,
        "masked_text": "",
        "mask_spans_json": "[]",
        "mask_count": 0,
        "annotation_status": "pending",
        "annotator_model": "",
        "notes": "",
    }


def write_outputs(
    clean_candidates: Sequence[Candidate],
    sampled_candidates: Sequence[Candidate],
    rejected_rows: Sequence[Dict[str, str]],
    output_dir: Path,
    chunk_size: int,
    pilot_chunks: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = output_dir / "clean_candidates.csv"
    write_csv(clean_path, (candidate_row(row) for row in clean_candidates), OUTPUT_COLUMNS)

    rejection_path = output_dir / "rejections.csv"
    write_csv(
        rejection_path,
        rejected_rows,
        ["source_row_id", "dialect", "original_text", "normalized_text", "reason"],
    )

    ordered = interleave_by_dialect(sampled_candidates, seed=17)
    chunks_dir = output_dir / "chunks"
    manifest_rows: List[Dict[str, object]] = []
    for chunk_index, start in enumerate(range(0, len(ordered), chunk_size), start=1):
        rows = ordered[start : start + chunk_size]
        chunk_name = f"chunk_{chunk_index:04d}.csv"
        chunk_path = chunks_dir / chunk_name
        write_csv(chunk_path, (candidate_row(row) for row in rows), OUTPUT_COLUMNS)

        counts = Counter(row.dialect for row in rows)
        manifest_rows.append(
            {
                "chunk_id": chunk_index,
                "file_name": chunk_name,
                "row_count": len(rows),
                "dialect_counts_json": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
                "sha256": sha256_file(chunk_path),
                "status": "pending",
            }
        )

    manifest_path = output_dir / "chunk_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        ["chunk_id", "file_name", "row_count", "dialect_counts_json", "sha256", "status"],
    )

    pilot_manifest_rows = manifest_rows[:pilot_chunks]
    pilot_dir = output_dir / "pilot"
    pilot_chunks_dir = pilot_dir / "chunks"
    for manifest_row in pilot_manifest_rows:
        chunk_path = chunks_dir / str(manifest_row["file_name"])
        pilot_chunk_path = pilot_chunks_dir / str(manifest_row["file_name"])
        pilot_chunk_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_chunk_path.write_bytes(chunk_path.read_bytes())

    pilot_manifest_path = pilot_dir / "pilot_manifest.csv"
    write_csv(
        pilot_manifest_path,
        pilot_manifest_rows,
        ["chunk_id", "file_name", "row_count", "dialect_counts_json", "sha256", "status"],
    )

    summary = {
        "clean_candidate_count": len(clean_candidates),
        "sampled_count": len(sampled_candidates),
        "chunk_count": len(manifest_rows),
        "chunk_size": chunk_size,
        "pilot_chunk_count": len(pilot_manifest_rows),
        "pilot_row_count": sum(int(row["row_count"]) for row in pilot_manifest_rows),
        "dialect_counts": dict(sorted(Counter(row.dialect for row in sampled_candidates).items())),
        "rejection_counts": dict(sorted(Counter(row["reason"] for row in rejected_rows).items())),
        "mask_types": MASK_TYPES,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/train/MultiTrain.Shuffled.csv",
        type=Path,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/masked_dataset",
        type=Path,
        help="Output directory for clean pool, chunks, and manifests.",
    )
    parser.add_argument(
        "--per-dialect-target",
        default=4000,
        type=int,
        help="Requested sample count per dialect bucket.",
    )
    parser.add_argument(
        "--chunk-size",
        default=80,
        type=int,
        help="Rows per annotation chunk file.",
    )
    parser.add_argument(
        "--min-words",
        default=5,
        type=int,
        help="Minimum normalized token count to keep a row.",
    )
    parser.add_argument(
        "--seed",
        default=13,
        type=int,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--pilot-chunks",
        default=2,
        type=int,
        help="Number of initial chunk files to copy into a pilot package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_candidates, rejected_rows = load_candidates(args.input, min_words=args.min_words)
    sampled_candidates = stratified_sample(
        clean_candidates,
        per_dialect_target=args.per_dialect_target,
        seed=args.seed,
    )
    write_outputs(
        clean_candidates=clean_candidates,
        sampled_candidates=sampled_candidates,
        rejected_rows=rejected_rows,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        pilot_chunks=args.pilot_chunks,
    )


if __name__ == "__main__":
    main()
