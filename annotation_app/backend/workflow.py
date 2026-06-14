#!/usr/bin/env python3
"""Shared services for the annotation coordinator and CLI scripts."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "app_state" / "config.json"

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

PLACEHOLDER_RE = re.compile(r"\[(?P<mask_type>[A-Z]+)_(?P<index>\d+)\]")


@dataclass
class ProjectPaths:
    project_id: str
    source_chunks_dir: Path
    source_manifest_path: Path
    accepted_dir: Path
    merged_dir: Path
    session_state_path: Path
    prompt_template_path: Path
    masking_guidelines_path: Path
    dataset_mode: str
    auto_advance: bool
    auto_merge: bool
    allow_pending_accept: bool


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_config(config_path: Path | None = None) -> Dict[str, object]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: Dict[str, object], config_path: Path | None = None) -> None:
    path = config_path or DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def get_project_paths(project_id: str, config: Dict[str, object] | None = None) -> ProjectPaths:
    config = config or load_config()
    projects = config.get("projects", {})
    if project_id not in projects:
        raise KeyError(f"Unknown project: {project_id}")
    project = projects[project_id]
    workspace_dir = _resolve_path(project["workspace_dir"])
    return ProjectPaths(
        project_id=project_id,
        source_chunks_dir=_resolve_path(project["source_chunks_dir"]),
        source_manifest_path=_resolve_path(project["source_manifest_path"]),
        accepted_dir=workspace_dir / "accepted",
        merged_dir=workspace_dir / "merged",
        session_state_path=workspace_dir / "session_state.json",
        prompt_template_path=_resolve_path(project["prompt_template_path"]),
        masking_guidelines_path=_resolve_path(project["masking_guidelines_path"]),
        dataset_mode=project.get("dataset_mode", project_id),
        auto_advance=bool(project.get("auto_advance", True)),
        auto_merge=bool(project.get("auto_merge", True)),
        allow_pending_accept=bool(project.get("allow_pending_accept", False)),
    )


def ensure_workspace(paths: ProjectPaths) -> None:
    paths.accepted_dir.mkdir(parents=True, exist_ok=True)
    paths.merged_dir.mkdir(parents=True, exist_ok=True)
    paths.session_state_path.parent.mkdir(parents=True, exist_ok=True)
    if not paths.session_state_path.exists():
        save_session_state(
            paths,
            {
                "project_id": paths.project_id,
                "current_chunk_id": None,
                "accepted_chunks": [],
                "last_merge_summary": None,
            },
        )


def load_session_state(paths: ProjectPaths) -> Dict[str, object]:
    ensure_workspace(paths)
    with paths.session_state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_session_state(paths: ProjectPaths, state: Dict[str, object]) -> None:
    ensure_workspace(paths)
    paths.session_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prompt_template(paths: ProjectPaths) -> str:
    return paths.prompt_template_path.read_text(encoding="utf-8")


def load_masking_guidelines(paths: ProjectPaths) -> str:
    return paths.masking_guidelines_path.read_text(encoding="utf-8")


def load_manifest_rows(paths: ProjectPaths) -> List[Dict[str, str]]:
    with paths.source_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_chunk_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_pasted_csv_text(csv_text: str) -> str:
    text = csv_text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if not lines:
        return text

    first_line = lines[0].strip()
    last_line = lines[-1].strip()
    if not first_line.startswith("```") or last_line != "```":
        return text

    return "\n".join(lines[1:-1]).strip()


def load_chunk_rows_from_text(csv_text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    buffer = io.StringIO(normalize_pasted_csv_text(csv_text))
    reader = csv.DictReader(buffer)
    fieldnames = reader.fieldnames or []
    return fieldnames, list(reader)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dump_rows_to_csv_text(rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_row(row: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    example_id = row.get("example_id") or "<missing_example_id>"

    def add_error(message: str) -> None:
        errors.append(f"{example_id}: {message}")

    status = (row.get("annotation_status") or "").strip()
    if status != "approved":
        status_label = repr(status) if status else "missing"
        add_error(
            "annotation_status must be 'approved' for a valid merged row; "
            f"received {status_label}"
        )

    masked_text = row.get("masked_text") or ""
    spans_raw = row.get("mask_spans_json") or "[]"
    mask_count_raw = row.get("mask_count") or "0"

    try:
        spans = json.loads(spans_raw)
        if not isinstance(spans, list):
            raise ValueError("mask_spans_json must be a list")
    except Exception as exc:
        return [f"{example_id}: mask_spans_json is not valid JSON list data: {exc}"]

    try:
        mask_count = int(mask_count_raw)
    except ValueError:
        add_error(f"mask_count must be an integer; received {mask_count_raw!r}")
        mask_count = -1

    placeholders = [match.group(0) for match in PLACEHOLDER_RE.finditer(masked_text)]
    if len(placeholders) != len(spans):
        add_error(
            "placeholder count does not match mask_spans_json length; "
            f"found {len(placeholders)} placeholder(s) in masked_text and {len(spans)} span object(s)"
        )
    if mask_count != len(spans):
        add_error(
            "mask_count does not match mask_spans_json length; "
            f"mask_count={mask_count} while spans={len(spans)}"
        )

    original_text = row.get("original_text") or ""
    seen_by_surface: Dict[Tuple[str, str], str] = {}
    indices_by_type: Dict[str, set[int]] = {}

    for span_index, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            add_error(f"span #{span_index} must be an object; received {type(span).__name__}")
            continue

        missing_keys = {"start", "end", "placeholder", "surface_form", "mask_type"} - set(span)
        if missing_keys:
            add_error(
                f"span #{span_index} is missing required key(s): {', '.join(sorted(missing_keys))}"
            )
            continue

        start = span["start"]
        end = span["end"]
        placeholder = span["placeholder"]
        surface_form = span["surface_form"]
        mask_type = span["mask_type"]

        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            add_error(
                f"span #{span_index} has invalid offsets start={start!r}, end={end!r}; "
                "expected non-negative integers with end >= start"
            )
            continue
        if end > len(original_text):
            add_error(
                f"span #{span_index} ends outside original_text; end={end}, text_length={len(original_text)}"
            )
            continue
        if original_text[start:end] != surface_form:
            add_error(
                f"span #{span_index} surface_form mismatch for {placeholder}: "
                f"original_text[{start}:{end}]={original_text[start:end]!r}, surface_form={surface_form!r}"
            )
        if placeholder not in masked_text:
            add_error(
                f"span #{span_index} placeholder {placeholder!r} does not appear in masked_text"
            )

        placeholder_match = PLACEHOLDER_RE.fullmatch(placeholder)
        if not placeholder_match:
            add_error(
                f"span #{span_index} placeholder {placeholder!r} has invalid format; "
                "expected values like [PERSON_1] or [LOC_2]"
            )
            continue

        placeholder_type = placeholder_match.group("mask_type")
        placeholder_index = int(placeholder_match.group("index"))
        if placeholder_type != mask_type:
            add_error(
                f"span #{span_index} placeholder type mismatch: placeholder={placeholder_type!r}, "
                f"mask_type={mask_type!r}"
            )

        seen_key = (mask_type, surface_form)
        previous_placeholder = seen_by_surface.get(seen_key)
        if previous_placeholder and previous_placeholder != placeholder:
            add_error(
                f"repeated surface form {surface_form!r} for mask_type {mask_type!r} uses "
                f"inconsistent placeholders {previous_placeholder!r} and {placeholder!r}"
            )
        seen_by_surface[seen_key] = placeholder
        indices_by_type.setdefault(mask_type, set()).add(placeholder_index)

    for mask_type, indices in indices_by_type.items():
        expected = set(range(1, len(indices) + 1))
        if indices != expected:
            add_error(
                f"placeholder numbering for mask_type {mask_type!r} is not contiguous; "
                f"found indices {sorted(indices)}, expected {sorted(expected)}"
            )

    return errors


def merge_chunks(chunks_dir: Path, output_dir: Path) -> Tuple[int, int]:
    rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    seen_ids: set[str] = set()

    for chunk_path in sorted(chunks_dir.glob("chunk_*.csv")):
        for row in load_chunk_rows(chunk_path):
            example_id = row.get("example_id") or ""
            row_errors = validate_row(row)
            if example_id in seen_ids:
                row_errors.append("duplicate_example_id")
            if row_errors:
                errors.append(
                    {
                        "chunk_file": chunk_path.name,
                        "example_id": example_id,
                        "errors": " | ".join(row_errors),
                    }
                )
                continue
            seen_ids.add(example_id)
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "masked_dataset_merged.csv", rows, OUTPUT_COLUMNS)
    with (output_dir / "masked_dataset_merged.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "merge_errors.csv", errors, ["chunk_file", "example_id", "errors"])
    summary = {
        "approved_rows": len(rows),
        "error_rows": len(errors),
        "dialect_counts": dict(sorted(Counter(row["dialect"] for row in rows).items())),
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(rows), len(errors)


def get_chunk_source_path(paths: ProjectPaths, chunk_file_name: str) -> Path:
    return paths.source_chunks_dir / chunk_file_name


def get_accepted_chunk_path(paths: ProjectPaths, chunk_file_name: str) -> Path:
    return paths.accepted_dir / chunk_file_name


def get_baseline_chunk_path(paths: ProjectPaths, chunk_file_name: str) -> Path:
    accepted = get_accepted_chunk_path(paths, chunk_file_name)
    return accepted if accepted.exists() else get_chunk_source_path(paths, chunk_file_name)


def _classify_errors(errors: List[str]) -> str:
    if not errors:
        return "approved"
    if all("annotation_status must be 'approved'" in error for error in errors):
        return "pending"
    return "invalid"


def _chunk_state(paths: ProjectPaths, row: Dict[str, str]) -> Dict[str, object]:
    chunk_file = row["file_name"]
    accepted_path = get_accepted_chunk_path(paths, chunk_file)
    merge_summary_path = paths.merged_dir / "merge_summary.json"
    session_state = load_session_state(paths)
    current_chunk_id = session_state.get("current_chunk_id")

    if accepted_path.exists():
        state = "accepted"
    elif str(row["chunk_id"]) == str(current_chunk_id):
        state = "in_progress"
    else:
        state = row.get("status", "pending")

    return {
        "chunk_id": int(row["chunk_id"]),
        "file_name": chunk_file,
        "row_count": int(row["row_count"]),
        "dialect_counts_json": row["dialect_counts_json"],
        "sha256": row["sha256"],
        "state": state,
        "source_path": str(get_chunk_source_path(paths, chunk_file)),
        "accepted_path": str(accepted_path),
        "has_accepted_copy": accepted_path.exists(),
        "included_in_last_merge": accepted_path.exists() and merge_summary_path.exists(),
    }


def list_chunks(project_id: str, config: Dict[str, object] | None = None) -> List[Dict[str, object]]:
    paths = get_project_paths(project_id, config)
    return [_chunk_state(paths, row) for row in load_manifest_rows(paths)]


def get_next_chunk(project_id: str, config: Dict[str, object] | None = None) -> Optional[Dict[str, object]]:
    for row in list_chunks(project_id, config):
        if row["state"] not in {"accepted", "completed"}:
            return row
    return None


def get_chunk_detail(project_id: str, chunk_id: int, config: Dict[str, object] | None = None) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    for row in load_manifest_rows(paths):
        if int(row["chunk_id"]) == int(chunk_id):
            state = _chunk_state(paths, row)
            baseline_path = get_baseline_chunk_path(paths, row["file_name"])
            state["rows"] = load_chunk_rows(baseline_path)
            state["prompt"] = build_prompt(project_id, chunk_id, config)
            return state
    raise KeyError(f"Unknown chunk id: {chunk_id}")


def build_prompt(project_id: str, chunk_id: int, config: Dict[str, object] | None = None) -> str:
    paths = get_project_paths(project_id, config)
    template = load_prompt_template(paths)
    guidelines = load_masking_guidelines(paths)
    for row in load_manifest_rows(paths):
        if int(row["chunk_id"]) == int(chunk_id):
            chunk_path = get_baseline_chunk_path(paths, row["file_name"])
            chunk_rows = load_chunk_rows(chunk_path)
            chunk_csv = dump_rows_to_csv_text(chunk_rows, OUTPUT_COLUMNS)
            return template.format(
                project_id=project_id,
                chunk_id=row["chunk_id"],
                chunk_file=row["file_name"],
                chunk_path=chunk_path,
                row_count=row["row_count"],
                dataset_mode=paths.dataset_mode,
                guidelines=guidelines.strip(),
                chunk_csv=chunk_csv.rstrip(),
            )
    raise KeyError(f"Unknown chunk id: {chunk_id}")


def _compare_rows(
    baseline_rows: List[Dict[str, str]],
    imported_rows: List[Dict[str, str]],
) -> Tuple[int, List[Dict[str, object]]]:
    baseline_by_id = {row["example_id"]: row for row in baseline_rows}
    changed_rows: List[Dict[str, object]] = []
    changed_count = 0

    for row in imported_rows:
        example_id = row["example_id"]
        baseline = baseline_by_id[example_id]
        changed_columns = []
        for column in OUTPUT_COLUMNS:
            if str(baseline.get(column, "")) != str(row.get(column, "")):
                changed_columns.append(
                    {
                        "column": column,
                        "before": baseline.get(column, ""),
                        "after": row.get(column, ""),
                    }
                )
        if changed_columns:
            changed_count += 1
            changed_rows.append(
                {
                    "example_id": example_id,
                    "changed_columns": changed_columns,
                }
            )

    return changed_count, changed_rows


def preview_import(
    project_id: str,
    chunk_id: int,
    csv_text: str,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    manifest_rows = load_manifest_rows(paths)
    manifest_row = next((row for row in manifest_rows if int(row["chunk_id"]) == int(chunk_id)), None)
    if manifest_row is None:
        raise KeyError(f"Unknown chunk id: {chunk_id}")

    fieldnames, imported_rows = load_chunk_rows_from_text(csv_text)
    if fieldnames != OUTPUT_COLUMNS:
        return {
            "ok": False,
            "error": "header_mismatch",
            "expected_headers": OUTPUT_COLUMNS,
            "received_headers": fieldnames,
        }

    baseline_rows = load_chunk_rows(get_baseline_chunk_path(paths, manifest_row["file_name"]))
    baseline_ids = [row["example_id"] for row in baseline_rows]
    imported_ids = [row.get("example_id", "") for row in imported_rows]

    if len(imported_rows) != len(baseline_rows):
        return {
            "ok": False,
            "error": "row_count_mismatch",
            "expected_row_count": len(baseline_rows),
            "received_row_count": len(imported_rows),
        }

    if imported_ids != baseline_ids:
        return {
            "ok": False,
            "error": "example_id_mismatch",
            "expected_example_ids": baseline_ids,
            "received_example_ids": imported_ids,
        }

    validation_rows: List[Dict[str, object]] = []
    counts = Counter()
    for index, row in enumerate(imported_rows, start=1):
        errors = validate_row(row)
        status = _classify_errors(errors)
        counts[status] += 1
        validation_rows.append(
            {
                "row_number": index,
                "example_id": row["example_id"],
                "status": status,
                "errors": errors,
            }
        )

    changed_count, changed_rows = _compare_rows(baseline_rows, imported_rows)
    return {
        "ok": True,
        "chunk_id": chunk_id,
        "file_name": manifest_row["file_name"],
        "summary": {
            "row_count": len(imported_rows),
            "approved_rows": counts["approved"],
            "pending_rows": counts["pending"],
            "invalid_rows": counts["invalid"],
            "changed_rows": changed_count,
        },
        "validation_rows": validation_rows,
        "changed_rows": changed_rows,
    }


def accept_import(
    project_id: str,
    chunk_id: int,
    csv_text: str,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    preview = preview_import(project_id, chunk_id, csv_text, config)
    if not preview.get("ok"):
        return preview

    summary = preview["summary"]
    if summary["invalid_rows"] > 0:
        return {"ok": False, "error": "invalid_rows_present", "preview": preview}
    if summary["pending_rows"] > 0 and not paths.allow_pending_accept:
        return {"ok": False, "error": "pending_rows_present", "preview": preview}

    manifest_row = next(row for row in load_manifest_rows(paths) if int(row["chunk_id"]) == int(chunk_id))
    _, imported_rows = load_chunk_rows_from_text(csv_text)
    accepted_path = get_accepted_chunk_path(paths, manifest_row["file_name"])
    write_csv(accepted_path, imported_rows, OUTPUT_COLUMNS)

    state = load_session_state(paths)
    accepted_chunks = set(state.get("accepted_chunks", []))
    accepted_chunks.add(int(chunk_id))
    state["accepted_chunks"] = sorted(accepted_chunks)
    next_chunk = get_next_chunk(project_id, config)
    state["current_chunk_id"] = next_chunk["chunk_id"] if next_chunk else None
    save_session_state(paths, state)

    response: Dict[str, object] = {
        "ok": True,
        "accepted_path": str(accepted_path),
        "chunk_id": chunk_id,
        "auto_merge": paths.auto_merge,
    }

    if paths.auto_merge:
        approved_rows, error_rows = merge_chunks(paths.accepted_dir, paths.merged_dir)
        merge_summary = json.loads((paths.merged_dir / "merge_summary.json").read_text(encoding="utf-8"))
        state["last_merge_summary"] = merge_summary
        save_session_state(paths, state)
        response["merge_summary"] = {
            "approved_rows": approved_rows,
            "error_rows": error_rows,
            "summary": merge_summary,
        }

    if paths.auto_advance:
        next_chunk = get_next_chunk(project_id, config)
        response["next_chunk"] = next_chunk
        state["current_chunk_id"] = next_chunk["chunk_id"] if next_chunk else None
        save_session_state(paths, state)

    return response


def get_project_summary(project_id: str, config: Dict[str, object] | None = None) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    chunks = list_chunks(project_id, config)
    counts = Counter(chunk["state"] for chunk in chunks)
    session_state = load_session_state(paths)
    merge_summary_path = paths.merged_dir / "merge_summary.json"
    merge_summary = json.loads(merge_summary_path.read_text(encoding="utf-8")) if merge_summary_path.exists() else None
    return {
        "project_id": project_id,
        "dataset_mode": paths.dataset_mode,
        "total_chunks": len(chunks),
        "state_counts": dict(sorted(counts.items())),
        "current_chunk_id": session_state.get("current_chunk_id"),
        "next_chunk": get_next_chunk(project_id, config),
        "auto_merge": paths.auto_merge,
        "auto_advance": paths.auto_advance,
        "merge_summary": merge_summary,
    }
