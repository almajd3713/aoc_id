#!/usr/bin/env python3
"""Shared services for the annotation coordinator and CLI scripts."""

from __future__ import annotations

import csv
import io
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
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
BACKLOG_COLUMNS = [
    "example_id",
    "source_row_id",
    "dialect",
    "original_text",
    "normalized_text",
    "chunk_id",
    "chunk_file_name",
    "attempt_count",
    "round_number",
    "last_attempted_masked_text",
    "latest_errors",
    "dropped_at",
]

AGENT_COLUMNS = [
    "original_text",
    "masked_text",
]
AGENT_ROW_ID_COLUMN = "row_id"
AGENT_COLUMNS_WITH_ROW_ID = [
    AGENT_ROW_ID_COLUMN,
    *AGENT_COLUMNS,
]
ROW_MATCHING_PREFER_ROW_ID = "prefer_row_id"
ROW_MATCHING_STRICT_ROW_ID = "strict_row_id"
ROW_MATCHING_STRICT_ORIGINAL_TEXT = "strict_original_text"
SUPPORTED_ROW_MATCHING = {
    ROW_MATCHING_PREFER_ROW_ID,
    ROW_MATCHING_STRICT_ROW_ID,
    ROW_MATCHING_STRICT_ORIGINAL_TEXT,
}
SUPPORTED_AGENT_IMPORT_SCHEMAS = {"original_masked_v1"}

PLACEHOLDER_RE = re.compile(r"\[(?P<mask_type>[A-Z]+)_(?P<index>\d+)\]")

AGENT_GUIDELINES = """Agent-facing masking rules:
- Return only two columns: `original_text` and `masked_text`.
- If the prompt includes a `row_id` column, treat it as reference context. Keep it unchanged if you return it.
- Keep `original_text` exactly unchanged.
- Fill only `masked_text`; all metadata will be generated automatically later.
- Use typed placeholders like `[PERSON_1]`, `[LOC_1]`, `[ORG_1]`, `[DATE_1]`, `[TIME_1]`, `[NUM_1]`.
- Restart numbering in each row.
- Keep numbering contiguous within each mask type in a row.
- Reuse the same placeholder when the same entity appears again in the same row.
- Preserve punctuation and all unmasked text exactly.
- Do not add or remove rows.
- Do not add extra columns.
- If uncertain, include `[UNCERTAIN]` or `[REVIEW]` somewhere in `masked_text`.

What to mask:
- PERSON: people names and person-identifying nicknames
- LOC: countries, cities, places, landmarks
- ORG: organizations, teams, companies, ministries, schools, newspapers, parties
- DATE: dates and date expressions
- TIME: times
- NUM: important numbers, money amounts, ages, rankings, counts when they function like entity-style spans
- HANDLE: usernames or tagged accounts
- URL: links
- EMAIL: email addresses
- PHONE: phone numbers
- ID: account, document, or reference numbers

What not to mask:
- general topic words
- sentiment words
- ordinary nouns and verbs
- dialect markers by themselves

Example:
original: `محمد قابل أحمد في الرباط ثم اتصل محمد بأحمد`
masked: `[PERSON_1] قابل [PERSON_2] في [LOC_1] ثم اتصل [PERSON_1] ب[PERSON_2]`
"""


@dataclass
class ProjectPaths:
    project_id: str
    base_source_chunks_dir: Path
    base_source_manifest_path: Path
    source_chunks_dir: Path
    source_manifest_path: Path
    accepted_dir: Path
    working_dir: Path
    merged_dir: Path
    backlog_path: Path
    refill_rounds_dir: Path
    session_state_path: Path
    prompt_template_path: Path
    masking_guidelines_path: Path
    dataset_mode: str
    auto_advance: bool
    auto_merge: bool
    allow_pending_accept: bool
    agent_import_schema: str
    row_matching: str
    uncertainty_markers: List[str]
    include_row_id_in_prompt: bool
    validate_original_text_with_row_id: bool
    active_round: str


class ConfigValidationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("Invalid config")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_config(config_path: Path | None = None) -> Dict[str, object]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return normalize_config(json.load(handle))


def save_config(config: Dict[str, object], config_path: Path | None = None) -> None:
    path = config_path or DEFAULT_CONFIG_PATH
    normalized = normalize_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_row_matching_value(value: object) -> str:
    row_matching = str(value or ROW_MATCHING_PREFER_ROW_ID)
    if row_matching == "exact_original_text":
        return ROW_MATCHING_STRICT_ORIGINAL_TEXT
    return row_matching


def normalize_config(config: Dict[str, object]) -> Dict[str, object]:
    errors: List[str] = []
    if not isinstance(config, dict):
        raise ConfigValidationError(["Config must be an object."])

    projects_raw = config.get("projects")
    if not isinstance(projects_raw, dict) or not projects_raw:
        raise ConfigValidationError(["Config must include a non-empty projects object."])

    normalized_projects: Dict[str, object] = {}
    required_project_paths = [
        "workspace_dir",
        "source_chunks_dir",
        "source_manifest_path",
        "base_source_chunks_dir",
        "base_source_manifest_path",
        "prompt_template_path",
        "masking_guidelines_path",
    ]

    for project_id, project_raw in projects_raw.items():
        if not isinstance(project_raw, dict):
            errors.append(f"projects.{project_id} must be an object.")
            continue

        project = dict(project_raw)
        project["base_source_chunks_dir"] = str(
            project.get("base_source_chunks_dir") or project.get("source_chunks_dir") or ""
        )
        project["base_source_manifest_path"] = str(
            project.get("base_source_manifest_path") or project.get("source_manifest_path") or ""
        )
        for required_key in required_project_paths:
            value = project.get(required_key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"projects.{project_id}.{required_key} must be a non-empty string.")

        label = project.get("label", project_id)
        if not isinstance(label, str) or not label.strip():
            errors.append(f"projects.{project_id}.label must be a non-empty string.")

        dataset_mode = project.get("dataset_mode", project_id)
        if not isinstance(dataset_mode, str) or not dataset_mode.strip():
            errors.append(f"projects.{project_id}.dataset_mode must be a non-empty string.")

        row_matching = _normalize_row_matching_value(project.get("row_matching", ROW_MATCHING_PREFER_ROW_ID))
        if row_matching not in SUPPORTED_ROW_MATCHING:
            errors.append(
                f"projects.{project_id}.row_matching must be one of {sorted(SUPPORTED_ROW_MATCHING)}."
            )

        agent_import_schema = str(project.get("agent_import_schema", "original_masked_v1"))
        if agent_import_schema not in SUPPORTED_AGENT_IMPORT_SCHEMAS:
            errors.append(
                f"projects.{project_id}.agent_import_schema must be one of {sorted(SUPPORTED_AGENT_IMPORT_SCHEMAS)}."
            )

        uncertainty_markers = project.get("uncertainty_markers", ["[UNCERTAIN]", "[REVIEW]"])
        if not isinstance(uncertainty_markers, list) or any(
            not isinstance(marker, str) or not marker.strip() for marker in uncertainty_markers
        ):
            errors.append(
                f"projects.{project_id}.uncertainty_markers must be a list of non-empty strings."
            )
            uncertainty_markers = ["[UNCERTAIN]", "[REVIEW]"]

        normalized_projects[project_id] = {
            **project,
            "label": label,
            "dataset_mode": dataset_mode,
            "auto_advance": bool(project.get("auto_advance", True)),
            "auto_merge": bool(project.get("auto_merge", True)),
            "allow_pending_accept": bool(project.get("allow_pending_accept", False)),
            "include_row_id_in_prompt": bool(project.get("include_row_id_in_prompt", False)),
            "agent_import_schema": agent_import_schema,
            "row_matching": row_matching,
            "uncertainty_markers": [marker.strip() for marker in uncertainty_markers],
            "validate_original_text_with_row_id": bool(
                project.get("validate_original_text_with_row_id", True)
            ),
            "active_round": str(project.get("active_round", "base") or "base"),
        }

    default_project = config.get("default_project")
    if not isinstance(default_project, str) or not default_project:
        errors.append("default_project must be a non-empty string.")
    elif default_project not in normalized_projects:
        errors.append("default_project must match a key in projects.")

    ui = config.get("ui", {})
    if ui is None:
        ui = {}
    if not isinstance(ui, dict):
        errors.append("ui must be an object when present.")
        ui = {}

    if errors:
        raise ConfigValidationError(errors)

    return {
        **config,
        "default_project": default_project,
        "ui": {
            **ui,
            "copy_compact_prompt": bool(ui.get("copy_compact_prompt", True)),
        },
        "projects": normalized_projects,
    }


def get_project_paths(project_id: str, config: Dict[str, object] | None = None) -> ProjectPaths:
    config = config or load_config()
    projects = config.get("projects", {})
    if project_id not in projects:
        raise KeyError(f"Unknown project: {project_id}")
    project = projects[project_id]
    workspace_dir = _resolve_path(project["workspace_dir"])
    row_matching = _normalize_row_matching_value(project.get("row_matching", ROW_MATCHING_PREFER_ROW_ID))
    if row_matching not in SUPPORTED_ROW_MATCHING:
        row_matching = ROW_MATCHING_PREFER_ROW_ID
    return ProjectPaths(
        project_id=project_id,
        base_source_chunks_dir=_resolve_path(project["base_source_chunks_dir"]),
        base_source_manifest_path=_resolve_path(project["base_source_manifest_path"]),
        source_chunks_dir=_resolve_path(project["source_chunks_dir"]),
        source_manifest_path=_resolve_path(project["source_manifest_path"]),
        accepted_dir=workspace_dir / "accepted",
        working_dir=workspace_dir / "working",
        merged_dir=workspace_dir / "merged",
        backlog_path=workspace_dir / "backlog_rows.csv",
        refill_rounds_dir=workspace_dir / "refill_rounds",
        session_state_path=workspace_dir / "session_state.json",
        prompt_template_path=_resolve_path(project["prompt_template_path"]),
        masking_guidelines_path=_resolve_path(project["masking_guidelines_path"]),
        dataset_mode=project.get("dataset_mode", project_id),
        auto_advance=bool(project.get("auto_advance", True)),
        auto_merge=bool(project.get("auto_merge", True)),
        allow_pending_accept=bool(project.get("allow_pending_accept", False)),
        agent_import_schema=str(project.get("agent_import_schema", "original_masked_v1")),
        row_matching=row_matching,
        uncertainty_markers=list(project.get("uncertainty_markers", ["[UNCERTAIN]", "[REVIEW]"])),
        include_row_id_in_prompt=bool(project.get("include_row_id_in_prompt", False)),
        validate_original_text_with_row_id=bool(project.get("validate_original_text_with_row_id", True)),
        active_round=str(project.get("active_round", "base") or "base"),
    )


def ensure_workspace(paths: ProjectPaths) -> None:
    paths.accepted_dir.mkdir(parents=True, exist_ok=True)
    paths.working_dir.mkdir(parents=True, exist_ok=True)
    paths.merged_dir.mkdir(parents=True, exist_ok=True)
    paths.refill_rounds_dir.mkdir(parents=True, exist_ok=True)
    paths.session_state_path.parent.mkdir(parents=True, exist_ok=True)
    if not paths.session_state_path.exists():
        save_session_state(
            paths,
            {
                "project_id": paths.project_id,
                "current_chunk_id": None,
                "accepted_chunks": [],
                "last_merge_summary": None,
                "invalid_retry_rows": [],
                "chunk_metrics": {},
            },
        )


def load_session_state(paths: ProjectPaths) -> Dict[str, object]:
    ensure_workspace(paths)
    with paths.session_state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("invalid_retry_rows", [])
    state.setdefault("chunk_metrics", {})
    return state


def save_session_state(paths: ProjectPaths, state: Dict[str, object]) -> None:
    ensure_workspace(paths)
    state.setdefault("invalid_retry_rows", [])
    state.setdefault("chunk_metrics", {})
    paths.session_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_backlog_rows(paths: ProjectPaths) -> List[Dict[str, str]]:
    ensure_workspace(paths)
    if not paths.backlog_path.exists():
        return []
    return load_chunk_rows(paths.backlog_path)


def save_backlog_rows(paths: ProjectPaths, rows: Sequence[Dict[str, object]]) -> None:
    write_csv(paths.backlog_path, rows, BACKLOG_COLUMNS)


def _list_refill_round_numbers(paths: ProjectPaths) -> List[int]:
    round_numbers = []
    for round_dir in paths.refill_rounds_dir.glob("round_*"):
        try:
            round_numbers.append(int(round_dir.name.split("_")[-1]))
        except ValueError:
            continue
    return sorted(round_numbers)


def _get_round_number(paths: ProjectPaths) -> int:
    if not paths.refill_rounds_dir.exists():
        return 0
    round_numbers = _list_refill_round_numbers(paths)
    return max(round_numbers, default=0)


def _get_dataset_root(paths: ProjectPaths) -> Path:
    source_root = paths.base_source_chunks_dir
    if source_root.parent.name == "pilot":
        return source_root.parent.parent
    return source_root.parent


def _load_clean_candidate_rows(paths: ProjectPaths) -> List[Dict[str, str]]:
    clean_path = _get_dataset_root(paths) / "clean_candidates.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean candidate pool: {clean_path}")
    return load_chunk_rows(clean_path)


def _load_attempted_example_ids(paths: ProjectPaths) -> set[str]:
    attempted_ids: set[str] = set()
    source_dirs = {paths.base_source_chunks_dir, paths.source_chunks_dir}
    for source_dir in sorted(source_dirs):
        for chunk_path in sorted(source_dir.glob("*.csv")):
            for row in load_chunk_rows(chunk_path):
                attempted_ids.add(row.get("example_id", ""))
    for chunk_path in sorted(paths.accepted_dir.glob("*.csv")):
        for row in load_chunk_rows(chunk_path):
            attempted_ids.add(row.get("example_id", ""))
    for row in load_backlog_rows(paths):
        attempted_ids.add(row.get("example_id", ""))
    for round_dir in sorted(paths.refill_rounds_dir.glob("round_*")):
        for chunk_path in sorted((round_dir / "chunks").glob("*.csv")):
            for row in load_chunk_rows(chunk_path):
                attempted_ids.add(row.get("example_id", ""))
    attempted_ids.discard("")
    return attempted_ids


def _append_backlog_rows(
    paths: ProjectPaths,
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, str]]:
    existing_rows = load_backlog_rows(paths)
    existing_by_id = {row["example_id"]: row for row in existing_rows}
    round_number = _get_round_number(paths)
    for row in rows:
        example_id = str(row.get("example_id", ""))
        prior = existing_by_id.get(example_id)
        attempt_count = int(prior["attempt_count"]) + 1 if prior and prior.get("attempt_count") else 1
        existing_by_id[example_id] = {
            "example_id": example_id,
            "source_row_id": str(row.get("source_row_id", "")),
            "dialect": str(row.get("dialect", "")),
            "original_text": str(row.get("original_text", "")),
            "normalized_text": str(row.get("normalized_text", "")),
            "chunk_id": str(row.get("chunk_id", "")),
            "chunk_file_name": str(row.get("chunk_file_name", "")),
            "attempt_count": str(attempt_count),
            "round_number": str(round_number),
            "last_attempted_masked_text": str(row.get("last_attempted_masked_text", "")),
            "latest_errors": str(row.get("latest_errors", "")),
            "dropped_at": str(row.get("dropped_at", _utc_now_iso())),
        }
    saved_rows = list(existing_by_id.values())
    save_backlog_rows(paths, saved_rows)
    return saved_rows


def load_prompt_template(paths: ProjectPaths) -> str:
    return paths.prompt_template_path.read_text(encoding="utf-8")


def load_masking_guidelines(paths: ProjectPaths) -> str:
    return paths.masking_guidelines_path.read_text(encoding="utf-8")


def load_agent_guidelines() -> str:
    return AGENT_GUIDELINES


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


def dump_agent_rows_to_csv_text(
    rows: Sequence[Dict[str, str]],
    *,
    include_row_id: bool = False,
) -> str:
    fieldnames = AGENT_COLUMNS_WITH_ROW_ID if include_row_id else AGENT_COLUMNS
    simplified_rows = []
    for row in rows:
        simplified_row = {
            "original_text": row.get("original_text", ""),
            "masked_text": row.get("masked_text", ""),
        }
        if include_row_id:
            simplified_row[AGENT_ROW_ID_COLUMN] = row.get("source_row_id", "")
        simplified_rows.append(simplified_row)
    return dump_rows_to_csv_text(simplified_rows, fieldnames)


def get_accepted_agent_header_sets(row_matching: str) -> List[List[str]]:
    if row_matching == ROW_MATCHING_STRICT_ROW_ID:
        return [AGENT_COLUMNS_WITH_ROW_ID]
    return [
        AGENT_COLUMNS,
        AGENT_COLUMNS_WITH_ROW_ID,
    ]


def _match_agent_header_set(fieldnames: Sequence[str], row_matching: str) -> Optional[List[str]]:
    fieldnames_list = list(fieldnames)
    for header_set in get_accepted_agent_header_sets(row_matching):
        if fieldnames_list == header_set:
            return header_set
    return None


def _normalize_agent_import_rows(
    fieldnames: Sequence[str],
    imported_rows: Sequence[Dict[str, str]],
    row_matching: str,
) -> Tuple[List[Dict[str, str]], bool]:
    header_set = _match_agent_header_set(fieldnames, row_matching)
    if header_set is None:
        raise ValueError("Unsupported agent import headers")

    includes_row_id = header_set == AGENT_COLUMNS_WITH_ROW_ID
    normalized_rows = []
    for row in imported_rows:
        normalized_row = {
            "original_text": row.get("original_text", ""),
            "masked_text": row.get("masked_text", ""),
        }
        if includes_row_id:
            normalized_row[AGENT_ROW_ID_COLUMN] = row.get(AGENT_ROW_ID_COLUMN, "")
        normalized_rows.append(normalized_row)
    return normalized_rows, includes_row_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _extract_uncertainty_markers(masked_text: str, markers: Sequence[str]) -> Tuple[str, List[str]]:
    cleaned = masked_text
    found: List[str] = []
    for marker in markers:
        if marker in cleaned:
            found.append(marker)
            cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, found


def reconstruct_mask_metadata(
    original_text: str,
    masked_text: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    errors: List[str] = []
    normalized_original = _normalize_newlines(original_text)
    normalized_masked = _normalize_newlines(masked_text)

    matches = list(PLACEHOLDER_RE.finditer(normalized_masked))
    if not matches:
        if normalized_masked != normalized_original:
            errors.append(
                "masked_text changes the original text but contains no typed placeholders; "
                "either keep the text unchanged or use placeholders like [PERSON_1]"
            )
        return [], errors

    literals: List[str] = []
    placeholders: List[Tuple[str, str, int]] = []
    last_end = 0
    for match in matches:
        literals.append(normalized_masked[last_end:match.start()])
        placeholders.append((match.group(0), match.group("mask_type"), int(match.group("index"))))
        last_end = match.end()
    literals.append(normalized_masked[last_end:])

    if normalized_original[: len(literals[0])] != literals[0]:
        errors.append(
            f"original_text does not start with the unmasked prefix {literals[0]!r} from masked_text"
        )
        return [], errors

    start_pos = len(literals[0])
    seen_surface_by_placeholder: Dict[str, str] = {}
    seen_indices_by_type: Dict[str, set[int]] = {}

    def align_from(
        placeholder_position: int,
        current_pos: int,
        solutions: List[List[Tuple[int, int]]],
        current_solution: List[Tuple[int, int]],
    ) -> None:
        if len(solutions) > 1:
            return
        if placeholder_position == len(placeholders):
            if current_pos == len(normalized_original):
                solutions.append(list(current_solution))
            return

        next_literal = literals[placeholder_position + 1]
        if next_literal == "" and placeholder_position < len(placeholders) - 1:
            return

        if next_literal == "":
            if current_pos >= len(normalized_original):
                return
            current_solution.append((current_pos, len(normalized_original)))
            align_from(placeholder_position + 1, len(normalized_original), solutions, current_solution)
            current_solution.pop()
            return

        search_pos = current_pos + 1
        while True:
            span_end = normalized_original.find(next_literal, search_pos)
            if span_end == -1:
                return
            current_solution.append((current_pos, span_end))
            align_from(
                placeholder_position + 1,
                span_end + len(next_literal),
                solutions,
                current_solution,
            )
            current_solution.pop()
            search_pos = span_end + 1

    solutions: List[List[Tuple[int, int]]] = []
    align_from(0, start_pos, solutions, [])
    if not solutions:
        errors.append(
            "could not align masked_text back to original_text using the placeholder sequence and surrounding literal text"
        )
        return [], errors
    if len(solutions) > 1:
        errors.append(
            "masked_text can be aligned to original_text in more than one valid way, so automatic span reconstruction is ambiguous"
        )
        return [], errors
    aligned_ranges = solutions[0]

    spans: List[Dict[str, object]] = []
    for index, ((placeholder, mask_type, placeholder_index), (span_start, span_end)) in enumerate(
        zip(placeholders, aligned_ranges),
        start=1,
    ):
        surface_form = normalized_original[span_start:span_end]
        if not surface_form:
            errors.append(f"placeholder {placeholder!r} maps to an empty span in original_text")
            return [], errors

        previous_surface = seen_surface_by_placeholder.get(placeholder)
        if previous_surface is not None and previous_surface != surface_form:
            errors.append(
                f"placeholder {placeholder!r} is reused for different surface forms: "
                f"{previous_surface!r} and {surface_form!r}"
            )

        seen_surface_by_placeholder[placeholder] = surface_form
        seen_indices_by_type.setdefault(mask_type, set()).add(placeholder_index)
        spans.append(
            {
                "start": span_start,
                "end": span_end,
                "placeholder": placeholder,
                "surface_form": surface_form,
                "mask_type": mask_type,
            }
        )

    for mask_type, indices in seen_indices_by_type.items():
        expected = set(range(1, len(indices) + 1))
        if indices != expected:
            errors.append(
                f"placeholder numbering for mask_type {mask_type!r} is not contiguous; "
                f"found indices {sorted(indices)}, expected {sorted(expected)}"
            )

    return spans, errors


def _build_reconstructed_row(
    source_row: Dict[str, str],
    masked_text: str,
    paths: ProjectPaths,
) -> Tuple[Dict[str, str], List[str]]:
    cleaned_masked_text, found_markers = _extract_uncertainty_markers(masked_text, paths.uncertainty_markers)
    spans, reconstruction_errors = reconstruct_mask_metadata(source_row["original_text"], cleaned_masked_text)

    status = "approved"
    notes = ""
    if found_markers:
        status = "pending"
        notes = f"Auto-marked pending from uncertainty marker(s): {', '.join(found_markers)}"

    row = dict(source_row)
    row["masked_text"] = cleaned_masked_text
    row["mask_spans_json"] = json.dumps(spans, ensure_ascii=False)
    row["mask_count"] = str(len(spans))
    row["annotation_status"] = status
    row["annotator_model"] = ""
    row["notes"] = notes
    return row, reconstruction_errors


def get_working_chunk_path(paths: ProjectPaths, chunk_file_name: str) -> Path:
    return paths.working_dir / chunk_file_name


def save_working_preview(paths: ProjectPaths, chunk_file_name: str, rows: Sequence[Dict[str, str]]) -> Path:
    working_path = get_working_chunk_path(paths, chunk_file_name)
    write_csv(working_path, rows, OUTPUT_COLUMNS)
    return working_path


def load_working_preview(paths: ProjectPaths, chunk_file_name: str) -> List[Dict[str, str]]:
    return load_chunk_rows(get_working_chunk_path(paths, chunk_file_name))


def _clear_invalid_retry_rows_for_chunk(
    state: Dict[str, object],
    chunk_id: int,
) -> None:
    invalid_rows = state.get("invalid_retry_rows", [])
    state["invalid_retry_rows"] = [
        row for row in invalid_rows if int(row.get("chunk_id", -1)) != int(chunk_id)
    ]


def _update_invalid_retry_rows(
    paths: ProjectPaths,
    chunk_id: int,
    chunk_file_name: str,
    validation_rows: Sequence[Dict[str, object]],
    imported_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    state = load_session_state(paths)
    _clear_invalid_retry_rows_for_chunk(state, chunk_id)

    imported_by_original = {row["original_text"]: row for row in imported_rows}
    invalid_rows: List[Dict[str, object]] = []
    for row in validation_rows:
        if row["status"] != "invalid":
            continue
        original_text = str(row.get("original_text") or "")
        attempted_masked_text = imported_by_original.get(original_text, {}).get("masked_text", "")
        invalid_rows.append(
            {
                "chunk_id": int(chunk_id),
                "chunk_file_name": chunk_file_name,
                "example_id": row["example_id"],
                "row_id": row.get("source_row_id", ""),
                "original_text": original_text,
                "attempted_masked_text": attempted_masked_text,
                "errors": row["errors"],
                "retry_status": "open",
                "last_seen_at": _utc_now_iso(),
            }
        )

    state["invalid_retry_rows"].extend(invalid_rows)
    save_session_state(paths, state)
    return state["invalid_retry_rows"]


def get_invalid_retry_rows(
    project_id: str,
    chunk_id: Optional[int] = None,
    config: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    paths = get_project_paths(project_id, config)
    state = load_session_state(paths)
    rows = state.get("invalid_retry_rows", [])
    if chunk_id is None:
        return rows
    return [row for row in rows if int(row.get("chunk_id", -1)) == int(chunk_id)]


def clear_invalid_retry_rows(
    project_id: str,
    chunk_id: Optional[int] = None,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    state = load_session_state(paths)
    if chunk_id is None:
        cleared_rows = len(state.get("invalid_retry_rows", []))
        state["invalid_retry_rows"] = []
    else:
        before = state.get("invalid_retry_rows", [])
        state["invalid_retry_rows"] = [
            row for row in before if int(row.get("chunk_id", -1)) != int(chunk_id)
        ]
        cleared_rows = len(before) - len(state["invalid_retry_rows"])
    save_session_state(paths, state)
    return {
        "ok": True,
        "cleared_rows": cleared_rows,
        "invalid_retry_rows": state["invalid_retry_rows"],
    }


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


def _build_validation_summary_from_rows(
    rows: Sequence[Dict[str, str]],
    cached_invalid_errors_by_example_id: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, int], List[Dict[str, object]]]:
    counts: Counter[str] = Counter()
    validation_rows: List[Dict[str, object]] = []
    cached_invalid_errors_by_example_id = cached_invalid_errors_by_example_id or {}

    for index, row in enumerate(rows, start=1):
        errors = cached_invalid_errors_by_example_id.get(row["example_id"])
        if errors is None:
            errors = validate_row(row)
        status = _classify_errors(errors)
        counts[status] += 1
        validation_rows.append(
            {
                "row_number": index,
                "example_id": row["example_id"],
                "original_text": row["original_text"],
                "status": status,
                "errors": errors,
            }
        )

    return {
        "row_count": len(rows),
        "approved_rows": counts["approved"],
        "pending_rows": counts["pending"],
        "invalid_rows": counts["invalid"],
    }, validation_rows


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
    chunk_metrics = session_state.get("chunk_metrics", {}).get(str(row["chunk_id"]), {})
    backlog_rows = load_backlog_rows(paths)
    backlog_count = sum(1 for backlog_row in backlog_rows if str(backlog_row.get("chunk_id", "")) == str(row["chunk_id"]))

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
        "accepted_row_count": int(chunk_metrics.get("accepted_row_count", int(row["row_count"]) if accepted_path.exists() else 0)),
        "dropped_backlog_row_count": int(chunk_metrics.get("dropped_backlog_row_count", backlog_count)),
    }


def _sample_replacement_rows(
    paths: ProjectPaths,
    backlog_rows: Sequence[Dict[str, str]],
    *,
    seed: int = 17,
) -> List[Dict[str, str]]:
    clean_rows = _load_clean_candidate_rows(paths)
    attempted_ids = _load_attempted_example_ids(paths)
    backlog_targets = Counter(row.get("dialect", "") for row in backlog_rows)
    available_by_dialect: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        example_id = row.get("example_id", "")
        if not example_id or example_id in attempted_ids:
            continue
        available_by_dialect[row.get("dialect", "")].append(row)

    rng = random.Random(seed)
    sampled_rows: List[Dict[str, str]] = []
    for dialect, count in sorted(backlog_targets.items()):
        rows = list(available_by_dialect.get(dialect, []))
        rng.shuffle(rows)
        sampled_rows.extend(rows[:count])
    return sampled_rows


def _round_chunk_name(round_number: int, chunk_index: int) -> str:
    return f"round_{round_number:04d}_chunk_{chunk_index:04d}.csv"


def _ensure_round_chunk_names(round_dir: Path, round_number: int) -> List[Dict[str, str]]:
    manifest_path = round_dir / "chunk_manifest.csv"
    manifest_rows = load_chunk_rows(manifest_path)
    chunks_dir = round_dir / "chunks"
    changed = False
    for row in manifest_rows:
        chunk_id = int(row["chunk_id"])
        expected_name = _round_chunk_name(round_number, chunk_id)
        current_name = row["file_name"]
        if current_name == expected_name:
            continue
        current_path = chunks_dir / current_name
        expected_path = chunks_dir / expected_name
        if current_path.exists() and current_path != expected_path:
            current_path.rename(expected_path)
        row["file_name"] = expected_name
        if expected_path.exists():
            row["sha256"] = sha256_file(expected_path)
        changed = True
    if changed:
        write_csv(
            manifest_path,
            manifest_rows,
            ["chunk_id", "file_name", "row_count", "dialect_counts_json", "sha256", "status"],
        )
    return manifest_rows


def generate_refill_round(
    project_id: str,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    backlog_rows = load_backlog_rows(paths)
    if not backlog_rows:
        return {"ok": False, "error": "empty_backlog"}

    round_number = _get_round_number(paths) + 1
    sampled_rows = _sample_replacement_rows(paths, backlog_rows, seed=round_number + 17)
    if not sampled_rows:
        return {"ok": False, "error": "no_replacements_available"}

    source_manifest_rows = load_manifest_rows(paths)
    chunk_size = max(int(source_manifest_rows[0]["row_count"]), 1)
    round_dir = paths.refill_rounds_dir / f"round_{round_number:04d}"
    chunks_dir = round_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    for chunk_index, start in enumerate(range(0, len(sampled_rows), chunk_size), start=1):
        rows = sampled_rows[start : start + chunk_size]
        chunk_name = _round_chunk_name(round_number, chunk_index)
        chunk_path = chunks_dir / chunk_name
        write_csv(chunk_path, rows, OUTPUT_COLUMNS)
        counts = Counter(row["dialect"] for row in rows)
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

    manifest_path = round_dir / "chunk_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        ["chunk_id", "file_name", "row_count", "dialect_counts_json", "sha256", "status"],
    )
    summary = {
        "round_number": round_number,
        "sampled_count": len(sampled_rows),
        "chunk_count": len(manifest_rows),
        "chunk_size": chunk_size,
        "dialect_counts": dict(sorted(Counter(row["dialect"] for row in sampled_rows).items())),
        "manifest_path": str(manifest_path),
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **summary, "round_dir": str(round_dir)}


def activate_refill_round(
    project_id: str,
    round_number: int | None = None,
    config: Dict[str, object] | None = None,
    config_path: Path | None = None,
) -> Dict[str, object]:
    config_data = config if config is not None else load_config(config_path)
    paths = get_project_paths(project_id, config_data)
    target_round_number = round_number or _get_round_number(paths)
    if target_round_number <= 0:
        return {"ok": False, "error": "missing_refill_round"}

    round_dir = paths.refill_rounds_dir / f"round_{target_round_number:04d}"
    manifest_path = round_dir / "chunk_manifest.csv"
    chunks_dir = round_dir / "chunks"
    if not manifest_path.exists() or not chunks_dir.exists():
        return {"ok": False, "error": "missing_refill_round"}

    manifest_rows = _ensure_round_chunk_names(round_dir, target_round_number)
    project_config = config_data["projects"][project_id]
    project_config["source_chunks_dir"] = str(chunks_dir)
    project_config["source_manifest_path"] = str(manifest_path)
    project_config["active_round"] = f"round_{target_round_number:04d}"

    if config is None:
        save_config(config_data, config_path)

    updated_paths = get_project_paths(project_id, config_data)
    state = load_session_state(updated_paths)
    next_chunk = get_next_chunk(project_id, config_data)
    state["current_chunk_id"] = next_chunk["chunk_id"] if next_chunk else None
    save_session_state(updated_paths, state)

    return {
        "ok": True,
        "active_round": updated_paths.active_round,
        "round_number": target_round_number,
        "manifest_path": str(updated_paths.source_manifest_path),
        "chunks_dir": str(updated_paths.source_chunks_dir),
        "chunk_count": len(manifest_rows),
        "next_chunk": next_chunk,
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
    guidelines = load_agent_guidelines()
    for row in load_manifest_rows(paths):
        if int(row["chunk_id"]) == int(chunk_id):
            chunk_path = get_baseline_chunk_path(paths, row["file_name"])
            chunk_rows = load_chunk_rows(chunk_path)
            include_row_id_in_prompt = (
                paths.include_row_id_in_prompt or paths.row_matching == ROW_MATCHING_STRICT_ROW_ID
            )
            chunk_csv = dump_agent_rows_to_csv_text(
                chunk_rows,
                include_row_id=include_row_id_in_prompt,
            )
            if paths.row_matching == ROW_MATCHING_STRICT_ROW_ID:
                output_csv_instruction = "Return only a three-column CSV with `row_id`, `original_text`, and `masked_text`."
                row_id_instruction = "The `row_id` column is required in your output because this project matches rows by row_id. Keep every row_id exactly unchanged."
            elif paths.include_row_id_in_prompt:
                output_csv_instruction = "Return a CSV with `original_text` and `masked_text`. You may also include `row_id` as the first column if you keep it unchanged."
                row_id_instruction = "The `row_id` column is included as reference context to help you keep track of rows. If you return it, keep it exactly unchanged."
            else:
                output_csv_instruction = "Return only a two-column CSV with `original_text` and `masked_text`."
                row_id_instruction = "Do not add a `row_id` column unless the prompt explicitly includes one."
            return template.format(
                project_id=project_id,
                chunk_id=row["chunk_id"],
                chunk_file=row["file_name"],
                chunk_path=chunk_path,
                row_count=row["row_count"],
                dataset_mode=paths.dataset_mode,
                guidelines=guidelines.strip(),
                include_row_id_in_prompt="yes" if include_row_id_in_prompt else "no",
                row_matching=paths.row_matching,
                validate_original_text_with_row_id="yes" if paths.validate_original_text_with_row_id else "no",
                output_csv_instruction=output_csv_instruction,
                row_id_instruction=row_id_instruction,
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


def _finalize_chunk_acceptance(
    project_id: str,
    chunk_id: int,
    preview: Dict[str, object],
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    manifest_row = next(row for row in load_manifest_rows(paths) if int(row["chunk_id"]) == int(chunk_id))

    summary = preview["summary"]
    if summary["invalid_rows"] > 0:
        return {"ok": False, "error": "invalid_rows_present", "preview": preview}
    if summary["pending_rows"] > 0 and not paths.allow_pending_accept:
        return {"ok": False, "error": "pending_rows_present", "preview": preview}

    accepted_path = get_accepted_chunk_path(paths, manifest_row["file_name"])
    write_csv(accepted_path, preview["reconstructed_rows"], OUTPUT_COLUMNS)

    state = load_session_state(paths)
    accepted_chunks = set(state.get("accepted_chunks", []))
    accepted_chunks.add(int(chunk_id))
    state["accepted_chunks"] = sorted(accepted_chunks)
    _clear_invalid_retry_rows_for_chunk(state, chunk_id)
    chunk_metrics = state.get("chunk_metrics", {})
    chunk_metrics[str(chunk_id)] = {
        "original_row_count": int(manifest_row["row_count"]),
        "accepted_row_count": len(preview["reconstructed_rows"]),
        "dropped_backlog_row_count": max(int(manifest_row["row_count"]) - len(preview["reconstructed_rows"]), 0),
    }
    state["chunk_metrics"] = chunk_metrics
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


def _error_headers(paths: ProjectPaths, fieldnames: Sequence[str]) -> Dict[str, object]:
    if paths.row_matching == ROW_MATCHING_STRICT_ROW_ID and list(fieldnames) == AGENT_COLUMNS:
        return {
            "ok": False,
            "error": "missing_row_id_column",
            "expected_headers": AGENT_COLUMNS,
            "accepted_header_sets": get_accepted_agent_header_sets(paths.row_matching),
            "received_headers": list(fieldnames),
        }
    return {
        "ok": False,
        "error": "header_mismatch",
        "expected_headers": AGENT_COLUMNS,
        "accepted_header_sets": get_accepted_agent_header_sets(paths.row_matching),
        "received_headers": list(fieldnames),
    }


def _build_rows_by_key(
    rows: Sequence[Dict[str, str]],
    key: str,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    rows_by_key: Dict[str, Dict[str, str]] = {}
    duplicates = set()
    for row in rows:
        value = row.get(key, "")
        if value in rows_by_key:
            duplicates.add(value)
        rows_by_key[value] = row
    return rows_by_key, sorted(duplicates)


def _match_rows_by_original_text(
    baseline_rows: Sequence[Dict[str, str]],
    imported_rows: Sequence[Dict[str, str]],
) -> Dict[str, object]:
    source_by_original, duplicate_originals = _build_rows_by_key(baseline_rows, "original_text")
    if duplicate_originals:
        return {
            "ok": False,
            "error": "ambiguous_source_original_text",
            "duplicate_original_texts": duplicate_originals,
        }

    imported_originals = [row.get("original_text", "") for row in imported_rows]
    duplicate_import_originals = sorted(
        original for original, count in Counter(imported_originals).items() if count > 1
    )
    if duplicate_import_originals:
        return {
            "ok": False,
            "error": "duplicate_original_text_in_import",
            "duplicate_original_texts": duplicate_import_originals,
        }

    missing_originals = [original for original in imported_originals if original not in source_by_original]
    if missing_originals:
        return {
            "ok": False,
            "error": "original_text_not_found",
            "missing_original_texts": missing_originals[:10],
        }

    baseline_order = [row["original_text"] for row in baseline_rows]
    if imported_originals != baseline_order:
        return {
            "ok": False,
            "error": "original_text_order_mismatch",
            "expected_original_texts": baseline_order,
            "received_original_texts": imported_originals,
        }

    return {
        "ok": True,
        "matched_source_rows": [source_by_original[row["original_text"]] for row in imported_rows],
        "imported_rows": list(imported_rows),
    }


def _match_rows_by_row_id(
    baseline_rows: Sequence[Dict[str, str]],
    imported_rows: Sequence[Dict[str, str]],
    *,
    require_order: bool,
    validate_original_text: bool,
) -> Dict[str, object]:
    source_by_row_id, duplicate_source_row_ids = _build_rows_by_key(baseline_rows, "source_row_id")
    if duplicate_source_row_ids:
        return {
            "ok": False,
            "error": "ambiguous_source_row_id",
            "duplicate_row_ids": duplicate_source_row_ids,
        }

    imported_row_ids = [row.get(AGENT_ROW_ID_COLUMN, "") for row in imported_rows]
    duplicate_import_row_ids = sorted(
        row_id for row_id, count in Counter(imported_row_ids).items() if count > 1
    )
    if duplicate_import_row_ids:
        return {
            "ok": False,
            "error": "duplicate_row_id_in_import",
            "duplicate_row_ids": duplicate_import_row_ids,
        }

    missing_row_ids = [row_id for row_id in imported_row_ids if row_id not in source_by_row_id]
    if missing_row_ids:
        return {
            "ok": False,
            "error": "row_id_not_found",
            "missing_row_ids": missing_row_ids[:10],
        }

    if require_order:
        expected_row_ids = [row.get("source_row_id", "") for row in baseline_rows]
        if imported_row_ids != expected_row_ids:
            return {
                "ok": False,
                "error": "row_id_order_mismatch",
                "expected_row_ids": expected_row_ids,
                "received_row_ids": imported_row_ids,
            }

    if validate_original_text:
        mismatched_rows = []
        for imported_row in imported_rows:
            source_row = source_by_row_id[imported_row[AGENT_ROW_ID_COLUMN]]
            if imported_row.get("original_text", "") != source_row.get("original_text", ""):
                mismatched_rows.append(
                    {
                        "row_id": imported_row.get(AGENT_ROW_ID_COLUMN, ""),
                        "expected_original_text": source_row.get("original_text", ""),
                        "received_original_text": imported_row.get("original_text", ""),
                    }
                )
        if mismatched_rows:
            return {
                "ok": False,
                "error": "row_id_original_text_mismatch",
                "mismatched_rows": mismatched_rows,
            }

    return {
        "ok": True,
        "matched_source_rows": [source_by_row_id[row[AGENT_ROW_ID_COLUMN]] for row in imported_rows],
        "imported_rows": list(imported_rows),
    }


def _match_full_import_rows(
    paths: ProjectPaths,
    baseline_rows: Sequence[Dict[str, str]],
    imported_rows: Sequence[Dict[str, str]],
    includes_row_id: bool,
) -> Dict[str, object]:
    if paths.row_matching == ROW_MATCHING_STRICT_ROW_ID and not includes_row_id:
        return {
            "ok": False,
            "error": "missing_row_id_column",
            "accepted_header_sets": get_accepted_agent_header_sets(paths.row_matching),
        }
    if includes_row_id and paths.row_matching in {
        ROW_MATCHING_PREFER_ROW_ID,
        ROW_MATCHING_STRICT_ROW_ID,
    }:
        return _match_rows_by_row_id(
            baseline_rows,
            imported_rows,
            require_order=True,
            validate_original_text=paths.validate_original_text_with_row_id,
        )
    return _match_rows_by_original_text(baseline_rows, imported_rows)


def _match_retry_rows(
    paths: ProjectPaths,
    working_rows: Sequence[Dict[str, str]],
    cached_retryable_rows: Sequence[Dict[str, object]],
    retry_rows: Sequence[Dict[str, str]],
    includes_row_id: bool,
) -> Dict[str, object]:
    if paths.row_matching == ROW_MATCHING_STRICT_ROW_ID and not includes_row_id:
        return {
            "ok": False,
            "error": "missing_row_id_column",
            "accepted_header_sets": get_accepted_agent_header_sets(paths.row_matching),
        }

    working_by_original, duplicate_working_originals = _build_rows_by_key(working_rows, "original_text")
    if duplicate_working_originals:
        return {
            "ok": False,
            "error": "ambiguous_source_original_text",
            "duplicate_original_texts": duplicate_working_originals,
        }

    if includes_row_id and paths.row_matching in {
        ROW_MATCHING_PREFER_ROW_ID,
        ROW_MATCHING_STRICT_ROW_ID,
    }:
        working_by_row_id, duplicate_working_row_ids = _build_rows_by_key(working_rows, "source_row_id")
        if duplicate_working_row_ids:
            return {
                "ok": False,
                "error": "ambiguous_source_row_id",
                "duplicate_row_ids": duplicate_working_row_ids,
            }

        cached_by_row_id: Dict[str, Dict[str, object]] = {}
        for cached_row in cached_retryable_rows:
            working_row = working_by_original.get(str(cached_row.get("original_text", "")))
            if not working_row:
                continue
            cached_by_row_id[working_row.get("source_row_id", "")] = cached_row

        imported_row_ids = [row.get(AGENT_ROW_ID_COLUMN, "") for row in retry_rows]
        duplicate_import_row_ids = sorted(
            row_id for row_id, count in Counter(imported_row_ids).items() if count > 1
        )
        if duplicate_import_row_ids:
            return {
                "ok": False,
                "error": "duplicate_row_id_in_import",
                "duplicate_row_ids": duplicate_import_row_ids,
            }

        unknown_retry_row_ids = [row_id for row_id in imported_row_ids if row_id not in cached_by_row_id]
        if unknown_retry_row_ids:
            return {
                "ok": False,
                "error": "unknown_retry_row_ids",
                "unknown_retry_row_ids": unknown_retry_row_ids,
            }

        matched_retry_pairs = []
        for retry_row in retry_rows:
            source_like_row = working_by_row_id[retry_row[AGENT_ROW_ID_COLUMN]]
            if paths.validate_original_text_with_row_id:
                if retry_row.get("original_text", "") != source_like_row.get("original_text", ""):
                    return {
                        "ok": False,
                        "error": "row_id_original_text_mismatch",
                        "mismatched_rows": [
                            {
                                "row_id": retry_row.get(AGENT_ROW_ID_COLUMN, ""),
                                "expected_original_text": source_like_row.get("original_text", ""),
                                "received_original_text": retry_row.get("original_text", ""),
                            }
                        ],
                    }
            matched_retry_pairs.append((retry_row, source_like_row))

        return {
            "ok": True,
            "matched_retry_pairs": matched_retry_pairs,
        }

    duplicate_originals = sorted(
        original for original, count in Counter(row.get("original_text", "") for row in retry_rows).items() if count > 1
    )
    if duplicate_originals:
        return {
            "ok": False,
            "error": "duplicate_original_text_in_import",
            "duplicate_original_texts": duplicate_originals,
        }

    cached_by_original = {row["original_text"]: row for row in cached_retryable_rows}
    unknown_retry_rows = [row["original_text"] for row in retry_rows if row["original_text"] not in cached_by_original]
    if unknown_retry_rows:
        return {
            "ok": False,
            "error": "unknown_retry_rows",
            "unknown_retry_rows": unknown_retry_rows,
        }

    return {
        "ok": True,
        "matched_retry_pairs": [(retry_row, working_by_original[retry_row["original_text"]]) for retry_row in retry_rows],
    }


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
    if _match_agent_header_set(fieldnames, paths.row_matching) is None:
        return _error_headers(paths, fieldnames)
    imported_rows, includes_row_id = _normalize_agent_import_rows(fieldnames, imported_rows, paths.row_matching)

    baseline_rows = load_chunk_rows(get_baseline_chunk_path(paths, manifest_row["file_name"]))
    if len(imported_rows) != len(baseline_rows):
        return {
            "ok": False,
            "error": "row_count_mismatch",
            "expected_row_count": len(baseline_rows),
            "received_row_count": len(imported_rows),
        }

    match_result = _match_full_import_rows(paths, baseline_rows, imported_rows, includes_row_id)
    if not match_result["ok"]:
        return match_result

    validation_rows: List[Dict[str, object]] = []
    counts = Counter()
    reconstructed_rows: List[Dict[str, str]] = []
    matched_source_rows = match_result["matched_source_rows"]
    for index, (imported_row, source_row) in enumerate(zip(imported_rows, matched_source_rows), start=1):
        reconstructed_row, reconstruction_errors = _build_reconstructed_row(
            source_row=source_row,
            masked_text=imported_row["masked_text"],
            paths=paths,
        )
        row_errors = [f"{reconstructed_row['example_id']}: {message}" for message in reconstruction_errors]
        if not reconstruction_errors:
            row_errors.extend(validate_row(reconstructed_row))
        errors = row_errors
        status = _classify_errors(errors)
        counts[status] += 1
        reconstructed_rows.append(reconstructed_row)
        validation_rows.append(
            {
                "row_number": index,
                "example_id": reconstructed_row["example_id"],
                "source_row_id": reconstructed_row["source_row_id"],
                "original_text": reconstructed_row["original_text"],
                "status": status,
                "errors": errors,
            }
        )

    changed_count, changed_rows = _compare_rows(baseline_rows, reconstructed_rows)
    working_path = save_working_preview(paths, manifest_row["file_name"], reconstructed_rows)
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
        "reconstructed_rows": reconstructed_rows,
        "working_preview_path": str(working_path),
        "invalid_retry_rows": _update_invalid_retry_rows(
            paths=paths,
            chunk_id=chunk_id,
            chunk_file_name=manifest_row["file_name"],
            validation_rows=validation_rows,
            imported_rows=imported_rows,
        ),
    }


def preview_retry_import(
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

    working_path = get_working_chunk_path(paths, manifest_row["file_name"])
    if not working_path.exists():
        return {
            "ok": False,
            "error": "missing_working_preview",
            "message": "This chunk does not have a working preview yet. Run a full chunk preview before retrying invalid rows.",
        }

    fieldnames, retry_rows = load_chunk_rows_from_text(csv_text)
    if _match_agent_header_set(fieldnames, paths.row_matching) is None:
        return _error_headers(paths, fieldnames)
    retry_rows, includes_row_id = _normalize_agent_import_rows(fieldnames, retry_rows, paths.row_matching)

    cached_rows = get_invalid_retry_rows(project_id, chunk_id, config)
    cached_retryable_rows = [
        row for row in cached_rows if row.get("retry_status", "open") in {"open", "fixed_in_preview"}
    ]
    cached_by_original = {row["original_text"]: row for row in cached_retryable_rows}

    if not cached_retryable_rows:
        return {
            "ok": False,
            "error": "no_open_invalid_rows",
            "message": "There are no open invalid rows cached for this chunk.",
        }

    working_rows = load_working_preview(paths, manifest_row["file_name"])
    match_result = _match_retry_rows(
        paths,
        working_rows,
        cached_retryable_rows,
        retry_rows,
        includes_row_id,
    )
    if not match_result["ok"]:
        return match_result

    merged_rows = [dict(row) for row in working_rows]
    merged_rows_by_original = {row["original_text"]: row for row in merged_rows}

    resolved_retry_rows: List[Dict[str, object]] = []
    still_invalid_retry_rows: List[Dict[str, object]] = []
    cache_updates: List[Dict[str, object]] = []
    cached_invalid_errors_by_example_id: Dict[str, List[str]] = {}

    for retry_row, source_like_row in match_result["matched_retry_pairs"]:
        original_text = source_like_row["original_text"]
        reconstructed_row, reconstruction_errors = _build_reconstructed_row(
            source_row=source_like_row,
            masked_text=retry_row["masked_text"],
            paths=paths,
        )
        row_errors = [f"{reconstructed_row['example_id']}: {message}" for message in reconstruction_errors]
        if not reconstruction_errors:
            row_errors.extend(validate_row(reconstructed_row))

        if row_errors:
            cached_invalid_errors_by_example_id[reconstructed_row["example_id"]] = row_errors
            still_invalid_retry_rows.append(
                {
                    "example_id": reconstructed_row["example_id"],
                    "original_text": original_text,
                    "attempted_masked_text": retry_row["masked_text"],
                    "errors": row_errors,
                }
            )
            cache_updates.append(
                {
                    **cached_by_original[original_text],
                    "attempted_masked_text": retry_row["masked_text"],
                    "errors": row_errors,
                    "retry_status": "open",
                    "last_seen_at": _utc_now_iso(),
                }
            )
            continue

        merged_rows_by_original[original_text].update(reconstructed_row)
        resolved_retry_rows.append(
            {
                "example_id": reconstructed_row["example_id"],
                "original_text": original_text,
            }
        )
        cache_updates.append(
            {
                **cached_by_original[original_text],
                "attempted_masked_text": retry_row["masked_text"],
                "errors": [],
                "retry_status": "fixed_in_preview",
                "last_seen_at": _utc_now_iso(),
            }
        )

    state = load_session_state(paths)
    retained_rows = [
        row
        for row in state.get("invalid_retry_rows", [])
        if int(row.get("chunk_id", -1)) != int(chunk_id)
        or row.get("retry_status", "open") != "open"
        and row["original_text"] not in cached_by_original
    ]
    unaffected_rows = [
        row
        for row in state.get("invalid_retry_rows", [])
        if int(row.get("chunk_id", -1)) != int(chunk_id)
    ]
    current_chunk_other_rows = [
        row
        for row in state.get("invalid_retry_rows", [])
        if int(row.get("chunk_id", -1)) == int(chunk_id)
        and row["original_text"] not in cached_by_original
    ]
    skipped_cached_rows = [
        row
        for row in cached_retryable_rows
        if row["original_text"] not in {retry_row["original_text"] for retry_row in retry_rows}
    ]
    state["invalid_retry_rows"] = unaffected_rows + current_chunk_other_rows + skipped_cached_rows + cache_updates
    save_session_state(paths, state)

    summary, validation_rows = _build_validation_summary_from_rows(
        merged_rows,
        cached_invalid_errors_by_example_id=cached_invalid_errors_by_example_id,
    )
    changed_count, changed_rows = _compare_rows(load_chunk_rows(get_chunk_source_path(paths, manifest_row["file_name"])), merged_rows)
    summary["changed_rows"] = changed_count
    merged_agent_csv = dump_agent_rows_to_csv_text(merged_rows).rstrip()

    return {
        "ok": True,
        "chunk_id": chunk_id,
        "file_name": manifest_row["file_name"],
        "summary": summary,
        "validation_rows": validation_rows,
        "changed_rows": changed_rows,
        "reconstructed_rows": merged_rows,
        "merged_agent_csv": merged_agent_csv,
        "resolved_retry_rows": resolved_retry_rows,
        "still_invalid_retry_rows": still_invalid_retry_rows,
        "skipped_cached_rows": skipped_cached_rows,
        "invalid_retry_rows": state["invalid_retry_rows"],
    }


def apply_retry_import(
    project_id: str,
    chunk_id: int,
    csv_text: str,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    preview = preview_retry_import(project_id, chunk_id, csv_text, config)
    if not preview.get("ok"):
        return preview

    paths = get_project_paths(project_id, config)
    manifest_row = next(row for row in load_manifest_rows(paths) if int(row["chunk_id"]) == int(chunk_id))
    working_rows = list(preview["reconstructed_rows"])
    resolved_ids = {row["example_id"] for row in preview.get("resolved_retry_rows", [])}
    current_chunk_invalid_rows = get_invalid_retry_rows(project_id, chunk_id, config)
    unresolved_cached_rows = [
        row
        for row in current_chunk_invalid_rows
        if row.get("example_id") not in resolved_ids
    ]
    unresolved_ids = {row.get("example_id", "") for row in unresolved_cached_rows}

    reduced_rows = [row for row in working_rows if row.get("example_id") not in unresolved_ids]
    dropped_backlog_rows = []
    working_by_id = {row["example_id"]: row for row in working_rows}
    for invalid_row in unresolved_cached_rows:
        source_row = working_by_id.get(str(invalid_row.get("example_id", "")))
        if not source_row:
            continue
        dropped_backlog_rows.append(
            {
                "example_id": source_row.get("example_id", ""),
                "source_row_id": source_row.get("source_row_id", ""),
                "dialect": source_row.get("dialect", ""),
                "original_text": source_row.get("original_text", ""),
                "normalized_text": source_row.get("normalized_text", ""),
                "chunk_id": chunk_id,
                "chunk_file_name": manifest_row["file_name"],
                "last_attempted_masked_text": invalid_row.get("attempted_masked_text", ""),
                "latest_errors": " | ".join(invalid_row.get("errors", [])),
                "dropped_at": _utc_now_iso(),
            }
        )

    backlog_rows = _append_backlog_rows(paths, dropped_backlog_rows)
    working_path = save_working_preview(paths, manifest_row["file_name"], reduced_rows)

    state = load_session_state(paths)
    state["invalid_retry_rows"] = [
        row for row in state.get("invalid_retry_rows", [])
        if int(row.get("chunk_id", -1)) != int(chunk_id)
    ]
    chunk_metrics = state.get("chunk_metrics", {})
    chunk_metrics[str(chunk_id)] = {
        "original_row_count": int(manifest_row["row_count"]),
        "accepted_row_count": len(reduced_rows),
        "dropped_backlog_row_count": len(dropped_backlog_rows),
    }
    state["chunk_metrics"] = chunk_metrics
    save_session_state(paths, state)

    summary, validation_rows = _build_validation_summary_from_rows(reduced_rows)
    changed_count, changed_rows = _compare_rows(
        load_chunk_rows(get_chunk_source_path(paths, manifest_row["file_name"])),
        reduced_rows,
    )
    summary["changed_rows"] = changed_count
    preview["working_preview_path"] = str(working_path)
    preview["ok"] = True
    preview["reconstructed_rows"] = reduced_rows
    preview["merged_agent_csv"] = dump_agent_rows_to_csv_text(reduced_rows).rstrip()
    preview["summary"] = summary
    preview["validation_rows"] = validation_rows
    preview["changed_rows"] = changed_rows
    preview["dropped_backlog_rows"] = dropped_backlog_rows
    preview["dropped_backlog_row_count"] = len(dropped_backlog_rows)
    preview["backlog_row_count"] = len(backlog_rows)
    preview["invalid_retry_rows"] = state["invalid_retry_rows"]
    return preview


def accept_import(
    project_id: str,
    chunk_id: int,
    csv_text: str,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    manifest_row = next(row for row in load_manifest_rows(paths) if int(row["chunk_id"]) == int(chunk_id))
    working_path = get_working_chunk_path(paths, manifest_row["file_name"])
    source_row_count = int(manifest_row["row_count"])
    if working_path.exists():
        working_rows = load_working_preview(paths, manifest_row["file_name"])
    else:
        working_rows = []
    use_working_preview = working_path.exists() and (
        len(working_rows) != source_row_count or not csv_text.strip()
    )
    if use_working_preview:
        summary, validation_rows = _build_validation_summary_from_rows(working_rows)
        changed_count, changed_rows = _compare_rows(
            load_chunk_rows(get_chunk_source_path(paths, manifest_row["file_name"])),
            working_rows,
        )
        summary["changed_rows"] = changed_count
        preview = {
            "ok": True,
            "chunk_id": chunk_id,
            "file_name": manifest_row["file_name"],
            "summary": summary,
            "validation_rows": validation_rows,
            "changed_rows": changed_rows,
            "reconstructed_rows": working_rows,
            "working_preview_path": str(working_path),
            "invalid_retry_rows": get_invalid_retry_rows(project_id, chunk_id, config),
        }
    else:
        preview = preview_import(project_id, chunk_id, csv_text, config)
        if not preview.get("ok"):
            return preview

    return _finalize_chunk_acceptance(project_id, chunk_id, preview, config)


def skip_invalid_retry_cache(
    project_id: str,
    chunk_id: int,
    config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    manifest_row = next(row for row in load_manifest_rows(paths) if int(row["chunk_id"]) == int(chunk_id))
    working_path = get_working_chunk_path(paths, manifest_row["file_name"])
    if not working_path.exists():
        return {
            "ok": False,
            "error": "missing_working_preview",
            "message": "This chunk does not have a working preview yet. Run a full chunk preview before skipping invalid rows.",
        }

    working_rows = load_working_preview(paths, manifest_row["file_name"])
    cached_rows = get_invalid_retry_rows(project_id, chunk_id, config)
    open_cached_rows = [row for row in cached_rows if row.get("retry_status", "open") == "open"]

    if not open_cached_rows:
        return accept_import(project_id, chunk_id, "", config)

    open_cached_ids = {row["example_id"] for row in open_cached_rows}
    reduced_rows = [row for row in working_rows if row.get("example_id") not in open_cached_ids]

    dropped_backlog_rows = []
    working_by_id = {row["example_id"]: row for row in working_rows}
    for invalid_row in open_cached_rows:
        source_row = working_by_id.get(str(invalid_row.get("example_id", "")))
        if not source_row:
            continue
        dropped_backlog_rows.append(
            {
                "example_id": source_row.get("example_id", ""),
                "source_row_id": source_row.get("source_row_id", ""),
                "dialect": source_row.get("dialect", ""),
                "original_text": source_row.get("original_text", ""),
                "normalized_text": source_row.get("normalized_text", ""),
                "chunk_id": chunk_id,
                "chunk_file_name": manifest_row["file_name"],
                "last_attempted_masked_text": invalid_row.get("attempted_masked_text", ""),
                "latest_errors": " | ".join(invalid_row.get("errors", [])),
                "dropped_at": _utc_now_iso(),
            }
        )

    backlog_rows = _append_backlog_rows(paths, dropped_backlog_rows)
    save_working_preview(paths, manifest_row["file_name"], reduced_rows)

    state = load_session_state(paths)
    state["invalid_retry_rows"] = [
        row
        for row in state.get("invalid_retry_rows", [])
        if int(row.get("chunk_id", -1)) != int(chunk_id)
    ]
    chunk_metrics = state.get("chunk_metrics", {})
    chunk_metrics[str(chunk_id)] = {
        "original_row_count": int(manifest_row["row_count"]),
        "accepted_row_count": len(reduced_rows),
        "dropped_backlog_row_count": len(dropped_backlog_rows),
    }
    state["chunk_metrics"] = chunk_metrics
    save_session_state(paths, state)

    summary, validation_rows = _build_validation_summary_from_rows(reduced_rows)
    changed_count, changed_rows = _compare_rows(
        load_chunk_rows(get_chunk_source_path(paths, manifest_row["file_name"])),
        reduced_rows,
    )
    summary["changed_rows"] = changed_count

    return _finalize_chunk_acceptance(
        project_id,
        chunk_id,
        {
            "ok": True,
            "chunk_id": chunk_id,
            "file_name": manifest_row["file_name"],
            "summary": summary,
            "validation_rows": validation_rows,
            "changed_rows": changed_rows,
            "reconstructed_rows": reduced_rows,
            "working_preview_path": str(working_path),
            "invalid_retry_rows": state["invalid_retry_rows"],
            "dropped_backlog_rows": dropped_backlog_rows,
            "dropped_backlog_row_count": len(dropped_backlog_rows),
            "backlog_row_count": len(backlog_rows),
        },
        config,
    )


def get_project_summary(project_id: str, config: Dict[str, object] | None = None) -> Dict[str, object]:
    paths = get_project_paths(project_id, config)
    chunks = list_chunks(project_id, config)
    counts = Counter(chunk["state"] for chunk in chunks)
    session_state = load_session_state(paths)
    merge_summary_path = paths.merged_dir / "merge_summary.json"
    merge_summary = json.loads(merge_summary_path.read_text(encoding="utf-8")) if merge_summary_path.exists() else None
    backlog_rows = load_backlog_rows(paths)
    base_manifest_rows = load_chunk_rows(paths.base_source_manifest_path)
    target_row_count = sum(int(row["row_count"]) for row in base_manifest_rows)
    approved_rows_total = int(merge_summary["approved_rows"]) if merge_summary else 0
    return {
        "project_id": project_id,
        "dataset_mode": paths.dataset_mode,
        "total_chunks": len(chunks),
        "target_row_count": target_row_count,
        "approved_rows_total": approved_rows_total,
        "rows_remaining_to_target": max(target_row_count - approved_rows_total, 0),
        "state_counts": dict(sorted(counts.items())),
        "current_chunk_id": session_state.get("current_chunk_id"),
        "next_chunk": get_next_chunk(project_id, config),
        "auto_merge": paths.auto_merge,
        "auto_advance": paths.auto_advance,
        "include_row_id_in_prompt": paths.include_row_id_in_prompt,
        "row_matching": paths.row_matching,
        "validate_original_text_with_row_id": paths.validate_original_text_with_row_id,
        "merge_summary": merge_summary,
        "backlog_rows": backlog_rows,
        "backlog_row_count": len(backlog_rows),
        "current_round": _get_round_number(paths),
        "active_round": paths.active_round,
        "active_source_manifest_path": str(paths.source_manifest_path),
        "base_source_manifest_path": str(paths.base_source_manifest_path),
        "invalid_retry_rows": session_state.get("invalid_retry_rows", []),
        "invalid_retry_row_count": len(session_state.get("invalid_retry_rows", [])),
    }
