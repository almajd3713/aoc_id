import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandas.api.types

import kaggle_metric_utilities


class ParticipantVisibleError(Exception):
    pass


PLACEHOLDER_RE = re.compile(r"\[(?P<mask_type>[A-Z]+)_(?P<index>\d+)\]")


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _parse_solution_spans(row: Dict[str, object]) -> List[Dict[str, object]]:
    spans_raw = row.get("mask_spans_json") or "[]"
    try:
        spans = json.loads(str(spans_raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(spans, list):
        return []
    return spans


def reconstruct_mask_metadata(original_text: str, masked_text: str) -> Tuple[List[Dict[str, object]], List[str]]:
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
    for (placeholder, mask_type, placeholder_index), (span_start, span_end) in zip(
        placeholders, aligned_ranges
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


def _span_tuples(row_index: int, spans: Iterable[Dict[str, object]]) -> set[Tuple[int, int, int, str]]:
    tuples: set[Tuple[int, int, int, str]] = set()
    for span in spans:
        tuples.add(
            (
                int(row_index),
                int(span["start"]),
                int(span["end"]),
                str(span["mask_type"]),
            )
        )
    return tuples


def _span_micro_f1(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    if "original_text" not in solution.columns or "mask_spans_json" not in solution.columns:
        raise ParticipantVisibleError("Solution must contain original_text and mask_spans_json columns")
    if "masked_text" not in submission.columns:
        raise ParticipantVisibleError("Submission must contain a masked_text column")
    if len(solution) != len(submission):
        raise ParticipantVisibleError("Solution and submission row counts do not match")

    solution_rows = solution.to_dict(orient="records")
    submission_rows = submission.to_dict(orient="records")
    gold_spans: set[Tuple[int, int, int, str]] = set()
    pred_spans: set[Tuple[int, int, int, str]] = set()

    for row_index, (solution_row, submission_row) in enumerate(zip(solution_rows, submission_rows)):
        gold_spans.update(_span_tuples(row_index, _parse_solution_spans(solution_row)))
        predicted_spans, errors = reconstruct_mask_metadata(
            str(solution_row.get("original_text") or ""),
            str(submission_row.get("masked_text") or ""),
        )
        if not errors:
            pred_spans.update(_span_tuples(row_index, predicted_spans))

    true_positives = len(gold_spans & pred_spans)
    false_positives = len(pred_spans - gold_spans)
    false_negatives = len(gold_spans - pred_spans)

    denominator = (2 * true_positives) + false_positives + false_negatives
    if denominator == 0:
        return float(1.0)
    return float((2 * true_positives) / denominator)


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    labels: Optional[Sequence] = None,
    pos_label: object = 1,
    average: str = "binary",
    weights_column_name: Optional[str] = None,
) -> float:
    """Compute span-level micro-F1 from Kaggle solution and submission tables."""
    del labels
    del pos_label
    del average

    solution = solution.copy()
    submission = submission.copy()

    if row_id_column_name not in solution.columns or row_id_column_name not in submission.columns:
        raise ParticipantVisibleError("The row ID column is missing from solution or submission")

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if weights_column_name:
        if weights_column_name not in solution.columns:
            raise ValueError(f"The solution weights column {weights_column_name} is not found")
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError("The solution weights are not numeric")

    score_result = kaggle_metric_utilities.safe_call_score(_span_micro_f1, solution, submission)
    return float(score_result)
