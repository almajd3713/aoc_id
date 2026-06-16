import json
import sys
import tempfile
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from annotation_app.backend.workflow import reconstruct_mask_metadata as workflow_reconstruct_mask_metadata
from scripts.kaggle_metric import build_metric_source, reconstruct_mask_metadata, score_rows, write_metric_artifact


class FakeSeries:
    def __init__(self, values):
        self.values = values


class FakeDataFrame:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]

    def copy(self):
        return FakeDataFrame(self._rows)

    @property
    def columns(self):
        if not self._rows:
            return []
        return list(self._rows[0].keys())

    def __delitem__(self, key):
        for row in self._rows:
            row.pop(key, None)

    def pop(self, key):
        values = [row.pop(key) for row in self._rows]
        return FakeSeries(values)

    def to_dict(self, orient="records"):
        if orient != "records":
            raise TypeError("unsupported orient")
        return [dict(row) for row in self._rows]

    def __len__(self):
        return len(self._rows)


class KaggleMetricTests(unittest.TestCase):
    def _solution_row(
        self,
        row_id: str,
        original_text: str,
        masked_text: str,
        spans: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "id": row_id,
            "original_text": original_text,
            "masked_text": masked_text,
            "mask_spans_json": json.dumps(spans, ensure_ascii=False),
        }

    def _stub_metric_dependencies(self):
        pandas_mod = types.ModuleType("pandas")
        pandas_mod.DataFrame = FakeDataFrame
        pandas_mod.api = types.SimpleNamespace(types=types.SimpleNamespace(is_numeric_dtype=lambda values: True))
        pandas_api_mod = types.ModuleType("pandas.api")
        pandas_api_types_mod = types.ModuleType("pandas.api.types")
        pandas_api_types_mod.is_numeric_dtype = lambda values: True
        numpy_mod = types.ModuleType("numpy")
        kmu_mod = types.ModuleType("kaggle_metric_utilities")
        kmu_mod.safe_call_score = lambda func, *args, **kwargs: func(*args, **kwargs)
        return {
            "pandas": pandas_mod,
            "pandas.api": pandas_api_mod,
            "pandas.api.types": pandas_api_types_mod,
            "numpy": numpy_mod,
            "kaggle_metric_utilities": kmu_mod,
        }

    def test_score_rows_returns_one_for_perfect_submission(self) -> None:
        solution = [
            self._solution_row(
                "row_000001",
                "زار محمد احمد الرباط",
                "زار [PERSON_1] [PERSON_2] [LOC_1]",
                [
                    {"start": 4, "end": 8, "placeholder": "[PERSON_1]", "surface_form": "محمد", "mask_type": "PERSON"},
                    {"start": 9, "end": 13, "placeholder": "[PERSON_2]", "surface_form": "احمد", "mask_type": "PERSON"},
                    {"start": 14, "end": 20, "placeholder": "[LOC_1]", "surface_form": "الرباط", "mask_type": "LOC"},
                ],
            )
        ]
        submission = [{"id": "row_000001", "masked_text": "زار [PERSON_1] [PERSON_2] [LOC_1]"}]
        self.assertEqual(score_rows(solution, submission), 1.0)

    def test_score_rows_penalizes_missing_span(self) -> None:
        solution = [
            self._solution_row(
                "row_000001",
                "زار محمد احمد الرباط",
                "زار [PERSON_1] [PERSON_2] [LOC_1]",
                [
                    {"start": 4, "end": 8, "placeholder": "[PERSON_1]", "surface_form": "محمد", "mask_type": "PERSON"},
                    {"start": 9, "end": 13, "placeholder": "[PERSON_2]", "surface_form": "احمد", "mask_type": "PERSON"},
                    {"start": 14, "end": 20, "placeholder": "[LOC_1]", "surface_form": "الرباط", "mask_type": "LOC"},
                ],
            )
        ]
        submission = [{"id": "row_000001", "masked_text": "زار [PERSON_1] احمد [LOC_1]"}]
        self.assertAlmostEqual(score_rows(solution, submission), 0.8)

    def test_score_rows_penalizes_wrong_mask_type(self) -> None:
        solution = [
            self._solution_row(
                "row_000001",
                "زار محمد الرباط",
                "زار [PERSON_1] [LOC_1]",
                [
                    {"start": 4, "end": 8, "placeholder": "[PERSON_1]", "surface_form": "محمد", "mask_type": "PERSON"},
                    {"start": 9, "end": 15, "placeholder": "[LOC_1]", "surface_form": "الرباط", "mask_type": "LOC"},
                ],
            )
        ]
        submission = [{"id": "row_000001", "masked_text": "زار [LOC_1] [PERSON_1]"}]
        self.assertEqual(score_rows(solution, submission), 0.0)

    def test_score_rows_treats_invalid_masked_text_as_zero_predicted_spans(self) -> None:
        solution = [
            self._solution_row(
                "row_000001",
                "زار محمد الرباط",
                "زار [PERSON_1] [LOC_1]",
                [
                    {"start": 4, "end": 8, "placeholder": "[PERSON_1]", "surface_form": "محمد", "mask_type": "PERSON"},
                    {"start": 9, "end": 15, "placeholder": "[LOC_1]", "surface_form": "الرباط", "mask_type": "LOC"},
                ],
            )
        ]
        submission = [{"id": "row_000001", "masked_text": "نص مختلف تماما [PERSON_1]"}]
        self.assertEqual(score_rows(solution, submission), 0.0)

    def test_score_rows_returns_zero_for_row_count_mismatch(self) -> None:
        solution = [self._solution_row("row_000001", "زار محمد", "زار [PERSON_1]", [{"start": 4, "end": 8, "placeholder": "[PERSON_1]", "surface_form": "محمد", "mask_type": "PERSON"}])]
        submission = []
        self.assertEqual(score_rows(solution, submission), 0.0)

    def test_standalone_reconstruction_matches_workflow_helper(self) -> None:
        original_text = "محمد قابل أحمد في الرباط ثم اتصل محمد بأحمد"
        masked_text = "[PERSON_1] قابل [PERSON_2] في [LOC_1] ثم اتصل [PERSON_1] ب[PERSON_2]"
        standalone_spans, standalone_errors = reconstruct_mask_metadata(original_text, masked_text)
        workflow_spans, workflow_errors = workflow_reconstruct_mask_metadata(original_text, masked_text)
        self.assertEqual(standalone_errors, workflow_errors)
        self.assertEqual(standalone_spans, workflow_spans)

    def test_build_metric_source_contains_kaggle_template_imports(self) -> None:
        source = build_metric_source()
        self.assertIn("import pandas as pd", source)
        self.assertIn("import kaggle_metric_utilities", source)
        self.assertIn("safe_call_score", source)
        self.assertIn("def score(", source)

    def test_metric_artifact_is_importable_with_stubbed_kaggle_dependencies(self) -> None:
        stubs = self._stub_metric_dependencies()
        previous = {name: sys.modules.get(name) for name in stubs}
        try:
            sys.modules.update(stubs)
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = write_metric_artifact(Path(tmpdir) / "metric.py")
                spec = spec_from_file_location("kaggle_metric_artifact", artifact_path)
                assert spec is not None and spec.loader is not None
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                solution = FakeDataFrame(
                    [
                        {
                            "id": "row_000001",
                            "original_text": "زار محمد",
                            "mask_spans_json": json.dumps(
                                [{"start": 4, "end": 8, "placeholder": "[PERSON_1]", "surface_form": "محمد", "mask_type": "PERSON"}],
                                ensure_ascii=False,
                            ),
                        }
                    ]
                )
                submission = FakeDataFrame([{"id": "row_000001", "masked_text": "زار [PERSON_1]"}])
                self.assertEqual(module.score(solution, submission, "id"), 1.0)
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
