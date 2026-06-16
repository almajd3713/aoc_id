import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_kaggle_dataset import (
    ID_MAPPING_COLUMNS,
    LABEL_COLUMNS,
    SOLUTION_COLUMNS,
    SUBMISSION_COLUMNS,
    SUPPORTED_MASK_TYPES,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
    prepare_kaggle_dataset,
)


class PrepareKaggleDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_csv = self.root / "masked_dataset_merged.csv"
        self.output_dir = self.root / "kaggle"
        self.rows = self._build_fixture_rows()
        with self.input_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_row(
        self,
        example_id: str,
        dialect: str,
        original_text: str,
        masked_text: str,
        mask_types: list[str],
        annotation_status: str = "approved",
    ) -> dict[str, str]:
        spans = []
        search_start = 0
        for index, mask_type in enumerate(mask_types, start=1):
            surface = f"اسم{index}"
            start = original_text.index(surface, search_start)
            end = start + len(surface)
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "placeholder": f"[{mask_type}_{index}]",
                    "surface_form": surface,
                    "mask_type": mask_type,
                }
            )
            search_start = end
        return {
            "example_id": example_id,
            "source_split": "train",
            "source_row_id": example_id.split("_")[-1],
            "dialect": dialect,
            "original_text": original_text,
            "normalized_text": original_text,
            "masked_text": masked_text,
            "mask_spans_json": json.dumps(spans, ensure_ascii=False),
            "mask_count": str(len(spans)),
            "annotation_status": annotation_status,
            "annotator_model": "",
            "notes": "",
        }

    def _build_fixture_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        dialects = ["MSA", "DIAL_EGY", "DIAL_GLF", "DIAL_LEV"]
        buckets = [
            ["PERSON"],
            ["PERSON", "LOC"],
            ["PERSON", "LOC", "ORG"],
        ]
        counter = 1
        for dialect in dialects:
            for mask_types in buckets:
                for variant in range(5):
                    surfaces = [f"اسم{i}" for i in range(1, len(mask_types) + 1)]
                    original_text = f"{dialect} مثال {variant} " + " ".join(surfaces)
                    masked_text = f"{dialect} مثال {variant} " + " ".join(
                        f"[{mask_type}_{index}]"
                        for index, mask_type in enumerate(mask_types, start=1)
                    )
                    rows.append(
                        self._make_row(
                            example_id=f"ex_{counter:03d}",
                            dialect=dialect,
                            original_text=original_text,
                            masked_text=masked_text,
                            mask_types=mask_types,
                        )
                    )
                    counter += 1

        for dialect in dialects:
            rows.append(
                self._make_row(
                    example_id=f"ex_{counter:03d}",
                    dialect=dialect,
                    original_text=f"{dialect} نص بدون كيانات",
                    masked_text=f"{dialect} نص بدون كيانات",
                    mask_types=[],
                )
            )
            counter += 1

        rows.append(
            self._make_row(
                example_id=f"ex_{counter:03d}",
                dialect="MSA",
                original_text="MSA نص مرفوض اسم1",
                masked_text="MSA نص مرفوض [EVENT_1]",
                mask_types=["EVENT"],
            )
        )
        counter += 1
        rows.append(
            self._make_row(
                example_id=f"ex_{counter:03d}",
                dialect="MSA",
                original_text="MSA نص معلق اسم1",
                masked_text="MSA نص معلق [PERSON_1]",
                mask_types=["PERSON"],
                annotation_status="pending",
            )
        )
        return rows

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_prepare_kaggle_dataset_builds_expected_files(self) -> None:
        summary = prepare_kaggle_dataset(
            input_csv=self.input_csv,
            output_dir=self.output_dir,
            seed=17,
            public_fraction=0.1,
            private_fraction=0.1,
        )

        train_rows = self._read_csv(self.output_dir / "train.csv")
        test_rows = self._read_csv(self.output_dir / "test.csv")
        sample_rows = self._read_csv(self.output_dir / "sample_submission.csv")
        public_labels = self._read_csv(self.output_dir / "public_labels.csv")
        private_labels = self._read_csv(self.output_dir / "private_labels.csv")
        solution_rows = self._read_csv(self.output_dir / "solution.csv")
        mapping_rows = self._read_csv(self.output_dir / "id_mapping.csv")
        excluded_rows = self._read_csv(self.output_dir / "excluded_rows.csv")
        metric_artifact = self.output_dir / "metric.py"

        self.assertEqual(list(train_rows[0].keys()), TRAIN_COLUMNS)
        self.assertEqual(list(test_rows[0].keys()), TEST_COLUMNS)
        self.assertEqual(list(sample_rows[0].keys()), SUBMISSION_COLUMNS)
        self.assertEqual(list(public_labels[0].keys()), LABEL_COLUMNS)
        self.assertEqual(list(solution_rows[0].keys()), SOLUTION_COLUMNS)
        self.assertEqual(list(mapping_rows[0].keys()), ID_MAPPING_COLUMNS)
        self.assertEqual(summary["participant_files_hide_dialect"], True)
        self.assertEqual(summary["excluded_row_count"], 1)
        self.assertEqual(summary["solution_split_column"], "Usage")
        self.assertEqual(len(excluded_rows), 1)
        self.assertEqual(excluded_rows[0]["exclusion_reason"], "unsupported_mask_types:EVENT")
        self.assertEqual(summary["train_id_format"], "train_XXXXXX")
        self.assertEqual(summary["test_id_format"], "test_XXXXXX")

        self.assertEqual(len(train_rows), 40)
        self.assertEqual(len(test_rows), 24)
        self.assertEqual(len(public_labels), 12)
        self.assertEqual(len(private_labels), 12)
        self.assertEqual(len(solution_rows), 24)
        self.assertEqual(len(mapping_rows), 64)
        self.assertEqual(len(sample_rows), len(test_rows))
        self.assertTrue(all(row["masked_text"] == "" for row in sample_rows))
        self.assertTrue(metric_artifact.exists())
        self.assertEqual(summary["metric_artifact"], "metric.py")

    def test_prepare_kaggle_dataset_hides_dialect_from_participants(self) -> None:
        prepare_kaggle_dataset(self.input_csv, self.output_dir, seed=17, public_fraction=0.1, private_fraction=0.1)

        train_rows = self._read_csv(self.output_dir / "train.csv")
        test_rows = self._read_csv(self.output_dir / "test.csv")
        mapping_rows = self._read_csv(self.output_dir / "id_mapping.csv")

        self.assertNotIn("dialect", train_rows[0])
        self.assertNotIn("dialect", test_rows[0])
        for row in train_rows[:5]:
            self.assertTrue(row["id"].startswith("train_"))
            self.assertNotIn("dial", row["id"].lower())
            self.assertNotIn("msa", row["id"].lower())
            self.assertNotIn("ex_", row["id"].lower())
        for row in test_rows[:5]:
            self.assertTrue(row["id"].startswith("test_"))
            self.assertNotIn("dial", row["id"].lower())
            self.assertNotIn("msa", row["id"].lower())
            self.assertNotIn("ex_", row["id"].lower())

        mapping_by_id = {row["id"]: row for row in mapping_rows}
        self.assertEqual(set(mapping_by_id), {row["id"] for row in train_rows} | {row["id"] for row in test_rows})
        self.assertEqual({row["split"] for row in mapping_rows}, {"train", "test"})

    def test_prepare_kaggle_dataset_keeps_test_rows_masked_and_supported(self) -> None:
        prepare_kaggle_dataset(self.input_csv, self.output_dir, seed=17, public_fraction=0.1, private_fraction=0.1)

        public_labels = self._read_csv(self.output_dir / "public_labels.csv")
        private_labels = self._read_csv(self.output_dir / "private_labels.csv")

        self.assertFalse({row["id"] for row in public_labels} & {row["id"] for row in private_labels})
        for row in [*public_labels, *private_labels]:
            spans = json.loads(row["mask_spans_json"])
            self.assertGreater(len(spans), 0)
            self.assertTrue(all(span["mask_type"] in SUPPORTED_MASK_TYPES for span in spans))

    def test_prepare_kaggle_dataset_builds_combined_solution_with_usage_split(self) -> None:
        prepare_kaggle_dataset(self.input_csv, self.output_dir, seed=17, public_fraction=0.1, private_fraction=0.1)

        solution_rows = self._read_csv(self.output_dir / "solution.csv")
        public_labels = {row["id"]: row for row in self._read_csv(self.output_dir / "public_labels.csv")}
        private_labels = {row["id"]: row for row in self._read_csv(self.output_dir / "private_labels.csv")}

        usage_values = {row["Usage"] for row in solution_rows}
        self.assertEqual(usage_values, {"Public", "Private"})
        self.assertEqual(len(solution_rows), len(public_labels) + len(private_labels))

        for row in solution_rows:
            if row["Usage"] == "Public":
                self.assertEqual(row["masked_text"], public_labels[row["id"]]["masked_text"])
            else:
                self.assertEqual(row["masked_text"], private_labels[row["id"]]["masked_text"])

    def test_prepare_kaggle_dataset_shuffles_participant_order_across_dialects(self) -> None:
        prepare_kaggle_dataset(self.input_csv, self.output_dir, seed=17, public_fraction=0.1, private_fraction=0.1)

        mapping_by_id = {
            row["id"]: row["dialect"] for row in self._read_csv(self.output_dir / "id_mapping.csv")
        }
        train_ids = [row["id"] for row in self._read_csv(self.output_dir / "train.csv")]
        test_ids = [row["id"] for row in self._read_csv(self.output_dir / "test.csv")]
        train_dialects = [mapping_by_id[row_id] for row_id in train_ids]
        test_dialects = [mapping_by_id[row_id] for row_id in test_ids]

        self.assertGreater(len(set(train_dialects[:8])), 1)
        self.assertGreater(len(set(test_dialects[:8])), 1)
        self.assertGreater(len(set(train_dialects)), 1)
        self.assertGreater(len(set(test_dialects)), 1)

    def test_prepare_kaggle_dataset_has_no_split_leakage(self) -> None:
        prepare_kaggle_dataset(self.input_csv, self.output_dir, seed=17, public_fraction=0.1, private_fraction=0.1)

        train_rows = self._read_csv(self.output_dir / "train.csv")
        test_rows = self._read_csv(self.output_dir / "test.csv")
        public_ids = {row["id"] for row in self._read_csv(self.output_dir / "public_labels.csv")}
        private_ids = {row["id"] for row in self._read_csv(self.output_dir / "private_labels.csv")}
        train_ids = {row["id"] for row in train_rows}
        test_ids = {row["id"] for row in test_rows}

        self.assertFalse(train_ids & test_ids)
        self.assertEqual(test_ids, public_ids | private_ids)
        self.assertEqual(len(test_ids), len(test_rows))

        zero_mask_example_ids = {
            row["example_id"]
            for row in self.rows
            if row["annotation_status"] == "approved" and int(row["mask_count"]) == 0
        }
        id_mapping = {
            row["example_id"]: row["id"] for row in self._read_csv(self.output_dir / "id_mapping.csv")
        }
        zero_mask_public_ids = {id_mapping[example_id] for example_id in zero_mask_example_ids}
        self.assertTrue(zero_mask_public_ids.issubset(train_ids))
        self.assertFalse(zero_mask_public_ids & test_ids)

    def test_prepare_kaggle_dataset_is_deterministic(self) -> None:
        first_output = self.root / "out_one"
        second_output = self.root / "out_two"

        prepare_kaggle_dataset(self.input_csv, first_output, seed=23, public_fraction=0.1, private_fraction=0.1)
        prepare_kaggle_dataset(self.input_csv, second_output, seed=23, public_fraction=0.1, private_fraction=0.1)

        for file_name in [
            "train.csv",
            "test.csv",
            "sample_submission.csv",
            "public_labels.csv",
            "private_labels.csv",
            "solution.csv",
            "id_mapping.csv",
            "excluded_rows.csv",
            "split_manifest.json",
        ]:
            self.assertEqual(
                (first_output / file_name).read_text(encoding="utf-8"),
                (second_output / file_name).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
