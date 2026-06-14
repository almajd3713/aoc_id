import csv
import tempfile
import unittest
from pathlib import Path

from annotation_app.backend.workflow import (
    OUTPUT_COLUMNS,
    accept_import,
    build_prompt,
    get_next_chunk,
    load_config,
    normalize_pasted_csv_text,
    preview_import,
    save_config,
)


class AnnotationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.project_dir = root / "project"
        self.source_chunks = self.project_dir / "source_chunks"
        self.workspace_dir = self.project_dir / "workspace"
        self.source_chunks.mkdir(parents=True)
        (self.workspace_dir / "accepted").mkdir(parents=True)
        (self.workspace_dir / "merged").mkdir(parents=True)
        (self.project_dir / "docs").mkdir(parents=True)
        (self.project_dir / "app_state").mkdir(parents=True)

        self.chunk_path = self.source_chunks / "chunk_0001.csv"
        with self.chunk_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "example_id": "ex1",
                    "source_split": "train",
                    "source_row_id": "1",
                    "dialect": "MSA",
                    "original_text": "زار محمد صلاح القاهرة",
                    "normalized_text": "زار محمد صلاح القاهره",
                    "masked_text": "",
                    "mask_spans_json": "[]",
                    "mask_count": "0",
                    "annotation_status": "pending",
                    "annotator_model": "",
                    "notes": "",
                }
            )

        manifest = self.project_dir / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["chunk_id", "file_name", "row_count", "dialect_counts_json", "sha256", "status"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "chunk_id": "1",
                    "file_name": "chunk_0001.csv",
                    "row_count": "1",
                    "dialect_counts_json": '{"MSA": 1}',
                    "sha256": "dummy",
                    "status": "pending",
                }
            )

        prompt_template = self.project_dir / "app_state" / "prompt.txt"
        prompt_template.write_text(
            "chunk {chunk_file} rows {row_count}\n{guidelines}\nCSV:\n{chunk_csv}\n",
            encoding="utf-8",
        )
        guidelines = self.project_dir / "docs" / "guidelines.md"
        guidelines.write_text("Rules here", encoding="utf-8")
        session_state = self.workspace_dir / "session_state.json"
        session_state.write_text(
            '{"project_id":"pilot","current_chunk_id":1,"accepted_chunks":[],"last_merge_summary":null}',
            encoding="utf-8",
        )

        self.config_path = self.project_dir / "app_state" / "config.json"
        save_config(
            {
                "default_project": "pilot",
                "projects": {
                    "pilot": {
                        "dataset_mode": "pilot",
                        "workspace_dir": str(self.workspace_dir),
                        "source_chunks_dir": str(self.source_chunks),
                        "source_manifest_path": str(manifest),
                        "prompt_template_path": str(prompt_template),
                        "masking_guidelines_path": str(guidelines),
                        "auto_advance": True,
                        "auto_merge": True,
                        "allow_pending_accept": False,
                    }
                },
            },
            self.config_path,
        )
        self.config = load_config(self.config_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preview_import_accepts_valid_csv(self) -> None:
        csv_text = (
            "example_id,source_split,source_row_id,dialect,original_text,normalized_text,masked_text,"
            "mask_spans_json,mask_count,annotation_status,annotator_model,notes\n"
            'ex1,train,1,MSA,زار محمد صلاح القاهرة,زار محمد صلاح القاهره,زار [PERSON_1] [LOC_1],"'
            '[{""start"": 4, ""end"": 13, ""placeholder"": ""[PERSON_1]"", ""surface_form"": ""محمد صلاح"", ""mask_type"": ""PERSON""}, '
            '{""start"": 14, ""end"": 21, ""placeholder"": ""[LOC_1]"", ""surface_form"": ""القاهرة"", ""mask_type"": ""LOC""}]'
            '",2,approved,test,\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)

    def test_accept_import_writes_managed_copy(self) -> None:
        csv_text = (
            "example_id,source_split,source_row_id,dialect,original_text,normalized_text,masked_text,"
            "mask_spans_json,mask_count,annotation_status,annotator_model,notes\n"
            'ex1,train,1,MSA,زار محمد صلاح القاهرة,زار محمد صلاح القاهره,زار [PERSON_1] [LOC_1],"'
            '[{""start"": 4, ""end"": 13, ""placeholder"": ""[PERSON_1]"", ""surface_form"": ""محمد صلاح"", ""mask_type"": ""PERSON""}, '
            '{""start"": 14, ""end"": 21, ""placeholder"": ""[LOC_1]"", ""surface_form"": ""القاهرة"", ""mask_type"": ""LOC""}]'
            '",2,approved,test,\n'
        )
        result = accept_import("pilot", 1, csv_text, self.config)
        self.assertTrue(result["ok"])
        self.assertTrue((self.workspace_dir / "accepted" / "chunk_0001.csv").exists())
        self.assertTrue((self.workspace_dir / "merged" / "merge_summary.json").exists())

    def test_next_chunk_returns_pending_chunk(self) -> None:
        next_chunk = get_next_chunk("pilot", self.config)
        self.assertEqual(next_chunk["chunk_id"], 1)

    def test_build_prompt_includes_raw_chunk_csv(self) -> None:
        prompt = build_prompt("pilot", 1, self.config)
        self.assertIn("chunk_0001.csv", prompt)
        self.assertIn("example_id,source_split,source_row_id,dialect", prompt)
        self.assertIn("ex1,train,1,MSA", prompt)

    def test_preview_import_returns_detailed_validation_error(self) -> None:
        csv_text = (
            "example_id,source_split,source_row_id,dialect,original_text,normalized_text,masked_text,"
            "mask_spans_json,mask_count,annotation_status,annotator_model,notes\n"
            'ex1,train,1,MSA,زار محمد صلاح القاهرة,زار محمد صلاح القاهره,زار [PERSON_1] القاهرة,"'
            '[{""start"": 0, ""end"": 5, ""placeholder"": ""[PERSON_1]"", ""surface_form"": ""محمد"", ""mask_type"": ""PERSON""}]'
            '",1,approved,test,\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["invalid_rows"], 1)
        self.assertTrue(
            any("surface_form mismatch" in error for error in preview["validation_rows"][0]["errors"])
        )

    def test_preview_import_marks_pending_only_rows_as_pending(self) -> None:
        csv_text = (
            "example_id,source_split,source_row_id,dialect,original_text,normalized_text,masked_text,"
            "mask_spans_json,mask_count,annotation_status,annotator_model,notes\n"
            'ex1,train,1,MSA,زار محمد صلاح القاهرة,زار محمد صلاح القاهره,,,0,pending,test,needs review\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["pending_rows"], 1)
        self.assertEqual(preview["summary"]["invalid_rows"], 0)

    def test_normalize_pasted_csv_text_strips_markdown_fences(self) -> None:
        csv_text = "```csv\nheader1,header2\nvalue1,value2\n```"
        normalized = normalize_pasted_csv_text(csv_text)
        self.assertEqual(normalized, "header1,header2\nvalue1,value2")

    def test_preview_import_accepts_markdown_fenced_csv(self) -> None:
        csv_text = (
            "```csv\n"
            "example_id,source_split,source_row_id,dialect,original_text,normalized_text,masked_text,"
            "mask_spans_json,mask_count,annotation_status,annotator_model,notes\n"
            'ex1,train,1,MSA,زار محمد صلاح القاهرة,زار محمد صلاح القاهره,زار [PERSON_1] [LOC_1],"'
            '[{""start"": 4, ""end"": 13, ""placeholder"": ""[PERSON_1]"", ""surface_form"": ""محمد صلاح"", ""mask_type"": ""PERSON""}, '
            '{""start"": 14, ""end"": 21, ""placeholder"": ""[LOC_1]"", ""surface_form"": ""القاهرة"", ""mask_type"": ""LOC""}]'
            '",2,approved,test,\n'
            "```"
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)


if __name__ == "__main__":
    unittest.main()
