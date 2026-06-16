import csv
import tempfile
import unittest
from pathlib import Path

from annotation_app.backend.workflow import (
    AGENT_COLUMNS,
    AGENT_COLUMNS_WITH_ROW_ID,
    ConfigValidationError,
    OUTPUT_COLUMNS,
    accept_import,
    activate_refill_round,
    apply_retry_import,
    build_prompt,
    clear_invalid_retry_rows,
    generate_refill_round,
    get_next_chunk,
    load_backlog_rows,
    load_config,
    get_project_paths,
    get_working_chunk_path,
    load_session_state,
    normalize_pasted_csv_text,
    preview_import,
    preview_retry_import,
    save_config,
    skip_invalid_retry_cache,
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

        clean_candidates_path = self.project_dir / "clean_candidates.csv"
        with clean_candidates_path.open("w", encoding="utf-8", newline="") as handle:
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
            writer.writerow(
                {
                    "example_id": "ex2",
                    "source_split": "train",
                    "source_row_id": "2",
                    "dialect": "MSA",
                    "original_text": "سافر أحمد إلى الرباط",
                    "normalized_text": "سافر احمد الى الرباط",
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
                        "include_row_id_in_prompt": False,
                        "agent_import_schema": "original_masked_v1",
                        "row_matching": "prefer_row_id",
                        "uncertainty_markers": ["[UNCERTAIN]", "[REVIEW]"],
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
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)
        reconstructed = preview["reconstructed_rows"][0]
        self.assertEqual(reconstructed["mask_count"], "1")
        self.assertEqual(reconstructed["annotation_status"], "approved")

    def test_load_config_normalizes_missing_optional_fields(self) -> None:
        self.assertEqual(self.config["ui"]["copy_compact_prompt"], True)
        self.assertEqual(self.config["projects"]["pilot"]["label"], "pilot")
        self.assertEqual(self.config["projects"]["pilot"]["validate_original_text_with_row_id"], True)
        self.assertEqual(
            self.config["projects"]["pilot"]["base_source_chunks_dir"],
            str(self.source_chunks),
        )
        self.assertEqual(
            self.config["projects"]["pilot"]["base_source_manifest_path"],
            str(self.project_dir / "manifest.csv"),
        )
        self.assertEqual(self.config["projects"]["pilot"]["active_round"], "base")

    def test_save_config_rejects_invalid_row_matching(self) -> None:
        bad_config = {
            "default_project": "pilot",
            "projects": {
                "pilot": {
                    "workspace_dir": str(self.workspace_dir),
                    "source_chunks_dir": str(self.source_chunks),
                    "source_manifest_path": str(self.project_dir / "manifest.csv"),
                    "prompt_template_path": str(self.project_dir / "app_state" / "prompt.txt"),
                    "masking_guidelines_path": str(self.project_dir / "docs" / "guidelines.md"),
                    "row_matching": "bad_value",
                }
            },
        }
        with self.assertRaises(ConfigValidationError):
            save_config(bad_config, self.config_path)

    def test_accept_import_writes_managed_copy(self) -> None:
        csv_text = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
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
        self.assertIn(",".join(AGENT_COLUMNS), prompt)
        self.assertNotIn("example_id,source_split,source_row_id,dialect", prompt)
        self.assertNotIn("mask_spans_json", prompt)
        self.assertNotIn("annotation_status", prompt)
        self.assertIn("زار محمد صلاح القاهرة,", prompt)

    def test_build_prompt_can_include_row_ids_when_enabled(self) -> None:
        self.config["projects"]["pilot"]["include_row_id_in_prompt"] = True
        prompt = build_prompt("pilot", 1, self.config)
        self.assertIn(",".join(AGENT_COLUMNS_WITH_ROW_ID), prompt)
        self.assertIn("1,زار محمد صلاح القاهرة,", prompt)

    def test_build_prompt_for_strict_row_id_includes_row_ids_even_when_prompt_toggle_is_off(self) -> None:
        self.config["projects"]["pilot"]["include_row_id_in_prompt"] = False
        self.config["projects"]["pilot"]["row_matching"] = "strict_row_id"
        prompt = build_prompt("pilot", 1, self.config)
        self.assertIn(",".join(AGENT_COLUMNS_WITH_ROW_ID), prompt)
        self.assertIn("1,زار محمد صلاح القاهرة,", prompt)

    def test_preview_import_returns_detailed_validation_error(self) -> None:
        csv_text = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["invalid_rows"], 1)
        self.assertTrue(
            any("automatic span reconstruction is ambiguous" in error for error in preview["validation_rows"][0]["errors"])
        )

    def test_preview_import_marks_pending_only_rows_as_pending(self) -> None:
        csv_text = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار محمد صلاح القاهرة [UNCERTAIN]\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["pending_rows"], 1)
        self.assertEqual(preview["summary"]["invalid_rows"], 0)
        self.assertEqual(preview["reconstructed_rows"][0]["annotation_status"], "pending")

    def test_normalize_pasted_csv_text_strips_markdown_fences(self) -> None:
        csv_text = "```csv\nheader1,header2\nvalue1,value2\n```"
        normalized = normalize_pasted_csv_text(csv_text)
        self.assertEqual(normalized, "header1,header2\nvalue1,value2")

    def test_preview_import_accepts_markdown_fenced_csv(self) -> None:
        csv_text = (
            "```csv\n"
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
            "```"
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)

    def test_preview_import_rejects_row_with_unknown_original_text(self) -> None:
        csv_text = (
            "original_text,masked_text\n"
            'نص غير موجود,نص غير موجود\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["error"], "original_text_not_found")

    def test_preview_import_rejects_wrong_headers(self) -> None:
        csv_text = (
            "example_id,masked_text\n"
            'ex1,زار [PERSON_1] [LOC_1]\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["expected_headers"], AGENT_COLUMNS)
        self.assertEqual(preview["accepted_header_sets"], [AGENT_COLUMNS, AGENT_COLUMNS_WITH_ROW_ID])

    def test_preview_import_accepts_optional_row_id_column(self) -> None:
        csv_text = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)

    def test_preview_import_rejects_unknown_row_id(self) -> None:
        csv_text = (
            "row_id,original_text,masked_text\n"
            '999,زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["error"], "row_id_not_found")
        self.assertEqual(preview["missing_row_ids"], ["999"])

    def test_preview_import_rejects_row_id_original_text_mismatch(self) -> None:
        csv_text = (
            "row_id,original_text,masked_text\n"
            '1,نص مختلف,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["error"], "row_id_original_text_mismatch")

    def test_preview_import_can_skip_original_text_validation_for_row_id(self) -> None:
        self.config["projects"]["pilot"]["validate_original_text_with_row_id"] = False
        csv_text = (
            "row_id,original_text,masked_text\n"
            '1,نص مختلف,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)

    def test_preview_import_strict_row_id_requires_row_id_column(self) -> None:
        self.config["projects"]["pilot"]["row_matching"] = "strict_row_id"
        csv_text = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertFalse(preview["ok"])
        self.assertEqual(preview["error"], "missing_row_id_column")

    def test_preview_import_strict_original_text_ignores_bad_row_id(self) -> None:
        self.config["projects"]["pilot"]["row_matching"] = "strict_original_text"
        csv_text = (
            "row_id,original_text,masked_text\n"
            '999,زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["summary"]["approved_rows"], 1)

    def test_preview_import_caches_invalid_rows_for_retry(self) -> None:
        csv_text = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview = preview_import("pilot", 1, csv_text, self.config)
        self.assertTrue(preview["ok"])
        self.assertEqual(len(preview["invalid_retry_rows"]), 1)
        self.assertEqual(preview["invalid_retry_rows"][0]["original_text"], "زار محمد صلاح القاهرة")
        self.assertEqual(preview["invalid_retry_rows"][0]["row_id"], "1")

        paths = get_project_paths("pilot", self.config)
        state = load_session_state(paths)
        self.assertEqual(len(state["invalid_retry_rows"]), 1)

    def test_accept_import_clears_cached_invalid_rows_for_chunk(self) -> None:
        invalid_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        valid_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        result = accept_import("pilot", 1, valid_csv, self.config)
        self.assertTrue(result["ok"])

        paths = get_project_paths("pilot", self.config)
        state = load_session_state(paths)
        self.assertEqual(state["invalid_retry_rows"], [])

    def test_preview_retry_import_requires_working_preview(self) -> None:
        retry_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        result = preview_retry_import("pilot", 1, retry_csv, self.config)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "missing_working_preview")

    def test_apply_retry_import_updates_working_preview_and_cache_status(self) -> None:
        invalid_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        retry_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        retry_preview = preview_retry_import("pilot", 1, retry_csv, self.config)
        self.assertTrue(retry_preview["ok"])
        self.assertEqual(len(retry_preview["resolved_retry_rows"]), 1)
        self.assertEqual(retry_preview["summary"]["approved_rows"], 1)

        apply_result = apply_retry_import("pilot", 1, retry_csv, self.config)
        self.assertTrue(apply_result["ok"])
        self.assertIn("merged_agent_csv", apply_result)

        paths = get_project_paths("pilot", self.config)
        working_path = get_working_chunk_path(paths, "chunk_0001.csv")
        self.assertTrue(working_path.exists())
        working_csv = working_path.read_text(encoding="utf-8")
        self.assertIn("زار [PERSON_1] القاهرة", working_csv)

        state = load_session_state(paths)
        self.assertEqual(state["invalid_retry_rows"], [])

    def test_preview_retry_import_strict_row_id_requires_row_id_column(self) -> None:
        self.config["projects"]["pilot"]["row_matching"] = "strict_row_id"
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        retry_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        result = preview_retry_import("pilot", 1, retry_csv, self.config)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "missing_row_id_column")

    def test_preview_retry_import_accepts_row_id_matching(self) -> None:
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        retry_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] القاهرة\n'
        )
        result = preview_retry_import("pilot", 1, retry_csv, self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["approved_rows"], 1)

    def test_preview_retry_import_can_skip_original_text_validation_for_row_id(self) -> None:
        self.config["projects"]["pilot"]["validate_original_text_with_row_id"] = False
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        retry_csv = (
            "row_id,original_text,masked_text\n"
            '1,نص مختلف,زار [PERSON_1] القاهرة\n'
        )
        result = preview_retry_import("pilot", 1, retry_csv, self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["approved_rows"], 1)

    def test_apply_retry_import_drops_unresolved_rows_to_backlog(self) -> None:
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        retry_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        result = apply_retry_import("pilot", 1, retry_csv, self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["dropped_backlog_row_count"], 1)
        self.assertEqual(result["summary"]["row_count"], 0)

        paths = get_project_paths("pilot", self.config)
        backlog_rows = load_backlog_rows(paths)
        self.assertEqual(len(backlog_rows), 1)
        self.assertEqual(backlog_rows[0]["example_id"], "ex1")

    def test_accept_import_uses_reduced_working_preview(self) -> None:
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)
        apply_retry_import("pilot", 1, invalid_csv, self.config)

        result = accept_import("pilot", 1, "", self.config)
        self.assertTrue(result["ok"])
        accepted_csv = (self.workspace_dir / "accepted" / "chunk_0001.csv").read_text(encoding="utf-8")
        self.assertIn("example_id", accepted_csv)

    def test_generate_refill_round_uses_backlog_count(self) -> None:
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)
        apply_retry_import("pilot", 1, invalid_csv, self.config)

        result = generate_refill_round("pilot", self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["sampled_count"], 1)
        manifest_path = Path(result["manifest_path"])
        self.assertTrue(manifest_path.exists())
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        self.assertEqual(manifest_rows[0]["file_name"], "round_0001_chunk_0001.csv")

    def test_activate_refill_round_switches_active_source_paths(self) -> None:
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)
        apply_retry_import("pilot", 1, invalid_csv, self.config)
        generate_refill_round("pilot", self.config)

        result = activate_refill_round("pilot", config=self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(self.config["projects"]["pilot"]["active_round"], "round_0001")
        self.assertTrue(self.config["projects"]["pilot"]["source_manifest_path"].endswith("round_0001/chunk_manifest.csv"))

        paths = get_project_paths("pilot", self.config)
        self.assertEqual(paths.active_round, "round_0001")
        self.assertTrue(paths.source_manifest_path.name == "chunk_manifest.csv")
        next_chunk = get_next_chunk("pilot", self.config)
        self.assertEqual(next_chunk["file_name"], "round_0001_chunk_0001.csv")

    def test_activate_refill_round_renames_legacy_chunk_names(self) -> None:
        invalid_csv = (
            "row_id,original_text,masked_text\n"
            '1,زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)
        apply_retry_import("pilot", 1, invalid_csv, self.config)
        result = generate_refill_round("pilot", self.config)
        round_dir = Path(result["round_dir"])
        legacy_path = round_dir / "chunks" / "chunk_0001.csv"
        new_path = round_dir / "chunks" / "round_0001_chunk_0001.csv"
        new_path.rename(legacy_path)
        manifest_path = Path(result["manifest_path"])
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        manifest_rows[0]["file_name"] = "chunk_0001.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["chunk_id", "file_name", "row_count", "dialect_counts_json", "sha256", "status"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        activate_refill_round("pilot", config=self.config)
        self.assertTrue(new_path.exists())
        self.assertFalse(legacy_path.exists())

    def test_copy_source_should_only_need_original_text(self) -> None:
        invalid_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview = preview_import("pilot", 1, invalid_csv, self.config)
        self.assertEqual(preview["invalid_retry_rows"][0]["original_text"], "زار محمد صلاح القاهرة")

    def test_clear_invalid_retry_rows_empties_cache(self) -> None:
        invalid_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)
        result = clear_invalid_retry_rows("pilot", config=self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["cleared_rows"], 1)

        paths = get_project_paths("pilot", self.config)
        state = load_session_state(paths)
        self.assertEqual(state["invalid_retry_rows"], [])

    def test_skip_invalid_retry_cache_advances_chunk_and_moves_rows_to_backlog(self) -> None:
        invalid_csv = (
            "original_text,masked_text\n"
            'زار محمد صلاح القاهرة,زار [PERSON_1] [LOC_1]\n'
        )
        preview_import("pilot", 1, invalid_csv, self.config)

        result = skip_invalid_retry_cache("pilot", 1, self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["chunk_id"], 1)
        self.assertIn("next_chunk", result)

        paths = get_project_paths("pilot", self.config)
        backlog_rows = load_backlog_rows(paths)
        self.assertEqual(len(backlog_rows), 1)
        self.assertEqual(backlog_rows[0]["example_id"], "ex1")

        state = load_session_state(paths)
        self.assertEqual(state["invalid_retry_rows"], [])
        self.assertEqual(state["current_chunk_id"], None)

        accepted_path = self.workspace_dir / "accepted" / "chunk_0001.csv"
        self.assertTrue(accepted_path.exists())


if __name__ == "__main__":
    unittest.main()
