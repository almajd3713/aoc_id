import json
import unittest

from scripts.merge_masked_chunks import validate_row


class MergeMaskedChunksTests(unittest.TestCase):
    def test_validate_row_accepts_consistent_annotation(self) -> None:
        row = {
            "example_id": "ex1",
            "source_split": "train",
            "source_row_id": "1",
            "dialect": "MSA",
            "original_text": "زار محمد صلاح القاهرة",
            "normalized_text": "زار محمد صلاح القاهره",
            "masked_text": "زار [PERSON_1] [LOC_1]",
            "mask_spans_json": json.dumps(
                [
                    {
                        "start": 4,
                        "end": 13,
                        "placeholder": "[PERSON_1]",
                        "surface_form": "محمد صلاح",
                        "mask_type": "PERSON",
                    },
                    {
                        "start": 14,
                        "end": 21,
                        "placeholder": "[LOC_1]",
                        "surface_form": "القاهرة",
                        "mask_type": "LOC",
                    },
                ],
                ensure_ascii=False,
            ),
            "mask_count": "2",
            "annotation_status": "approved",
            "annotator_model": "test-model",
            "notes": "",
        }
        self.assertEqual(validate_row(row), [])

    def test_validate_row_rejects_bad_offsets(self) -> None:
        row = {
            "example_id": "ex2",
            "source_split": "train",
            "source_row_id": "2",
            "dialect": "MSA",
            "original_text": "زار محمد صلاح القاهرة",
            "normalized_text": "زار محمد صلاح القاهره",
            "masked_text": "زار [PERSON_1] القاهرة",
            "mask_spans_json": json.dumps(
                [
                    {
                        "start": 0,
                        "end": 5,
                        "placeholder": "[PERSON_1]",
                        "surface_form": "محمد",
                        "mask_type": "PERSON",
                    }
                ],
                ensure_ascii=False,
            ),
            "mask_count": "1",
            "annotation_status": "approved",
            "annotator_model": "test-model",
            "notes": "",
        }
        errors = validate_row(row)
        self.assertTrue(any("surface_form mismatch" in error for error in errors))

    def test_validate_row_rejects_non_contiguous_indices(self) -> None:
        row = {
            "example_id": "ex3",
            "source_split": "train",
            "source_row_id": "3",
            "dialect": "MSA",
            "original_text": "زار محمد احمد",
            "normalized_text": "زار محمد احمد",
            "masked_text": "زار [PERSON_1] [PERSON_3]",
            "mask_spans_json": json.dumps(
                [
                    {
                        "start": 4,
                        "end": 8,
                        "placeholder": "[PERSON_1]",
                        "surface_form": "محمد",
                        "mask_type": "PERSON",
                    },
                    {
                        "start": 9,
                        "end": 13,
                        "placeholder": "[PERSON_3]",
                        "surface_form": "احمد",
                        "mask_type": "PERSON",
                    },
                ],
                ensure_ascii=False,
            ),
            "mask_count": "2",
            "annotation_status": "approved",
            "annotator_model": "test-model",
            "notes": "",
        }
        errors = validate_row(row)
        self.assertTrue(any("not contiguous" in error and "'PERSON'" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
