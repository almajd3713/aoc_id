import unittest

from scripts.build_masked_dataset import (
    build_example_id,
    build_near_duplicate_key,
    normalize_for_comparison,
    stratified_sample,
    Candidate,
)


class BuildMaskedDatasetTests(unittest.TestCase):
    def test_normalize_for_comparison(self) -> None:
        text = "آلاءُ   ذهبتـ إلى المدرسة!!!"
        normalized = normalize_for_comparison(text)
        self.assertEqual(normalized, "الاء ذهبت الي المدرسه")

    def test_near_duplicate_key_collapses_minor_variants(self) -> None:
        first = "هذا مثال بسيط جدا لاختبار تكرار النص في النظام"
        second = "هذا مثال بسيط جدا، لاختبار تكرار النص في النظام!!!"
        self.assertEqual(
            build_near_duplicate_key(normalize_for_comparison(first)),
            build_near_duplicate_key(normalize_for_comparison(second)),
        )

    def test_example_id_is_stable(self) -> None:
        normalized = normalize_for_comparison("أهلا وسهلا بكم")
        self.assertEqual(
            build_example_id("MSA", normalized),
            build_example_id("MSA", normalized),
        )

    def test_stratified_sample_balances_small_pool(self) -> None:
        candidates = []
        for dialect in ("MSA", "DIAL_EGY"):
            for index in range(3):
                normalized = f"{dialect} نص تجريبي رقم {index} مكون من كلمات عديدة"
                candidates.append(
                    Candidate(
                        source_row_id=str(index),
                        dialect=dialect,
                        original_text=normalized,
                        normalized_text=normalized,
                        example_id=f"{dialect}_{index}",
                        word_count=8 + index,
                        char_count=len(normalized),
                        near_duplicate_key=f"{dialect}_{index}",
                    )
                )

        sampled = stratified_sample(candidates, per_dialect_target=2, seed=5)
        counts = {}
        for row in sampled:
            counts[row.dialect] = counts.get(row.dialect, 0) + 1
        self.assertEqual(counts, {"DIAL_EGY": 2, "MSA": 2})


if __name__ == "__main__":
    unittest.main()
