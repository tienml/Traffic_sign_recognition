"""Synthetic in-memory tables only; no image/model/data access or fitting."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)

from src import evaluate as evaluation


def test_tables(actual=(0, 0, 1, 2), predicted=(0, 1, 1, 0)):
    """Small fake official-test identities with inclusive, valid ROI metadata."""
    n = len(actual)
    metadata = pd.DataFrame({
        "CSVRow": np.arange(n, dtype=np.int64),
        "Path": [f"Test/{index:05d}.png" for index in range(n)],
        "ClassId": np.asarray(actual, dtype=np.int64),
        "Width": np.full(n, 40), "Height": np.full(n, 40),
        "Roi.X1": np.zeros(n, dtype=np.int64),
        "Roi.Y1": np.zeros(n, dtype=np.int64),
        "Roi.X2": np.full(n, 31), "Roi.Y2": np.full(n, 31),
    })
    predictions = metadata[["CSVRow", "Path", "ClassId"]].copy()
    predictions["PredictedClassId"] = np.asarray(predicted, dtype=np.int64)
    return predictions, metadata


def oof_tables():
    # One represented class suffices; all three folds and whole tracks are required.
    metadata = pd.DataFrame({
        "CSVRow": np.arange(6, dtype=np.int64),
        "Path": [f"Train/0/00000_{track:05d}_{frame:05d}.png"
                 for track in range(3) for frame in range(2)],
        "ClassId": np.zeros(6, dtype=np.int64),
    })
    manifest = metadata.assign(
        Group=[f"0:{track}" for track in range(3) for _ in range(2)],
        ValidationFold=np.repeat(np.arange(3, dtype=np.int64), 2))
    predictions = manifest.assign(PredictedClassId=np.zeros(6, dtype=np.int64))
    return predictions, metadata, manifest


class SavedEvaluationTests(unittest.TestCase):
    def test_canonical_api_keeps_legacy_adapter_available(self):
        for function in (evaluation.validate_predictions, evaluation.summarize_predictions,
                         evaluation.paired_comparison, evaluation.group_error_summary,
                         evaluation.select_error_examples, evaluation.evaluate_and_analyze):
            self.assertEqual(function.__module__, "src.evaluate")

    def test_saved_metrics_do_not_infer_or_access_models_files_or_plotting(self):
        from unittest.mock import patch
        from pathlib import Path
        import joblib
        from sklearn.neighbors import KNeighborsClassifier
        frame, metadata = test_tables()
        forbidden = AssertionError("Saved evaluation must be in-memory only")
        with patch("builtins.open", side_effect=forbidden), \
             patch.object(Path, "open", side_effect=forbidden), \
             patch.object(pd, "read_csv", side_effect=forbidden), \
             patch.object(joblib, "load", side_effect=forbidden), \
             patch.object(KNeighborsClassifier, "predict", side_effect=forbidden), \
             patch.object(evaluation, "evaluate_and_analyze", side_effect=forbidden):
            checked = evaluation.validate_predictions(frame, metadata, split="official_test")
            self.assertEqual(evaluation.summarize_predictions(checked)["metrics"]["accuracy"], .5)
            evaluation.group_error_summary(checked, metadata)
            evaluation.select_error_examples(checked)
            evaluation.paired_comparison(checked, checked, name_first="a", name_second="b")

    def test_validation_returns_copy_and_checks_split_size_and_empty(self):
        frame, metadata = test_tables()
        frame.index = [11, 12, 13, 14]  # DataFrame index is not CSVRow.
        original = frame.copy(deep=True)
        validated = evaluation.validate_predictions(
            frame, metadata, split="official_test", expected_size=4)
        assert_frame_equal(validated, original)
        validated.iloc[0, validated.columns.get_loc("PredictedClassId")] = 42
        assert_frame_equal(frame, original)
        for kwargs in ({"expected_size": 3}, {"expected_size": 4.0},
                       {"expected_size": True}, {"manifest": metadata}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluation.validate_predictions(frame, metadata, split="official_test", **kwargs)
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(frame, metadata, split="test")
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(frame.iloc[:0], metadata.iloc[:0], split="official_test")

    def test_strict_integer_labels_and_rows(self):
        frame, metadata = test_tables()
        invalid_values = ("0", 0.0, True, None, np.nan, -1, 43)
        for column in ("ClassId", "PredictedClassId", "CSVRow"):
            for value in invalid_values:
                broken = frame.copy()
                broken[column] = broken[column].astype(object)
                broken.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    evaluation.validate_predictions(broken, metadata, split="official_test")
        for column in ("ClassId", "CSVRow"):
            broken_metadata = metadata.copy()
            broken_metadata[column] = broken_metadata[column].astype(float)
            with self.subTest(metadata_column=column), self.assertRaises(ValueError):
                evaluation.validate_predictions(frame, broken_metadata, split="official_test")
        with self.assertRaises(ValueError):
            evaluation.summarize_predictions(frame.drop(columns="PredictedClassId"))
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(
                pd.concat([frame, frame[["ClassId"]]], axis=1), metadata, split="official_test")

    def test_exact_keys_row_order_and_duplicate_rejection(self):
        frame, metadata = test_tables()
        mutations = []
        duplicate_row = frame.copy()
        duplicate_row.loc[1, "CSVRow"] = 0
        mutations.append(duplicate_row)
        duplicate_path = frame.copy()
        duplicate_path.loc[1, "Path"] = duplicate_path.loc[0, "Path"]
        mutations.append(duplicate_path)
        wrong_class = frame.copy()
        wrong_class.loc[0, "ClassId"] = 42
        mutations.append(wrong_class)
        wrong_path = frame.copy()
        wrong_path.loc[0, "Path"] = "Test/new.png"
        mutations.extend([wrong_path, frame.iloc[::-1], frame.iloc[:-1]])
        for index, broken in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                evaluation.validate_predictions(broken, metadata, split="official_test")
        reordered = metadata.iloc[::-1].reset_index(drop=True)
        reordered["CSVRow"] = np.arange(len(reordered))
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(frame, reordered, split="official_test")

    def test_path_strings_split_traversal_and_absolute_guards(self):
        frame, metadata = test_tables()
        unsafe = (None, 7, "", "../Test/0.png", "/Test/0.png", "C:/Test/0.png",
                  "\\\\host\\Test\\0.png", "Test/../0.png", "Test/./0.png",
                  "Test//0.png", "Test/a/0.png", "Test/a.png/", "Test/a.jpg",
                  "Test/a\x00.png", "Test/a.png:stream", "Test/a?.png",
                  "Train/0/00000_00000_00000.png")
        for path in unsafe:
            broken, bad_metadata = frame.copy(), metadata.copy()
            broken.loc[0, "Path"] = path
            bad_metadata.loc[0, "Path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                evaluation.validate_predictions(broken, bad_metadata, split="official_test")
        aliases, aliases_metadata = frame.copy(), metadata.copy()
        aliases.loc[1, "Path"] = "Test\\00000.png"
        aliases_metadata.loc[1, "Path"] = "Test\\00000.png"
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(aliases, aliases_metadata, split="official_test")
        windows_paths, windows_metadata = frame.copy(), metadata.copy()
        windows_paths["Path"] = windows_paths["Path"].str.replace("/", "\\", regex=False)
        windows_metadata["Path"] = windows_paths["Path"]
        evaluation.validate_predictions(windows_paths, windows_metadata, split="official_test")

    def test_metrics_match_sklearn_fixed_43_labels(self):
        frame, _ = test_tables(actual=(0, 0, 1, 1, 2, 2), predicted=(0, 1, 1, 2, 2, 0))
        result = evaluation.summarize_predictions(frame)
        actual, predicted = frame.ClassId, frame.PredictedClassId
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=evaluation.LABELS, zero_division=0)
        for name, expected in (("precision_macro", precision.mean()),
                               ("recall_macro", recall.mean()), ("f1_macro", f1.mean()),
                               ("accuracy", accuracy_score(actual, predicted)),
                               ("balanced_accuracy", balanced_accuracy_score(actual, predicted))):
            self.assertAlmostEqual(result["metrics"][name], expected)
        self.assertNotEqual(result["metrics"]["balanced_accuracy"], result["metrics"]["recall_macro"])
        np.testing.assert_allclose(result["classes"].precision, precision)
        np.testing.assert_allclose(result["classes"].recall, recall)
        np.testing.assert_allclose(result["classes"].f1, f1)
        np.testing.assert_array_equal(result["classes"].support, support)
        np.testing.assert_array_equal(result["confusion"], confusion_matrix(
            actual, predicted, labels=evaluation.LABELS))
        self.assertEqual(result["confusion"].shape, (43, 43))
        self.assertEqual(result["confusion"].dtype, np.dtype("int64"))
        self.assertEqual(result["confusion_normalized"].dtype, np.dtype("float64"))

    def test_confusion_zero_rows_error_order_and_custom_labels(self):
        frame, _ = test_tables(actual=(0, 0, 0, 1, 1), predicted=(1, 1, 2, 2, 1))
        result = evaluation.summarize_predictions(frame)
        np.testing.assert_allclose(result["confusion_normalized"].sum(axis=1)[:2], [1, 1])
        np.testing.assert_array_equal(result["confusion_normalized"][2:], np.zeros((41, 43)))
        self.assertTrue(np.isfinite(result["confusion_normalized"]).all())
        self.assertEqual(result["errors"][["true_class", "predicted_class", "count"]].values.tolist(),
                         [[0, 1, 2], [0, 2, 1], [1, 2, 1]])
        np.testing.assert_allclose(result["errors"].error_share_within_true, [2 / 3, 1 / 3, 1 / 2])
        custom = evaluation.summarize_predictions(frame, labels=(2, 0, 1))
        self.assertEqual(custom["classes"].class_id.tolist(), [2, 0, 1])
        self.assertEqual(custom["confusion"].shape, (3, 3))
        for labels in ((), (0, 0, 1), (0.0, 1, 2), (True, 1, 2), (0, 1), (0, 1, 43)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluation.summarize_predictions(frame, labels=labels)

    def test_paired_counts_and_exact_alignment(self):
        first, _ = test_tables(actual=(0, 0, 1, 1), predicted=(0, 0, 0, 0))
        second = first.copy()
        second["PredictedClassId"] = [0, 1, 1, 2]
        result = evaluation.paired_comparison(first, second, name_first="A", name_second="B")
        self.assertEqual(result, {
            "n": 4, "both_correct": 1, "first_only_correct": 1,
            "second_only_correct": 1, "both_wrong": 1,
            "accuracy_difference_first_minus_second": 0.0, "discordant": 2})
        second["PredictedClassId"] = [0, 1, 0, 2]
        self.assertEqual(evaluation.paired_comparison(
            first, second, name_first="A", name_second="B")[
                "accuracy_difference_first_minus_second"], 0.25)
        for broken in (second.iloc[::-1], second.iloc[:-1],
                       second.assign(ClassId=[1, 0, 1, 1]),
                       second.assign(PredictedClassId=[0.0, 1.0, 0.0, 2.0]),
                       second.assign(Path=["Test/other.png", *second.Path.iloc[1:]])):
            with self.assertRaises(ValueError):
                evaluation.paired_comparison(first, broken, name_first="A", name_second="B")
        with self.assertRaises(ValueError):
            evaluation.paired_comparison(first, second, name_first="A", name_second="A")

    def test_group_sizes_inclusive_roi_speed_family_and_categories(self):
        frame, metadata = test_tables(actual=(0, 6, 8, 42), predicted=(1, 6, 8, 0))
        metadata.loc[0, ["Width", "Height", "Roi.X2", "Roi.Y2"]] = [20, 30, 19, 29]
        metadata.loc[2, ["Roi.X2", "Roi.Y2"]] = [30, 30]
        metadata["ColorId"] = ["red", "blue", "red", None]
        metadata["ShapeId"] = [0, 1, 0, 1]
        metadata["BlurScore"] = [0.1, 0.2, 0.3, 0.4]  # No adaptive threshold here.
        saved = metadata.copy(deep=True)
        result = evaluation.group_error_summary(frame, metadata)
        lookup = result.set_index(["dimension", "group"])
        self.assertEqual(tuple(lookup.loc[("raw_min_dimension", "lt32"), ["n", "errors"]]), (1, 1))
        self.assertEqual(tuple(lookup.loc[("raw_min_dimension", "ge32"), ["n", "errors"]]), (3, 1))
        self.assertEqual(tuple(lookup.loc[("roi_min_dimension", "lt32"), ["n", "errors"]]), (2, 1))
        self.assertEqual(tuple(lookup.loc[("roi_min_dimension", "ge32"), ["n", "errors"]]), (2, 1))
        self.assertEqual(lookup.loc[("sign_family", "speed_limit"), "n"], 2)
        self.assertEqual(lookup.loc[("sign_family", "other"), "errors"], 1)
        self.assertEqual(lookup.loc[("ColorId", "unknown"), "n"], 1)
        self.assertEqual(lookup.loc[("ShapeId", "0"), "n"], 2)
        self.assertNotIn("BlurScore", result.dimension.tolist())
        self.assertTrue((result.groupby("dimension").n.sum() == 4).all())
        np.testing.assert_allclose(result.error_rate, result.errors / result.n)
        assert_frame_equal(metadata, saved)
        for column, value in (("Width", 0), ("Height", np.nan), ("Roi.X1", -1),
                              ("Roi.X2", 40), ("Roi.Y1", 32), ("Width", "40")):
            broken = metadata.copy()
            broken[column] = broken[column].astype(object)
            broken.loc[1, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                evaluation.group_error_summary(frame, broken)
        with self.assertRaises(ValueError):
            evaluation.group_error_summary(frame, metadata.assign(ClassId=[0, 6, 8, 41]))

    def test_error_examples_stable_top_pairs_and_no_mutation(self):
        frame, _ = test_tables(actual=(0, 0, 0, 1, 1, 1, 2, 2),
                               predicted=(1, 1, 1, 2, 2, 1, 0, 2))
        frame["CandidateId"] = "synthetic"
        original = frame.copy(deep=True)
        selected = evaluation.select_error_examples(frame, max_pairs=2, per_pair=2, seed=7)
        assert_frame_equal(selected, evaluation.select_error_examples(
            frame, max_pairs=2, per_pair=2, seed=7))
        self.assertEqual(selected.ClassId.tolist(), [0, 0, 1, 1])
        self.assertEqual(selected.pair_rank.tolist(), [1, 1, 2, 2])
        self.assertEqual(selected.example_rank.tolist(), [1, 2, 1, 2])
        self.assertTrue((selected.ClassId != selected.PredictedClassId).all())
        self.assertTrue(selected.CSVRow.is_unique)
        self.assertEqual(selected.columns.tolist(), [*frame.columns, "pair_rank", "example_rank"])
        assert_frame_equal(frame, original)
        for kwargs in ({"max_pairs": 0}, {"per_pair": 0}, {"seed": True}, {"seed": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluation.select_error_examples(frame, **kwargs)

    def test_all_correct_examples_and_errors_have_expected_empty_schema(self):
        frame, _ = test_tables(actual=(0, 1), predicted=(0, 1))
        examples = evaluation.select_error_examples(frame)
        self.assertTrue(examples.empty)
        self.assertEqual(examples.columns.tolist(), [*frame.columns, "pair_rank", "example_rank"])
        self.assertEqual(examples.pair_rank.dtype, np.dtype("int64"))
        summary = evaluation.summarize_predictions(frame)
        self.assertTrue(summary["errors"].empty)
        self.assertEqual(summary["errors"].columns.tolist(), [
            "true_class", "predicted_class", "count", "true_support", "error_share_within_true"])
        self.assertEqual(summary["metrics"]["accuracy"], 1)
        self.assertEqual(summary["metrics"]["balanced_accuracy"], 1)

    def test_oof_accepts_missing_classes_but_requires_frozen_manifest_identity(self):
        frame, metadata, manifest = oof_tables()
        validated = evaluation.validate_predictions(frame, metadata, split="train_oof", manifest=manifest)
        assert_frame_equal(frame, validated)
        for broken_manifest in (None, manifest.drop(columns="Group"), manifest.iloc[::-1],
                                manifest.assign(Group="wrong"),
                                manifest.assign(ValidationFold=[1, 1, 0, 0, 2, 2])):
            with self.assertRaises(ValueError):
                evaluation.validate_predictions(frame, metadata, split="train_oof", manifest=broken_manifest)
        wrong_groups = frame.assign(Group="wrong")
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(wrong_groups, metadata, split="train_oof", manifest=manifest)
        bad_metadata, bad_frame, bad_manifest = metadata.copy(), frame.copy(), manifest.copy()
        for table in (bad_metadata, bad_frame, bad_manifest):
            table.loc[0, "Path"] = "Train/1/00000_00000_00000.png"
        with self.assertRaises(ValueError):
            evaluation.validate_predictions(bad_frame, bad_metadata, split="train_oof", manifest=bad_manifest)

    def test_oof_rejects_whole_track_leakage_missing_or_invalid_fold_ids(self):
        frame, metadata, manifest = oof_tables()
        assignments = ([0, 1, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 3, 3],
                       [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [False, False, True, True, 2, 2])
        for assignment in assignments:
            broken, bad_manifest = frame.copy(), manifest.copy()
            broken["ValidationFold"] = pd.Series(assignment, dtype=object)
            bad_manifest["ValidationFold"] = pd.Series(assignment, dtype=object)
            with self.subTest(assignment=assignment), self.assertRaises(ValueError):
                evaluation.validate_predictions(broken, metadata, split="train_oof", manifest=bad_manifest)


if __name__ == "__main__":
    unittest.main()
