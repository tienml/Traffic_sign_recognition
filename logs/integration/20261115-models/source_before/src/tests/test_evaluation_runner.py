"""Synthetic provenance/no-overwrite tests for member4's saved-evaluation runner."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import evaluate_saved_models as runner
from src.evaluate import summarize_predictions


class EvaluationRunnerTests(unittest.TestCase):
    def fixture(self, root):
        tune = root / "report/knn/full-knn-01"
        test = root / "report/knn/20261113T194939Z-test-adf82d0d"
        tune.mkdir(parents=True); test.mkdir(parents=True)
        labels = np.arange(12630) % 43
        metadata = pd.DataFrame({"CSVRow": np.arange(12630),
                                  "Path": [f"Test/{i:05d}.png" for i in range(12630)],
                                  "ClassId": labels})
        predictions = metadata.assign(PredictedClassId=labels)
        metric = summarize_predictions(predictions)["metrics"]
        selection = {"status": "frozen", "winner": "hog", "methods": [
            {"method": method, "candidate_index": 0, "cv_f1_macro": 1. if method == "hog" else .9,
             "cv_f1_std": 0., "params": {}} for method in runner.METHODS]}
        (tune / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
        claim = {"selection_sha256": runner.digest(tune / "selection.json"),
                 "evaluation_run": test.name}
        (tune / "official_test_claim.json").write_text(json.dumps(claim), encoding="utf-8")
        metrics = {"selection_sha256": claim["selection_sha256"], "winner_fixed_by_cv": "hog",
                   "methods": {method: {k: metric[k] for k in runner.METRICS} for method in runner.METHODS}}
        (test / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        for method in runner.METHODS:
            predictions.to_csv(test / f"predictions_{method}.csv", index=False)
            pd.DataFrame({"mean_test_f1_macro": [1. if method == "hog" else .9]}).to_csv(tune / f"cv_{method}.csv", index=False)
        return tune, test, metadata

    def test_saved_only_metrics_claim_and_current_row_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, metadata = self.fixture(root)
            sources = set()
            import joblib
            from sklearn.neighbors import KNeighborsClassifier
            with patch.object(joblib, "load", side_effect=AssertionError("No pickle")), \
                 patch.object(KNeighborsClassifier, "predict", side_effect=AssertionError("No inference")):
                frames, summary, _ = runner.historical_predictions(root, "knn", metadata, sources)
            self.assertEqual(len(summary), 3)
            self.assertEqual(set(frames), set(runner.METHODS))
            self.assertTrue(all(row["f1_macro"] == 1. for row in summary))
            broken = metadata.copy()
            broken.loc[0, "ClassId"] = 3
            with self.assertRaises(ValueError):
                runner.historical_predictions(root, "knn", broken, set())

    def test_modified_claim_or_saved_metric_is_rejected(self):
        for field in ("claim", "metric"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                tune, test, metadata = self.fixture(root)
                if field == "claim":
                    (tune / "official_test_claim.json").write_text(json.dumps({
                        "selection_sha256": "bad", "evaluation_run": test.name}), encoding="utf-8")
                else:
                    path = test / "test_metrics.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["methods"]["hog"]["f1_macro"] = .5
                    path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.historical_predictions(root, "knn", metadata, set())

    def test_incomplete_upgrade_rejected_before_evaluation_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            upgrade = root / "upgrade"
            upgrade.mkdir()
            (upgrade / "selection.json").write_text(json.dumps({"status": "smoke_only"}), encoding="utf-8")
            with self.assertRaises(ValueError), patch.object(runner, "_evaluate") as evaluate:
                runner.evaluate(root / "dataset", upgrade, "fresh", project_root=root)
            evaluate.assert_not_called()
            self.assertFalse((root / "report/evaluation").exists())

    def test_exact_overlap_hashes_and_split_left_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Test").mkdir()
            image = root / "Test/00000.png"
            image.write_bytes(b"same-image-bytes")
            fingerprint = runner.digest(image)
            audit = root / "train_audit.csv"
            pd.DataFrame({"Path": ["Train/0/00000_00000_00000.png"], "ClassId": [0],
                          "image_sha256": [fingerprint]}).to_csv(audit, index=False)
            meta = pd.DataFrame({"CSVRow": [0], "Path": ["Test/00000.png"], "ClassId": [0]})
            pairs, hashes = runner.duplicate_audit(root, audit, meta)
            self.assertEqual(len(pairs), 1)
            self.assertTrue(pairs.iloc[0].equal_label)
            self.assertEqual(len(hashes), 1)
            self.assertEqual(image.read_bytes(), b"same-image-bytes")


if __name__ == "__main__":
    unittest.main()
