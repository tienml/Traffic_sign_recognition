"""Synthetic runner regressions for Tiến's upgrades in canonical src.models."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import train_models as runner
from scripts import summarize_models as report
from src.models import grouped_folds, track_groups


class UpgradeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"CSVRow": (c * 3 + t) * 2 + f,
             "Path": f"Train/{c}/{c:05d}_{t:05d}_{f:05d}.png", "ClassId": c,
             "Width": 40, "Height": 40, "Roi.X1": 2, "Roi.Y1": 2,
             "Roi.X2": 37, "Roi.Y2": 37}
            for c in range(43) for t in range(3) for f in range(2)])
        self.groups = track_groups(self.frame)
        self.folds = grouped_folds(self.frame.ClassId.to_numpy(), self.groups)

    def test_run_is_exclusive_and_old_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sentinel = root / "report/knn/full-knn-01/selection.json"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b"historical selection")
            _, output, logs, models = runner.new_run("smoke", "new-run", root)
            self.assertTrue(all(p.is_dir() for paths in (output, logs, models) for p in paths.values()))
            self.assertEqual(output["knn"], root / "report/knn/new-run")
            self.assertEqual(output["dt"], root / "report/dt/new-run")
            self.assertFalse((root / "report/model3_upgrade").exists())
            with self.assertRaises(FileExistsError):
                runner.new_run("smoke", "new-run", root)
            with self.assertRaises(ValueError):
                runner.new_run("smoke", "../escape", root)
            self.assertEqual(sentinel.read_bytes(), b"historical selection")

    def test_train_runner_writes_selected_algorithm_into_each_existing_area(self):
        from types import SimpleNamespace
        raw = np.zeros((len(self.frame), 324), dtype=np.float32)
        quality = self.frame[["CSVRow", "Path", "ClassId"]].assign(image_sha256="synthetic")
        def fake_cv(*args, **kwargs):
            return {"candidates": [{"candidate_id": "knn_baseline", "params": {"n_neighbors": 7}, "mean_f1_macro": 1.}],
                    "fold_scores": [], "transforms": [], "oof": {"knn_baseline": self.frame.ClassId.to_numpy()}}
        knn_result = fake_cv()
        dt_result = {"candidates": [{"candidate_id": "dt_baseline", "params": {"ccp_alpha": 0.}, "mean_f1_macro": 1.}],
                     "fold_scores": [], "transforms": [], "oof": {"dt_baseline": self.frame.ClassId.to_numpy()}}
        def fake_dump(bundle, stream, **kwargs):
            stream.write(b"synthetic-bundle")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, outputs, _, models = runner.new_run("tune", "new", root)
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "extract_train", return_value=(raw, quality, .1)), \
                 patch.object(runner, "evaluate_knn_cv", return_value=knn_result), \
                 patch.object(runner, "evaluate_dt_cv", return_value=dt_result), \
                 patch.object(runner, "refit_model", return_value=object()), \
                 patch.object(runner.joblib, "dump", side_effect=fake_dump):
                runner.train_run(SimpleNamespace(mode="tune", batch_size=8), root, self.frame,
                                 self.groups, self.folds, outputs, models, {"run_id": "new"})
            for algorithm in ("knn", "dt"):
                stored = json.loads((outputs[algorithm] / "selection.json").read_text(encoding="utf-8"))
                self.assertEqual([item["algorithm"] for item in stored["algorithms"]], [algorithm])
                self.assertEqual(stored["algorithms"][0]["model_file"], f"data/models/{algorithm}/new/hog.joblib")
                self.assertTrue((outputs[algorithm] / f"oof_{algorithm}_selected.csv").is_file())
                self.assertFalse((outputs[algorithm] / f"oof_{'dt' if algorithm == 'knn' else 'knn'}_selected.csv").exists())
            self.assertEqual((outputs["knn"] / "feature_audit.json").read_bytes(),
                             (outputs["dt"] / "feature_audit.json").read_bytes())

    def test_relocated_evidence_requires_exact_hash_and_rejects_path_escape(self):
        from scripts import model_artifacts as evidence
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "report/knn").mkdir(parents=True)
            moved = root / "report/knn/evidence.csv"
            moved.write_bytes(b"recorded predictions")
            snapshot = root / "archived_source.py"
            snapshot.write_bytes(b"old code")
            manifest = root / evidence.RELOCATIONS
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({
                "paths": {"old/evidence.csv": moved.relative_to(root).as_posix()},
                "historical_sources": [{"original_path": "src/models.py",
                    "sha256": evidence.digest(snapshot), "archived_path": snapshot.name}]
            }), encoding="utf-8")
            self.assertEqual(evidence.resolve_evidence("old/evidence.csv", evidence.digest(moved), project_root=root), moved)
            self.assertEqual(evidence.resolve_evidence("src/models.py", evidence.digest(snapshot), project_root=root), snapshot)
            with self.assertRaises(ValueError):
                evidence.resolve_evidence("old/evidence.csv", "wrong-hash", project_root=root)
            with self.assertRaises(ValueError):
                evidence.resolve_evidence("../outside", project_root=root)

    def test_shared_run_loader_filters_each_algorithm_and_rejects_audit_drift(self):
        from scripts.model_artifacts import digest, load_model_runs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = {a: root / a / "hog-full" for a in ("knn", "dt")}
            for run in runs.values():
                run.mkdir(parents=True)
                for name in ("fold_manifest.csv", "feature_audit.json", "train_image_audit.csv"):
                    (run / name).write_bytes(b"shared bytes")
                (run / "config.json").write_text(json.dumps({
                    "official_test_read": False, "train_sha256": "same"}), encoding="utf-8")
                (run / "selection.json").write_text(json.dumps({
                    "status": "completed_train_cv", "official_test_read": False,
                    "official_test_evaluated": False,
                    "algorithms": [{"algorithm": "knn"}, {"algorithm": "dt"}],
                    "fold_manifest_sha256": digest(run / "fold_manifest.csv"),
                    "feature_audit_sha256": digest(run / "feature_audit.json")
                }), encoding="utf-8")
            loaded, selection, _ = load_model_runs(runs["knn"], runs["dt"])
            self.assertEqual(loaded, runs)
            self.assertEqual([x["algorithm"] for x in selection["algorithms"]], ["knn", "dt"])
            (runs["dt"] / "train_image_audit.csv").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_model_runs(runs["knn"], runs["dt"])

    def test_source_audit_distinguishes_verified_history_and_current_refactor(self):
        from scripts import check_model_artifacts as audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current, archived = root / "current", root / "archived"
            for directory in (current, archived):
                (directory / "src").mkdir(parents=True)
                (directory / "src/models.py").write_bytes(b"historical implementation")
            expected = {"src/models.py": audit.sha256(current / "src/models.py")}
            with patch.object(audit, "ROOT", current):
                self.assertTrue(audit.check_source_hashes(expected)["current_source_matches"])
                (current / "src/models.py").write_bytes(b"canonical implementation")
                with self.assertRaises(ValueError):
                    audit.check_source_hashes(expected)
                result = audit.check_source_hashes(expected, source_root=archived)
                self.assertTrue(result["historical_source_verified"])
                self.assertFalse(result["current_source_matches"])
                (current / "src/models.py").unlink()
                self.assertIsNone(audit.check_source_hashes(
                    expected, source_root=archived)["checks"][0]["current_sha256"])
                (archived / "src/models.py").write_bytes(b"wrong archive")
                with self.assertRaises(ValueError):
                    audit.check_source_hashes(expected, source_root=archived)

    def test_preservation_reports_authorized_edits_and_rejects_unlisted_changes(self):
        from scripts import verify_models as verify
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = {"README.md": b"before", "historical.csv": b"evidence"}
            for name, content in original.items():
                (root / name).write_bytes(content)
            snapshot = root / "baseline.json"
            snapshot.write_text(json.dumps({"before_sha256": {
                name: hashlib.sha256(content).hexdigest() for name, content in original.items()
            }}), encoding="utf-8")
            (root / "README.md").write_bytes(b"authorized refactor docs")
            with patch.object(verify, "ROOT", root):
                rejected = verify.preservation(snapshot, root / "rejected.json")
                self.assertFalse(rejected["protected_unchanged"])
                accepted = verify.preservation(snapshot, root / "accepted.json",
                                                allowed_changes=["README.md"])
                self.assertTrue(accepted["protected_unchanged"])
                self.assertFalse(accepted["all_sources_unchanged"])
                self.assertEqual(accepted["expected_refactor_changes"][0]["path"], "README.md")
                (root / "historical.csv").write_bytes(b"tampered evidence")
                detected = verify.preservation(snapshot, root / "detected.json",
                                                allowed_changes=["README.md"])
                self.assertFalse(detected["protected_unchanged"])
                self.assertEqual(detected["unexpected_changes"][0]["path"], "historical.csv")
                archive = root / "history.csv"
                archive.write_bytes(original["historical.csv"])
                moves = root / "relocations.json"
                moves.write_text(json.dumps({"paths": {"historical.csv": "history.csv"}}), encoding="utf-8")
                archived = verify.preservation(snapshot, root / "relocated.json",
                    allowed_changes=["README.md"], relocations=moves)
                self.assertTrue(archived["protected_unchanged"])
                self.assertEqual(archived["relocated_byte_identical"][0]["to"], "history.csv")

    def test_cli_has_no_official_test_mode(self):
        with self.assertRaises(SystemExit), patch.object(runner, "metadata") as load:
            runner.main(["test"])
        load.assert_not_called()

    def test_manifest_validation_reads_only_train_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "prior/fold_manifest.csv"
            manifest.parent.mkdir()
            manifest.write_bytes(b"manifest")
            (manifest.parent / "config.json").write_text(json.dumps({
                "mode": "tune", "seed": 42, "train_sha256": "train-hash"}), encoding="utf-8")
            full = pd.DataFrame(index=range(39209))
            with patch.object(runner, "metadata", return_value=full) as metadata, \
                 patch.object(runner, "digest", return_value="train-hash"), \
                 patch.object(runner, "frozen_folds", return_value=("groups", "folds")):
                actual, groups, folds = runner.validate_manifest(root, manifest)
            metadata.assert_called_once_with(root, "Train")
            self.assertIs(actual, full)
            self.assertEqual((groups, folds), ("groups", "folds"))

    def test_stream_extraction_calls_unchanged_public_hog(self):
        image = np.ones((32, 32, 3), dtype=np.float32) * .5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            small = self.frame.iloc[:3].copy()
            for relative in small.Path:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic-image")
            def fake_hog(images, **kwargs):
                self.assertEqual(kwargs, runner.HOG_DEFAULTS)
                return np.full((len(images), 324), .25, dtype=np.float32)
            with patch.object(runner, "load_image", return_value=image) as loader, \
                 patch.object(runner, "extract_hog", side_effect=fake_hog) as hog:
                raw, quality, seconds = runner.extract_train(root, small, batch_size=2,
                                                               progress=lambda _: None)
            self.assertEqual(raw.shape, (3, 324))
            self.assertEqual(loader.call_count, 3)
            self.assertEqual(hog.call_count, 2)
            self.assertTrue((quality.image_sha256 == hashlib.sha256(b"synthetic-image").hexdigest()).all())
            np.testing.assert_allclose(quality.roi32_brightness, .5)
            self.assertGreaterEqual(seconds, 0)

    def test_export_validates_oof_no_model_or_test_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "report/knn/synthetic"
            run.mkdir(parents=True)
            manifest = self.frame[["CSVRow", "Path", "ClassId"]].copy()
            manifest["Group"] = self.groups
            assignment = np.full(len(manifest), -1)
            for fold, (_, valid) in enumerate(self.folds):
                assignment[valid] = fold
            manifest["ValidationFold"] = assignment
            manifest.to_csv(run / "fold_manifest.csv", index=False)
            audit = self.frame[["CSVRow", "Path", "ClassId", "Width", "Height"]].copy()
            audit["RoiWidth"] = 36; audit["RoiHeight"] = 36
            audit["roi32_brightness"] = .5; audit["roi32_contrast"] = .2
            audit["image_sha256"] = "synthetic"
            audit.to_csv(run / "train_image_audit.csv", index=False)
            (run / "feature_audit.json").write_text("{}", encoding="utf-8")
            (run / "config.json").write_text(json.dumps({"official_test_read": False}), encoding="utf-8")
            algorithms = []
            for algorithm in ("knn", "dt"):
                params = ({"n_neighbors": 7, "weights": "distance", "metric": "manhattan"}
                          if algorithm == "knn" else {"min_samples_leaf": 1})
                item = {"candidate_id": f"{algorithm}_baseline", "params": params,
                        "mean_f1_macro": 1., "std_f1_macro": 0., "mean_accuracy": 1.}
                pd.DataFrame([{**item, "params": json.dumps(params)}]).to_csv(
                    run / f"{algorithm}_candidates.csv", index=False)
                pd.DataFrame([{"candidate_id": item["candidate_id"], "fold": f,
                               "accuracy": 1., "precision_macro": 1., "recall_macro": 1.,
                               "f1_macro": 1.} for f in range(3)]).to_csv(
                    run / f"{algorithm}_fold_scores.csv", index=False)
                (run / f"{algorithm}_transforms.json").write_text("[]", encoding="utf-8")
                for role in ("baseline", "selected"):
                    pred = manifest.assign(PredictedClassId=manifest.ClassId,
                                           CandidateId=item["candidate_id"])
                    pred.to_csv(run / f"oof_{algorithm}_{role}.csv", index=False)
                algorithms.append({"algorithm": algorithm, "baseline": item, "selected": item,
                                   "cv_wall_seconds": .1, "model_bytes": 100})
            selection = {"status": "smoke_only", "run_id": "synthetic",
                         "official_test_read": False, "official_test_evaluated": False,
                         "algorithms": algorithms,
                         "feature_audit_sha256": report.digest(run / "feature_audit.json"),
                         "fold_manifest_sha256": report.digest(run / "fold_manifest.csv")}
            (run / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            with patch.object(report, "plot_outputs", return_value=[]) as plots, \
                 patch.object(report, "__file__", str(run / "config.json")):
                result = report.export(run, project_root=root)
            self.assertEqual(len(result["summary"]), 4)
            self.assertEqual(len(result["oof_classes"]), 172)
            self.assertEqual(plots.call_count, 2)
            with self.assertRaises(FileExistsError):
                report.export(run, project_root=root)
            # The production layout has no DT candidate/OOF files in the KNN run.
            # Verify export consumes each algorithm's own directory, not both
            # algorithms from each byte-preserved historical selection.json.
            import shutil
            dt_run = root / "report/dt/synthetic"
            dt_run.mkdir(parents=True)
            for path in list(run.iterdir()):
                if path.name.startswith(("dt_", "oof_dt_")):
                    path.rename(dt_run / path.name)
                elif not path.name.startswith(("knn_", "oof_knn_")):
                    shutil.copyfile(path, dt_run / path.name)
            with patch.object(report, "plot_outputs", return_value=[]), \
                 patch.object(report, "__file__", str(run / "config.json")):
                separate = report.export(run, dt_run=dt_run, project_root=root, export_id="split")
            pd.testing.assert_frame_equal(result["summary"], separate["summary"])
            self.assertFalse((root / "report/model3_upgrade").exists())
            broken = manifest.assign(PredictedClassId=manifest.ClassId, CandidateId="x")
            broken.loc[0, "ValidationFold"] = (assignment[0] + 1) % 3
            with self.assertRaises(ValueError):
                report.validate_oof(broken, audit, manifest)


if __name__ == "__main__":
    unittest.main()
