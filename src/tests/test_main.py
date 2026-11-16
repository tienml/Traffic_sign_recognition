"""CLI/deployment regressions on small synthetic Train-only directories."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

import main as cli
from src import deployment
from src.inference import ModelRegistry


def synthetic_training(root, per_class=2):
    """Create generated rasters, not copies or reads of the real dataset."""
    root.mkdir(parents=True, exist_ok=True)
    columns = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
    with (root / "Train.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        rng = np.random.default_rng(42)
        for label in range(43):
            directory = root / "Train" / str(label)
            directory.mkdir(parents=True)
            for index in range(per_class):
                relative = f"Train/{label}/{label:05d}_00000_{index:05d}.png"
                pixels = rng.integers(0, 256, (16, 19, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(root / relative)
                writer.writerow(dict(Width=19, Height=16, **{
                    "Roi.X1": 1, "Roi.Y1": 2, "Roi.X2": 17, "Roi.Y2": 14,
                    "ClassId": label, "Path": relative,
                }))
    # Test.csv deliberately does not exist. Any attempt to use it must fail.
    return root


class CLITests(unittest.TestCase):
    def test_parser_defaults_and_explicit_demo(self):
        parser = cli.build_parser()
        options = parser.parse_args(["train"])
        self.assertEqual(options.features, "hog")
        self.assertEqual(options.models, ["knn", "dt"])
        self.assertIsNone(options.max_per_class)
        self.assertIsNone(options.dataset_dir)
        options = parser.parse_args(["train", "--features", "all", "--models", "dt", "--max-per-class", "3"])
        self.assertEqual(options.max_per_class, 3)
        options = parser.parse_args(["serve"])
        self.assertEqual(options.host, "127.0.0.1")
        self.assertEqual(options.port, 5000)
        options = parser.parse_args(["predict", "image.png", "--roi", "1", "2", "3", "4"])
        self.assertEqual(options.roi, [1, 2, 3, 4])

    def test_no_command_starts_local_web_without_training(self):
        from unittest.mock import Mock
        app = Mock()
        fake_web = types.ModuleType("web.app")
        fake_web.create_app = Mock(return_value=app)
        with patch.dict("sys.modules", {"web.app": fake_web}), patch.object(deployment, "train_models") as train:
            self.assertEqual(cli.main([]), 0)
            train.assert_not_called()
        app.run.assert_called_once_with(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
        self.assertEqual(fake_web.create_app.call_args.kwargs["model_dir"], cli.DEFAULT_MODELS)

    def test_default_web_resolves_flat_and_archive_data_layouts(self):
        from web.app import _dataset_root
        self.assertEqual(cli.DEFAULT_DATASET, cli.ROOT / "data")
        with tempfile.TemporaryDirectory() as temp:
            flat = Path(temp) / "data"
            flat.mkdir()
            (flat / "Train.csv").write_text("placeholder", encoding="utf-8")
            self.assertEqual(_dataset_root(flat), deployment.resolve_dataset_dir(flat))

    def test_models_is_json_and_missing_model_cli_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["models", "--model-dir", temp]), 0)
            self.assertEqual(json.loads(output.getvalue()), [])
            image = Path(temp) / "image.png"
            Image.new("RGB", (12, 12)).save(image)
            with contextlib.redirect_stderr(errors):
                self.assertEqual(cli.main(["predict", str(image), "--model-dir", temp]), 1)
            self.assertIn("train", errors.getvalue())

    def test_positive_arguments_and_port_are_validated(self):
        parser = cli.build_parser()
        for arguments in (["serve", "--port", "0"], ["serve", "--port", "65536"],
                          ["train", "--max-per-class", "-1"], ["train", "--max-per-class", "2.5"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)


class DeploymentTests(unittest.TestCase):
    def test_dataset_resolution_requires_only_train(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "data"
            archive = parent / "archive"
            archive.mkdir(parents=True)
            (archive / "Train.csv").write_text("placeholder", encoding="utf-8")
            self.assertEqual(deployment.resolve_dataset_dir(parent), archive.resolve())
            self.assertEqual(deployment.resolve_dataset_dir(archive), archive.resolve())
            with self.assertRaises(ValueError):
                deployment.resolve_dataset_dir(Path(temp) / "missing")

    def test_saved_parameter_sources_do_not_copy_metrics(self):
        root = Path(__file__).resolve().parents[2]
        knn, origin = deployment.selected_parameters("hog", "knn", root)
        self.assertEqual(knn, {"n_neighbors": 15, "weights": "distance", "metric": "manhattan"})
        self.assertIn("hog-full-20261115", origin)
        tree, _ = deployment.selected_parameters("hog", "dt", root)
        self.assertEqual(tree["ccp_alpha"], 0.0005)
        pixel, _ = deployment.selected_parameters("pixel", "knn", root)
        self.assertEqual(pixel["n_neighbors"], 3)
        self.assertFalse(any("score" in key or "f1" in key for key in knn))
        with tempfile.TemporaryDirectory() as temp:
            params, origin = deployment.selected_parameters("hog", "knn", Path(temp))
            self.assertEqual(params["n_neighbors"], 7)
            self.assertIn("defaults", origin)

    def test_fixed_refit_roundtrip_full_demo_fresh_runs_and_no_test_access(self):
        from sklearn.model_selection import GridSearchCV
        from src import models
        from src import preprocessing
        with tempfile.TemporaryDirectory() as temp:
            root = synthetic_training(Path(temp) / "archive")
            model_dir = Path(temp) / "models"
            sentinel = model_dir / "historical.txt"
            model_dir.mkdir()
            sentinel.write_bytes(b"unchanged historical artifact")
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.name.lower() == "test.csv" or "Test" in path.parts:
                    raise AssertionError("Deployment must never read official Test")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", guarded_open), \
                 patch.object(GridSearchCV, "fit", side_effect=AssertionError("no retuning")), \
                 patch.object(models, "train_and_tune", side_effect=AssertionError("no legacy tuning")), \
                 patch.object(preprocessing, "load_and_preprocess", side_effect=AssertionError("no mixed image cache")):
                result = deployment.train_models(dataset_dir=root, model_dir=model_dir, features="hog",
                                                 models=("knn", "dt"), progress=None)
                demo = deployment.train_models(dataset_dir=root, model_dir=model_dir, features="pixel",
                                               models=("dt",), max_per_class=1, progress=None)
            self.assertEqual(result["status"], "train_refit")
            self.assertEqual(result["training_state"], "completed")
            self.assertEqual(result["training_samples"], 86)
            self.assertIsNone(result["metrics"])
            self.assertFalse(result["official_test_read"])
            self.assertFalse(result["official_test_evaluated"])
            self.assertFalse(result["cross_validation_performed"])
            self.assertIn("scikit-learn", result["versions"]["packages"])
            self.assertEqual(demo["status"], "demo")
            self.assertEqual(demo["training_samples"], 43)
            self.assertNotEqual(result["run_dir"], demo["run_dir"])
            self.assertEqual(sentinel.read_bytes(), b"unchanged historical artifact")
            registry = ModelRegistry(model_dir)
            records = registry.list_models()
            self.assertEqual(len(records), 3)
            self.assertTrue(all(row["ready"] for row in records))
            self.assertTrue(all(row["n_classes"] == 43 for row in records))
            for artifact in result["bundles"]:
                stored = joblib.load(Path(result["run_dir"]) / artifact["file"])
                self.assertEqual(stored["raw_dimensions"], 324)
                self.assertEqual(stored["estimator"].n_features_in_, 324)
                self.assertIn("scaler", stored["estimator"].named_steps)
                self.assertEqual(stored["metadata"]["training_samples"], 86)
                self.assertIsNone(stored["metadata"]["metrics"])
                self.assertIn("parameter_origin", stored["metadata"])
            buffer = io.BytesIO()
            Image.new("RGB", (20, 20), "red").save(buffer, format="PNG")
            prediction = registry.predict(buffer.getvalue())
            self.assertEqual(prediction["model"]["method"], "hog")
            self.assertTrue(0 <= prediction["prediction"]["score"] <= 1)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["models", "--model-dir", str(model_dir)]), 0)
            self.assertEqual(len(json.loads(output.getvalue())), 3)

    def test_train_rejects_csv_dimensions_that_hide_invalid_actual_roi(self):
        with tempfile.TemporaryDirectory() as temp:
            root = synthetic_training(Path(temp) / "archive", per_class=1)
            csv_path = root / "Train.csv"
            with csv_path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames
                rows = list(reader)
            rows[0].update({"Width": "64", "Height": "64", "Roi.X1": "32",
                            "Roi.Y1": "32", "Roi.X2": "63", "Roi.Y2": "63"})
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            output = Path(temp) / "models"
            with self.assertRaisesRegex(ValueError, "kích thước thực"):
                deployment.train_models(dataset_dir=root, model_dir=output, features="pixel",
                                        models=("dt",), max_per_class=1, progress=None)
            self.assertFalse(list(output.rglob("*.joblib")))

    def test_train_rejects_test_path_and_missing_class_for_full(self):
        with tempfile.TemporaryDirectory() as temp:
            root = synthetic_training(Path(temp))
            csv_path = root / "Train.csv"
            original = csv_path.read_text(encoding="utf-8")
            csv_path.write_text(original.replace("Train/0/", "Test/0/", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                deployment.train_models(dataset_dir=root, model_dir=root / "out", progress=None)
            csv_path.write_text("\n".join(line for line in original.splitlines() if "Train/42/" not in line) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                deployment.train_models(dataset_dir=root, model_dir=root / "out", progress=None)


if __name__ == "__main__":
    unittest.main()
