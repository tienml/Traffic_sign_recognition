"""Synthetic service regressions: no real training data or official test access."""
import base64
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.inference import (HOG_CONTRACT, IMAGE_CONTRACT, InferenceError,
                           MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, ModelRegistry, get_classes,
                           prepare_image, raw_features)

ROOT = Path(__file__).resolve().parents[2]


def encoded(array=None, format="PNG", **options):
    if array is None:
        array = np.zeros((32, 32, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=format, **options)
    return stream.getvalue()


def bundle(path, *, method="pixel", algorithm="dt", classes=(2, 7), status="train_refit",
           expanded=False, updates=None):
    dimensions = 3072 if method == "pixel" else 324
    rng = np.random.default_rng(17)
    raw = rng.random((len(classes) * 3, dimensions), dtype=np.float32)
    labels = np.repeat(classes, 3)
    steps = [] if method == "pixel" else [("scaler", StandardScaler())]
    if method == "hog_pca":
        from sklearn.decomposition import PCA
        steps.append(("pca", PCA(n_components=.95)))
    classifier = (KNeighborsClassifier(n_neighbors=1, algorithm="brute", n_jobs=1)
                  if algorithm == "knn" else DecisionTreeClassifier(random_state=42))
    model = Pipeline(steps + [(algorithm, classifier)]).fit(raw, labels)
    stored = dict(schema_version=1, estimator=model, algorithm=algorithm,
                  method=method, labels=list(classes), raw_dimensions=dimensions,
                  hog_params=dict(HOG_CONTRACT), image_contract=dict(IMAGE_CONTRACT), status=status)
    if expanded:
        stored.pop("method")
        stored.pop("labels")
        stored["params"] = {"n_neighbors": 1} if algorithm == "knn" else {"max_depth": None}
    stored.update(updates or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stored, path)
    return model


class ImageValidationTests(unittest.TestCase):
    def test_labels_align_original_class_names(self):
        from src.preprocessing import CLASS_NAMES
        catalog = get_classes()
        self.assertEqual(len(catalog), 43)
        self.assertEqual([row["class_id"] for row in catalog], list(range(43)))
        self.assertEqual([row["label_en"] for row in catalog], CLASS_NAMES)
        self.assertTrue(all(row["label"] and row["category"] for row in catalog))
        catalog[0]["label"] = "changed"
        self.assertNotEqual(get_classes()[0]["label"], "changed")

    def test_import_and_help_without_ml_dependencies(self):
        script = f'''
import importlib.abc, runpy, sys
sys.path.insert(0, {str(ROOT)!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {{'cv2', 'skimage', 'sklearn', 'numpy', 'pandas', 'joblib'}}:
            raise ImportError('intentionally unavailable')
sys.meta_path.insert(0, Block())
from src.inference import get_classes, ModelRegistry
assert len(get_classes()) == 43
assert ModelRegistry({str(ROOT / 'nonexistent-models')!r}).list_models() == []
sys.argv = ['main.py', '--help']
runpy.run_path({str(ROOT / 'main.py')!r}, run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", "-c", script], capture_output=True,
                                text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("serve", result.stdout)

    def test_exact_area_rgb_float32_and_inclusive_roi(self):
        import cv2
        rng = np.random.default_rng(8)
        pixels = rng.integers(0, 256, (51, 43, 3), dtype=np.uint8)
        raw, size, preview = prepare_image(encoded(pixels), [3, 4, 30, 42])
        expected = cv2.resize(pixels[4:43, 3:31], (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(raw, expected)
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(size, {"width": 43, "height": 51})
        shown = Image.open(io.BytesIO(base64.b64decode(preview.split(",", 1)[1])))
        self.assertEqual(shown.size, (32, 32))
        pixel, _, _ = prepare_image(encoded(pixels), [4, 4, 4, 4])
        np.testing.assert_array_equal(pixel[0, 0], pixels[4, 4].astype(np.float32) / np.float32(255))

    def test_huge_integer_roi_is_rejected_without_float_overflow(self):
        with self.assertRaises(InferenceError) as caught:
            prepare_image(encoded(), [0, 0, 10 ** 400, 31])
        self.assertEqual(caught.exception.status_code, 400)

    def test_invalid_rois_not_clipped_or_silently_ignored(self):
        for roi in ([], [0, 0, 32, 31], [-1, 0, 31, 31], [8, 0, 7, 31],
                    [0, 0, 31, float("nan")], [0, 0, 31, 2.5], [True, 0, 31, 31],
                    ["0", 0, 31, 31], [0, 0, 31, float("inf")]):
            with self.subTest(roi=roi), self.assertRaises(InferenceError):
                prepare_image(encoded(), roi)

    def test_raster_validation_and_byte_limit(self):
        for data in (b"", b"not an image", b"<svg></svg>", encoded(format="BMP")):
            with self.subTest(data=data[:20]), self.assertRaises(InferenceError) as caught:
                prepare_image(data)
            self.assertEqual(caught.exception.status_code, 400)
        with self.assertRaises(InferenceError) as caught:
            prepare_image(b"0" * (MAX_IMAGE_BYTES + 1))
        self.assertEqual(caught.exception.status_code, 413)
        for format in ("JPEG", "WEBP"):
            self.assertEqual(prepare_image(encoded(format=format))[0].shape, (32, 32, 3))

    def test_ten_megabyte_boundary_accepts_padded_static_png(self):
        self.assertEqual(MAX_IMAGE_BYTES, 10 * 1024 * 1024)
        self.assertEqual(MAX_IMAGE_PIXELS, 16_000_000)
        data = encoded()
        for size in (9 * 1024 * 1024, MAX_IMAGE_BYTES):
            with self.subTest(size=size):
                padded = data + b"\0" * (size - len(data))
                image, dimensions, preview = prepare_image(padded)
                self.assertEqual(image.shape, (32, 32, 3))
                self.assertEqual(dimensions, {"width": 32, "height": 32})
                self.assertTrue(preview.startswith("data:image/png;base64,"))
        oversized = data + b"\0" * (MAX_IMAGE_BYTES + 1 - len(data))
        with patch("PIL.Image.open", side_effect=AssertionError("Check byte limit before opening")):
            with self.assertRaises(InferenceError) as caught:
                prepare_image(oversized)
        self.assertEqual(caught.exception.status_code, 413)
        self.assertIn("10 MB", caught.exception.message)

    def test_oversize_header_rejected_before_cv2_decode(self):
        def chunk(name, value):
            return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value))
        self.assertEqual(MAX_IMAGE_PIXELS, 16_000_000)
        for width, height in ((4000, 4001), (5000, 4000)):
            with self.subTest(width=width, height=height):
                data = (b"\x89PNG\r\n\x1a\n"
                        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                        + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))
                with patch("cv2.imdecode", side_effect=AssertionError("must inspect dimensions first")):
                    with self.assertRaises(InferenceError) as caught:
                        prepare_image(data)
                self.assertEqual(caught.exception.status_code, 413)
                self.assertIn("16 triệu pixel", caught.exception.message)

    def test_all_exif_orientations_match_pillow(self):
        import cv2
        rng = np.random.default_rng(3)
        pixels = rng.integers(0, 256, (19, 28, 3), dtype=np.uint8)
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = Image.Exif()
                exif[274] = orientation
                data = encoded(pixels, format="JPEG", exif=exif, quality=100, subsampling=0)
                with Image.open(io.BytesIO(data)) as image:
                    oriented = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
                expected = cv2.resize(oriented, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(255)
                actual, size, _ = prepare_image(data)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(size, {"width": oriented.shape[1], "height": oriented.shape[0]})


class ModelRegistryTests(unittest.TestCase):
    def test_missing_models_is_helpful_503(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = ModelRegistry(Path(temp) / "absent")
            self.assertEqual(registry.list_models(), [])
            with self.assertRaises(InferenceError) as caught:
                registry.predict(encoded())
            self.assertEqual(caught.exception.status_code, 503)
            self.assertIn("main.py train", caught.exception.message)

    def test_cache_lazy_and_invalidated_by_changed_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run/a.joblib"
            bundle(path)
            with patch("joblib.load", wraps=joblib.load) as load:
                registry = ModelRegistry(temp)
                self.assertEqual(load.call_count, 0)
                self.assertTrue(registry.list_models()[0]["ready"])
                registry.list_models()
                registry.predict(encoded())
                self.assertEqual(load.call_count, 1)
                path.write_bytes(b"broken replacement")
                self.assertFalse(registry.list_models()[0]["ready"])
                self.assertEqual(load.call_count, 2)
                path.unlink()
                self.assertEqual(registry.list_models(), [])

    def test_explicit_selection_only_loads_requested_and_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ("a", "b", "c", "d"):
                bundle(Path(temp) / f"{name}.joblib")
            with patch("joblib.load", wraps=joblib.load) as load:
                registry = ModelRegistry(temp)
                registry.predict(encoded(), model_id="d")
                self.assertEqual(load.call_count, 1)
                self.assertEqual(Path(load.call_args.args[0]).stem, "d")
                self.assertEqual(len(registry.list_models()), 4)
                self.assertLessEqual(sum(entry["estimator"] is not None for entry in registry._cache.values()), 2)
                registry.predict(encoded(), model_id="b")
                registry.predict(encoded(), model_id="c")
                self.assertLessEqual(sum(entry["estimator"] is not None for entry in registry._cache.values()), 2)
                count = load.call_count
                registry.predict(encoded(), model_id="c")
                self.assertEqual(load.call_count, count)

    def test_prediction_keeps_pipeline_reference_during_other_model_eviction(self):
        import src.inference as inference
        with tempfile.TemporaryDirectory() as temp:
            for name in ("a", "b", "c"):
                bundle(Path(temp) / f"{name}.joblib")
            registry = ModelRegistry(temp)
            original = inference.raw_features
            def evict_selected(image, method, hog_params):
                registry._select("b")
                registry._select("c")
                self.assertIsNone(registry._cache["a"]["estimator"])
                return original(image, method, hog_params)
            with patch.object(inference, "raw_features", side_effect=evict_selected):
                self.assertEqual(registry.predict(encoded(), model_id="a")["model"]["id"], "a")

    def test_model_directory_expands_home(self):
        expected = (Path.home() / "signal-lab-models").resolve()
        self.assertEqual(ModelRegistry("~/signal-lab-models").model_dir, expected)

    def test_prediction_maps_noncontiguous_classes_and_preserves_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tie.joblib"
            x = np.zeros((2, 3072), dtype=np.float32)
            model = Pipeline([("knn", KNeighborsClassifier(n_neighbors=2, algorithm="brute"))]).fit(x, [7, 2])
            joblib.dump(dict(estimator=model, method="pixel", labels=[2, 7]), path)
            result = ModelRegistry(temp).predict(encoded(), top_k=2)
            self.assertEqual(result["prediction"]["class_id"], int(model.predict(x[:1])[0]))
            self.assertEqual([row["class_id"] for row in result["top_k"]], [2, 7])
            self.assertEqual(result["prediction"]["score"], .5)
            self.assertEqual(result["top_k"][0], result["prediction"])
            self.assertTrue(all(0 <= row["score"] <= 1 for row in result["top_k"]))
            self.assertIn("chưa hiệu chuẩn", " ".join(result["warnings"]))

    def test_hog_expanded_and_pca_pass_raw_features_to_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in ("hog", "hog_pca"):
                for algorithm in ("dt", "knn"):
                    with self.subTest(method=method, algorithm=algorithm):
                        path = Path(temp) / f"{method}_{algorithm}.joblib"
                        model = bundle(path, method=method, algorithm=algorithm,
                                       expanded=method == "hog")
                        registry = ModelRegistry(temp)
                        result = registry.predict(encoded(), model_id=path.stem)
                        image, _, _ = prepare_image(encoded())
                        raw = raw_features(image, method)
                        self.assertEqual(result["prediction"]["class_id"], int(model.predict(raw)[0]))
                        self.assertEqual(result["model"]["method"], method)

    def test_default_prefers_full_hog_then_newest_not_demo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle(root / "old_hog.joblib", method="hog", classes=tuple(range(43)))
            bundle(root / "new_hog.joblib", method="hog", classes=tuple(range(43)))
            bundle(root / "pixel.joblib", classes=tuple(range(43)))
            bundle(root / "demo.joblib", method="hog", status="demo", classes=tuple(range(43)))
            os.utime(root / "old_hog.joblib", ns=(1000000000, 1000000000))
            result = ModelRegistry(temp).predict(encoded())
            self.assertEqual(result["model"]["id"], "new_hog")

    def test_default_id_prioritizes_knn_before_newer_tree_and_newest_knn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle(root / "old_knn.joblib", method="hog", algorithm="knn", classes=tuple(range(43)))
            bundle(root / "new_knn.joblib", method="hog", algorithm="knn", classes=tuple(range(43)))
            bundle(root / "newer_tree.joblib", method="hog", algorithm="dt", classes=tuple(range(43)))
            os.utime(root / "old_knn.joblib", ns=(1_000_000_000, 1_000_000_000))
            os.utime(root / "new_knn.joblib", ns=(2_000_000_000, 2_000_000_000))
            os.utime(root / "newer_tree.joblib", ns=(3_000_000_000, 3_000_000_000))
            registry = ModelRegistry(temp)
            self.assertEqual(registry.default_model_id(), "new_knn")
            self.assertEqual(registry.predict(encoded())["model"]["id"], registry.default_model_id())
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(ModelRegistry(temp).default_model_id())

    def test_smoke_status_is_not_upgraded_and_has_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle(Path(temp) / "smoke.joblib", method="hog", status="smoke_only", expanded=True)
            registry = ModelRegistry(temp)
            self.assertEqual(registry.list_models()[0]["status"], "smoke_only")
            self.assertIn("không phải mô hình đầy đủ", " ".join(registry.predict(encoded())["warnings"]))

    def test_explicit_broken_unknown_or_traversal_id_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle(Path(temp) / "good.joblib")
            (Path(temp) / "broken.joblib").write_bytes(b"not a joblib file")
            registry = ModelRegistry(temp)
            self.assertEqual(len(registry.list_models()), 2)
            for model_id in ("broken", "unknown", "../good", "good.joblib"):
                with self.subTest(model_id=model_id), self.assertRaises(InferenceError):
                    registry.predict(encoded(), model_id=model_id)

    def test_bad_contracts_algorithms_dimensions_versions_unfit_remain_visible(self):
        cases = [
            {"method": "other"}, {"algorithm": "knn"}, {"raw_dimensions": 100},
            {"image_contract": {**IMAGE_CONTRACT, "color": "BGR"}},
            {"hog_params": {**HOG_CONTRACT, "orientations": 18}},
            {"labels": [True]}, {"labels": [2]},
            {"metadata": {"versions": {"packages": {"scikit-learn": "0.0.1"}}}},
            {"estimator": Pipeline([("dt", DecisionTreeClassifier())])},
            {"estimator": DecisionTreeClassifier().fit([[0], [1]], [2, 7])},
            {"status": "in_progress"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            for index, updates in enumerate(cases):
                bundle(Path(temp) / f"bad_{index}.joblib", updates=updates)
            records = ModelRegistry(temp).list_models()
            self.assertEqual(len(records), len(cases))
            self.assertTrue(all(not item["ready"] and item["error"] for item in records))
            self.assertTrue(all(temp not in item["error"] for item in records))

    def test_invalid_top_k(self):
        for value in (0, 44, 2.5, True, "5"):
            with self.subTest(value=value), self.assertRaises(InferenceError):
                ModelRegistry("missing").predict(encoded(), top_k=value)


if __name__ == "__main__":
    unittest.main()
