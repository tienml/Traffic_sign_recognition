"""Flask API regression tests: synthetic images, stub models, no official Test."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import MultiDict

from src.inference import InferenceError, MAX_IMAGE_BYTES as INFERENCE_MAX_IMAGE_BYTES
from web.app import MAX_IMAGE_BYTES, create_app


class StubRegistry:
    """Exercise the HTTP/service boundary without loading or fitting any model."""
    def __init__(self, records=None):
        self.records = ([{
            "id": "synthetic/run/hog_knn", "name": "KNN · HOG · synthetic",
            "algorithm": "knn", "method": "hog", "status": "demo", "ready": True,
            "error": None, "n_classes": 2, "training_samples": 8,
        }] if records is None else records)
        self.calls = []
        self.error = None

    def list_models(self):
        return copy.deepcopy(self.records)

    def predict(self, image_bytes, model_id=None, roi=None, top_k=5):
        self.calls.append({"bytes": image_bytes, "model_id": model_id,
                           "roi": roi, "top_k": top_k})
        if self.error:
            raise self.error
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            raise InferenceError("Ảnh không hợp lệ.") from None
        if roi is not None:
            x1, y1, x2, y2 = roi
            if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
                raise InferenceError("ROI phải nằm trong ảnh.")
        ready = [model for model in self.records if model["ready"]]
        if not ready:
            raise InferenceError("Chưa có mô hình sẵn sàng.", 503)
        record = next((model for model in ready if model["id"] == model_id), ready[0])
        prediction = {"class_id": 1, "label": "Biển tổng hợp", "label_en": "Synthetic sign", "score": 0.75}
        digest = hashlib.sha256(image_bytes + repr(tuple(roi) if roi else None).encode()).hexdigest()
        return {"model": {key: record[key] for key in ("id", "name", "algorithm", "method")},
                "prediction": prediction, "top_k": [prediction][:top_k], "elapsed_ms": 1.0,
                "image": {"width": width, "height": height},
                "processed_image": "data:image/png;base64," + digest,
                "warnings": ["Điểm minh họa từ stub, không phải đánh giá mô hình."]}


def png_bytes(size=(24, 20), color=(180, 80, 40)):
    image = Image.new("RGB", size, color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data" / "archive"
        self.dataset.mkdir(parents=True)
        (self.dataset / "Meta").mkdir()
        (self.dataset / "Meta" / "1.png").write_bytes(png_bytes())
        self.image = png_bytes()
        self.rows = []
        for class_id in range(12):
            for frame, size, roi in ((0, (24, 20), (2, 3, 20, 16)),
                                     (1, (48, 40), (3, 4, 42, 35))):
                relative = f"Train/{class_id}/{class_id:05d}_00000_{frame:05d}.png"
                path = self.dataset / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png_bytes(size, (160 + class_id, 80, 40)))
                self.rows.append(dict(zip(
                    ("Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"),
                    (*size, *roi, class_id, relative))))
        self.write_manifest()
        self.registry = StubRegistry()
        self.app = self.new_app()
        self.client = self.app.test_client()

    def write_manifest(self):
        with (self.dataset / "Train.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"))
            writer.writeheader()
            writer.writerows(self.rows)

    def new_app(self, *, registry=None, dataset=None, **config):
        return create_app(model_dir=self.root / "models", dataset_dir=dataset or self.root / "data",
                          registry=self.registry if registry is None else registry,
                          test_config={"TESTING": True, **config})

    def upload(self, data=None, image=None, client=None):
        form = {"image": (io.BytesIO(self.image if image is None else image), "../../private.png")}
        form.update(data or {})
        return (client or self.client).post("/api/predict", data=form, content_type="multipart/form-data")

    def assert_error(self, response, status):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        self.assertEqual(set(response.get_json()), {"error"})
        self.assertIsInstance(response.get_json()["error"], str)
        self.assertTrue(response.get_json()["error"])
        self.assertNotIn(self.temporary.name, response.get_data(as_text=True))
        self.assertNotIn("Traceback", response.get_data(as_text=True))

    def test_root_health_models_classes_and_security_headers(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.content_type)
        html = page.get_data(as_text=True)
        for removed in ('href="#thuc-nghiem"', 'data-view="thuc-nghiem"',
                        'id="view-thuc-nghiem"', 'id="download-result"', "Tải JSON"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, html)
        self.assertIn("script-src 'self'", page.headers["Content-Security-Policy"])
        self.assertNotIn("unsafe-inline", page.headers["Content-Security-Policy"])
        self.assertEqual(page.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(page.headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(self.client.get("/api/health").get_json(),
                         {"status": "ok", "model_count": 1, "ready_models": 1})
        models = self.client.get("/api/models").get_json()
        self.assertEqual(models["default_model_id"], self.registry.records[0]["id"])
        self.assertEqual(len(models["models"]), 1)
        classes = self.client.get("/api/classes").get_json()["classes"]
        self.assertEqual([item["class_id"] for item in classes], list(range(43)))
        self.assertEqual(self.client.get("/api/health").headers["Cache-Control"], "no-store")
        self.assertFalse(self.app.debug)
        self.assertFalse(self.registry.calls)

    def test_sample_catalog_is_deterministic_diverse_and_uses_larger_roi(self):
        response = self.client.get("/api/samples")
        samples = response.get_json()["samples"]
        self.assertEqual(len(samples), 10)
        self.assertEqual(len({sample["class_id"] for sample in samples}), 10)
        for sample in samples:
            self.assertEqual(set(sample), {"id", "label", "class_id", "image_url"})
            self.assertIn("00001", sample["id"])
            self.assertEqual(self.client.get(sample["image_url"]).status_code, 200)
        # Input order does not affect ranking or public IDs.
        self.rows.reverse()
        self.write_manifest()
        other = self.new_app().test_client().get("/api/samples").get_json()["samples"]
        self.assertEqual(samples, other)
        self.assertFalse(self.registry.calls)

    def test_sample_listing_retries_metadata_and_clears_failure_note(self):
        success_note = self.client.get("/api/samples").get_json()["note"]
        manifest = self.dataset / "Train.csv"
        for invalid in (None, "wrong,columns\n1,2\n"):
            with self.subTest(metadata=invalid):
                if invalid is None:
                    manifest.unlink()
                else:
                    manifest.write_text(invalid, encoding="utf-8")
                client = self.new_app(dataset=self.dataset).test_client()
                failed = client.get("/api/samples").get_json()
                self.assertEqual(failed["samples"], [])
                self.assertNotEqual(failed["note"], success_note)
                self.write_manifest()
                recovered = client.get("/api/samples").get_json()
                self.assertEqual(len(recovered["samples"]), 10)
                self.assertEqual(recovered["note"], success_note)

    def test_sample_listing_refreshes_metadata_and_added_or_deleted_files(self):
        smaller, larger = [row for row in self.rows if row["ClassId"] == 1]
        self.rows = [smaller]
        self.write_manifest()
        initial = self.client.get("/api/samples").get_json()
        first = initial["samples"][0]
        self.rows.append(larger)
        self.write_manifest()
        catalog = self.app.extensions["sample_catalog"]
        with patch.object(catalog, "_build", side_effect=AssertionError("Lookups must reuse catalog")):
            self.assertEqual(self.client.get(first["image_url"]).status_code, 200)
            self.assertEqual(self.client.post("/api/predict", data={"sample_id": first["id"]}).status_code, 200)
        refreshed = self.client.get("/api/samples").get_json()["samples"][0]
        self.assertNotEqual(refreshed["id"], first["id"])
        larger_path = self.dataset / larger["Path"]
        larger_path.unlink()
        self.assertEqual(self.client.get("/api/samples").get_json()["samples"], [first])
        (self.dataset / smaller["Path"]).unlink()
        self.assertEqual(self.client.get("/api/samples").get_json()["samples"], [])
        larger_path.write_bytes(png_bytes((48, 40)))
        recovered = self.client.get("/api/samples").get_json()
        self.assertEqual(recovered["samples"], [refreshed])
        self.assertEqual(recovered["note"], initial["note"])

    def test_sample_listing_falls_back_to_ninth_metadata_candidate(self):
        self.rows = []
        for frame in range(9):
            size = 100 - frame
            relative = f"Train/42/00042_00000_{frame:05d}.png"
            self.rows.append({"ClassId": 42, "Width": size, "Height": size,
                              "Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": size - 1,
                              "Roi.Y2": size - 1, "Path": relative})
        only_image = self.dataset / self.rows[-1]["Path"]
        only_image.parent.mkdir(parents=True)
        only_image.write_bytes(self.image)
        self.write_manifest()
        samples = self.client.get("/api/samples").get_json()["samples"]
        self.assertEqual([sample["id"] for sample in samples], ["train-00042_00000_00008"])

    def test_sample_listing_skips_empty_or_oversized_files_at_configured_limit(self):
        self.rows = [row for row in self.rows if row["ClassId"] == 1]
        self.write_manifest()
        larger_path = self.dataset / self.rows[1]["Path"]
        (self.dataset / self.rows[0]["Path"]).write_bytes(self.image)
        client = self.new_app(MAX_IMAGE_BYTES=len(self.image)).test_client()
        for content in (b"", self.image + b"x"):
            with self.subTest(size=len(content)):
                larger_path.write_bytes(content)
                samples = client.get("/api/samples").get_json()["samples"]
                self.assertEqual([sample["id"] for sample in samples], ["train-00001_00000_00000"])
        larger_path.write_bytes(self.image)
        samples = client.get("/api/samples").get_json()["samples"]
        self.assertEqual([sample["id"] for sample in samples], ["train-00001_00000_00001"])

    def test_tilde_paths_expand_for_models_and_dataset(self):
        with patch.dict("os.environ", {"HOME": str(self.root), "USERPROFILE": str(self.root)}):
            app = self.new_app(dataset="~/data", MODEL_DIR="~/models")
            self.assertEqual(app.extensions["model_registry"].model_dir, self.root / "models")
            self.assertEqual(app.extensions["sample_catalog"].root, self.dataset)
            client = app.test_client()
            self.assertEqual(len(client.get("/api/samples").get_json()["samples"]), 10)

    def test_sample_and_upload_share_bytes_inclusive_roi_and_prediction(self):
        sample = self.client.get("/api/samples").get_json()["samples"][0]
        sample_bytes = self.client.get(sample["image_url"]).data
        response = self.client.post("/api/predict", data={"sample_id": sample["id"], "top_k": "3"})
        self.assertEqual(response.status_code, 200)
        call = self.registry.calls[-1]
        self.assertEqual(call["bytes"], sample_bytes)
        self.assertEqual(tuple(call["roi"]), (3, 4, 42, 35))
        self.assertEqual(call["top_k"], 3)
        upload = self.upload({"roi": json.dumps(list(call["roi"])), "top_k": "3"}, sample_bytes)
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(upload.get_json(), response.get_json())
        # Uploaded filenames never become local filesystem paths.
        self.assertFalse((self.root / "private.png").exists())

    def test_upload_defaults_and_opaque_model_keys(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.registry.calls[-1]["roi"])
        self.assertIsNone(self.registry.calls[-1]["model_id"])
        self.assertEqual(self.registry.calls[-1]["top_k"], 5)
        opaque = "trusted_run/" + "x" * 200 + "/hog_knn"
        self.registry.records[0]["id"] = opaque
        self.assertEqual(self.upload({"model_id": opaque}).status_code, 200)
        self.assertEqual(self.registry.calls[-1]["model_id"], opaque)

    def test_api_default_model_id_uses_shared_registry_selector(self):
        # The first listed demo must not shadow a full model preferred by service.
        record = dict(self.registry.records[0], id="full/hog_knn", status="train_refit", n_classes=43)
        self.registry.records.append(record)
        with patch.object(self.registry, "default_model_id", return_value=record["id"], create=True) as selector:
            payload = self.client.get("/api/models").get_json()
        self.assertEqual(payload["default_model_id"], "full/hog_knn")
        selector.assert_called_once_with()
        self.assertFalse(self.registry.calls)

    def test_sample_selection_reads_training_metadata_without_decoding_images(self):
        opened = []
        real_open = Path.open
        def tracked_open(path, *args, **kwargs):
            opened.append(path)
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", tracked_open), \
                patch.object(Image, "open", side_effect=AssertionError("No pixel decode for catalog")):
            response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["samples"]), 10)
        self.assertEqual(opened, [self.dataset / "Train.csv"])
        self.assertFalse(self.registry.calls)

    def test_sample_metadata_read_is_bounded_and_retryable(self):
        with patch("web.app._MAX_SAMPLE_METADATA_BYTES", 32):
            payload = self.client.get("/api/samples").get_json()
        self.assertEqual(payload["samples"], [])
        self.assertEqual(len(self.client.get("/api/samples").get_json()["samples"]), 10)
        self.assertFalse(self.registry.calls)

    def test_roi_validation_and_inclusive_single_pixel(self):
        invalid = ("", "null", "{}", "[0,1,2]", "[0,0,1,1,2]", "[0,0,true,1]",
                   "[0,0,1.0,1]", "[-1,0,1,1]", "[2,0,1,1]", "[0,2,1,1]",
                   "[0,0,Infinity,1]", "[[0],0,1,1]", "not json", "[" * 400)
        for roi in invalid:
            with self.subTest(roi=roi[:40]):
                self.assert_error(self.upload({"roi": roi}), 400)
        self.assertFalse(self.registry.calls)
        self.assertEqual(self.upload({"roi": "[0,0,0,0]"}).status_code, 200)
        self.assertEqual(self.registry.calls[-1]["roi"], [0, 0, 0, 0])
        self.assert_error(self.upload({"roi": "[0,0,10000,10000]"}), 400)

    def test_top_k_validation_rejects_ambiguous_and_out_of_range_values(self):
        for value in ("", "0", "44", "-1", "1.0", "1e1", " 5", "5 ", "05", "NaN", "true", "999"):
            with self.subTest(value=value):
                self.assert_error(self.upload({"top_k": value}), 400)
        self.assertFalse(self.registry.calls)
        for value in ("1", "43"):
            self.assertEqual(self.upload({"top_k": value}).status_code, 200)
            self.assertEqual(self.registry.calls[-1]["top_k"], int(value))

    def test_form_requires_exactly_one_source_and_rejects_unknown_fields(self):
        self.assert_error(self.client.post("/api/predict", data={}), 400)
        self.assert_error(self.upload({"sample_id": "any"}), 400)
        self.assert_error(self.upload({"path": "../../secret"}), 400)
        self.assert_error(self.upload({"image": "not-a-file"}), 400)
        self.assert_error(self.client.post("/api/predict", json={"sample_id": "any"}), 400)
        self.assert_error(self.client.post("/api/predict?top_k=2", data={"sample_id": "any"}), 400)
        form = MultiDict([("image", (io.BytesIO(self.image), "a.png")),
                          ("model", (io.BytesIO(self.image), "b.png"))])
        self.assert_error(self.client.post("/api/predict", data=form), 400)
        self.assertFalse(self.registry.calls)

    def test_repeated_form_and_file_fields_are_rejected(self):
        for name, first, second in (("sample_id", "a", "b"), ("top_k", "5", "6"),
                                    ("model_id", "one", "two"), ("roi", "[0,0,1,1]", "[0,0,2,2]")):
            with self.subTest(name=name):
                fields = MultiDict([(name, first), (name, second)])
                if name != "sample_id":
                    fields.add("image", (io.BytesIO(self.image), "x.png"))
                self.assert_error(self.client.post("/api/predict", data=fields), 400)
        files = MultiDict([("image", (io.BytesIO(self.image), "a.png")),
                           ("image", (io.BytesIO(self.image), "b.png"))])
        self.assert_error(self.client.post("/api/predict", data=files), 400)
        self.assertFalse(self.registry.calls)

    def test_sample_rejects_replacement_roi_unknown_id_and_traversal(self):
        sample = self.client.get("/api/samples").get_json()["samples"][0]
        self.assert_error(self.client.post("/api/predict", data={
            "sample_id": sample["id"], "roi": "[0,0,1,1]"}), 400)
        for sample_id in ("", "../../Train.csv", "Meta/1.png", "Test/00000.png", str(self.dataset / "Train.csv")):
            with self.subTest(sample_id=sample_id):
                self.assert_error(self.client.post("/api/predict", data={"sample_id": sample_id}), 400)
        for url in ("/api/samples/unknown/image", "/api/samples/..%2f..%2fTrain.csv/image",
                    "/api/classes/43/image", "/api/classes/-1/image", "/api/classes/..%2fTrain.csv/image"):
            self.assert_error(self.client.get(url), 404)
        self.assertFalse(self.registry.calls)

    def test_dataset_metadata_cannot_catalog_test_meta_or_outside_paths(self):
        self.rows = []
        malicious = ("../outside.png", "Meta/1.png", "Test/00000.png", "Train/1/../../Meta/1.png",
                     "Train/1/00001_00000_00000.png/extra", "Train\\1\\00001_00000_00000.png",
                     "Train/1//00001_00000_00000.png", "Train/1/00002_00000_00000.png",
                     str(self.root / "secret.png"), "C:/secret.png")
        for relative in malicious:
            self.rows.append({"ClassId": 1, "Width": 24, "Height": 20, "Roi.X1": 1,
                              "Roi.Y1": 1, "Roi.X2": 20, "Roi.Y2": 18, "Path": relative})
        self.write_manifest()
        response = self.new_app().test_client().get("/api/samples")
        self.assertEqual(response.get_json()["samples"], [])
        self.assertFalse(self.registry.calls)

    def test_sample_symlink_into_official_test_is_never_allowed(self):
        # Even an in-dataset link must not turn a Train allowlist into Test input.
        target = self.dataset / "Test" / "00000.png"
        target.parent.mkdir()
        target.write_bytes(self.image)
        relative = "Train/42/00042_00000_00000.png"
        link = self.dataset / relative
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation requires Windows developer mode or privilege.")
        self.rows = [{"ClassId": 42, "Width": 24, "Height": 20, "Roi.X1": 1,
                      "Roi.Y1": 1, "Roi.X2": 20, "Roi.Y2": 18, "Path": relative}]
        self.write_manifest()
        self.assertEqual(self.new_app().test_client().get("/api/samples").get_json()["samples"], [])

    def test_missing_dataset_file_and_reference_only_meta(self):
        self.assertEqual(self.client.get("/api/classes/1/image").data, self.image)
        self.assertFalse(self.registry.calls)
        self.assert_error(self.client.get("/api/classes/42/image"), 404)
        samples = self.client.get("/api/samples").get_json()["samples"]
        entry = self.app.extensions["sample_catalog"].lookup(samples[0]["id"])
        (self.dataset / str(entry["relative"])).unlink()
        self.assert_error(self.client.get(samples[0]["image_url"]), 404)
        self.assert_error(self.client.post("/api/predict", data={"sample_id": samples[0]["id"]}), 400)
        missing = self.new_app(dataset=self.root / "absent").test_client()
        response = missing.get("/api/samples").get_json()
        self.assertEqual(response["samples"], [])
        self.assertTrue(response["note"])
        self.assertEqual(missing.get("/").status_code, 200)
        self.assertEqual(missing.get("/api/health").status_code, 200)

    def test_ten_megabyte_upload_limit_and_multipart_overhead(self):
        self.assertEqual(MAX_IMAGE_BYTES, 10 * 1024 * 1024)
        self.assertEqual(INFERENCE_MAX_IMAGE_BYTES, MAX_IMAGE_BYTES)
        self.assertEqual(self.app.config["MAX_IMAGE_BYTES"], MAX_IMAGE_BYTES)
        self.assertEqual(self.app.config["MAX_CONTENT_LENGTH"], MAX_IMAGE_BYTES + 128 * 1024)
        self.assertEqual(self.app.config["MAX_FORM_MEMORY_SIZE"], 256 * 1024)
        self.assertEqual(self.app.config["MAX_FORM_PARTS"], 12)
        # Pad a tiny static PNG after IEND; do not allocate huge image pixels.
        for size in (9 * 1024 * 1024, MAX_IMAGE_BYTES):
            with self.subTest(size=size):
                content = self.image + b"\0" * (size - len(self.image))
                response = self.upload({"roi": "[0,0,0,0]", "top_k": "3",
                                        "model_id": self.registry.records[0]["id"]}, image=content)
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                self.assertEqual(self.registry.calls[-1]["bytes"], content)
                self.assertEqual(response.get_json()["image"], {"width": 24, "height": 20})

    def test_sample_catalog_and_prediction_share_ten_megabyte_limit(self):
        self.assertEqual(self.app.extensions["sample_catalog"].max_image_bytes, MAX_IMAGE_BYTES)
        self.rows = [row for row in self.rows if row["ClassId"] == 1]
        self.write_manifest()
        larger_path = self.dataset / self.rows[1]["Path"]
        data = png_bytes((48, 40))
        content = data + b"\0" * (MAX_IMAGE_BYTES - len(data))
        larger_path.write_bytes(content)
        sample = self.client.get("/api/samples").get_json()["samples"][0]
        self.assertEqual(sample["id"], "train-00001_00000_00001")
        self.assertEqual(self.client.get(sample["image_url"]).data, content)
        self.assertEqual(self.client.post("/api/predict", data={"sample_id": sample["id"]}).status_code, 200)
        self.assertEqual(self.registry.calls[-1]["bytes"], content)
        with larger_path.open("ab") as stream:
            stream.write(b"\0")
        self.assert_error(self.client.get(sample["image_url"]), 413)
        self.assert_error(self.client.post("/api/predict", data={"sample_id": sample["id"]}), 413)
        fallback = self.client.get("/api/samples").get_json()["samples"][0]
        self.assertEqual(fallback["id"], "train-00001_00000_00000")

    def test_bad_or_oversized_uploads_and_multipart_overhead(self):
        self.assert_error(self.upload(image=b""), 400)
        self.assert_error(self.upload(image=b"<svg onload=alert(1)></svg>"), 400)
        self.assert_error(self.upload(image=b"not an image"), 400)
        for size in (MAX_IMAGE_BYTES + 1, MAX_IMAGE_BYTES + 200 * 1024):
            with self.subTest(size=size):
                content = self.image + b"\0" * (size - len(self.image))
                response = self.upload(image=content)
                self.assert_error(response, 413)
                self.assertIn("10 MB", response.get_json()["error"])
        self.assertEqual(len(self.registry.calls), 2)
        # Malformed data at the exact limit still reaches inference, not a 413.
        response = self.upload(image=b"x" * MAX_IMAGE_BYTES)
        self.assert_error(response, 400)
        self.assertEqual(len(self.registry.calls[-1]["bytes"]), MAX_IMAGE_BYTES)
        self.assert_error(self.client.post("/api/predict", data={"sample_id": "x" * (300 * 1024)}), 413)

    def test_no_models_and_unknown_model_are_clear_errors(self):
        client = self.new_app(registry=StubRegistry([])).test_client()
        self.assertEqual(client.get("/api/models").get_json(), {"models": [], "default_model_id": None})
        self.assertEqual(client.get("/api/health").get_json()["ready_models"], 0)
        self.assert_error(self.upload(client=client), 503)
        self.assert_error(self.upload({"model_id": "../../private.joblib"}), 400)
        self.assertFalse(self.registry.calls)
        for model_id in ("", "x" * 513, "bad\nmodel", "bad\x7fmodel"):
            self.assert_error(self.upload({"model_id": model_id}), 400)

    def test_missing_inference_dependencies_do_not_break_pages_or_status(self):
        with patch("web.app.importlib.import_module", side_effect=ImportError("private dependency location")) as importer:
            app = create_app(model_dir=self.root / "models", dataset_dir=self.root / "absent",
                             test_config={"TESTING": True})
            self.assertFalse(importer.called)
            client = app.test_client()
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/api/models").get_json(), {"models": [], "default_model_id": None})
            self.assertEqual(client.get("/api/health").get_json(),
                             {"status": "ok", "model_count": 0, "ready_models": 0})
            self.assertEqual(client.get("/api/classes").get_json()["classes"], [])
            self.assertEqual(client.get("/api/samples").get_json()["samples"], [])
            self.assert_error(self.upload(client=client), 503)

    def test_internal_error_and_model_error_are_sanitized(self):
        for error in (RuntimeError(str(self.root / "secret.joblib")),
                      InferenceError(f"Cannot open {self.root / 'secret.joblib'}", 503)):
            with self.subTest(error=type(error).__name__):
                self.registry.error = error
                self.assert_error(self.upload(), 503 if isinstance(error, InferenceError) else 500)
        self.registry.error = None
        record = self.registry.records[0]
        record["ready"] = False
        record["error"] = "Traceback: private model failed"
        record["name"] = str(self.root / "secret.joblib")
        record["private_path"] = str(self.root / "secret.joblib")
        body = self.client.get("/api/models").get_json()
        self.assertIsNone(body["default_model_id"])
        text = json.dumps(body)
        self.assertNotIn(self.temporary.name, text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn("private_path", body["models"][0])

    def test_removed_benchmarks_endpoint_is_404_without_reading_reports_or_models(self):
        self.assertNotIn("BENCHMARK_DIR", self.app.config)
        with patch.object(Path, "open", side_effect=AssertionError("Removed route must not read reports")), \
                patch.object(self.registry, "list_models", side_effect=AssertionError("Must not load models")), \
                patch.object(self.registry, "predict", side_effect=AssertionError("Must not predict")):
            self.assert_error(self.client.get("/api/benchmarks"), 404)
        self.assertFalse(self.registry.calls)

    def test_http_errors(self):
        self.assert_error(self.client.get("/no-such-route"), 404)
        response = self.client.get("/api/predict")
        self.assert_error(response, 405)
        self.assertIn("POST", response.headers["Allow"])


if __name__ == "__main__":
    unittest.main()
