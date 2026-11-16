"""Flask transport for local, single-image traffic-sign inference.

No training, model deserialization, official-test images, or upload persistence
happens in this module. Model dependencies are imported only when needed.
"""
from __future__ import annotations

import csv
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from threading import Lock

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.inference import MAX_IMAGE_BYTES

ROOT = Path(__file__).resolve().parents[1]
_MAX_SAMPLE_METADATA_BYTES = 16 * 1024 * 1024
_ROI_COLUMNS = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
_SAMPLE_NAME = re.compile(r"(\d{5})_\d{5}_\d{5}\.png\Z")
_SAMPLE_NOTE = "Ảnh minh họa lấy từ tập Train, không phải tập Test để đánh giá."
_ERROR_MESSAGES = {
    400: "Yêu cầu không hợp lệ. Vui lòng kiểm tra ảnh và các tùy chọn.",
    404: "Không tìm thấy tài nguyên yêu cầu.",
    405: "Phương thức không được hỗ trợ.",
    413: "Ảnh vượt quá giới hạn 10 MB hoặc biểu mẫu quá lớn.",
    500: "Không thể xử lý yêu cầu. Vui lòng thử lại.",
    503: "Chưa có mô hình sẵn sàng hoặc thiếu thư viện chạy mô hình.",
}


class _ApiError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _public_text(value, fallback: str) -> str:
    """Do not leak dependency exceptions, absolute paths, or tracebacks."""
    if not isinstance(value, str) or not value or len(value) > 500:
        return fallback
    if (re.search(r"[A-Za-z]:[\\/]|(?:^|\s)/\S|\\|[\x00-\x1f]", value)
            or "traceback" in value.lower() or 'File "' in value):
        return fallback
    return value


def _class_records() -> list[dict]:
    try:
        return importlib.import_module("src.inference").get_classes()
    except ImportError:
        return []


def _dataset_root(explicit) -> Path:
    base = Path(explicit) if explicit is not None else ROOT / "data"
    base = base.expanduser().resolve()
    if (base / "Train.csv").is_file():
        return base
    if (base / "archive" / "Train.csv").is_file():
        return base / "archive"
    return base


def _confined_file(root: Path, relative: PurePosixPath) -> Path | None:
    """Reject traversal and redirection, including Train links into Test/Meta."""
    if relative.is_absolute() or any(p in (".", "..") for p in relative.parts):
        return None
    try:
        base = root.resolve()
        current = base
        for part in relative.parts:
            current /= part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                return None
        candidate = current.resolve()
        candidate.relative_to(base)
        return candidate if candidate.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


class _SampleCatalog:
    """A small, deterministic allowlist built solely from Train.csv metadata.

    Prefer large annotated sign areas and diverse classes. Brightness cannot be
    inferred from Train.csv, so no brightness or quality guarantee is made.
    Opening candidate image pixels is deliberately not part of selection.
    """
    def __init__(self, root: Path, max_image_bytes: int = MAX_IMAGE_BYTES):
        self.root = root
        self.max_image_bytes = max_image_bytes
        self._entries = None
        self._lock = Lock()
        self.note = _SAMPLE_NOTE

    def entries(self, *, refresh: bool = False) -> dict:
        with self._lock:
            if refresh or self._entries is None:
                self.note = _SAMPLE_NOTE
                self._entries = self._build()
            return self._entries

    def _build(self) -> dict:
        candidates = {}
        try:
            metadata = _confined_file(self.root, PurePosixPath("Train.csv"))
            if metadata is None:
                self.note = "Chưa tìm thấy Train.csv; bạn vẫn có thể tải ảnh riêng lên."
                return {}
            with metadata.open("rb") as stream:
                # Bound metadata memory without discarding lower-ranked rows.
                content = stream.read(_MAX_SAMPLE_METADATA_BYTES + 1)
                if len(content) > _MAX_SAMPLE_METADATA_BYTES:
                    raise ValueError("Training metadata too large")
                reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
                required = {"Path", "ClassId", "Width", "Height", *_ROI_COLUMNS}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError("Invalid training metadata")
                for row in reader:
                    try:
                        class_id = int(row["ClassId"])
                        width, height = int(row["Width"]), int(row["Height"])
                        roi = tuple(int(row[name]) for name in _ROI_COLUMNS)
                        raw_path = row["Path"]
                        relative = PurePosixPath(raw_path)
                        match = _SAMPLE_NAME.fullmatch(relative.name)
                        if (not 0 <= class_id < 43 or "\\" in raw_path
                                or relative.as_posix() != raw_path
                                or len(relative.parts) != 3
                                or relative.parts[:2] != ("Train", str(class_id))
                                or match is None or int(match[1]) != class_id
                                or not (0 <= roi[0] <= roi[2] < width
                                        and 0 <= roi[1] <= roi[3] < height)):
                            continue
                        area = (roi[2] - roi[0] + 1) * (roi[3] - roi[1] + 1)
                        score = (-area, -min(width, height), raw_path)
                        group = candidates.setdefault(class_id, [])
                        group.append((score, relative, roi))
                    except (TypeError, ValueError, KeyError):
                        continue
        except (OSError, UnicodeError, ValueError, csv.Error):
            self.note = "Không đọc được danh mục Train; hãy tải ảnh riêng lên."
            return {}

        # Different sign families, with a stable fallback for small datasets.
        preferred = [14, 13, 17, 1, 2, 38, 35, 25, 12, 28]
        order = preferred + [c for c in sorted(candidates) if c not in preferred]
        result = {}
        for class_id in order:
            for _, relative, roi in sorted(candidates.get(class_id, []), key=lambda item: item[0]):
                path = _confined_file(self.root, relative)
                if path is None:
                    continue
                try:
                    if not 0 < path.stat().st_size <= self.max_image_bytes:
                        continue
                except OSError:
                    continue
                sample_id = "train-" + relative.stem
                result[sample_id] = {
                    "id": sample_id, "class_id": class_id,
                    "relative": relative, "roi": roi,
                }
                break
            if len(result) == 10:
                break
        if not result:
            self.note = "Chưa có ảnh Train hợp lệ; bạn có thể tải ảnh riêng lên."
        return result

    def lookup(self, sample_id: str) -> dict:
        entry = self.entries().get(sample_id)
        if entry is None:
            raise _ApiError("Ảnh mẫu không thuộc danh mục cho phép.")
        return entry


class _RegistryProvider:
    def __init__(self, model_dir: Path, registry):
        self.model_dir = model_dir
        self.instance = registry
        self._lock = Lock()

    def get(self):
        with self._lock:
            if self.instance is None:
                try:
                    module = importlib.import_module("src.inference")
                    self.instance = module.ModelRegistry(self.model_dir)
                except (ImportError, OSError, ValueError):
                    return None
            return self.instance


def _model_records(provider: _RegistryProvider) -> list[dict]:
    registry = provider.get()
    if registry is None:
        return []
    records = []
    for item in registry.list_models():
        record = {key: item.get(key) for key in (
            "id", "name", "algorithm", "method", "status", "ready",
            "error", "n_classes", "training_samples",
        )}
        record["error"] = (_public_text(record["error"], "Không thể nạp mô hình.")
                           if record["error"] else None)
        record["name"] = _public_text(record["name"], "Mô hình nhận dạng biển báo")
        records.append(record)
    return records


def _parse_prediction_form():
    if request.args:
        raise _ApiError("Các tùy chọn dự đoán phải gửi trong biểu mẫu, không qua URL.")
    if request.mimetype not in ("multipart/form-data", "application/x-www-form-urlencoded"):
        raise _ApiError("Gửi ảnh hoặc mã ảnh mẫu bằng biểu mẫu.")
    form, files = request.form, request.files
    allowed = {"sample_id", "model_id", "top_k", "roi"}
    if (set(form) - allowed or set(files) - {"image"}
            or any(len(form.getlist(key)) != 1 for key in form)
            or any(len(files.getlist(key)) != 1 for key in files)):
        raise _ApiError("Biểu mẫu có trường không hỗ trợ hoặc bị lặp.")
    if ("image" in files) == ("sample_id" in form):
        raise _ApiError("Chọn đúng một nguồn ảnh: tải ảnh lên hoặc chọn ảnh mẫu.")

    model_id = form.get("model_id")
    if model_id is not None:
        # IDs are exact registry keys, never used to construct filesystem paths.
        if not model_id or len(model_id) > 512 or re.search(r"[\x00-\x1f\x7f]", model_id):
            raise _ApiError("Mã mô hình không hợp lệ.")

    raw_top_k = form.get("top_k", "5")
    if not re.fullmatch(r"[1-9][0-9]?", raw_top_k) or not 1 <= int(raw_top_k) <= 43:
        raise _ApiError("top_k phải là số nguyên từ 1 đến 43.")
    roi = None
    if "roi" in form:
        if "sample_id" in form:
            raise _ApiError("Ảnh mẫu tự dùng ROI trong Train.csv; không gửi ROI khác.")
        try:
            if len(form["roi"]) > 256:
                raise ValueError("ROI too long")
            roi = json.loads(form["roi"])
            if (not isinstance(roi, list) or len(roi) != 4
                    or any(type(value) is not int or value < 0 for value in roi)
                    or roi[0] > roi[2] or roi[1] > roi[3]):
                raise ValueError("Invalid ROI")
        except (ValueError, TypeError, RecursionError):
            raise _ApiError("ROI phải là mảng 4 số nguyên [x1, y1, x2, y2], tính cả điểm cuối.") from None
    return form, files, model_id, int(raw_top_k), roi


def _read_image(stream, limit: int) -> bytes:
    image_bytes = stream.read(limit + 1)
    if len(image_bytes) > limit:
        raise _ApiError(_ERROR_MESSAGES[413], 413)
    if not image_bytes:
        raise _ApiError("Ảnh trống hoặc không đọc được.")
    return image_bytes


def create_app(*, model_dir=None, dataset_dir=None, registry=None, test_config=None):
    """Create the app without training or eagerly loading model dependencies."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(
        DEBUG=False, MODEL_DIR=model_dir if model_dir is not None else ROOT / "data" / "models",
        DATASET_DIR=dataset_dir,
        MAX_IMAGE_BYTES=MAX_IMAGE_BYTES,
        MAX_FORM_MEMORY_SIZE=256 * 1024, MAX_FORM_PARTS=12,
    )
    if test_config:
        app.config.update(test_config)
    if not test_config or "MAX_CONTENT_LENGTH" not in test_config:
        # Keep room for multipart framing without weakening the file-byte cap.
        app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_IMAGE_BYTES"] + 128 * 1024
    app.config["DEBUG"] = False
    app.json.ensure_ascii = False
    provider = _RegistryProvider(Path(app.config["MODEL_DIR"]).expanduser().resolve(), registry)
    catalog = _SampleCatalog(_dataset_root(app.config["DATASET_DIR"]), app.config["MAX_IMAGE_BYTES"])
    app.extensions["model_registry"] = provider
    app.extensions["sample_catalog"] = catalog

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(_ApiError)
    def api_error(error):
        return jsonify(error=error.message), error.status_code

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error=_ERROR_MESSAGES[413]), 413

    @app.errorhandler(HTTPException)
    def http_error(error):
        response = error.get_response()
        response.set_data(app.json.dumps({"error": _ERROR_MESSAGES.get(error.code, _ERROR_MESSAGES[400])}))
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.error("Request failed (%s).", type(error).__name__)
        return jsonify(error=_ERROR_MESSAGES[500]), 500

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        models = _model_records(provider)
        return jsonify(status="ok", model_count=len(models),
                       ready_models=sum(bool(model["ready"]) for model in models))

    @app.get("/api/models")
    def models():
        items = _model_records(provider)
        service = provider.get()
        selector = getattr(service, "default_model_id", None)
        default_id = (selector() if callable(selector) else
                      next((model["id"] for model in items if model["ready"]), None))
        return jsonify(models=items, default_model_id=default_id)

    @app.get("/api/classes")
    def classes():
        records = _class_records()
        payload = {"classes": records}
        if not records:
            payload["note"] = "Chưa nạp được danh mục nhãn; hãy kiểm tra thư viện mô hình."
        return jsonify(payload)

    @app.get("/api/samples")
    def samples():
        entries = catalog.entries(refresh=True)
        labels = {item["class_id"]: item["label"] for item in _class_records()}
        items = [{"id": entry["id"], "class_id": entry["class_id"],
                  "label": labels.get(entry["class_id"], f"Biển báo {entry['class_id']}"),
                  "image_url": f"/api/samples/{entry['id']}/image"}
                 for entry in entries.values()]
        return jsonify(samples=items, note=catalog.note)

    @app.get("/api/samples/<sample_id>/image")
    def sample_image(sample_id):
        try:
            entry = catalog.lookup(sample_id)
        except _ApiError:
            raise _ApiError(_ERROR_MESSAGES[404], 404) from None
        path = _confined_file(catalog.root, entry["relative"])
        if path is None:
            raise _ApiError(_ERROR_MESSAGES[404], 404)
        try:
            with path.open("rb") as stream:
                content = _read_image(stream, app.config["MAX_IMAGE_BYTES"])
        except OSError:
            raise _ApiError(_ERROR_MESSAGES[404], 404) from None
        return send_file(io.BytesIO(content), mimetype="image/png", download_name="training-sample.png")

    @app.get("/api/classes/<int:class_id>/image")
    def class_image(class_id):
        # Meta prototypes are reference-only: never a prediction source.
        if not 0 <= class_id < 43:
            raise _ApiError(_ERROR_MESSAGES[404], 404)
        path = _confined_file(catalog.root, PurePosixPath(f"Meta/{class_id}.png"))
        if path is None:
            raise _ApiError(_ERROR_MESSAGES[404], 404)
        try:
            with path.open("rb") as stream:
                content = _read_image(stream, app.config["MAX_IMAGE_BYTES"])
        except OSError:
            raise _ApiError(_ERROR_MESSAGES[404], 404) from None
        return send_file(io.BytesIO(content), mimetype="image/png", download_name=f"class-{class_id}.png")

    @app.post("/api/predict")
    def predict():
        form, files, model_id, top_k, roi = _parse_prediction_form()
        if "sample_id" in form:
            entry = catalog.lookup(form["sample_id"])
            path = _confined_file(catalog.root, entry["relative"])
            if path is None:
                raise _ApiError("Ảnh mẫu không còn sẵn; hãy chọn ảnh khác hoặc tải ảnh lên.")
            try:
                with path.open("rb") as stream:
                    image_bytes = _read_image(stream, app.config["MAX_IMAGE_BYTES"])
            except OSError:
                raise _ApiError("Không đọc được ảnh mẫu. Hãy chọn ảnh khác.") from None
            roi = entry["roi"]
        else:
            image_bytes = _read_image(files["image"].stream, app.config["MAX_IMAGE_BYTES"])
        service = provider.get()
        if service is None:
            raise _ApiError(_ERROR_MESSAGES[503], 503)
        if model_id is not None and not any(model.get("id") == model_id for model in service.list_models()):
            raise _ApiError("Không tìm thấy mô hình đã chọn. Hãy tải lại danh sách mô hình.")
        try:
            result = service.predict(image_bytes, model_id=model_id, roi=roi, top_k=top_k)
        except Exception as error:
            try:
                known_error = isinstance(error, importlib.import_module("src.inference").InferenceError)
            except (ImportError, AttributeError):
                known_error = False
            code = getattr(error, "status_code", None)
            if known_error and code in (400, 413, 503, 500):
                message = _public_text(getattr(error, "message", None), _ERROR_MESSAGES[code])
                raise _ApiError(message, code) from None
            raise
        return jsonify(result)

    return app
