"""Trusted-local model registry and the shared, train-compatible image contract.

Importing this module does not import the ML stack. Joblib is pickle: only place
files that you trust in ``model_dir``; uploaded images are never deserialized.
"""
from __future__ import annotations

import base64
import io
import math
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from time import perf_counter
import warnings

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_CONTRACT = {
    "size": [32, 32], "color": "RGB", "roi": True,
    "scale": "divide255", "augmentation": False,
}
HOG_CONTRACT = {
    "orientations": 9, "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2), "block_norm": "L2-Hys",
}
# Class IDs are the original GTSRB IDs, NOT positions in predict_proba output.
_CLASS_DATA = (
    ("Giới hạn tốc độ 20 km/h", "Speed limit 20", "Giới hạn tốc độ"),
    ("Giới hạn tốc độ 30 km/h", "Speed limit 30", "Giới hạn tốc độ"),
    ("Giới hạn tốc độ 50 km/h", "Speed limit 50", "Giới hạn tốc độ"),
    ("Giới hạn tốc độ 60 km/h", "Speed limit 60", "Giới hạn tốc độ"),
    ("Giới hạn tốc độ 70 km/h", "Speed limit 70", "Giới hạn tốc độ"),
    ("Giới hạn tốc độ 80 km/h", "Speed limit 80", "Giới hạn tốc độ"),
    ("Hết giới hạn tốc độ 80 km/h", "End of speed limit 80", "Hết hạn chế"),
    ("Giới hạn tốc độ 100 km/h", "Speed limit 100", "Giới hạn tốc độ"),
    ("Giới hạn tốc độ 120 km/h", "Speed limit 120", "Giới hạn tốc độ"),
    ("Cấm vượt", "No passing", "Biển cấm"),
    ("Cấm xe trên 3,5 tấn vượt", "No passing for vehicles over 3.5 tons", "Biển cấm"),
    ("Ưu tiên tại giao lộ kế tiếp", "Right-of-way at next intersection", "Quyền ưu tiên"),
    ("Đường ưu tiên", "Priority road", "Quyền ưu tiên"),
    ("Nhường đường", "Yield", "Quyền ưu tiên"),
    ("Dừng lại", "Stop", "Quyền ưu tiên"),
    ("Cấm các loại xe", "No vehicles", "Biển cấm"),
    ("Cấm xe trên 3,5 tấn", "Vehicles over 3.5 tons prohibited", "Biển cấm"),
    ("Cấm đi ngược chiều", "No entry", "Biển cấm"),
    ("Nguy hiểm khác", "General caution", "Cảnh báo nguy hiểm"),
    ("Chỗ ngoặt nguy hiểm bên trái", "Dangerous curve left", "Cảnh báo nguy hiểm"),
    ("Chỗ ngoặt nguy hiểm bên phải", "Dangerous curve right", "Cảnh báo nguy hiểm"),
    ("Nhiều chỗ ngoặt liên tiếp", "Double curve", "Cảnh báo nguy hiểm"),
    ("Đường không bằng phẳng", "Bumpy road", "Cảnh báo nguy hiểm"),
    ("Đường trơn trượt", "Slippery road", "Cảnh báo nguy hiểm"),
    ("Đường hẹp bên phải", "Road narrows on the right", "Cảnh báo nguy hiểm"),
    ("Công trường", "Road work", "Cảnh báo nguy hiểm"),
    ("Giao nhau có tín hiệu đèn", "Traffic signals", "Cảnh báo nguy hiểm"),
    ("Người đi bộ", "Pedestrians", "Cảnh báo nguy hiểm"),
    ("Trẻ em qua đường", "Children crossing", "Cảnh báo nguy hiểm"),
    ("Xe đạp qua đường", "Bicycles crossing", "Cảnh báo nguy hiểm"),
    ("Cẩn thận băng, tuyết", "Beware of ice/snow", "Cảnh báo nguy hiểm"),
    ("Động vật hoang dã qua đường", "Wild animals crossing", "Cảnh báo nguy hiểm"),
    ("Hết mọi giới hạn tốc độ và cấm vượt", "End of all speed and passing limits", "Hết hạn chế"),
    ("Bắt buộc rẽ phải phía trước", "Turn right ahead", "Hướng đi bắt buộc"),
    ("Bắt buộc rẽ trái phía trước", "Turn left ahead", "Hướng đi bắt buộc"),
    ("Chỉ được đi thẳng", "Ahead only", "Hướng đi bắt buộc"),
    ("Đi thẳng hoặc rẽ phải", "Go straight or right", "Hướng đi bắt buộc"),
    ("Đi thẳng hoặc rẽ trái", "Go straight or left", "Hướng đi bắt buộc"),
    ("Đi về bên phải", "Keep right", "Hướng đi bắt buộc"),
    ("Đi về bên trái", "Keep left", "Hướng đi bắt buộc"),
    ("Vòng xuyến bắt buộc", "Roundabout mandatory", "Hướng đi bắt buộc"),
    ("Hết cấm vượt", "End of no passing", "Hết hạn chế"),
    ("Hết cấm xe trên 3,5 tấn vượt", "End of no passing by vehicles over 3.5 tons", "Hết hạn chế"),
)


def get_classes() -> list[dict]:
    """Dependency-free Vietnamese/English catalog aligned with CLASS_NAMES."""
    return [dict(class_id=i, label=vi, label_en=en, category=category)
            for i, (vi, en, category) in enumerate(_CLASS_DATA)]


class InferenceError(Exception):
    """An actionable, public-safe error; never expose an underlying file path."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class _BundleError(ValueError):
    pass


def _validate_roi(roi, width: int, height: int):
    if roi is None:
        return None
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise InferenceError("ROI phải gồm đúng 4 tọa độ [x1, y1, x2, y2].")
    if any(isinstance(v, bool) or not isinstance(v, Real)
           or (not isinstance(v, Integral) and (not math.isfinite(v) or int(v) != v))
           for v in roi):
        raise InferenceError("Tọa độ ROI phải là các số nguyên hữu hạn.")
    x1, y1, x2, y2 = (int(v) for v in roi)
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise InferenceError("ROI phải nằm trong ảnh và x1 ≤ x2, y1 ≤ y2 (tính cả pixel cuối).")
    return x1, y1, x2, y2


def prepare_image(image_bytes: bytes, roi=None):
    """Inspect before decode, orient, strict inclusive crop, INTER_AREA RGB32.

    EXIF is inspected by Pillow; OpenCV's implicit rotation is disabled and the
    same 8 EXIF transpose operations are applied explicitly. ROI uses the
    oriented image coordinates, matching what a browser displays.
    """
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise InferenceError("Vui lòng chọn một ảnh PNG, JPEG hoặc WebP.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InferenceError("Ảnh vượt quá giới hạn 10 MB.", 413)
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise InferenceError("Thiếu Pillow; hãy cài các thư viện của ứng dụng.", 503) from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as header:
                if header.format not in {"PNG", "JPEG", "WEBP"}:
                    raise InferenceError("Chỉ hỗ trợ ảnh raster PNG, JPEG và WebP.")
                width, height = header.size
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise InferenceError("Ảnh không được vượt quá 16 triệu pixel.", 413)
                if getattr(header, "n_frames", 1) != 1:
                    raise InferenceError("Vui lòng dùng ảnh tĩnh, không dùng ảnh động.")
                orientation = header.getexif().get(274, 1)
            # PNG getexif()/frame inspection may consume the underlying stream.
            # verify() must be called immediately after a FRESH open in Pillow.
            with Image.open(io.BytesIO(image_bytes)) as verification:
                verification.verify()
    except InferenceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InferenceError("Ảnh không được vượt quá 16 triệu pixel.", 413) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InferenceError("Không đọc được ảnh; tập tin có thể hỏng hoặc không đúng định dạng.") from exc
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise InferenceError("Thiếu thư viện xử lý ảnh; hãy cài requirements.txt.", 503) from exc
    try:
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8),
                               cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if decoded is None or decoded.shape[:2] != (height, width):
            raise InferenceError("Không giải mã được ảnh hợp lệ.")
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        # Equivalent to Pillow ImageOps.exif_transpose, without a second decode.
        if orientation == 2:
            rgb = rgb[:, ::-1]
        elif orientation == 3:
            rgb = rgb[::-1, ::-1]
        elif orientation == 4:
            rgb = rgb[::-1]
        elif orientation == 5:
            rgb = rgb.transpose(1, 0, 2)
        elif orientation == 6:
            rgb = np.rot90(rgb, 3)
        elif orientation == 7:
            rgb = rgb.transpose(1, 0, 2)[::-1, ::-1]
        elif orientation == 8:
            rgb = np.rot90(rgb, 1)
        height, width = rgb.shape[:2]
        coordinates = _validate_roi(roi, width, height)
        if coordinates is not None:
            x1, y1, x2, y2 = coordinates
            rgb = rgb[y1:y2 + 1, x1:x2 + 1]
        resized = cv2.resize(np.ascontiguousarray(rgb), (32, 32), interpolation=cv2.INTER_AREA)
        image = resized.astype(np.float32) / np.float32(255.0)
        stream = io.BytesIO()
        Image.fromarray(resized).save(stream, format="PNG")
    except InferenceError:
        raise
    except (cv2.error, ValueError, OSError) as exc:
        raise InferenceError("Không xử lý được ảnh; hãy thử một ảnh khác.") from exc
    return image, {"width": width, "height": height}, (
        "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii"))


def raw_features(image, method: str, hog_params=None):
    """Extract ONLY stateless raw pixels/HOG; pipeline owns scaling and PCA."""
    if method == "pixel":
        return image.reshape(1, 3072)
    try:
        from src.features import extract_hog
    except ImportError as exc:
        raise InferenceError("Thiếu thư viện HOG; hãy cài requirements.txt.", 503) from exc
    return extract_hog(image[None], **(hog_params or HOG_CONTRACT))


def _version_packages(bundle):
    metadata = bundle.get("metadata") or {}
    config = bundle.get("config") or metadata.get("config") or {}
    candidates = (bundle.get("versions"), metadata.get("versions"),
                  config.get("versions"), config.get("environment"))
    for version in candidates:
        if isinstance(version, dict):
            return version.get("packages", version)
    return {}


def _check_versions(bundle, method):
    from importlib.metadata import PackageNotFoundError, version
    recorded = _version_packages(bundle)
    # Pickle layout and learned transformations must not silently cross versions.
    relevant = ["scikit-learn", "numpy"]
    if method != "pixel":
        relevant += ["scikit-image", "opencv-python"]
    for package in relevant:
        expected = recorded.get(package)
        if not expected:
            continue  # Legacy bundles have no environment manifest.
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise _BundleError(f"Thiếu thư viện {package} để chạy mô hình.") from exc
        if str(expected) != installed:
            raise _BundleError(f"Phiên bản {package} không khớp mô hình; cần refit bằng môi trường hiện tại.")


def _bundle_status(bundle):
    metadata = bundle.get("metadata") or {}
    config = bundle.get("config") or metadata.get("config") or {}
    if config.get("mode") == "smoke" or metadata.get("status") == "smoke_only":
        return "smoke_only"
    status = bundle.get("status", metadata.get("status", "legacy"))
    if status not in {"legacy", "train_refit", "train_cv_selected", "smoke_only", "demo", "frozen"}:
        raise _BundleError("Trạng thái mô hình không được hỗ trợ hoặc chưa huấn luyện xong.")
    return status


def _validate_bundle(bundle):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(bundle, dict) or bundle.get("schema_version", 1) != 1:
        raise _BundleError("Bundle mô hình không đúng schema được hỗ trợ.")
    model = bundle.get("estimator")
    if not isinstance(model, Pipeline) or not model.steps:
        raise _BundleError("Mô hình phải là Pipeline đã lưu đầy đủ scaler/PCA.")
    classifier = model.steps[-1][1]
    if isinstance(classifier, KNeighborsClassifier):
        algorithm = "knn"
    elif isinstance(classifier, DecisionTreeClassifier):
        algorithm = "dt"
    else:
        raise _BundleError("Chỉ hỗ trợ bộ phân loại KNN và Decision Tree.")
    declared = bundle.get("algorithm", algorithm)
    if declared == "decision_tree":
        declared = "dt"
    if declared != algorithm:
        raise _BundleError("Loại thuật toán không khớp bộ phân loại đã lưu.")
    method = bundle.get("method")
    if method is None and all(key in bundle for key in (
            "algorithm", "params", "status", "image_contract", "hog_params", "raw_dimensions")):
        method = "hog"  # Expanded train_models.py bundles intentionally omit method.
    if method not in {"pixel", "hog", "hog_pca"}:
        raise _BundleError("Không xác định được biểu diễn pixel/HOG/PCA của mô hình.")
    _check_versions(bundle, method)
    contract = bundle.get("image_contract")
    if contract is not None and contract != IMAGE_CONTRACT:
        raise _BundleError("Hợp đồng ảnh không tương thích RGB 32×32, ROI, chia 255.")
    hog_params = bundle.get("hog_params", HOG_CONTRACT)
    if not isinstance(hog_params, dict) or set(hog_params) != set(HOG_CONTRACT):
        raise _BundleError("Tham số HOG thiếu hoặc không được hỗ trợ.")
    try:
        hog_params = {**hog_params, "pixels_per_cell": tuple(hog_params["pixels_per_cell"]),
                      "cells_per_block": tuple(hog_params["cells_per_block"])}
    except TypeError as exc:
        raise _BundleError("Tham số HOG không hợp lệ.") from exc
    if hog_params != HOG_CONTRACT:
        raise _BundleError("Mô hình yêu cầu hợp đồng HOG khác HOG324 hiện tại.")
    expected = 3072 if method == "pixel" else 324
    if (getattr(model, "n_features_in_", None) != expected
            or bundle.get("raw_dimensions", expected) != expected):
        raise _BundleError("Số chiều đầu vào thô không khớp pixel3072/HOG324.")
    transforms = [step for _, step in model.steps[:-1]]
    if method == "pixel":
        valid = not transforms
    elif method == "hog":
        valid = len(transforms) == 1 and isinstance(transforms[0], StandardScaler)
    else:
        valid = (len(transforms) in (2, 3) and isinstance(transforms[0], StandardScaler)
                 and isinstance(transforms[1], PCA))
        if valid and len(transforms) == 3:
            cast = transforms[2]
            function = getattr(cast, "func", None)
            valid = (isinstance(cast, FunctionTransformer)
                     and (getattr(function, "__module__", ""), getattr(function, "__name__", ""))
                     in {("src.models", "_float32"), ("src.decision_tree", "float32")})
    if not valid:
        raise _BundleError("Pipeline không đúng thứ tự scaler/PCA của biểu diễn đã khai báo.")
    dimensions = expected
    for transform in transforms:
        if isinstance(transform, (StandardScaler, PCA)):
            check_is_fitted(transform)
            if transform.n_features_in_ != dimensions:
                raise _BundleError("Scaler/PCA không khớp số chiều của pipeline.")
        if isinstance(transform, StandardScaler):
            if not transform.with_mean or not transform.with_std:
                raise _BundleError("Scaler không đúng hợp đồng StandardScaler của dự án.")
        elif isinstance(transform, PCA):
            dimensions = int(transform.n_components_)
    check_is_fitted(classifier)
    if classifier.n_features_in_ != dimensions:
        raise _BundleError("Bộ phân loại không khớp đầu ra scaler/PCA.")
    declared_params = bundle.get("params")
    if declared_params is not None:
        fitted_params = classifier.get_params(deep=False)
        if (not isinstance(declared_params, dict)
                or any(key not in fitted_params or fitted_params[key] != value
                       for key, value in declared_params.items())):
            raise _BundleError("Tham số khai báo không khớp bộ phân loại đã lưu.")
    classes = np.asarray(classifier.classes_)
    if (classes.ndim != 1 or not len(classes) or not np.issubdtype(classes.dtype, np.integer)
            or len(np.unique(classes)) != len(classes) or np.any(classes < 0) or np.any(classes > 42)):
        raise _BundleError("Nhãn mô hình phải là các ClassId nguyên, khác nhau, trong 0..42.")
    if getattr(classifier, "n_outputs_", 1) != 1:
        raise _BundleError("Không hỗ trợ mô hình đa đầu ra.")
    if "labels" in bundle:
        labels = np.asarray(bundle["labels"])
        if (labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer)
                or len(np.unique(labels)) != len(labels) or np.any(labels < 0) or np.any(labels > 42)
                or not np.isin(classes, labels).all()):
            raise _BundleError("Danh sách nhãn lưu trong bundle không khớp mô hình.")
    if "class_names" in bundle and bundle["class_names"] != [row[1] for row in _CLASS_DATA]:
        raise _BundleError("Tên lớp lưu trong bundle không khớp danh mục GTSRB.")
    if isinstance(classifier, KNeighborsClassifier):
        training_samples = int(classifier.n_samples_fit_)
        if classifier.n_neighbors > training_samples or classifier._fit_X.shape[1] != dimensions:
            raise _BundleError("KNN không đủ mẫu hoặc sai số chiều dữ liệu huấn luyện.")
    else:
        training_samples = int(classifier.tree_.n_node_samples[0])
    status = _bundle_status(bundle)
    messages = []
    if status in {"demo", "smoke_only"}:
        messages.append("Mô hình demo/smoke huấn luyện trên dữ liệu giới hạn, không phải mô hình đầy đủ.")
    if len(classes) != 43:
        messages.append(f"Mô hình chỉ nhận diện {len(classes)}/43 lớp GTSRB.")
    if contract is None:
        messages.append("Bundle lịch sử thiếu hợp đồng ảnh; sử dụng hợp đồng gốc RGB32/ROI/chia255.")
    if not _version_packages(bundle):
        messages.append("Bundle lịch sử không ghi phiên bản thư viện; nên refit để tái lập môi trường.")
    return model, dict(algorithm=algorithm, method=method, status=status,
                       n_classes=len(classes), training_samples=training_samples), hog_params, messages


class ModelRegistry:
    """Validate local bundles, retaining at most two fitted pipelines in memory.

    Catalog metadata stays cached after estimator eviction. An explicit prediction
    loads only its requested model; a first catalog scan validates bundles one by
    one without retaining every potentially large KNN training matrix. Broken
    requested IDs never silently fall back to a different model.
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self._cache = {}
        self._loaded_order = []
        self._max_loaded = 2
        self._lock = RLock()
        self._prediction_lock = RLock()

    def _retain(self, model_id):
        if model_id in self._loaded_order:
            self._loaded_order.remove(model_id)
        self._loaded_order.append(model_id)
        while len(self._loaded_order) > self._max_loaded:
            evicted = self._loaded_order.pop(0)
            if evicted in self._cache:
                self._cache[evicted]["estimator"] = None

    def _paths(self):
        if not self.model_dir.is_dir():
            return {}
        return {path.relative_to(self.model_dir).with_suffix("").as_posix(): path
                for path in sorted(self.model_dir.rglob("*.joblib")) if path.is_file()}

    def _load(self, model_id, path, *, for_prediction=False):
        with self._lock:
            fingerprint = (path.stat().st_mtime_ns, path.stat().st_size)
            cached = self._cache.get(model_id)
            if (cached and cached["fingerprint"] == fingerprint
                    and (not for_prediction or cached["estimator"] is not None
                         or not cached["record"]["ready"])):
                if cached["estimator"] is not None:
                    self._retain(model_id)
                return cached
            record = dict(id=model_id, name=path.stem, algorithm=None, method=None,
                          status="unavailable", ready=False, error=None,
                          n_classes=None, training_samples=None)
            entry = dict(fingerprint=fingerprint, record=record, warnings=[], estimator=None)
            try:
                if not path.resolve().is_relative_to(self.model_dir):
                    raise _BundleError("Không nạp liên kết mô hình nằm ngoài thư mục tin cậy.")
                import joblib
                from sklearn.exceptions import InconsistentVersionWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    bundle = joblib.load(path)
                estimator, information, hog_params, messages = _validate_bundle(bundle)
                record.update(information, ready=True)
                record["name"] = f"{information['algorithm'].upper()} · {information['method'].upper()} · {model_id}"
                entry.update(estimator=estimator, hog_params=hog_params, warnings=messages)
            except _BundleError as exc:
                record["error"] = str(exc)
            except ImportError:
                record["error"] = "Thiếu thư viện để nạp mô hình; hãy cài requirements.txt."
            except Exception as exc:
                if type(exc).__name__ == "InconsistentVersionWarning":
                    record["error"] = "Phiên bản scikit-learn không khớp mô hình; hãy refit trong môi trường hiện tại."
                else:
                    record["error"] = "Mô hình hỏng, chưa fit hoặc không tương thích; hãy kiểm tra/refit bundle tin cậy."
            self._cache[model_id] = entry
            if model_id in self._loaded_order:
                self._loaded_order.remove(model_id)
            if entry["estimator"] is not None:
                if for_prediction or len(self._loaded_order) < self._max_loaded:
                    self._retain(model_id)
                else:
                    # Keep the validated lightweight record, not every KNN matrix.
                    entry["estimator"] = None
            elif model_id in self._loaded_order:
                self._loaded_order.remove(model_id)
            return entry

    def list_models(self) -> list[dict]:
        with self._lock:
            try:
                paths = self._paths()
            except OSError:
                return []
            self._cache = {key: value for key, value in self._cache.items() if key in paths}
            self._loaded_order = [key for key in self._loaded_order if key in self._cache
                                  and self._cache[key]["estimator"] is not None]
            records = []
            for model_id, path in paths.items():
                try:
                    entry = self._load(model_id, path)
                    records.append(dict(entry["record"]))
                except OSError:
                    records.append(dict(id=model_id, name=path.stem, algorithm=None,
                                        method=None, status="unavailable", ready=False,
                                        error="Không đọc được tập tin mô hình.",
                                        n_classes=None, training_samples=None))
            return records

    def _select(self, model_id):
        if model_id is not None:
            if not isinstance(model_id, str) or not model_id:
                raise InferenceError("Mã mô hình không hợp lệ.")
            try:
                path = self._paths().get(model_id)
                if path is None:
                    raise InferenceError("Không tìm thấy mã mô hình đã chọn; hãy tải lại danh sách.")
                entry = self._load(model_id, path, for_prediction=True)
            except OSError as error:
                raise InferenceError("Không đọc được tập tin mô hình đã chọn.", 503) from error
            if not entry["record"]["ready"]:
                raise InferenceError(entry["record"]["error"] or "Mô hình đã chọn chưa sẵn sàng.", 503)
            return entry
        records = self.list_models()
        ready = [record for record in records if record["ready"]]
        if not ready:
            raise InferenceError("Chưa có mô hình sẵn sàng. Chạy python main.py train từ Train.csv rồi thử lại.", 503)
        def rank(record):
            is_demo = record["status"] in {"demo", "smoke_only"} or record["n_classes"] != 43
            return (is_demo, {"hog": 0, "hog_pca": 1, "pixel": 2}[record["method"]],
                    record["algorithm"] != "knn",
                    -self._cache[record["id"]]["fingerprint"][0], record["id"])
        return self._select(min(ready, key=rank)["id"])

    def default_model_id(self) -> str | None:
        """Share the exact inference preference with the web's selected option."""
        with self._lock:
            try:
                return self._select(None)["record"]["id"]
            except InferenceError as error:
                if error.status_code == 503:
                    return None
                raise

    def predict(self, image_bytes: bytes, model_id: str | None = None,
                roi: list[int] | None = None, top_k: int = 5) -> dict:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 43:
            raise InferenceError("top_k phải là số nguyên trong 1..43.")
        started = perf_counter()
        image, dimensions, preview = prepare_image(image_bytes, roi)
        with self._lock:
            # Keep a strong local estimator reference if another request evicts it.
            entry = dict(self._select(model_id))
        record = entry["record"]
        try:
            import numpy as np
            from threadpoolctl import threadpool_limits
            features = raw_features(image, record["method"], entry["hog_params"])
            estimator = entry["estimator"]
            with self._prediction_lock, threadpool_limits(limits=2):
                predicted = estimator.predict(features)
                probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
            classes = np.asarray(estimator.classes_)
            if (predicted.shape != (1,) or probabilities.shape != (1, len(classes))
                    or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                    or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1.0)):
                raise ValueError("Invalid classifier output")
            class_id = int(predicted[0])
            where = np.flatnonzero(classes == class_id)
            if len(where) != 1:
                raise ValueError("Prediction absent from classes_")
            predicted_column = int(where[0])
            # Always preserve estimator.predict's tie-handling, not JSON/index order.
            ordering = sorted(range(len(classes)), key=lambda i: (-probabilities[0, i], i))
            ordering.remove(predicted_column)
            ordering.insert(0, predicted_column)
            def result(column):
                identifier = int(classes[column])
                vi, en, _ = _CLASS_DATA[identifier]
                return dict(class_id=identifier, label=vi, label_en=en,
                            score=float(probabilities[0, column]))
            prediction = result(predicted_column)
            candidates = [result(column) for column in ordering[:top_k]]
        except InferenceError:
            raise
        except ImportError as exc:
            raise InferenceError("Thiếu thư viện suy luận; hãy cài requirements.txt.", 503) from exc
        except Exception as exc:
            raise InferenceError("Mô hình không suy luận được; hãy kiểm tra hợp đồng hoặc refit mô hình đã chọn.", 503) from exc
        messages = list(entry["warnings"])
        messages.append("Điểm là predict_proba chưa hiệu chuẩn, không phải độ tin cậy hay độ chính xác đã kiểm định.")
        if roi is None:
            messages.append("Đang dùng toàn ảnh. Nên dùng ảnh chỉ có một biển báo hoặc chọn ROI bao quanh biển.")
        return dict(model={key: record[key] for key in ("id", "name", "algorithm", "method")},
                    prediction=prediction, top_k=candidates,
                    elapsed_ms=round((perf_counter() - started) * 1000, 2),
                    image=dimensions, processed_image=preview, warnings=messages)
