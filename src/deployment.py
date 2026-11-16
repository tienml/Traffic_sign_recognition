"""Explicit deployment refit, separate from the group's historical experiments.

Only Train.csv is read. There is no CV, tuning, official-test inference, mixed
image cache or automatic training when the web server starts.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import tempfile
from time import perf_counter
import uuid

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pixel", "hog", "hog_pca")
ALGORITHMS = ("knn", "dt")


def resolve_dataset_dir(dataset_dir=None) -> Path:
    """Resolve only known locations; never scan archives or require Test.csv."""
    if dataset_dir is not None:
        supplied = Path(dataset_dir).expanduser().resolve()
        choices = [candidate for candidate in (supplied, supplied / "archive")
                   if (candidate / "Train.csv").is_file()]
    else:
        choices = [candidate.resolve() for candidate in (ROOT / "data/archive", ROOT / "data")
                   if (candidate / "Train.csv").is_file()]
    if len(choices) != 1:
        raise ValueError("Không tìm thấy Train.csv hoặc nguồn bị trùng; chỉ rõ --dataset-dir.")
    return choices[0]


def environment_versions() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scikit-image", "opencv-python",
                 "scipy", "joblib", "threadpoolctl", "Pillow"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def _integer(value, description):
    try:
        number = float(value)
        if not math.isfinite(number) or int(number) != number:
            raise ValueError
        return int(number)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Train.csv chứa {description} không phải số nguyên hữu hạn.") from exc


def _train_rows(root: Path, max_per_class=None):
    """Keep only small metadata, not the image dataset; preserve CSV row order."""
    rows, seen = [], set()
    counts = [0] * 43
    required = {"Path", "ClassId", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"}
    with (root / "Train.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Train.csv thiếu cột ảnh, nhãn, kích thước hoặc ROI.")
        for number, row in enumerate(reader, start=2):
            label = _integer(row.get("ClassId"), "ClassId")
            if not 0 <= label <= 42:
                raise ValueError("ClassId phải thuộc 0..42.")
            relative = PurePosixPath(str(row.get("Path") or "").replace("\\", "/"))
            if (relative.is_absolute() or not relative.parts or relative.parts[0] != "Train"
                    or any(part in {"..", "."} or ":" in part for part in relative.parts)):
                raise ValueError(f"Train.csv dòng {number}: chỉ được đọc ảnh trong Train/.")
            path = (root / relative).resolve()
            if not path.is_relative_to(root / "Train"):
                raise ValueError(f"Train.csv dòng {number}: ảnh nằm ngoài thư mục Train/.")
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f"Train.csv dòng {number}: đường dẫn ảnh bị trùng.")
            seen.add(key)
            width, height = (_integer(row.get(name), name) for name in ("Width", "Height"))
            roi = [_integer(row.get(name), name) for name in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")]
            if not (width > 0 and height > 0 and 0 <= roi[0] <= roi[2] < width
                    and 0 <= roi[1] <= roi[3] < height):
                raise ValueError(f"Train.csv dòng {number}: kích thước/ROI không hợp lệ.")
            if max_per_class is None or counts[label] < max_per_class:
                rows.append((path, label, roi, (width, height)))
                counts[label] += 1
    if not rows:
        raise ValueError("Train.csv không có ảnh huấn luyện.")
    if max_per_class is None and any(count == 0 for count in counts):
        raise ValueError("Refit đầy đủ yêu cầu đủ 43 lớp; dùng --max-per-class cho demo có giới hạn.")
    return rows, counts


def _defaults(method, algorithm):
    # Fixed fallbacks are explicit, not tuning. Prefer saved full selections below.
    if algorithm == "knn":
        return {"n_neighbors": 3 if method == "pixel" else 7,
                "weights": "distance", "metric": "manhattan" if method == "hog" else "euclidean"}
    return {"criterion": "entropy" if method != "hog_pca" else "gini",
            "max_depth": None if method == "pixel" else 20, "min_samples_split": 5}


def selected_parameters(method: str, algorithm: str, project_root=None):
    """Reuse params, never reuse scores or claim a previous model's evaluation."""
    root = ROOT if project_root is None else Path(project_root)
    # These are curated selection locations, not an arbitrary filesystem sweep.
    locations = []
    if method == "hog":
        locations.append(root / "report" / algorithm / "hog-full-20261115/selection.json")
    locations.append(root / "report" / algorithm / f"full-{algorithm}-01/selection.json")
    for path in locations:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved.get("status") not in {"frozen", "completed_train_cv"}:
            continue
        params = None
        if method == "hog" and "algorithms" in saved:
            for item in saved["algorithms"]:
                if item.get("algorithm") == algorithm:
                    params = item.get("selected", {}).get("params")
                    break
        else:
            for item in saved.get("methods", []):
                if item.get("method") == method:
                    params = item.get("params")
                    break
        if isinstance(params, dict) and params:
            prefix = f"{algorithm}__"
            normalized = {key[len(prefix):] if key.startswith(prefix) else key: value
                          for key, value in params.items()}
            return normalized, path.relative_to(root).as_posix()
    return _defaults(method, algorithm), "fixed deployment defaults (no saved full selection)"


def _new_run(model_dir):
    directory = Path(model_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-refit-{uuid.uuid4().hex[:8]}"
    run = directory / run_id
    run.mkdir(exist_ok=False)
    return run


def train_models(*, dataset_dir=None, model_dir=None, features="hog", models=("knn", "dt"),
                 max_per_class=None, progress=print) -> dict:
    """One fixed-parameter fit per requested pipeline on streamed raw features.

    Raw features live in a temporary disk memmap (HOG ~51 MB, pixel ~482 MB
    for all 39209 rows), rather than storing the ~482 MB RGB image batch.
    sklearn's scaler/PCA/classifier may allocate the numeric arrays it requires;
    only one pipeline is fitted at once and numeric thread count is capped at 2.
    """
    if features not in (*METHODS, "all"):
        raise ValueError("--features phải là pixel, hog, hog_pca hoặc all.")
    models = tuple(dict.fromkeys(models))
    if not models or any(model not in ALGORITHMS for model in models):
        raise ValueError("--models chỉ nhận knn và dt.")
    if max_per_class is not None and (isinstance(max_per_class, bool)
            or not isinstance(max_per_class, int) or max_per_class < 1):
        raise ValueError("--max-per-class phải là số nguyên dương, chỉ dùng cho demo.")
    root = resolve_dataset_dir(dataset_dir)
    rows, counts = _train_rows(root, max_per_class)
    import cv2
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from src.decision_tree import dt_pipeline
    from src.features import extract_hog, HOG_DEFAULTS
    from src.inference import IMAGE_CONTRACT, _validate_bundle
    from src.models import knn_pipeline
    from src.preprocessing import crop_roi, read_rgb

    def report(message):
        if progress is not None:
            progress(message)

    methods = METHODS if features == "all" else (features,)
    planned, parameter_sources = {}, {}
    for method in methods:
        for algorithm in models:
            estimator = knn_pipeline(method) if algorithm == "knn" else dt_pipeline(method)
            params, origin = selected_parameters(method, algorithm)
            # Saved selections may only configure this classifier, never pipeline IO.
            allowed = set(estimator.steps[-1][1].get_params(deep=False)) - {"random_state", "n_jobs", "algorithm"}
            if not set(params).issubset(allowed):
                raise ValueError("Tham số trong selection không phù hợp bộ phân loại được yêu cầu.")
            if algorithm == "knn" and params.get("n_neighbors", 5) > len(rows):
                if max_per_class is None:
                    raise ValueError("Số mẫu train không đủ cho k đã chọn.")
                previous = params.get("n_neighbors", 5)
                params = {**params, "n_neighbors": len(rows)}
                origin += f"; demo: k capped from {previous} to {len(rows)}"
            estimator.set_params(**{f"{algorithm}__{key}": value for key, value in params.items()})
            planned[method, algorithm] = estimator
            parameter_sources[f"{method}_{algorithm}"] = {"params": params, "origin": origin}
    run = _new_run(model_dir or ROOT / "data/models")
    started = perf_counter()
    versions = environment_versions()
    manifest = {
        "schema_version": 1, "run_id": run.name,
        "status": "demo" if max_per_class is not None else "train_refit",
        "training_state": "in_progress", "training_samples": len(rows),
        "class_counts": counts, "max_per_class": max_per_class,
        "train_csv_sha256": hashlib.sha256((root / "Train.csv").read_bytes()).hexdigest(),
        "features": list(methods), "algorithms": list(models),
        "parameter_sources": parameter_sources, "versions": versions,
        "official_test_read": False, "official_test_evaluated": False,
        "cross_validation_performed": False, "metrics": None, "bundles": [],
    }
    manifest_path = run / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    report(f"Refit {manifest['status']}: {len(rows)} ảnh Train.csv, {sum(c > 0 for c in counts)}/43 lớp.")
    report("Không đọc Test.csv/cache; không tìm kiếm tham số/CV; luồng tính toán tối đa 2.")
    try:
        # Temp storage is outside the model registry, and deleted on success/failure.
        with tempfile.TemporaryDirectory(prefix="gtsrb-refit-") as temporary:
            groups = []
            if "pixel" in methods:
                groups.append(("pixel", ["pixel"]))
            hog_methods = [method for method in methods if method != "pixel"]
            if hog_methods:
                groups.append(("hog", hog_methods))
            for representation, group_methods in groups:
                dimensions = 3072 if representation == "pixel" else 324
                matrix = np.memmap(Path(temporary) / f"{representation}.f32", dtype=np.float32,
                                   mode="w+", shape=(len(rows), dimensions))
                try:
                    report(f"Trích {representation.upper()} thô → {dimensions} chiều, từng ảnh một.")
                    for index, (path, _, roi, declared_size) in enumerate(rows):
                        rgb = read_rgb(path)
                        height, width = rgb.shape[:2]
                        if (width, height) != declared_size:
                            raise ValueError(f"Ảnh Train thứ {index + 1}: kích thước thực không khớp Train.csv.")
                        if not (0 <= roi[0] <= roi[2] < width and 0 <= roi[1] <= roi[3] < height):
                            raise ValueError(f"Ảnh Train thứ {index + 1}: ROI nằm ngoài ảnh thực.")
                        # Same crop/AREA/float32 contract, but no silent ROI fallback.
                        image = cv2.resize(crop_roi(rgb, roi), (32, 32), interpolation=cv2.INTER_AREA)
                        image = image.astype(np.float32) / np.float32(255.0)
                        vector = image.reshape(-1) if representation == "pixel" else extract_hog(image[None], **HOG_DEFAULTS)[0]
                        if vector.shape != (dimensions,) or not np.isfinite(vector).all():
                            raise ValueError(f"Đặc trưng ảnh thứ {index + 1} không hợp lệ.")
                        matrix[index] = vector
                        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
                            report(f"  {representation}: {index + 1}/{len(rows)} ảnh")
                    matrix.flush()
                    for method in group_methods:
                        for algorithm in models:
                            name = f"{method}_{algorithm}"
                            source = parameter_sources[name]
                            report(f"Fit {name}: {source['params']} | nguồn: {source['origin']}")
                            estimator = planned.pop((method, algorithm))
                            fit_started = perf_counter()
                            with threadpool_limits(limits=2):
                                estimator.fit(matrix, labels)
                            fit_seconds = perf_counter() - fit_started
                            metadata = {
                                "status": manifest["status"], "training_samples": len(rows),
                                "class_counts": counts, "run_id": run.name,
                                "parameter_origin": source["origin"], "versions": versions,
                                "train_csv_sha256": manifest["train_csv_sha256"],
                                "official_test_read": False, "official_test_evaluated": False,
                                "cross_validation_performed": False, "metrics": None,
                            }
                            bundle = dict(schema_version=1, algorithm=algorithm, method=method,
                                          estimator=estimator, params=source["params"], metadata=metadata,
                                          status=manifest["status"], raw_dimensions=dimensions,
                                          image_contract=dict(IMAGE_CONTRACT), hog_params=dict(HOG_DEFAULTS),
                                          labels=[int(label) for label in estimator.classes_])
                            _validate_bundle(bundle)
                            # A .part file is not discoverable; publish completed bundles atomically.
                            target = run / f"{name}.joblib"
                            temporary_model = run / f"{name}.part"
                            with temporary_model.open("xb") as stream:
                                joblib.dump(bundle, stream, compress=3)
                            temporary_model.replace(target)
                            manifest["bundles"].append({"file": target.name, "method": method,
                                                        "algorithm": algorithm, "fit_seconds": round(fit_seconds, 3)})
                            report(f"Đã lưu {target.name} ({fit_seconds:.2f}s fit).")
                            del bundle, estimator
                finally:
                    # Explicit close is necessary before temp cleanup on Windows.
                    matrix._mmap.close()
                    del matrix
        manifest["training_state"] = "completed"
    except BaseException:
        manifest["training_state"] = "failed"
        raise
    finally:
        manifest["elapsed_seconds"] = round(perf_counter() - started, 3)
        # Only the freshly, exclusively-created run's own manifest is updated.
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    report(f"Hoàn tất: {run}. Không gán điểm thực nghiệm lịch sử cho các model refit này.")
    return {"run_dir": str(run), **manifest}
