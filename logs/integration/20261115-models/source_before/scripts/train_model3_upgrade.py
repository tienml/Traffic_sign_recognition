"""Additive person-3 upgrade: fixed HOG, frozen track CV, never official test.

Run from the repository root with data/.venv-model3/Scripts/python.exe.
Existing features, models, selections, test claims and reports are not overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.train_knn import digest, metadata, resolve_dataset_dir, save_folds
from src.config import IMAGE_SIZE, RANDOM_STATE, USE_ROI
from src.decision_tree import frozen_folds, smoke_from_frozen
from src.features import HOG_DEFAULTS, extract_hog
from src.models import (expanded_knn_candidates, regularized_dt_candidates,
                        evaluate_dt_cv, evaluate_knn_cv, refit_model, select_best)
from src.preprocessing import ROI_COLUMNS, load_image

IMAGE_CONTRACT = {"size": [32, 32], "color": "RGB", "roi": True,
                  "scale": "divide255", "augmentation": False}
PACKAGES = ("numpy", "pandas", "scikit-learn", "scikit-image", "opencv-python",
            "scipy", "joblib", "threadpoolctl", "matplotlib", "nbformat", "nbclient")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


def new_run(mode, run_id=None, project_root=ROOT):
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{mode}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id must contain only letters, numbers, _ and -")
    output = project_root / "report/model3_upgrade" / run_id
    logs = project_root / "logs/model3_upgrade" / run_id
    models = project_root / "data/models/model3_upgrade" / run_id
    if any(p.exists() for p in (output, logs, models)):
        raise FileExistsError("Run already exists; choose a new run-id")
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    if mode != "preflight":
        models.mkdir(parents=True)
    return run_id, output, logs, models


def validate_manifest(root, manifest):
    """Only read Train.csv and the historical train-fold provenance."""
    frame = metadata(root, "Train")
    source = manifest.parent / "config.json"
    prior = json.loads(source.read_text(encoding="utf-8"))
    if (len(frame) != 39209 or prior.get("mode") != "tune"
            or prior.get("seed") != 42
            or prior.get("train_sha256") != digest(root / "Train.csv")):
        raise ValueError("Expected the full 39209-row Train.csv of the frozen KNN run")
    groups, folds = frozen_folds(frame, manifest)
    return frame, groups, folds


def extract_train(root, frame, *, batch_size=256, progress=print):
    """Stream only train images through unchanged public person-1/2 functions.

    Image arrays are bounded by batch_size, rather than retaining 459 MiB of RGB.
    The returned SHA256 fingerprints each image's bytes (not only CSV metadata).
    Diagnostic brightness/contrast are measured on the processed ROI, unlike the
    historical EDA whole-image brightness sample; neither is used as a feature.
    """
    raw = np.empty((len(frame), 324), dtype=np.float32)
    quality = []
    started = time.perf_counter()
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start:start + batch_size]
        images = []
        for row in part.to_dict("records"):
            path = (root / row["Path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image path escapes dataset")
            image_hash = digest(path)
            image = load_image(path, [row[k] for k in ROI_COLUMNS], use_roi=True)
            images.append(image)
            gray = (image[..., 0] * .299 + image[..., 1] * .587
                    + image[..., 2] * .114)
            quality.append({"CSVRow": int(row["CSVRow"]), "Path": row["Path"],
                            "ClassId": int(row["ClassId"]),
                            "Width": int(row["Width"]), "Height": int(row["Height"]),
                            "RoiWidth": int(row["Roi.X2"] - row["Roi.X1"] + 1),
                            "RoiHeight": int(row["Roi.Y2"] - row["Roi.Y1"] + 1),
                            "roi32_brightness": float(gray.mean()),
                            "roi32_contrast": float(gray.std()),
                            "image_sha256": image_hash})
        batch = np.asarray(images, dtype=np.float32)
        raw[start:start + len(batch)] = extract_hog(batch, **HOG_DEFAULTS)
        if start == 0 or start // batch_size % 20 == 0 or start + len(batch) == len(frame):
            progress(f"HOG train {start + len(batch)}/{len(frame)}; "
                     f"elapsed={time.perf_counter() - started:.1f}s")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite raw HOG")
    return raw, pd.DataFrame(quality), time.perf_counter() - started


def serializable_candidates(rows):
    return [{**row, "params": json.dumps(row["params"], sort_keys=True)} for row in rows]


def train_run(args, root, frame, groups, folds, output, models, config):
    smoke = args.mode == "smoke"
    if smoke:
        frame, groups, folds = smoke_from_frozen(frame, groups, folds)
    save_folds(output, frame, groups, folds)
    raw, quality, extraction_seconds = extract_train(root, frame, batch_size=args.batch_size)
    write_csv(output / "train_image_audit.csv", quality)
    feature_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    write_json(output / "feature_audit.json", {
        "rows": len(raw), "dimensions": raw.shape[1], "dtype": str(raw.dtype),
        "raw_hog_sha256": feature_hash, "extraction_seconds": extraction_seconds,
        "hog_params": HOG_DEFAULTS, "image_contract": IMAGE_CONTRACT,
        "diagnostics_are_model_features": False,
        "diagnostic_definition": "RGB-weighted gray mean/std on resized ROI32 [0,1]",
        "official_test_read": False})
    summaries = []
    for algorithm, evaluate in (("knn", evaluate_knn_cv), ("dt", evaluate_dt_cv)):
        started = time.perf_counter()
        print(f"START {algorithm}: {len(raw)} images, 3 frozen track folds", flush=True)
        result = evaluate(raw, frame.ClassId.to_numpy(), groups, folds,
                          batch_size=args.batch_size, smoke=smoke,
                          progress=lambda message: print(message, flush=True))
        elapsed = time.perf_counter() - started
        winner = select_best(result)
        baseline_id = f"{algorithm}_baseline"
        baseline = next(r for r in result["candidates"] if r["candidate_id"] == baseline_id)
        write_csv(output / f"{algorithm}_candidates.csv",
                  pd.DataFrame(serializable_candidates(result["candidates"])))
        write_csv(output / f"{algorithm}_fold_scores.csv", pd.DataFrame(result["fold_scores"]))
        write_json(output / f"{algorithm}_transforms.json", result["transforms"])
        assignment = np.full(len(frame), -1, dtype=np.int64)
        for fold, (_, valid) in enumerate(folds):
            assignment[valid] = fold
        for name, item in (("baseline", baseline), ("selected", winner)):
            predictions = frame[["CSVRow", "Path", "ClassId"]].copy()
            predictions["Group"] = groups
            predictions["ValidationFold"] = assignment
            predictions["PredictedClassId"] = result["oof"][item["candidate_id"]]
            predictions["CandidateId"] = item["candidate_id"]
            write_csv(output / f"oof_{algorithm}_{name}.csv", predictions)
        started = time.perf_counter()
        fitted = refit_model(raw, frame.ClassId.to_numpy(), algorithm, winner["params"])
        refit_seconds = time.perf_counter() - started
        model_path = models / f"hog_{algorithm}.joblib"
        with model_path.open("xb") as stream:
            joblib.dump({"schema_version": 1, "algorithm": algorithm,
                         "estimator": fitted, "params": winner["params"],
                         "image_contract": IMAGE_CONTRACT, "hog_params": HOG_DEFAULTS,
                         "raw_dimensions": 324, "config": config,
                         "status": "smoke_only" if smoke else "train_cv_selected",
                         "official_test_evaluated": False}, stream, compress=3)
        summary = {"algorithm": algorithm, "baseline": baseline, "selected": winner,
                   "delta_cv_f1_macro": winner["mean_f1_macro"] - baseline["mean_f1_macro"],
                   "cv_wall_seconds": elapsed, "refit_seconds": refit_seconds,
                   "model_file": str(model_path.relative_to(ROOT)).replace(chr(92), "/"),
                   "model_sha256": digest(model_path), "model_bytes": model_path.stat().st_size,
                   "n_train": len(frame), "n_tracks": len(np.unique(groups)),
                   "candidate_count": len(result["candidates"])}
        write_json(output / f"{algorithm}_selection.json", summary)
        summaries.append(summary)
        print(f"DONE {algorithm}: baseline F1={baseline['mean_f1_macro']:.6f}, "
              f"selected F1={winner['mean_f1_macro']:.6f}, "
              f"delta={summary['delta_cv_f1_macro']:+.6f}, CV={elapsed:.1f}s", flush=True)
    write_json(output / "selection.json", {
        "schema_version": 1, "run_id": config["run_id"], "mode": args.mode,
        "status": "smoke_only" if smoke else "completed_train_cv",
        "algorithms": summaries, "train_images": len(frame),
        "feature_audit_sha256": digest(output / "feature_audit.json"),
        "fold_manifest_sha256": digest(output / "fold_manifest.csv"),
        "official_test_read": False, "official_test_evaluated": False,
        "selection_caveat": "OOF of candidates selected on these same folds is not nested/unbiased evaluation.",
        "historical_test_caveat": "Earlier EDA and model reports explored official test; this upgrade does not read it."})


class Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream
    def write(self, text):
        self.console.write(text)
        self.stream.write(text)
        self.stream.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "tune"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--fold-manifest", type=Path,
                        default=ROOT / "report/knn/full-knn-01/fold_manifest.csv")
    parser.add_argument("--run-id")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if IMAGE_SIZE != (32, 32) or USE_ROI is not True or RANDOM_STATE != 42:
        raise ValueError("The existing image/seed contract must remain unchanged")
    root = resolve_dataset_dir(args.dataset_dir)
    frame, groups, folds = validate_manifest(root, args.fold_manifest.resolve())
    run_id, output, logs, models = new_run(args.mode, args.run_id)
    config = {"run_id": run_id, "mode": args.mode, "dataset_root": str(root),
              "train_sha256": digest(root / "Train.csv"),
              "source_manifest_sha256": digest(args.fold_manifest),
              "environment": environment(), "seed": 42, "n_jobs": 1,
              "numeric_threads": 2, "batch_size": args.batch_size,
              "hog_params": HOG_DEFAULTS, "image_contract": IMAGE_CONTRACT,
              "representations": ["hog"], "person2_unchanged": True,
              "knn_candidates": expanded_knn_candidates(args.mode == "smoke"),
              "dt_candidates": regularized_dt_candidates(args.mode == "smoke"),
              "source_sha256": {p: digest(ROOT / p) for p in (
                  "scripts/train_model3_upgrade.py",
                  "src/models.py", "src/decision_tree.py", "src/features.py",
                  "src/preprocessing.py", "src/config.py")},
              "official_test_read": False}
    write_json(output / "config.json", config)
    write_json(logs / "environment.json", environment())
    handler = logging.FileHandler(logs / "preprocessing.log", mode="x", encoding="utf-8")
    handler.set_name("preprocessing.file")
    logger = logging.getLogger("src.preprocessing")
    logger.addHandler(handler)
    try:
        with (logs / "console.log").open("x", encoding="utf-8") as console:
            with redirect_stdout(Tee(sys.stdout, console)), redirect_stderr(Tee(sys.stderr, console)):
                print(f"Run: {run_id}; mode={args.mode}; official test access DISABLED", flush=True)
                try:
                    with threadpool_limits(limits=2):
                        if args.mode == "preflight":
                            save_folds(output, frame, groups, folds)
                            write_json(output / "preflight.json", {
                                "train_images": len(frame), "tracks": len(np.unique(groups)),
                                "folds": 3, "raw_features": 324,
                                "knn_candidates": len(expanded_knn_candidates()),
                                "dt_candidates": len(regularized_dt_candidates()), "official_test_read": False})
                            print("Preflight passed: train/frozen folds match; no images decoded.")
                        else:
                            train_run(args, root, frame, groups, folds, output, models, config)
                    print("Run completed.", flush=True)
                except BaseException as error:
                    traceback.print_exc()
                    write_json(output / "failure.json", {"type": type(error).__name__,
                               "message": str(error), "official_test_read": False})
                    raise
    finally:
        logger.removeHandler(handler)
        handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
