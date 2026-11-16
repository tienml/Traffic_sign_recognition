"""Đặng Quang Tiến (member 4): complete evaluation from saved predictions only.

No estimator is loaded, fit or asked to predict. Historical official test and new
training OOF remain separate. Existing sources, claims and outputs are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.evaluation_figures import numeric_figures, error_gallery
from scripts.train_models import environment, Tee
from scripts.model_artifacts import load_model_runs
from src.evaluate import (validate_predictions, summarize_predictions,
                                  paired_comparison, group_error_summary, select_error_examples)
from src.preprocessing import CLASS_NAMES

METHODS = ("pixel", "hog", "hog_pca")
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def metadata(dataset, split):
    frame = pd.read_csv(dataset / f"{split}.csv")
    frame.insert(0, "CSVRow", np.arange(len(frame)))
    return frame


def historical_predictions(project, algorithm, test_metadata, sources):
    """Validate frozen CV choice, test claim/hash, and actual CSV alignment."""
    tune = project / ("report/knn/full-knn-01" if algorithm == "knn" else "report/dt/full-dt-01")
    test = project / ("report/knn/20261113T194939Z-test-adf82d0d" if algorithm == "knn" else "report/dt/test-dt-01")
    def load(path):
        sources.add(path)
        return json.loads(path.read_text(encoding="utf-8"))
    selection = load(tune / "selection.json")
    metrics = load(test / "test_metrics.json")
    claim = load(tune / "official_test_claim.json")
    if (selection.get("status") != "frozen" or selection["winner"] != "hog"
            or [m["method"] for m in selection["methods"]] != list(METHODS)
            or metrics["selection_sha256"] != digest(tune / "selection.json")
            or claim["selection_sha256"] != metrics["selection_sha256"]
            or claim["evaluation_run"] != test.name
            or metrics["winner_fixed_by_cv"] != selection["winner"]):
        raise ValueError("Historical frozen selection/claim mismatch")
    predictions, summary = {}, []
    for item in selection["methods"]:
        method = item["method"]
        path = test / f"predictions_{method}.csv"
        frame = validate_predictions(pd.read_csv(path), test_metadata,
                                     split="official_test", expected_size=12630)
        sources.add(path)
        result = summarize_predictions(frame)
        for key in METRICS:
            if not np.isclose(result["metrics"][key], metrics["methods"][method][key], atol=1e-12, rtol=0):
                raise ValueError(f"Historical metric mismatch {algorithm}/{method}/{key}")
        cv_path = tune / f"cv_{method}.csv"
        cv = pd.read_csv(cv_path)
        sources.add(cv_path)
        if (item["candidate_index"] != int(np.argmax(cv.mean_test_f1_macro.to_numpy()))
                or not np.isclose(cv.iloc[item["candidate_index"]].mean_test_f1_macro,
                                  item["cv_f1_macro"], atol=1e-12, rtol=0)):
            raise ValueError("Historical candidate differs from CV winner")
        selected_cv = cv.iloc[item["candidate_index"]]
        for key, value in item["params"].items():
            observed = selected_cv[f"param_{key}"]
            if not (pd.isna(observed) if value is None else observed == value):
                raise ValueError("Historical selected parameters differ from saved CV")
        predictions[method] = frame
        summary.append({"split": "historical_official_test", "algorithm": algorithm,
                        "method": method, "n": len(frame), **result["metrics"],
                        "cv_selected_method": selection["winner"],
                        "representation_is_cv_winner": method == selection["winner"],
                        "historical_cv_f1_mean": item["cv_f1_macro"],
                        "historical_cv_f1_std": item["cv_f1_std"],
                        "params": json.dumps(item["params"], sort_keys=True),
                        "prediction_source": str(path.relative_to(project)).replace(chr(92), "/")})
    if selection["winner"] != max(selection["methods"], key=lambda m: m["cv_f1_macro"])["method"]:
        raise ValueError("Historical representation winner differs from CV")
    return predictions, summary, metrics


def duplicate_audit(dataset, train_audit, test_metadata):
    """Exact byte duplicates only; not a near-duplicate detector or split repair."""
    train = pd.read_csv(train_audit)
    by_hash = {}
    for row in train.itertuples(index=False):
        by_hash.setdefault(row.image_sha256, []).append(row)
    overlap = []
    test_hashes = []
    for row in test_metadata.itertuples(index=False):
        path = (dataset / row.Path).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError("Test image path escapes dataset")
        fingerprint = digest(path)
        test_hashes.append({"CSVRow": int(row.CSVRow), "Path": row.Path,
                            "ClassId": int(row.ClassId), "image_sha256": fingerprint})
        for match in by_hash.get(fingerprint, []):
            overlap.append({"train_path": match.Path, "train_class": int(match.ClassId),
                            "test_path": row.Path, "test_class": int(row.ClassId),
                            "test_csv_row": int(row.CSVRow), "image_sha256": fingerprint,
                            "equal_label": int(match.ClassId) == int(row.ClassId)})
    return pd.DataFrame(overlap, columns=["train_path", "train_class", "test_path", "test_class",
                         "test_csv_row", "image_sha256", "equal_label"]), pd.DataFrame(test_hashes)


def evaluate(dataset, knn_run, run_id, *, dt_run=None, project_root=ROOT):
    project = Path(project_root).resolve()
    dataset, knn_run = Path(dataset).resolve(), Path(knn_run).resolve()
    upgrade_run = knn_run  # Shared audit below; algorithm OOF inputs use runs mapping.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run-id")
    selection = json.loads((upgrade_run / "selection.json").read_text(encoding="utf-8"))
    if (selection.get("status") != "completed_train_cv" or selection.get("official_test_read") is not False
            or selection.get("official_test_evaluated") is not False):
        raise ValueError("Finish train-only full upgrade before evaluation")
    runs, selection, config = load_model_runs(knn_run, dt_run)
    if config["train_sha256"] != digest(dataset / "Train.csv"):
        raise ValueError("Upgrade Train.csv differs from current dataset")
    run = project / "report/evaluation" / run_id
    logs = project / "logs/evaluation" / run_id
    prefix = f"evaluation_{run_id}"
    if run.exists() or logs.exists():
        raise FileExistsError("Evaluation run exists; no overwrite")
    if list((project / "report/tables").glob(prefix + "_*")) or list((project / "report/figures").glob(prefix + "_*")):
        raise FileExistsError("Evaluation namespaced outputs already exist")
    run.mkdir(parents=True)
    logs.mkdir(parents=True)
    started = time.perf_counter()
    with (logs / "console.log").open("x", encoding="utf-8") as console:
        with redirect_stdout(Tee(sys.stdout, console)), redirect_stderr(Tee(sys.stderr, console)):
            try:
                return _evaluate(dataset, runs, selection, run, logs, project, prefix, started)
            except BaseException as error:
                traceback.print_exc()
                write_json(run / "failure.json", {"type": type(error).__name__, "message": str(error)})
                raise


def _evaluate(dataset, model_runs, selection, run, logs, project, prefix, started):
    print("Đặng Quang Tiến — member4 evaluation: saved predictions only", flush=True)
    upgrade_run = model_runs["knn"]
    protected_folders = [project / "src", project / "scripts", project / "report/knn",
                         project / "report/dt", *model_runs.values()]
    frozen_inputs = {p: digest(p) for folder in protected_folders for p in folder.rglob("*")
                     if p.is_file() and p.suffix in (".json", ".csv", ".py", ".ipynb", ".md", ".png")}
    sources = {dataset / "Train.csv", dataset / "Test.csv", dataset / "Meta.csv",
               upgrade_run / "selection.json", upgrade_run / "config.json",
               upgrade_run / "train_image_audit.csv", upgrade_run / "fold_manifest.csv"}
    sources.update(directory / name for directory in model_runs.values() for name in
                   ("selection.json", "config.json", "train_image_audit.csv", "fold_manifest.csv", "feature_audit.json"))
    implementation_sources = {project / "src/evaluate.py", Path(__file__).resolve(),
                              project / "scripts/evaluation_figures.py", project / "scripts/model_artifacts.py"}
    test_metadata = metadata(dataset, "Test")
    train_metadata = metadata(dataset, "Train")
    meta = pd.read_csv(dataset / "Meta.csv")[["ClassId", "ColorId", "ShapeId"]]
    diagnostic_meta = test_metadata.merge(meta, on="ClassId", how="left", validate="many_to_one", sort=False)
    if len(test_metadata) != 12630 or len(train_metadata) != 39209:
        raise ValueError("Expected full official dataset")
    summaries, class_tables, group_tables, error_tables, cm_tables = [], [], [], [], []
    predictions, confusions, examples = {}, {}, []
    for algorithm in ("knn", "dt"):
        frames, rows, test_metrics = historical_predictions(project, algorithm, test_metadata, sources)
        if test_metrics["test_sha256"] != digest(dataset / "Test.csv"):
            raise ValueError("Test.csv fingerprint differs from historical evaluation")
        summaries.extend(rows)
        predictions[algorithm] = frames
        for method, frame in frames.items():
            result = summarize_predictions(frame)
            classes = result["classes"].assign(algorithm=algorithm, method=method,
                                                split="historical_official_test")
            classes["class_name"] = classes.class_id.map(dict(enumerate(CLASS_NAMES)))
            train_counts = train_metadata.ClassId.value_counts()
            classes["train_images"] = classes.class_id.map(train_counts)
            class_tables.append(classes)
            group_tables.append(group_error_summary(frame, diagnostic_meta).assign(
                algorithm=algorithm, method=method, split="historical_official_test"))
            error_tables.append(result["errors"].assign(algorithm=algorithm, method=method,
                                                         split="historical_official_test"))
            cm = result["confusion"]
            confusions[f"{algorithm}_{method}"] = cm
            normalized = result["confusion_normalized"]
            cm_tables.append(pd.DataFrame([{"algorithm": algorithm, "method": method,
                "true_class": a, "predicted_class": b, "count": int(cm[a, b]),
                "row_normalized": float(normalized[a, b])} for a in range(43) for b in range(43)]))
        chosen = select_error_examples(frames["hog"], max_pairs=4, per_pair=2)
        chosen["algorithm"] = algorithm
        chosen["method"] = "hog"
        examples.append(chosen)
        print(f"Validated {algorithm}: 3 x 12630 saved predictions and historical metrics", flush=True)
    summary = pd.DataFrame(summaries)
    classes = pd.concat(class_tables, ignore_index=True)
    groups = pd.concat(group_tables, ignore_index=True)
    errors = pd.concat(error_tables, ignore_index=True)
    paired = []
    for method in METHODS:
        paired.append({"split": "historical_official_test", "method": method,
                       "name_first": "knn", "name_second": "dt",
                       **paired_comparison(predictions["knn"][method], predictions["dt"][method],
                                           name_first="knn", name_second="dt")})
    manifest = pd.read_csv(upgrade_run / "fold_manifest.csv")
    if (selection["fold_manifest_sha256"] != digest(upgrade_run / "fold_manifest.csv")
            or selection["feature_audit_sha256"] != digest(upgrade_run / "feature_audit.json")):
        raise ValueError("Upgrade audit/manifest fingerprint mismatch")
    sources.add(upgrade_run / "feature_audit.json")
    upgrade_summaries, upgrade_classes, upgrade_pairs = [], [], []
    for item in selection["algorithms"]:
        algorithm = item["algorithm"]
        oof_frames = {}
        for role in ("baseline", "selected"):
            path = model_runs[algorithm] / f"oof_{algorithm}_{role}.csv"
            sources.add(path)
            frame = validate_predictions(pd.read_csv(path), train_metadata,
                                         split="train_oof", manifest=manifest, expected_size=39209)
            if (frame.CandidateId.nunique() != 1
                    or frame.CandidateId.iloc[0] != item[role]["candidate_id"]):
                raise ValueError("OOF candidate differs from upgrade selection")
            fold_f1 = [summarize_predictions(frame.loc[frame.ValidationFold.eq(fold)]
                       .assign(CSVRow=lambda f: np.arange(len(f))))["metrics"]["f1_macro"]
                       for fold in range(3)]
            if not np.isclose(np.mean(fold_f1), item[role]["mean_f1_macro"], atol=1e-12, rtol=0):
                raise ValueError("OOF fold metrics differ from upgrade selection")
            oof_frames[role] = frame
            result = summarize_predictions(frame)
            upgrade_summaries.append({"split": "train_oof_same_fold_selection", "algorithm": algorithm,
                                      "role": role, "n": len(frame), **result["metrics"],
                                      "mean_fold_f1_macro": item[role]["mean_f1_macro"],
                                      "std_fold_f1_macro": item[role]["std_f1_macro"],
                                      "params": json.dumps(item[role]["params"], sort_keys=True)})
            upgrade_classes.append(result["classes"].assign(algorithm=algorithm, role=role,
                class_name=lambda f: f.class_id.map(dict(enumerate(CLASS_NAMES)))))
        upgrade_pairs.append({"split": "train_oof_same_fold_selection", "algorithm": algorithm,
            "name_first": "selected", "name_second": "baseline",
            **paired_comparison(oof_frames["selected"], oof_frames["baseline"],
                                name_first="selected", name_second="baseline")})
    train_audit = pd.read_csv(upgrade_run / "train_image_audit.csv")
    if (not train_audit[["CSVRow", "Path", "ClassId"]].equals(train_metadata[["CSVRow", "Path", "ClassId"]])
            or train_audit.Path.duplicated().any()
            or not train_audit.image_sha256.str.fullmatch(r"[0-9a-f]{64}").fillna(False).all()):
        raise ValueError("Train image audit is incomplete or differs from current Train.csv")
    overlap, test_hashes = duplicate_audit(dataset, upgrade_run / "train_image_audit.csv", test_metadata)
    print(f"Exact byte overlap: {len(overlap)} train/test pairs. Official split unchanged.", flush=True)
    tables = {
        "official_summary": summary, "official_per_class": classes,
        "official_confusions": pd.concat(cm_tables, ignore_index=True),
        "official_error_pairs": errors, "official_error_groups": groups,
        "official_paired": pd.DataFrame(paired), "error_examples": pd.concat(examples, ignore_index=True),
        "upgrade_oof_summary": pd.DataFrame(upgrade_summaries),
        "upgrade_oof_per_class": pd.concat(upgrade_classes, ignore_index=True),
        "upgrade_oof_paired": pd.DataFrame(upgrade_pairs), "exact_split_overlap": overlap,
        "class_mapping": pd.DataFrame({"class_id": range(43), "class_name": CLASS_NAMES}),
    }
    outputs = []
    for name, frame in tables.items():
        path = project / "report/tables" / f"{prefix}_{name}.csv"
        write_csv(path, frame)
        outputs.append(path)
    write_csv(run / "test_image_audit.csv", test_hashes)
    outputs.append(run / "test_image_audit.csv")
    figures = project / "report/figures"
    outputs.extend(numeric_figures(summary, classes, groups, pd.DataFrame(paired), confusions,
                                   figures, prefix))
    for algorithm in ("knn", "dt"):
        outputs.extend(error_gallery(tables["error_examples"].loc[tables["error_examples"].algorithm.eq(algorithm)],
                                     test_metadata, dataset, CLASS_NAMES, figures, prefix, algorithm))
    provenance = {"schema_version": 1, "run_id": run.name, "contributor": "Đặng Quang Tiến",
        "member_number": 4, "ai_assisted": True, "started_from_saved_predictions": True,
        "fit": False, "predict": False, "joblib_deserialized": False,
        "official_test_read": True, "new_official_test_inference": False,
        "historical_test_previously_explored": True, "upgrade_test_evaluated": False,
        "selection_rule_changed": False, "split_unchanged": True,
        "test_images": len(test_metadata), "train_images": len(train_metadata),
        "exact_duplicate_pairs": len(overlap),
        "model_runs": {a: p.relative_to(project).as_posix() for a, p in model_runs.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "sources_sha256": {str(p.relative_to(project)).replace(chr(92), "/"): digest(p) for p in sorted(sources)},
        "outputs_sha256": {str(p.relative_to(project)).replace(chr(92), "/"): digest(p) for p in outputs},
        "implementation_sha256": {str(p.relative_to(project)).replace(chr(92), "/"): digest(p)
                                   for p in sorted(implementation_sources)},
        "limitations": ["Test explored historically; no claim of an untouched holdout.",
            "OOF winner selected on same folds is not nested/unbiased evaluation.",
            "Grouped error rates are descriptive and not causal; no blur/lighting cause labels.",
            "Gallery uses deterministic SHA256(seed42|Path)-ranked examples of frequent pairs, not representative random sampling.",
            "Exact byte hashing does not detect all perceptual/near duplicates.",
            "Paired correctness counts are descriptive; no IID significance assumption.",
            "No new official-test score exists for upgraded models."]}
    modified = [str(p.relative_to(project)).replace(chr(92), "/")
                for p, before in frozen_inputs.items() if not p.is_file() or digest(p) != before]
    if modified:
        raise RuntimeError(f"Read-only inputs changed during evaluation: {modified}")
    write_json(logs / "preservation.json", {"protected_files": len(frozen_inputs),
        "changed": modified, "all_original_inputs_unchanged": True})
    write_json(run / "provenance.json", provenance)
    write_json(logs / "environment.json", environment())
    write_json(logs / "evaluation_result.json", {"status": "passed", "run_id": run.name,
                "metrics_recomputed": True, "all_six_align_current_test": True,
                "figures": sum(p.suffix == ".png" for p in outputs),
                "tables": len(tables), "contributor": "Đặng Quang Tiến", "member_number": 4,
                "fit_or_predict": False, "elapsed_seconds": time.perf_counter() - started})
    print(summary[["algorithm", "method", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False))
    print(f"Evaluation complete: {len(tables)} tables, {sum(p.suffix == '.png' for p in outputs)} figures; original files untouched.", flush=True)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--knn-run", type=Path, required=True)
    parser.add_argument("--dt-run", type=Path, required=True)
    parser.add_argument("--run-id", default="tien-evaluation-20261115")
    args = parser.parse_args(argv)
    evaluate(args.dataset_dir, args.knn_run, args.run_id, dt_run=args.dt_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
