"""Export additive person-3 train-CV results, without loading models or official test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from src.models import classification_metrics, track_groups, validate_folds

PALETTES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "grid": "#e1e0d9",
              "series": ("#2a78d6", "#eb6834")},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "grid": "#383835",
             "series": ("#3987e5", "#d95926")},
}
SPEED_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def validate_oof(predictions, audit, manifest):
    """Reject missing, reordered, duplicated or non-OOF rows before exporting."""
    keys = ["CSVRow", "Path", "ClassId"]
    if (len(predictions) != len(manifest) or predictions.Path.duplicated().any()
            or not predictions[keys].equals(manifest[keys])
            or not predictions[keys].equals(audit[keys])
            or not np.array_equal(predictions.Group.to_numpy(), manifest.Group.to_numpy())
            or not np.array_equal(predictions.ValidationFold.to_numpy(), manifest.ValidationFold.to_numpy())
            or not predictions.PredictedClassId.isin(range(43)).all()
            or not np.issubdtype(predictions.PredictedClassId.dtype, np.integer)
            or predictions.CandidateId.nunique() != 1):
        raise ValueError("OOF predictions do not align with audited train/fold rows")
    groups = track_groups(manifest)
    if not np.array_equal(groups, manifest.Group.to_numpy()):
        raise ValueError("Manifest groups differ from image track names")
    assignment = manifest.ValidationFold.to_numpy()
    if set(assignment) != {0, 1, 2}:
        raise ValueError("Expected three frozen validation folds")
    folds = [(np.flatnonzero(assignment != f), np.flatnonzero(assignment == f)) for f in range(3)]
    validate_folds(manifest.ClassId.to_numpy(), groups, folds)
    return predictions.merge(audit.drop(columns="ClassId"), on=["CSVRow", "Path"],
                             validate="one_to_one", sort=False)


def diagnostic_tables(joined, algorithm, role, manifest):
    """Descriptive OOF diagnostics, not a causal test or additional selection rule."""
    target = joined.ClassId.to_numpy()
    predicted = joined.PredictedClassId.to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=list(range(43)), zero_division=0)
    train_counts = manifest.ClassId.value_counts()
    track_counts = manifest.groupby("ClassId").Group.nunique()
    class_rows = [{"algorithm": algorithm, "role": role, "class_id": c,
                   "train_images": int(train_counts.get(c, 0)),
                   "train_tracks": int(track_counts.get(c, 0)), "oof_support": int(support[c]),
                   "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])}
                  for c in range(43)]
    group_rows = []
    group_definitions = {
        "raw_min_dimension": np.where(joined[["Width", "Height"]].min(axis=1) < 32, "lt32", "ge32"),
        "roi_min_dimension": np.where(joined[["RoiWidth", "RoiHeight"]].min(axis=1) < 32, "lt32", "ge32"),
        "sign_family": np.where(joined.ClassId.isin(SPEED_CLASSES), "speed_limit", "other"),
    }
    # Quantile thresholds are fitted on each fold's TRAIN side, never its validation.
    thresholds = []
    for column in ("roi32_brightness", "roi32_contrast"):
        group = np.empty(len(joined), dtype=object)
        assignment = joined.ValidationFold.to_numpy()
        for fold in range(3):
            train_values = joined.loc[assignment != fold, column]
            low, high = train_values.quantile([.25, .75]).to_numpy()
            values = joined.loc[assignment == fold, column].to_numpy()
            group[assignment == fold] = np.where(values <= low, "low_train_quartile",
                np.where(values >= high, "high_train_quartile", "middle"))
            thresholds.append({"algorithm": algorithm, "role": role, "fold": fold,
                               "measurement": column, "q25_fold_train": float(low),
                               "q75_fold_train": float(high)})
        group_definitions[column] = group
    for dimension, values in group_definitions.items():
        for value in sorted(set(values)):
            mask = values == value
            n = int(np.count_nonzero(mask))
            group_rows.append({"algorithm": algorithm, "role": role,
                               "dimension": dimension, "group": value, "n": n,
                               "errors": int(np.count_nonzero(target[mask] != predicted[mask])),
                               "error_rate": float(np.mean(target[mask] != predicted[mask]))})
    cm = confusion_matrix(target, predicted, labels=list(range(43)))
    pairs = [{"algorithm": algorithm, "role": role, "true_class": i, "predicted_class": j,
              "count": int(cm[i, j]), "within_speed_classes": i in SPEED_CLASSES and j in SPEED_CLASSES}
             for i in range(43) for j in range(43) if i != j and cm[i, j] > 0]
    return class_rows, group_rows, pairs, thresholds


def error_pair_frame(rows):
    columns = ["algorithm", "role", "true_class", "predicted_class", "count", "within_speed_classes"]
    return pd.DataFrame(rows, columns=columns).sort_values("count", ascending=False)


def plot_outputs(summary, candidates, groups, figures, prefix, theme):
    smoke = len(candidates) == 2
    title_prefix = "SMOKE ONLY — " if smoke else ""
    tokens = PALETTES[theme]
    colors = tokens["series"]
    settings = {"figure.facecolor": tokens["surface"], "axes.facecolor": tokens["surface"],
                "text.color": tokens["ink"], "axes.labelcolor": tokens["ink"],
                "xtick.color": tokens["ink"], "ytick.color": tokens["ink"],
                "axes.edgecolor": tokens["grid"], "grid.color": tokens["grid"],
                "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                "axes.spines.right": False, "savefig.facecolor": tokens["surface"]}
    suffix = "" if theme == "light" else "_dark"
    paths = []
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
        for ax, algorithm in zip(axes, ("knn", "dt")):
            part = summary.loc[summary.algorithm.eq(algorithm)]
            for i, role in enumerate(("baseline", "selected")):
                row = part.loc[part.role.eq(role)].iloc[0]
                ax.errorbar(i, row.cv_f1_macro_mean, yerr=row.cv_f1_macro_std,
                            fmt=("o" if i == 0 else "D"), color=colors[i],
                            markersize=8, markeredgecolor=tokens["surface"],
                            markeredgewidth=1.5, capsize=5, linewidth=2, label=role)
                ax.annotate(f"{row.cv_f1_macro_mean:.4f}", (i, row.cv_f1_macro_mean),
                            xytext=(12, 0), textcoords="offset points", va="center")
            ax.set(xticks=[0, 1], xticklabels=["Baseline", "CV selected"], xlim=(-.5, 1.65),
                   ylim=(0, 1), ylabel="Macro-F1 (mean of 3 folds)", title=algorithm.upper())
            ax.grid(axis="y", linewidth=.6)
            ax.legend(frameon=False, loc="lower right")
        fig.suptitle(title_prefix + "Fixed HOG: baseline vs selected — SD is not a confidence interval")
        path = figures / f"{prefix}_cv_comparison{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        knn = candidates.loc[candidates.algorithm.eq("knn")].copy()
        expanded = pd.json_normalize(knn.params.map(json.loads)).set_axis(knn.index)
        knn = pd.concat([knn, expanded], axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
        for ax, metric in zip(axes, ("euclidean", "manhattan")):
            present = knn.loc[knn.metric.eq(metric)]
            for i, weights in enumerate(("uniform", "distance")):
                part = present.loc[present.weights.eq(weights)].sort_values("n_neighbors")
                if part.empty:
                    continue
                ax.plot(part.n_neighbors, part.mean_f1_macro, marker=("o" if i == 0 else "D"),
                        markersize=8, markeredgecolor=tokens["surface"], markeredgewidth=1.5,
                        linewidth=2, linestyle=("-" if i == 0 else "--"),
                        color=colors[i], label=weights)
                if len(part):
                    end = part.iloc[-1]
                    ax.annotate(weights, (end.n_neighbors, end.mean_f1_macro),
                                xytext=(16, 15 if i else -16), textcoords="offset points",
                                fontsize=9, arrowprops={"arrowstyle": "-", "color": tokens["grid"], "lw": .8})
            ax.set(title=metric, xlabel="Number of neighbors k", ylabel="CV macro-F1",
                   xlim=(1, 43))
            ax.grid(axis="y", linewidth=.6)
            if not present.empty:
                ax.legend(frameon=False)
            else:
                ax.text(.5, .5, "Not evaluated in baseline-only smoke", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(title_prefix + "KNN search on fixed folds — no official-test scores")
        path = figures / f"{prefix}_knn_k_curve{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
        for ax, algorithm in zip(axes, ("knn", "dt")):
            for i, role in enumerate(("baseline", "selected")):
                part = groups.loc[groups.algorithm.eq(algorithm) & groups.role.eq(role)
                                  & groups.dimension.eq("raw_min_dimension")].set_index("group")
                vals = [part.loc[g, "error_rate"] for g in ("lt32", "ge32")]
                ax.plot([0, 1], vals, marker=("o" if i == 0 else "D"), markersize=8,
                        markeredgecolor=tokens["surface"], markeredgewidth=1.5,
                        color=colors[i], linewidth=2, linestyle=("-" if i == 0 else "--"), label=role)
                ax.annotate(role, (1, vals[-1]), xytext=(-65, 14 if i else -17),
                            textcoords="offset points", fontsize=9,
                            arrowprops={"arrowstyle": "-", "color": tokens["grid"], "lw": .8})
            ax.set(title=algorithm.upper(), xticks=[0, 1], xticklabels=["min(W,H) < 32", "min(W,H) ≥ 32"],
                   ylabel="Pooled OOF error rate", ylim=(0, 1), xlim=(-.2, 1.2))
            ax.grid(axis="y", linewidth=.6)
            ax.legend(frameon=False)
        fig.suptitle(title_prefix + "OOF size diagnostics — descriptive, not causal or nested evaluation")
        path = figures / f"{prefix}_oof_size_errors{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def export(run_dir, *, project_root=ROOT, export_id=None):
    run_dir = Path(run_dir).resolve()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (selection.get("status") not in ("completed_train_cv", "smoke_only")
            or selection.get("official_test_read") is not False
            or selection.get("official_test_evaluated") is not False
            or config.get("official_test_read") is not False
            or selection.get("feature_audit_sha256") != digest(run_dir / "feature_audit.json")
            or selection.get("fold_manifest_sha256") != digest(run_dir / "fold_manifest.csv")):
        raise ValueError("Not a complete compatible train-only upgrade run")
    for algorithm in ("knn", "dt"):
        if sum(item.get("algorithm") == algorithm for item in selection.get("algorithms", [])) != 1:
            raise ValueError("Expected exactly one KNN and one DT selection")
    run_id = selection["run_id"]
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Invalid run-id")
    if export_id is not None and (not export_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in export_id)):
        raise ValueError("Invalid export-id")
    prefix = f"model3_{run_id}" + (f"_{export_id}" if export_id else "")
    tables = project_root / "report/tables"
    figures = project_root / "report/figures"
    logs = project_root / "logs/model3_upgrade" / run_id
    expected_tables = ("summary", "candidates", "fold_scores", "oof_classes", "oof_groups",
                       "oof_error_pairs", "quality_thresholds", "fold_deltas")
    targets = [tables / f"{prefix}_{name}.csv" for name in expected_tables]
    targets += [figures / f"{prefix}_{name}{suffix}.png" for name in
                ("cv_comparison", "knn_k_curve", "oof_size_errors") for suffix in ("", "_dark")]
    provenance_path = logs / (f"export_provenance_{export_id}.json" if export_id else "export_provenance.json")
    if any(p.exists() for p in targets + [provenance_path]):
        raise FileExistsError("Derived export exists; outputs will not be overwritten")
    manifest = pd.read_csv(run_dir / "fold_manifest.csv")
    audit = pd.read_csv(run_dir / "train_image_audit.csv")
    summaries, candidate_frames, fold_frames = [], [], []
    class_rows, group_rows, pair_rows, thresholds, deltas = [], [], [], [], []
    sources = [run_dir / p for p in ("selection.json", "config.json", "fold_manifest.csv",
                                      "feature_audit.json", "train_image_audit.csv")]
    sources.append(Path(__file__).resolve())
    for item in selection["algorithms"]:
        algorithm = item["algorithm"]
        candidate_path = run_dir / f"{algorithm}_candidates.csv"
        fold_path = run_dir / f"{algorithm}_fold_scores.csv"
        candidates = pd.read_csv(candidate_path)
        scores = pd.read_csv(fold_path)
        if candidates.candidate_id.duplicated().any() or not np.isfinite(candidates.mean_f1_macro).all():
            raise ValueError("Invalid candidate results")
        expected_count = (1 if selection["status"] == "smoke_only" else (28 if algorithm == "knn" else 11))
        if len(candidates) != expected_count or len(scores) != 3 * expected_count:
            raise ValueError("Incomplete candidate/fold search")
        if candidates.iloc[0].candidate_id != f"{algorithm}_baseline":
            raise ValueError("Baseline must be the first stable tie-break candidate")
        for candidate in candidates.itertuples(index=False):
            part = scores.loc[scores.candidate_id.eq(candidate.candidate_id)]
            if len(part) != 3 or set(part.fold) != {0, 1, 2}:
                raise ValueError("Missing or duplicate candidate fold scores")
            if not np.isclose(part.f1_macro.mean(), candidate.mean_f1_macro, atol=1e-12, rtol=0):
                raise ValueError("Candidate mean does not match fold scores")
        expected = candidates.iloc[int(np.argmax(candidates.mean_f1_macro.to_numpy()))]
        if (expected.candidate_id != item["selected"]["candidate_id"]
                or json.loads(expected.params) != item["selected"]["params"]):
            raise ValueError("Selection is not deterministic CV winner")
        if not np.isclose(expected.mean_f1_macro, item["selected"]["mean_f1_macro"], atol=1e-12, rtol=0):
            raise ValueError("Selected score mismatch")
        candidate_frames.append(candidates.assign(algorithm=algorithm))
        fold_frames.append(scores.assign(algorithm=algorithm))
        sources.extend([candidate_path, fold_path, run_dir / f"{algorithm}_transforms.json"])
        role_ids = {}
        for role in ("baseline", "selected"):
            chosen = item[role]
            role_ids[role] = chosen["candidate_id"]
            prediction_path = run_dir / f"oof_{algorithm}_{role}.csv"
            predictions = pd.read_csv(prediction_path)
            if set(predictions.CandidateId) != {chosen["candidate_id"]}:
                raise ValueError("OOF candidate differs from selection")
            joined = validate_oof(predictions, audit, manifest)
            actual = []
            for fold in range(3):
                part = joined.loc[joined.ValidationFold.eq(fold)]
                metrics = classification_metrics(part.ClassId, part.PredictedClassId)
                stored = scores.loc[scores.candidate_id.eq(chosen["candidate_id"]) & scores.fold.eq(fold)]
                if len(stored) != 1 or any(not np.isclose(metrics[k], stored.iloc[0][k], atol=1e-12, rtol=0)
                                          for k in metrics):
                    raise ValueError("OOF metrics differ from saved fold scores")
                actual.append(metrics["f1_macro"])
            if not np.isclose(np.mean(actual), chosen["mean_f1_macro"], atol=1e-12, rtol=0):
                raise ValueError("OOF mean differs from chosen candidate")
            pooled = classification_metrics(joined.ClassId, joined.PredictedClassId)
            summaries.append({"algorithm": algorithm, "role": role, "candidate_id": chosen["candidate_id"],
                              "params": json.dumps(chosen["params"], sort_keys=True),
                              "cv_f1_macro_mean": chosen["mean_f1_macro"],
                              "cv_f1_macro_std": chosen["std_f1_macro"],
                              "cv_accuracy_mean": chosen["mean_accuracy"],
                              "pooled_oof_f1_macro": pooled["f1_macro"],
                              "pooled_oof_accuracy": pooled["accuracy"],
                              "train_images": len(manifest), "tracks": manifest.Group.nunique(),
                              "cv_wall_seconds_all_candidates": item["cv_wall_seconds"],
                              "selected_model_bytes": item["model_bytes"],
                              "official_test_evaluated": False})
            cr, gr, pr, tr = diagnostic_tables(joined, algorithm, role, manifest)
            class_rows.extend(cr); group_rows.extend(gr); pair_rows.extend(pr); thresholds.extend(tr)
            sources.append(prediction_path)
        for fold in range(3):
            base = scores.loc[scores.candidate_id.eq(role_ids["baseline"]) & scores.fold.eq(fold)].iloc[0]
            best = scores.loc[scores.candidate_id.eq(role_ids["selected"]) & scores.fold.eq(fold)].iloc[0]
            deltas.append({"algorithm": algorithm, "fold": fold,
                           "baseline_f1_macro": base.f1_macro, "selected_f1_macro": best.f1_macro,
                           "delta_f1_macro": best.f1_macro - base.f1_macro})
    frames = {"summary": pd.DataFrame(summaries), "candidates": pd.concat(candidate_frames, ignore_index=True),
              "fold_scores": pd.concat(fold_frames, ignore_index=True),
              "oof_classes": pd.DataFrame(class_rows), "oof_groups": pd.DataFrame(group_rows),
              "oof_error_pairs": error_pair_frame(pair_rows),
              "quality_thresholds": pd.DataFrame(thresholds), "fold_deltas": pd.DataFrame(deltas)}
    for directory in (tables, figures, logs):
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        write_csv(tables / f"{prefix}_{name}.csv", frame)
    figure_paths = []
    for theme in ("light", "dark"):
        figure_paths.extend(plot_outputs(frames["summary"], frames["candidates"], frames["oof_groups"],
                                          figures, prefix, theme))
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump({"exported_utc": datetime.now(timezone.utc).isoformat(),
                   "run_id": run_id, "status": selection["status"], "official_test_read": False,
                   "models_loaded": False, "fit_or_predict": False,
                   "sources_sha256": {str(p.relative_to(project_root)).replace(chr(92), "/"): digest(p) for p in sources},
                   "outputs": [str(p.relative_to(project_root)).replace(chr(92), "/") for p in targets],
                   "timing_caveat": "CV wall time includes shared work once; candidate-only timing must not be summed as full pipeline latency.",
                   "statistical_caveat": "Selection and OOF share folds; descriptive, not nested/unbiased evaluation. SD is not CI."},
                  stream, ensure_ascii=False, indent=2)
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-id", help="Optional new suffix to re-export without overwriting")
    args = parser.parse_args(argv)
    frames = export(args.run_dir, export_id=args.export_id)
    print(frames["summary"].to_string(index=False))
    print("Exported new namespaced tables/figures; no model or official-test access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
