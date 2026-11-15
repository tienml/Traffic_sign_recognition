"""KNN/Decision Tree training and tuning on caller-supplied features.

Nguyễn Hoàng: original grids, grouped-CV contracts and KNN bundles.
Đặng Quang Tiến (member 4): expanded HOG tuning, shared neighbor queries,
fold-local scaling and OOF diagnostics. No deep learning or feature changes.

``train_and_tune`` retains the legacy precomputed-feature interface used by
main.py. Use the grouped APIs for track-aware experiments; the expanded HOG
APIs require three frozen training folds and never read datasets or test files.
OOF scores used to select a winner are not an independent generalization test.
"""

from hashlib import sha256
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.config import RANDOM_STATE
from src.features import HOG_DEFAULTS, PCA_VARIANCE
from src.preprocessing import CLASS_NAMES


KNN_METHODS = ("pixel", "hog", "hog_pca")
KNN_LABELS = tuple(range(len(CLASS_NAMES)))
HOG_DIMENSIONS = 324
_CV_METRICS = ("f1_macro", "accuracy", "precision_macro", "recall_macro")

def train_and_tune(X_train_feat: np.ndarray, y_train: np.ndarray, model_type: str = "knn"):
    """
    Input:
        - X_train_feat, y_train: Đặc trưng và nhãn tập train
        - model_type: 'knn' hoặc 'dt' (Decision Tree)
    Output:
        - best_model: Mô hình tốt nhất sau khi tìm tham số
        - cv_results_df: Bảng kết quả thử nghiệm các tham số
    """
    print(f"[Người 3] Đang huấn luyện và tối ưu mô hình: {model_type.upper()}...")

    if model_type == "knn":
        base_model = KNeighborsClassifier()
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    elif model_type == "dt":
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5]
        }
    else:
        raise ValueError(f"Loại mô hình {model_type} không hợp lệ!")

    grid = GridSearchCV(base_model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train_feat, y_train)

    print(f" -> Tham số tốt nhất ({model_type}): {grid.best_params_}")
    cv_results_df = pd.DataFrame(grid.cv_results_)
    return grid.best_estimator_, cv_results_df

# Original grouped-CV interfaces remain compatible with train_knn.py/train_dt.py.


def track_groups(metadata, labels=None):
    """Validate CSV order/class/path agreement; never fall back to image CV."""
    y = metadata["ClassId"].to_numpy()
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, KNN_LABELS).all():
        raise ValueError("ClassId must contain integers 0..42")
    if labels is not None and not np.array_equal(y, labels):
        raise ValueError("Loader labels are not aligned with CSV rows")
    if metadata["Path"].duplicated().any():
        raise ValueError("Duplicate image paths")
    groups = []
    for path, label in zip(metadata["Path"], y):
        normalized = PurePosixPath(str(path).replace("\\", "/"))
        match = re.fullmatch(r"(\d+)_(\d+)_(\d+)\.png", normalized.name)
        if (not match or len(normalized.parts) != 3
                or normalized.parts[0] != "Train"
                or normalized.parts[1] != str(int(label))
                or int(match[1]) != label):
            raise ValueError(f"Invalid train track/class path: {path}")
        groups.append(f"{int(label)}:{int(match[2])}")
    return np.asarray(groups)


def validate_folds(y, groups, folds, labels=KNN_LABELS):
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape or not np.array_equal(np.unique(y), labels):
        raise ValueError("Expected aligned labels/groups and all required classes")
    seen = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        train, valid = np.asarray(train), np.asarray(valid)
        if (len(train) == 0 or len(valid) == 0
                or len(np.unique(train)) != len(train)
                or len(np.unique(valid)) != len(valid)
                or np.any(train < 0) or np.any(valid < 0)
                or np.any(train >= len(y)) or np.any(valid >= len(y))):
            raise ValueError("Invalid fold indices")
        if (len(np.intersect1d(train, valid))
                or len(train) + len(valid) != len(y)
                or len(np.intersect1d(groups[train], groups[valid]))):
            raise ValueError("Fold overlap, missing rows or group leakage")
        if any(not np.array_equal(np.unique(y[idx]), labels) for idx in (train, valid)):
            raise ValueError("Every fold side must contain every class")
        seen[valid] += 1
    if not len(folds) or not np.all(seen == 1):
        raise ValueError("Each row must be validation exactly once")


def grouped_folds(y, groups, n_splits=3, labels=KNN_LABELS):
    y, groups = np.asarray(y), np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("Groups and labels are not aligned")
    for label in labels:
        if len(np.unique(groups[y == label])) < n_splits:
            raise ValueError(f"Class {label} needs at least {n_splits} tracks")
    folds = list(StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    ).split(np.zeros(len(y)), y, groups))
    validate_folds(y, groups, folds, labels)
    return folds


def _float32(values):
    return np.asarray(values, dtype=np.float32)


def knn_pipeline(method):
    if method not in KNN_METHODS:
        raise ValueError(f"Unknown representation: {method}")
    steps = []
    if method != "pixel":
        steps.append(("scaler", StandardScaler()))
    if method == "hog_pca":
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)))
        # Match person 2's feature contract after PCA, including dtype.
        steps.append(("float32", FunctionTransformer(_float32)))
    steps.append(("knn", KNeighborsClassifier(algorithm="brute", n_jobs=1)))
    return Pipeline(steps)


def knn_candidates(smoke=False):
    # ParameterGrid's stable order is the declared tie-break, not elapsed time.
    return {"knn__n_neighbors": [3] if smoke else [3, 5, 7],
            "knn__weights": ["uniform"] if smoke else ["uniform", "distance"],
            "knn__metric": ["euclidean"] if smoke else ["euclidean", "manhattan"]}


def batch_predict(estimator, features, batch_size=128):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if len(features) == 0:
        raise ValueError("Empty prediction input")
    return np.concatenate([estimator.predict(features[start:start + batch_size])
                           for start in range(0, len(features), batch_size)])


def classification_metrics(y, predicted, labels=KNN_LABELS):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=list(labels), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1)}


class BatchScorer:
    """One batched prediction per fold/candidate for all scoring metrics."""
    def __init__(self, batch_size=128, labels=KNN_LABELS):
        self.batch_size, self.labels = batch_size, labels

    def __call__(self, estimator, features, y):
        result = classification_metrics(y, batch_predict(
            estimator, features, self.batch_size), self.labels)
        pca = estimator.named_steps.get("pca")
        result["feature_dimensions"] = (float(pca.n_components_) if pca is not None
                                        else float(features.shape[1]))
        return result


def first_best_index(results):
    scores = np.asarray(results["mean_test_f1_macro"], dtype=float)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("CV scores must all be finite")
    return int(np.argmax(scores))


def tune_grouped_knn(features, y, groups, folds, method, *, smoke=False,
                     batch_size=128, n_jobs=1, verbose=3, labels=KNN_LABELS):
    validate_folds(y, groups, folds, labels)
    if len(features) != len(y) or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Invalid raw feature matrix")
    candidates = knn_candidates(smoke)
    if max(candidates["knn__n_neighbors"]) > min(len(t) for t, _ in folds):
        raise ValueError("k exceeds a fold's training size")
    if n_jobs != 1:
        raise ValueError("Use n_jobs=1 to avoid duplicating image/feature arrays")
    grid = GridSearchCV(knn_pipeline(method), candidates, cv=folds,
                        scoring=BatchScorer(batch_size, labels),
                        refit=first_best_index, n_jobs=1, error_score="raise",
                        return_train_score=False, verbose=verbose)
    with threadpool_limits(limits=2):
        grid.fit(features, y)
    return grid


def save_knn_bundle(path, estimator, method, metadata):
    """Exclusive creation; joblib files must only be loaded from trusted sources."""
    if method not in KNN_METHODS:
        raise ValueError("Unknown representation")
    bundle = {"schema_version": 1, "method": method, "estimator": estimator,
              "class_names": CLASS_NAMES.copy(), "labels": list(KNN_LABELS),
              "raw_dimensions": int(estimator.n_features_in_),
              "hog_params": HOG_DEFAULTS.copy(), "seed": RANDOM_STATE,
              "image_contract": {"size": [32, 32], "color": "RGB", "roi": True,
                                 "scale": "divide255", "augmentation": False},
              "metadata": metadata}
    with Path(path).open("xb") as stream:
        joblib.dump(bundle, stream, compress=3)


def load_knn_bundle(path):
    bundle = joblib.load(path)  # Trusted local files only: pickle can execute code.
    if (bundle.get("schema_version") != 1 or bundle.get("class_names") != CLASS_NAMES
            or bundle.get("labels") != list(KNN_LABELS)
            or bundle.get("method") not in KNN_METHODS
            or bundle.get("hog_params") != HOG_DEFAULTS
            or bundle.get("seed") != RANDOM_STATE
            or bundle.get("image_contract") != {
                "size": [32, 32], "color": "RGB", "roi": True,
                "scale": "divide255", "augmentation": False}
            or bundle.get("raw_dimensions") != bundle["estimator"].n_features_in_):
        raise ValueError("Incompatible K-NN bundle")
    return bundle


# Expanded HOG tuning — Đặng Quang Tiến (member 4), 2026-11-15.
# Explicit names distinguish these ordered candidate records from the original
# Pipeline-prefixed parameter grids. Do not change the historical grid contract.
def expanded_knn_candidates(smoke=False):
    """Return the baseline first, then the remaining stable 28-point HOG grid."""
    baseline = {"n_neighbors": 7, "weights": "distance", "metric": "manhattan"}
    result = [{"candidate_id": "knn_baseline", "params": baseline}]
    if not smoke:
        for k in (3, 5, 7, 11, 15, 21, 31):
            for weights in ("uniform", "distance"):
                for metric in ("euclidean", "manhattan"):
                    params = {"n_neighbors": k, "weights": weights, "metric": metric}
                    if params != baseline:
                        result.append({"candidate_id": f"knn_k{k}_{weights}_{metric}",
                                       "params": params})
    return result


def regularized_dt_candidates(smoke=False):
    """Target leaf size, balancing and light pruning; retain the HOG baseline."""
    baseline = {"criterion": "entropy", "max_depth": 20, "min_samples_split": 5,
                "min_samples_leaf": 1, "class_weight": None, "ccp_alpha": 0.0}
    result = [{"candidate_id": "dt_baseline", "params": baseline}]
    if not smoke:
        for leaf in (1, 2, 5, 10):
            for weight in (None, "balanced"):
                params = {**baseline, "min_samples_leaf": leaf, "class_weight": weight}
                if params != baseline:
                    name = "none" if weight is None else weight
                    result.append({"candidate_id": f"dt_leaf{leaf}_{name}", "params": params})
        for name, alpha in (("1e-4", 1e-4), ("5e-4", 5e-4), ("1e-3", 1e-3)):
            result.append({"candidate_id": f"dt_alpha_{name}",
                           "params": {**baseline, "ccp_alpha": alpha}})
    return result


def _integer_values(values, name, *, ndim=1):
    values = np.asarray(values)
    if (values.ndim != ndim or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError(f"{name} must be a {ndim}D integer array")
    if values.size and np.issubdtype(values.dtype, np.unsignedinteger):
        if np.max(values) > np.iinfo(np.int64).max:
            raise ValueError(f"{name} exceeds int64 range")
    return values.astype(np.int64, copy=False)


def _label_values(labels):
    values = _integer_values(labels, "labels")
    if not len(values) or len(np.unique(values)) != len(values):
        raise ValueError("labels must contain distinct integer classes")
    return np.sort(values)


def _raw_features(raw):
    raw = np.asarray(raw)
    if (raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != HOG_DIMENSIONS
            or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all()):
        raise ValueError("raw must be a nonempty finite floating HOG324 matrix")
    # Normalize floating input, not arbitrary integer/image data, to the HOG dtype.
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.ascontiguousarray(raw, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("raw cannot be represented as finite float32")
    return raw


def _batch_size(value):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError("batch_size must be a positive integer")
    return int(value)


def _cv_inputs(raw, y, groups, folds, labels, batch_size, progress):
    raw = _raw_features(raw)
    y = _integer_values(y, "y")
    groups = np.asarray(groups)
    labels = tuple(int(c) for c in _label_values(labels))
    if len(y) != len(raw) or groups.shape != y.shape:
        raise ValueError("raw, y and groups must be row-aligned")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    batch_size = _batch_size(batch_size)
    supplied_folds = list(folds)
    if len(supplied_folds) != 3:
        raise ValueError("Exactly three frozen folds are required")
    checked = []
    for pair in supplied_folds:
        if len(pair) != 2:
            raise ValueError("Each fold must contain train and validation indices")
        checked.append((_integer_values(pair[0], "train indices"),
                        _integer_values(pair[1], "validation indices")))
    validate_folds(y, groups, checked, labels=labels)
    return raw, y, checked, labels, batch_size


def _vote(distances, neighbor_labels, weights, labels):
    # Preserve reciprocal dtype/rounding, then sum in float64 like sklearn's
    # weighted_mode. With zero distances only zero-distance neighbors get votes.
    if weights == "distance":
        with np.errstate(divide="ignore", over="ignore"):
            voting_weights = 1.0 / distances
        infinite = np.isinf(voting_weights)
        zero_rows = infinite.any(axis=1)
        voting_weights[zero_rows] = infinite[zero_rows]
        voting_weights = voting_weights.astype(np.float64, copy=False)
    counts = np.empty((len(distances), len(labels)), dtype=np.float64)
    for column, label in enumerate(labels):
        match = neighbor_labels == label
        if weights == "uniform":
            counts[:, column] = np.count_nonzero(match, axis=1)
        else:
            counts[:, column] = np.where(match, voting_weights, 0.0).sum(axis=1)
    # labels are sorted, and argmax retains the first maximum.
    return labels[counts.argmax(axis=1)].astype(np.int64, copy=False)


def vote_neighbors(distances, neighbor_labels, *, weights, labels):
    """Vote over supplied nearest neighbors, matching sklearn class/zero ties.

    Arrays must have identical nonempty (queries, neighbors) shapes; distances
    must be finite, nonnegative real numbers and neighbor labels integer members
    of ``labels``. Label order supplied by the caller does not affect tie breaks.
    The caller is responsible for selecting the correct neighbors at a k cutoff.
    """
    distances = np.asarray(distances)
    neighbor_labels = _integer_values(neighbor_labels, "neighbor_labels", ndim=2)
    labels = _label_values(labels)
    if weights not in ("uniform", "distance"):
        raise ValueError("weights must be uniform or distance")
    if (distances.ndim != 2 or 0 in distances.shape
            or distances.shape != neighbor_labels.shape
            or not (np.issubdtype(distances.dtype, np.integer)
                    or np.issubdtype(distances.dtype, np.floating))
            or not np.isfinite(distances).all() or np.any(distances < 0)
            or not np.isin(neighbor_labels, labels).all()):
        raise ValueError("Invalid neighbor distances or labels")
    if not np.issubdtype(distances.dtype, np.floating):
        distances = distances.astype(np.float64)
    return _vote(distances, neighbor_labels, weights, labels)


def _notify(progress, message):
    if progress is not None:
        progress(message)


def _scale_fold(raw, train, valid, fold):
    scaler = StandardScaler()
    training = raw[train]
    started = perf_counter()
    scaler.fit(training)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    training = scaler.transform(training)
    validation = scaler.transform(raw[valid])
    transform_seconds = perf_counter() - started
    diagnostic = {"fold": fold, "n_train": len(train), "n_valid": len(valid),
                  "scaler_fit_samples": int(scaler.n_samples_seen_),
                  "mean_checksum": sha256(scaler.mean_.tobytes()).hexdigest(),
                  "scaler_fit_seconds": fit_seconds,
                  "scaler_transform_seconds": transform_seconds}
    return training, validation, diagnostic


def _aggregate(candidates, fold_scores, oof, transforms, *, is_knn):
    rows = []
    for candidate in candidates:
        scores = [r for r in fold_scores if r["candidate_id"] == candidate["candidate_id"]]
        row = {"candidate_id": candidate["candidate_id"], "params": candidate["params"].copy()}
        for metric in _CV_METRICS:
            row[f"mean_{metric}"] = float(np.mean([r[metric] for r in scores]))
        row["std_f1_macro"] = float(np.std([r["f1_macro"] for r in scores], ddof=0))
        row["cv_seconds_sum"] = float(sum(r["fit_seconds"] + r["predict_seconds"] for r in scores))
        row["timing_scope"] = ("candidate_only_excludes_shared_scaler_and_neighbors" if is_knn
                               else "fit_and_train_valid_predict_excludes_shared_scaler")
        if not is_knn:
            row["mean_train_f1_macro"] = float(np.mean([r["train_f1_macro"] for r in scores]))
        rows.append(row)
    return {"candidates": rows, "fold_scores": fold_scores, "oof": oof,
            "transforms": transforms}


def evaluate_knn_cv(raw, y, groups, folds, *, batch_size=256, labels=KNN_LABELS,
                    smoke=False, progress=None):
    """Evaluate the expanded grid with fold-local scaling and bounded queries.

    Neighbor buffers are at most batch_size x (max_k + 1); no full validation
    distance matrix is materialized. At equal kth/(k+1)st distances, max-k
    neighbor prefixes can disagree with sklearn's k-specific selection. Only
    those rows fall back to direct, brute KNeighborsClassifier.predict.

    Candidate timings EXCLUDE shared scaler/neighbor work, recorded once in
    transforms[].neighbor_searches. Use a wall timer for end-to-end comparisons.
    No feature extraction, fold generation, dataset access or serialization.
    """
    raw, y, folds, labels, batch_size = _cv_inputs(
        raw, y, groups, folds, labels, batch_size, progress)
    candidates = expanded_knn_candidates(smoke)
    max_k = max(c["params"]["n_neighbors"] for c in candidates)
    if any(len(train) < max_k for train, _ in folds):
        raise ValueError("k exceeds a frozen fold's training size")
    label_array = np.asarray(labels, dtype=np.int64)
    oof = {c["candidate_id"]: np.full(len(y), -1, dtype=np.int64) for c in candidates}
    fold_scores, transforms = [], []
    with threadpool_limits(limits=2):
        for fold, (train, valid) in enumerate(folds):
            _notify(progress, f"knn fold={fold} scaler")
            training, validation, diagnostic = _scale_fold(raw, train, valid, fold)
            diagnostic["neighbor_searches"] = []
            transforms.append(diagnostic)
            y_train = y[train]
            # Baseline-first metric order is stable, just like the candidate list.
            for metric in dict.fromkeys(c["params"]["metric"] for c in candidates):
                _notify(progress, f"knn fold={fold} metric={metric} shared_neighbors")
                relevant = [c for c in candidates if c["params"]["metric"] == metric]
                query_k = min(max_k + 1, len(train))
                search = NearestNeighbors(n_neighbors=query_k, metric=metric,
                                          algorithm="brute", n_jobs=1)
                started = perf_counter()
                search.fit(training)
                shared = {"metric": metric, "n_neighbors": query_k,
                          "fit_seconds": perf_counter() - started,
                          "query_seconds": 0.0, "query_batches": 0,
                          "timing_scope": "shared_once_per_metric_per_fold"}
                diagnostic["neighbor_searches"].append(shared)
                scores = {c["candidate_id"]: {"candidate_id": c["candidate_id"], "fold": fold,
                          "fit_seconds": 0.0, "predict_seconds": 0.0,
                          "tie_fallback_rows": 0,
                          "timing_scope": "candidate_only_excludes_shared_scaler_and_neighbors",
                          "shared_neighbor_timing_key": f"fold={fold}/metric={metric}"}
                          for c in relevant}
                fallback_models = {}
                for start in range(0, len(valid), batch_size):
                    stop = min(start + batch_size, len(valid))
                    chunk = validation[start:stop]
                    started = perf_counter()
                    distances, indices = search.kneighbors(chunk)
                    shared["query_seconds"] += perf_counter() - started
                    shared["query_batches"] += 1
                    neighbor_labels = y_train[indices]
                    for candidate in relevant:
                        cid, params = candidate["candidate_id"], candidate["params"]
                        score = scores[cid]
                        k = params["n_neighbors"]
                        started = perf_counter()
                        predicted = _vote(distances[:, :k], neighbor_labels[:, :k],
                                          params["weights"], label_array)
                        boundary_ties = (distances[:, k - 1] == distances[:, k]
                                         if k < query_k else np.zeros(len(chunk), dtype=bool))
                        score["predict_seconds"] += perf_counter() - started
                        if boundary_ties.any():
                            if cid not in fallback_models:
                                started = perf_counter()
                                model = KNeighborsClassifier(**params, algorithm="brute", n_jobs=1)
                                fallback_models[cid] = model.fit(training, y_train)
                                score["fit_seconds"] += perf_counter() - started
                            started = perf_counter()
                            predicted[boundary_ties] = fallback_models[cid].predict(chunk[boundary_ties])
                            score["predict_seconds"] += perf_counter() - started
                            score["tie_fallback_rows"] += int(np.count_nonzero(boundary_ties))
                        oof[cid][valid[start:stop]] = predicted
                for candidate in relevant:
                    cid = candidate["candidate_id"]
                    scores[cid].update(classification_metrics(y[valid], oof[cid][valid], labels=labels))
                    fold_scores.append(scores[cid])
                    _notify(progress, f"knn fold={fold} metric={metric} candidate={cid} complete")
    return _aggregate(candidates, fold_scores, oof, transforms, is_knn=True)


def evaluate_dt_cv(raw, y, groups, folds, *, batch_size=256, labels=KNN_LABELS,
                   smoke=False, progress=None):
    """Fit trees sequentially with fold-local HOG scaling; no dataset/file I/O.

    Prediction timings include validation and training predictions, excluding
    shared scaler work. Training scores diagnose the train/validation gap.
    """
    raw, y, folds, labels, batch_size = _cv_inputs(
        raw, y, groups, folds, labels, batch_size, progress)
    candidates = regularized_dt_candidates(smoke)
    oof = {c["candidate_id"]: np.full(len(y), -1, dtype=np.int64) for c in candidates}
    fold_scores, transforms = [], []
    with threadpool_limits(limits=2):
        for fold, (train, valid) in enumerate(folds):
            _notify(progress, f"dt fold={fold} scaler")
            training, validation, diagnostic = _scale_fold(raw, train, valid, fold)
            transforms.append(diagnostic)
            for candidate in candidates:
                cid = candidate["candidate_id"]
                _notify(progress, f"dt fold={fold} candidate={cid} fit")
                estimator = DecisionTreeClassifier(**candidate["params"], random_state=RANDOM_STATE)
                started = perf_counter()
                estimator.fit(training, y[train])
                fit_seconds = perf_counter() - started
                started = perf_counter()
                predicted = batch_predict(estimator, validation, batch_size).astype(np.int64, copy=False)
                valid_seconds = perf_counter() - started
                started = perf_counter()
                train_predicted = batch_predict(estimator, training, batch_size).astype(np.int64, copy=False)
                train_seconds = perf_counter() - started
                oof[cid][valid] = predicted
                row = {"candidate_id": cid, "fold": fold,
                       **classification_metrics(y[valid], predicted, labels=labels),
                       "train_f1_macro": classification_metrics(
                           y[train], train_predicted, labels=labels)["f1_macro"],
                       "fit_seconds": fit_seconds,
                       "predict_seconds": valid_seconds + train_seconds,
                       "valid_predict_seconds": valid_seconds,
                       "train_predict_seconds": train_seconds,
                       "timing_scope": "fit_and_train_valid_predict_excludes_shared_scaler"}
                fold_scores.append(row)
                _notify(progress, f"dt fold={fold} candidate={cid} complete")
    return _aggregate(candidates, fold_scores, oof, transforms, is_knn=False)


def select_best(result):
    """Return the first finite maximum-scoring aggregate row, never a time tie-break."""
    finite = [row for row in result["candidates"] if np.isfinite(row["mean_f1_macro"])]
    if not finite:
        raise ValueError("No finite CV candidate scores")
    return max(finite, key=lambda row: row["mean_f1_macro"])


def refit_model(raw, y, algorithm, params):
    """Fit the selected scaler/estimator pipeline on training data, with no I/O."""
    raw = _raw_features(raw)
    y = _integer_values(y, "y")
    if len(y) != len(raw):
        raise ValueError("raw and y must be row-aligned")
    if algorithm == "knn":
        estimator = KNeighborsClassifier(**{**params, "algorithm": "brute", "n_jobs": 1})
    elif algorithm == "dt":
        estimator = DecisionTreeClassifier(**{**params, "random_state": RANDOM_STATE})
    else:
        raise ValueError("algorithm must be knn or dt")
    model = Pipeline([("scaler", StandardScaler()), (algorithm, estimator)])
    with threadpool_limits(limits=2):
        model.fit(raw, y)
    return model


if __name__ == "__main__":
    X_dummy = np.random.rand(100, 128)
    y_dummy = np.random.randint(0, 5, size=100)
    model, df_log = train_and_tune(X_dummy, y_dummy, model_type="knn")
    print("Test thành công! Số cấu hình đã thử:", len(df_log))
