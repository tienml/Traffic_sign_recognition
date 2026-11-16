"""ML-only tuning on raw HOG324 and caller-supplied frozen training folds.

No image/feature extraction, fold generation, test access, logging configuration,
serialization, or other file I/O belongs here. StandardScaler is fit separately
on each fold's training rows. Selecting and reporting the winner on these same
OOF folds is optimistic; its OOF score is not an independent generalization test.

KNN shares one bounded neighbor query stream per metric/fold. Its candidate
``fit_seconds``, ``predict_seconds`` and ``cv_seconds_sum`` EXCLUDE shared scaler
and neighbor work. Shared timings live once in ``transforms[].neighbor_searches``;
use an external wall timer for end-to-end algorithm comparisons. Candidate time
includes voting and any exact sklearn prediction needed at a tied k boundary.
DT candidate prediction time includes both validation and training predictions.
"""

from hashlib import sha256
from numbers import Integral
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.models import classification_metrics, validate_folds


LABELS = tuple(range(43))
HOG_DIMENSIONS = 324
_METRICS = ("f1_macro", "accuracy", "precision_macro", "recall_macro")


def knn_candidates(smoke=False):
    """Return the baseline first, then the remaining stable 28-point grid."""
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


def dt_candidates(smoke=False):
    """Target leaf size, balancing and light pruning; retain the legacy baseline."""
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
        for metric in _METRICS:
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


def evaluate_knn_cv(raw, y, groups, folds, *, batch_size=256, labels=LABELS,
                    smoke=False, progress=None):
    """Evaluate the expanded grid without fitting transforms outside each fold.

    Neighbor buffers are at most batch_size x (max_k + 1); no full validation
    distance matrix is materialized. At equal kth/(k+1)st distances, max-k
    neighbor prefixes can disagree with sklearn's k-specific selection. Only
    those rows fall back to direct, brute KNeighborsClassifier.predict.
    """
    raw, y, folds, labels, batch_size = _cv_inputs(
        raw, y, groups, folds, labels, batch_size, progress)
    candidates = knn_candidates(smoke)
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


def _predict_batches(estimator, features, batch_size):
    return np.concatenate([estimator.predict(features[start:start + batch_size])
                           for start in range(0, len(features), batch_size)]).astype(np.int64, copy=False)


def evaluate_dt_cv(raw, y, groups, folds, *, batch_size=256, labels=LABELS,
                   smoke=False, progress=None):
    """Fit trees sequentially, with the same fold-local scaling as legacy HOG."""
    raw, y, folds, labels, batch_size = _cv_inputs(
        raw, y, groups, folds, labels, batch_size, progress)
    candidates = dt_candidates(smoke)
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
                estimator = DecisionTreeClassifier(**candidate["params"], random_state=42)
                started = perf_counter()
                estimator.fit(training, y[train])
                fit_seconds = perf_counter() - started
                started = perf_counter()
                predicted = _predict_batches(estimator, validation, batch_size)
                valid_seconds = perf_counter() - started
                started = perf_counter()
                train_predicted = _predict_batches(estimator, training, batch_size)
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
        estimator = DecisionTreeClassifier(**{**params, "random_state": 42})
    else:
        raise ValueError("algorithm must be knn or dt")
    model = Pipeline([("scaler", StandardScaler()), (algorithm, estimator)])
    with threadpool_limits(limits=2):
        model.fit(raw, y)
    return model
