"""Synthetic-only upgrade tests: no datasets, feature extraction, or file outputs."""

from hashlib import sha256
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src import model_upgrade as upgrade
from src.models import classification_metrics


LABELS = (0, 1)


def synthetic_training(*, tied=False):
    """Two classes, six whole tracks/class, four frames/track, three fixed folds."""
    rng = np.random.default_rng(42)
    y = np.repeat(np.arange(2, dtype=np.int64), 24)
    groups = np.array([f"{c}:{track}" for c in range(2) for track in range(6)
                       for _ in range(4)])
    assignment = np.tile(np.repeat(np.arange(6) % 3, 4), 2)
    folds = [(np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold))
             for fold in range(3)]
    if tied:
        # Every training/validation fold contains conflicting labels at identical
        # feature vectors, including zero distances and k-boundary ties.
        raw = np.zeros((len(y), 324), dtype=np.float32)
        raw[:, 0] = np.tile([0.0, 0.0, 1.0, -1.0], 12)
    else:
        raw = rng.normal(size=(len(y), 324)).astype(np.float32)
        raw[:, :5] += (y * 0.5)[:, None]
    return raw, y, groups, folds


def sklearn_oof(raw, y, folds, candidates, algorithm, batch_size):
    result = {}
    for candidate in candidates:
        predicted = np.full(len(y), -1, dtype=np.int64)
        for train, valid in folds:
            estimator = (KNeighborsClassifier(**candidate["params"], algorithm="brute", n_jobs=1)
                         if algorithm == "knn" else
                         DecisionTreeClassifier(**candidate["params"], random_state=42))
            model = Pipeline([("scaler", StandardScaler()), (algorithm, estimator)])
            model.fit(raw[train], y[train])
            predicted[valid] = np.concatenate([
                model.predict(raw[valid[start:start + batch_size]])
                for start in range(0, len(valid), batch_size)])
        result[candidate["candidate_id"]] = predicted
    return result


class UpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limits = threadpool_limits(limits=2)
        cls.thread_limits.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.thread_limits.__exit__(None, None, None)

    def test_exact_grids_baseline_first_and_smoke(self):
        knn = upgrade.knn_candidates()
        self.assertEqual(len(knn), 28)
        self.assertEqual(len({row["candidate_id"] for row in knn}), 28)
        self.assertEqual(knn[0], {"candidate_id": "knn_baseline", "params": {
            "n_neighbors": 7, "weights": "distance", "metric": "manhattan"}})
        actual = {(r["params"]["n_neighbors"], r["params"]["weights"], r["params"]["metric"])
                  for r in knn}
        self.assertEqual(actual, {(k, w, m) for k in (3, 5, 7, 11, 15, 21, 31)
                                 for w in ("uniform", "distance")
                                 for m in ("euclidean", "manhattan")})
        dt = upgrade.dt_candidates()
        self.assertEqual(len(dt), 11)
        self.assertEqual(len({row["candidate_id"] for row in dt}), 11)
        self.assertEqual(dt[0], {"candidate_id": "dt_baseline", "params": {
            "criterion": "entropy", "max_depth": 20, "min_samples_split": 5,
            "min_samples_leaf": 1, "class_weight": None, "ccp_alpha": 0.0}})
        regularization = {(r["params"]["min_samples_leaf"], r["params"]["class_weight"],
                           r["params"]["ccp_alpha"]) for r in dt}
        self.assertEqual(regularization,
                         {(leaf, weight, 0.0) for leaf in (1, 2, 5, 10)
                          for weight in (None, "balanced")} |
                         {(1, None, alpha) for alpha in (1e-4, 5e-4, 1e-3)})
        for row in dt:
            self.assertEqual(row["params"]["criterion"], "entropy")
            self.assertEqual(row["params"]["max_depth"], 20)
            self.assertEqual(row["params"]["min_samples_split"], 5)
        for function in (upgrade.knn_candidates, upgrade.dt_candidates):
            self.assertEqual(function(True), function()[:1])
            self.assertEqual(function(), function())
            changed = function()
            changed[0]["params"].clear()
            self.assertTrue(function()[0]["params"])

    def test_voting_sklearn_parity_zero_distance_and_sorted_ties(self):
        rng = np.random.default_rng(7)
        training = rng.normal(size=(15, 4)).astype(np.float32)
        training[1] = training[0]
        y = np.array([9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5, 9, 2, 5], dtype=np.int64)
        query = np.vstack([training[0], training[3], rng.normal(size=(8, 4))]).astype(np.float32)
        for metric in ("euclidean", "manhattan"):
            for weights in ("uniform", "distance"):
                with self.subTest(metric=metric, weights=weights):
                    model = KNeighborsClassifier(n_neighbors=7, algorithm="brute", n_jobs=1,
                                                 metric=metric, weights=weights).fit(training, y)
                    distances, indices = model.kneighbors(query)
                    actual = upgrade.vote_neighbors(distances, y[indices], weights=weights,
                                                    labels=(9, 2, 5))
                    np.testing.assert_array_equal(actual, model.predict(query))
                    self.assertEqual(actual.dtype, np.dtype("int64"))
        distances = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float64)
        neighbor_labels = np.array([[9, 2, 9, 9], [9, 2, 9, 2]])
        np.testing.assert_array_equal(upgrade.vote_neighbors(
            distances, neighbor_labels, weights="distance", labels=(9, 2)), [2, 2])
        np.testing.assert_array_equal(upgrade.vote_neighbors(
            distances, neighbor_labels, weights="uniform", labels=(9, 2)), [9, 2])
        invalid = [
            ([-1.0], [0], "uniform", (0,)),
            ([[np.nan]], [[0]], "uniform", (0,)),
            ([[np.inf]], [[0]], "uniform", (0,)),
            ([[1 + 2j]], [[0]], "uniform", (0,)),
            ([[1.0]], [[2]], "uniform", (0,)),
            ([[1.0]], [[0.0]], "uniform", (0,)),
            ([[1.0]], [[0]], "other", (0,)),
            ([[1.0]], [[0]], "uniform", (0, 0)),
            (np.empty((0, 1)), np.empty((0, 1), dtype=int), "uniform", (0,)),
            ([[1.0, 2.0]], [[0]], "uniform", (0,)),
        ]
        for distances, neighbors, weights, labels in invalid:
            with self.subTest(invalid=repr(distances)), self.assertRaises(ValueError):
                upgrade.vote_neighbors(distances, neighbors, weights=weights, labels=labels)

    def test_knn_full_grid_matches_reference_and_queries_are_bounded_and_shared(self):
        raw, y, groups, folds = synthetic_training()
        expected = sklearn_oof(raw, y, folds, upgrade.knn_candidates(), "knn", 5)
        query_sizes, fitted_metrics = [], []
        original_query, original_fit = NearestNeighbors.kneighbors, NearestNeighbors.fit

        def inspect_query(search, features, *args, **kwargs):
            query_sizes.append(len(features))
            return original_query(search, features, *args, **kwargs)

        def inspect_fit(search, features, *args, **kwargs):
            fitted_metrics.append(search.metric)
            return original_fit(search, features, *args, **kwargs)

        messages = []
        with patch.object(NearestNeighbors, "kneighbors", inspect_query), \
             patch.object(NearestNeighbors, "fit", inspect_fit):
            result = upgrade.evaluate_knn_cv(raw, y, groups, folds, batch_size=5,
                                             labels=LABELS, progress=messages.append)
        self.assertEqual(fitted_metrics, ["manhattan", "euclidean"] * 3)
        self.assertEqual(len(query_sizes), 6 * 4)
        self.assertLessEqual(max(query_sizes), 5)
        self.assertEqual(len(result["candidates"]), 28)
        self.assertEqual(len(result["fold_scores"]), 28 * 3)
        self.assertEqual(len(result["transforms"]), 3)
        self.assertTrue(any("metric=" in message for message in messages))
        self.assertEqual(sum("candidate=" in message for message in messages), 28 * 3)
        for cid, predicted in result["oof"].items():
            np.testing.assert_array_equal(predicted, expected[cid], err_msg=cid)
            self.assertEqual(predicted.dtype, np.dtype("int64"))
            self.assertEqual(predicted.shape, y.shape)
            self.assertTrue(np.isin(predicted, LABELS).all())
        for diagnostic in result["transforms"]:
            self.assertEqual(len(diagnostic["neighbor_searches"]), 2)
            for shared in diagnostic["neighbor_searches"]:
                self.assertEqual(shared["n_neighbors"], 32)
                self.assertEqual(shared["query_batches"], 4)
                self.assertGreaterEqual(shared["query_seconds"], 0)
        self.assert_aggregation(result, y, folds)

    def test_boundary_tie_fallback_is_exact_for_every_candidate_and_batching(self):
        raw, y, groups, folds = synthetic_training(tied=True)
        expected = sklearn_oof(raw, y, folds, upgrade.knn_candidates(), "knn", 5)
        direct_calls = []
        original_predict = KNeighborsClassifier.predict

        def inspect_predict(model, features):
            direct_calls.append(len(features))
            return original_predict(model, features)

        with patch.object(KNeighborsClassifier, "predict", inspect_predict):
            result = upgrade.evaluate_knn_cv(raw, y, groups, folds, batch_size=5, labels=LABELS)
        self.assertTrue(direct_calls)
        self.assertLessEqual(max(direct_calls), 5)
        self.assertGreater(sum(row["tie_fallback_rows"] for row in result["fold_scores"]), 0)
        for cid, predicted in result["oof"].items():
            np.testing.assert_array_equal(predicted, expected[cid], err_msg=cid)
        # Rechunking must not change predictions, including sklearn tie fallbacks.
        again = upgrade.evaluate_knn_cv(raw, y, groups, folds, batch_size=64, labels=LABELS)
        for cid in result["oof"]:
            np.testing.assert_array_equal(result["oof"][cid], again["oof"][cid], err_msg=cid)
        # If train size == maximum k, querying k+1 is deliberately impossible.
        # Keep the frozen split complete by removing one row globally.
        keep = np.arange(len(y)) != 0
        reduced_assignment = np.full(len(y), -1)
        for fold, (_, valid) in enumerate(folds):
            reduced_assignment[valid] = fold
        reduced_assignment = reduced_assignment[keep]
        smaller = [(np.flatnonzero(reduced_assignment != fold),
                    np.flatnonzero(reduced_assignment == fold)) for fold in range(3)]
        reduced = upgrade.evaluate_knn_cv(raw[keep], y[keep], groups[keep], smaller,
                                          batch_size=8, labels=LABELS)
        self.assertIn(31, [s["n_neighbors"] for d in reduced["transforms"]
                           for s in d["neighbor_searches"]])
        reduced_expected = sklearn_oof(raw[keep], y[keep], smaller,
                                      upgrade.knn_candidates(), "knn", 8)
        for cid in reduced["oof"]:
            np.testing.assert_array_equal(reduced["oof"][cid], reduced_expected[cid], err_msg=cid)

    def assert_aggregation(self, result, y, folds):
        for row in result["candidates"]:
            scores = [r for r in result["fold_scores"] if r["candidate_id"] == row["candidate_id"]]
            self.assertEqual(len(scores), 3)
            for score in scores:
                valid = folds[score["fold"]][1]
                expected = classification_metrics(y[valid], result["oof"][row["candidate_id"]][valid],
                                                   labels=LABELS)
                for metric, value in expected.items():
                    self.assertEqual(score[metric], value)
                self.assertGreaterEqual(score["fit_seconds"], 0)
                self.assertGreaterEqual(score["predict_seconds"], 0)
            for metric in ("f1_macro", "accuracy", "precision_macro", "recall_macro"):
                self.assertEqual(row[f"mean_{metric}"], float(np.mean([r[metric] for r in scores])))
            self.assertEqual(row["std_f1_macro"], float(np.std([r["f1_macro"] for r in scores], ddof=0)))
            self.assertEqual(row["cv_seconds_sum"],
                             sum(r["fit_seconds"] + r["predict_seconds"] for r in scores))
            self.assertIn("excludes_shared_scaler", row["timing_scope"])

    def test_dt_full_grid_matches_reference_training_scores_and_progress(self):
        raw, y, groups, folds = synthetic_training()
        messages = []
        result = upgrade.evaluate_dt_cv(raw, y, groups, folds, batch_size=5,
                                        labels=LABELS, progress=messages.append)
        expected = sklearn_oof(raw, y, folds, upgrade.dt_candidates(), "dt", 5)
        self.assertEqual(len(result["candidates"]), 11)
        self.assertEqual(len(result["fold_scores"]), 33)
        for cid, predicted in result["oof"].items():
            np.testing.assert_array_equal(predicted, expected[cid], err_msg=cid)
            self.assertEqual(predicted.dtype, np.dtype("int64"))
        for score in result["fold_scores"]:
            candidate = next(c for c in upgrade.dt_candidates() if c["candidate_id"] == score["candidate_id"])
            train, _ = folds[score["fold"]]
            model = Pipeline([("scaler", StandardScaler()), ("dt", DecisionTreeClassifier(
                **candidate["params"], random_state=42))]).fit(raw[train], y[train])
            train_score = classification_metrics(y[train], model.predict(raw[train]), labels=LABELS)
            self.assertEqual(score["train_f1_macro"], train_score["f1_macro"])
            self.assertEqual(score["predict_seconds"],
                             score["valid_predict_seconds"] + score["train_predict_seconds"])
        self.assertEqual(sum(message.endswith("complete") for message in messages), 33)
        self.assert_aggregation(result, y, folds)

    def test_scalers_fit_only_training_rows_frozen_indices_unchanged_and_float_cast(self):
        raw, y, groups, folds = synthetic_training()
        # Extreme validation values make fitting on the full matrix detectable.
        raw[folds[0][1], 0] += 1000
        original_raw = raw.copy()
        original_folds = [(train.copy(), valid.copy()) for train, valid in folds]
        for evaluate in (upgrade.evaluate_knn_cv, upgrade.evaluate_dt_cv):
            observed = []
            original_fit = StandardScaler.fit

            def inspect_fit(scaler, features, *args, **kwargs):
                observed.append(features.copy())
                return original_fit(scaler, features, *args, **kwargs)

            with patch.object(StandardScaler, "fit", inspect_fit):
                result = evaluate(raw.astype(np.float64), y, groups, folds, smoke=True, labels=LABELS)
            self.assertEqual(len(observed), 3)
            for fold, ((train, valid), actual) in enumerate(zip(folds, observed)):
                np.testing.assert_array_equal(actual, raw[train])
                self.assertEqual(actual.dtype, np.dtype("float32"))
                expected_scaler = StandardScaler().fit(raw[train])
                diagnostic = result["transforms"][fold]
                self.assertEqual(diagnostic["fold"], fold)
                self.assertEqual(diagnostic["n_train"], len(train))
                self.assertEqual(diagnostic["n_valid"], len(valid))
                self.assertEqual(diagnostic["scaler_fit_samples"], len(train))
                self.assertEqual(diagnostic["mean_checksum"], sha256(expected_scaler.mean_.tobytes()).hexdigest())
                self.assertFalse(set(groups[train]) & set(groups[valid]))
            self.assertEqual(len(result["oof"]), 1)
        np.testing.assert_array_equal(raw, original_raw)
        for expected, actual in zip(original_folds, folds):
            np.testing.assert_array_equal(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])

    def test_invalid_inputs_and_crossing_tracks_fail_before_fitting(self):
        raw, y, groups, folds = synthetic_training()
        wrong_groups = groups.copy()
        wrong_groups[folds[0][0][0]] = groups[folds[0][1][0]]
        crossed = [(np.r_[train, valid[0]], valid) if i == 0 else (train, valid)
                   for i, (train, valid) in enumerate(folds)]
        invalid = [
            (raw[:, :323], y, groups, folds, {}),
            (raw.reshape(-1), y, groups, folds, {}),
            (raw.astype(np.int32), y, groups, folds, {}),
            (raw[:-1], y, groups, folds, {}),
            (raw, y.astype(float), groups, folds, {}),
            (raw, y, groups[:-1], folds, {}),
            (raw, y, wrong_groups, folds, {}),
            (raw, y, groups, folds[:2], {}),
            (raw, y, groups, crossed, {}),
            (raw, y, groups, [(t.astype(float), v) for t, v in folds], {}),
            (raw, y, groups, [folds[0]] * 3, {}),
            (raw, y, groups, folds, {"batch_size": 0}),
            (raw, y, groups, folds, {"batch_size": True}),
            (raw, y, groups, folds, {"progress": "not callable"}),
            (raw, y, groups, folds, {"labels": (0, 1, 2)}),
        ]
        for value in (np.nan, np.inf):
            broken = raw.copy()
            broken[0, 0] = value
            invalid.append((broken, y, groups, folds, {}))
        for evaluate in (upgrade.evaluate_knn_cv, upgrade.evaluate_dt_cv):
            for matrix, target, tracks, split, kwargs in invalid:
                options = {"smoke": True, "labels": LABELS, **kwargs}
                with self.subTest(algorithm=evaluate.__name__, shape=matrix.shape, options=options), \
                     patch.object(StandardScaler, "fit", side_effect=AssertionError("fit before validation")), \
                     self.assertRaises(ValueError):
                    evaluate(matrix, target, tracks, split, **options)
        # Complete three folds but fewer than 31 training rows rejects the full grid.
        small_assignment = np.tile(np.repeat(np.arange(3), 4), 2)
        small_folds = [(np.flatnonzero(small_assignment != f), np.flatnonzero(small_assignment == f))
                       for f in range(3)]
        with self.assertRaisesRegex(ValueError, "k exceeds"):
            upgrade.evaluate_knn_cv(raw[:24], np.repeat([0, 1], 12),
                                   np.arange(24), small_folds, labels=LABELS)

    def test_selection_finite_stable_and_refit_pipeline_parity(self):
        rows = [{"candidate_id": "bad", "mean_f1_macro": np.nan},
                {"candidate_id": "knn_baseline", "mean_f1_macro": 0.8},
                {"candidate_id": "tied", "mean_f1_macro": 0.8},
                {"candidate_id": "infinite", "mean_f1_macro": np.inf}]
        self.assertIs(upgrade.select_best({"candidates": rows}), rows[1])
        rows[2]["mean_f1_macro"] = 0.9
        self.assertIs(upgrade.select_best({"candidates": rows}), rows[2])
        for candidates in ([], [rows[0]], [rows[3]]):
            with self.assertRaises(ValueError):
                upgrade.select_best({"candidates": candidates})
        raw, y, _, _ = synthetic_training()
        for algorithm, candidate in (("knn", upgrade.knn_candidates()[0]),
                                     ("dt", upgrade.dt_candidates()[0])):
            model = upgrade.refit_model(raw.astype(float), y, algorithm, candidate["params"])
            self.assertIsInstance(model, Pipeline)
            self.assertEqual(list(model.named_steps), ["scaler", algorithm])
            self.assertEqual(model.named_steps["scaler"].n_samples_seen_, len(y))
            expected_scaler = StandardScaler().fit(raw)
            np.testing.assert_array_equal(model.named_steps["scaler"].mean_, expected_scaler.mean_)
            estimator = (KNeighborsClassifier(**candidate["params"], algorithm="brute", n_jobs=1)
                         if algorithm == "knn" else
                         DecisionTreeClassifier(**candidate["params"], random_state=42))
            reference = Pipeline([("scaler", StandardScaler()), (algorithm, estimator)]).fit(raw, y)
            np.testing.assert_array_equal(model.predict(raw), reference.predict(raw))
            if algorithm == "knn":
                self.assertEqual(model.named_steps[algorithm].algorithm, "brute")
                self.assertEqual(model.named_steps[algorithm].n_jobs, 1)
            else:
                self.assertEqual(model.named_steps[algorithm].random_state, 42)
        with self.assertRaises(ValueError):
            upgrade.refit_model(raw, y, "invalid", {})
        with self.assertRaises(ValueError):
            upgrade.refit_model(raw[:-1], y, "dt", {})

    def test_no_dataset_access_extraction_serialization_or_global_logging(self):
        raw, y, groups, folds = synthetic_training()
        # Import first so patching I/O checks our API, not dependency import machinery.
        import joblib
        import logging
        import pandas as pd
        from src import features, preprocessing

        forbidden = AssertionError("Dataset/feature/output side effect")
        with patch("builtins.open", side_effect=forbidden), \
             patch.object(pd, "read_csv", side_effect=forbidden), \
             patch.object(preprocessing, "load_split", side_effect=forbidden), \
             patch.object(preprocessing, "load_and_preprocess", side_effect=forbidden), \
             patch.object(features, "extract_hog", side_effect=forbidden), \
             patch.object(features, "extract_pixel", side_effect=forbidden), \
             patch.object(joblib, "dump", side_effect=forbidden), \
             patch.object(logging, "basicConfig", side_effect=forbidden):
            for evaluate in (upgrade.evaluate_knn_cv, upgrade.evaluate_dt_cv):
                result = evaluate(raw, y, groups, folds, smoke=True, labels=LABELS)
                chosen = upgrade.select_best(result)
                algorithm = "knn" if evaluate is upgrade.evaluate_knn_cv else "dt"
                upgrade.refit_model(raw, y, algorithm, chosen["params"])


if __name__ == "__main__":
    unittest.main()
