"""Person 3 tests: synthetic data only, no official GTSRB images."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from scripts.train_knn import resolve_dataset_dir, smoke_subset, new_run, run_test
import argparse
import json
from src.features import FeatureExtractor, extract_hog, extract_pixel
from src.models import (KNN_LABELS, KNN_METHODS, batch_predict, classification_metrics,
                        first_best_index, grouped_folds, knn_candidates, knn_pipeline,
                        load_knn_bundle, save_knn_bundle, track_groups,
                        tune_grouped_knn, validate_folds)


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {'ClassId': c, 'Path': f'Train/{c}/{c:05d}_{t:05d}_{f:05d}.png'}
            for c in KNN_LABELS for t in range(6) for f in range(2)])
        self.y = self.frame.ClassId.to_numpy()
        self.groups = track_groups(self.frame, self.y)

    def test_group_identity_alignment_and_fail_closed(self):
        self.assertEqual(len(np.unique(self.groups)), 43 * 6)
        self.assertNotEqual(self.groups[0], self.groups[12])
        with self.assertRaises(ValueError):
            track_groups(self.frame, self.y[::-1])
        for path in ('Train/0/bad.png', 'Train/0/00001_00000_00000.png',
                     'Train/1/00000_00000_00000.png', '../00000_00000_00000.png'):
            broken = self.frame.copy()
            broken.loc[0, 'Path'] = path
            with self.assertRaises(ValueError):
                track_groups(broken)
        with self.assertRaises(ValueError):
            track_groups(pd.concat([self.frame, self.frame.iloc[:1]]))

    def test_shared_folds_reproducible_complete_and_disjoint(self):
        folds = grouped_folds(self.y, self.groups)
        again = grouped_folds(self.y, self.groups)
        seen = np.zeros(len(self.y), dtype=int)
        for (train, valid), (train2, valid2) in zip(folds, again):
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(valid, valid2)
            self.assertFalse(set(self.groups[train]) & set(self.groups[valid]))
            self.assertEqual(len(np.unique(self.y[train])), 43)
            self.assertEqual(len(np.unique(self.y[valid])), 43)
            seen[valid] += 1
        np.testing.assert_array_equal(seen, 1)
        with self.assertRaises(ValueError):
            validate_folds(self.y, self.groups, [(folds[0][0], folds[0][0])])
        with self.assertRaises(ValueError):
            grouped_folds(self.y[self.y != 42], self.groups[self.y != 42])
        with self.assertRaises(ValueError):
            grouped_folds(self.y, np.asarray([str(c) for c in self.y]))

    def test_smoke_selects_whole_tracks_all_classes(self):
        subset, groups = smoke_subset(self.frame, self.groups)
        self.assertEqual(len(np.unique(groups)), 43 * 3)
        for group in np.unique(groups):
            self.assertEqual(sum(groups == group), sum(self.groups == group))
        grouped_folds(subset.ClassId.to_numpy(), groups)
        subset2, groups2 = smoke_subset(self.frame, self.groups)
        np.testing.assert_array_equal(groups, groups2)
        self.assertTrue(subset.equals(subset2))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=2)
        cls.limit.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.limit.__exit__(None, None, None)

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.images = self.rng.random((24, 32, 32, 3), dtype=np.float32)
        self.y = np.arange(24) % 2

    def test_person2_equivalence_and_validation_cannot_mutate_fit(self):
        for method in KNN_METHODS:
            raw = extract_pixel(self.images) if method == 'pixel' else extract_hog(self.images)
            model = knn_pipeline(method).fit(raw[:18], self.y[:18])
            reference = FeatureExtractor(method)
            expected = reference.fit_transform(self.images[:18])
            # PCA fit_transform can differ at float rounding from separate transform.
            actual = model[:-1].transform(raw[:18]) if method != 'pixel' else raw[:18]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            if method != 'pixel':
                scaler = model.named_steps['scaler']
                before = scaler.mean_.copy()
                np.testing.assert_allclose(before, raw[:18].mean(axis=0), atol=1e-6)
                model.predict(raw[18:] * 100)
                np.testing.assert_array_equal(before, scaler.mean_)
                changed = raw.copy()
                changed[18:] += 1000
                second = clone(model).fit(changed[:18], self.y[:18])
                np.testing.assert_array_equal(before, second.named_steps['scaler'].mean_)
                if method == 'hog_pca':
                    np.testing.assert_array_equal(model.named_steps['pca'].components_,
                                                  second.named_steps['pca'].components_)

    def test_candidates_ties_metrics_and_batches(self):
        self.assertEqual(len(list(ParameterGrid(knn_candidates()))), 12)
        self.assertEqual(first_best_index({'mean_test_f1_macro': [0.5, 0.5]}), 0)
        with self.assertRaises(ValueError):
            first_best_index({'mean_test_f1_macro': [0.5, np.nan]})
        self.assertAlmostEqual(classification_metrics([0], [0])['f1_macro'], 1 / 43)
        model = knn_pipeline('pixel').fit(extract_pixel(self.images), self.y)
        raw = extract_pixel(self.images)
        np.testing.assert_array_equal(model.predict(raw), batch_predict(model, raw, 5))
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                batch_predict(model, raw, bad)

    def test_grid_fold_local_scaler_and_error_propagation(self):
        y = np.repeat([0, 1], 12)
        groups = np.array([f'{c}:{i // 2}' for c in (0, 1) for i in range(12)])
        folds = grouped_folds(y, groups, labels=(0, 1))
        raw = self.rng.random((24, 8), dtype=np.float32)
        observed = []
        from src.models import BatchScorer
        original = BatchScorer.__call__

        def inspect(scorer, estimator, x, labels):
            observed.append(estimator.named_steps['scaler'].mean_.copy())
            return original(scorer, estimator, x, labels)

        with patch.object(BatchScorer, '__call__', inspect):
            grid = tune_grouped_knn(raw, y, groups, folds, 'hog', smoke=True,
                                    verbose=0, labels=(0, 1))
        self.assertEqual(len(observed), 3)
        for mean, (train, _) in zip(observed, folds):
            np.testing.assert_allclose(mean, raw[train].mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(grid.best_estimator_.named_steps['scaler'].mean_,
                                   raw.mean(axis=0), atol=1e-7)
        self.assertEqual(len(grid.cv_results_['params']), 1)
        with self.assertRaises(ValueError):
            tune_grouped_knn(raw, y, groups, folds, 'hog', n_jobs=2, labels=(0, 1))
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            tune_grouped_knn(raw, y, groups, folds, 'hog', labels=(0, 1))
        with self.assertRaises(ValueError):
            tune_grouped_knn(np.ones((6, 2)), np.array([0, 0, 0, 1, 1, 1]),
                            np.arange(6), [(np.array([0, 3]), np.array([1, 2, 4, 5])),
                                           (np.array([1, 2, 4, 5]), np.array([0, 3]))],
                            'pixel', labels=(0, 1))

    def test_bundle_roundtrip_and_no_overwrite(self):
        for method in KNN_METHODS:
            raw = extract_pixel(self.images) if method == 'pixel' else extract_hog(self.images)
            model = knn_pipeline(method).fit(raw, self.y)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'model.joblib'
                save_knn_bundle(path, model, method, {'selection': 'test'})
                loaded = load_knn_bundle(path)
                self.assertEqual(loaded['raw_dimensions'], raw.shape[1])
                self.assertEqual(len(loaded['class_names']), 43)
                np.testing.assert_array_equal(batch_predict(model, raw, 3),
                                              batch_predict(loaded['estimator'], raw, 7))
                with self.assertRaises(FileExistsError):
                    save_knn_bundle(path, model, method, {})


class LayoutTests(unittest.TestCase):
    def test_smoke_selection_rejected_before_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'schema_version': 1, 'status': 'smoke_only'}))
            args = argparse.Namespace(selection=selection)
            with patch('scripts.train_knn.metadata', side_effect=AssertionError('test read')):
                with self.assertRaises(ValueError):
                    run_test(args, root, pd.DataFrame(), root, {'run_id': 'test'})

    def test_direct_archive_ambiguous_and_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / 'data'
            direct.mkdir()
            for name in ('Train.csv', 'Test.csv', 'Meta.csv'):
                (direct / name).touch()
            self.assertEqual(resolve_dataset_dir(project_root=root), direct.resolve())
            archive = direct / 'archive'
            archive.mkdir()
            for name in ('Train.csv', 'Test.csv', 'Meta.csv'):
                (archive / name).touch()
            with self.assertRaises(ValueError):
                resolve_dataset_dir(project_root=root)
            self.assertEqual(resolve_dataset_dir(direct, root), direct.resolve())
            with self.assertRaises(ValueError):
                new_run('smoke', '../escape')


if __name__ == '__main__':
    unittest.main()
