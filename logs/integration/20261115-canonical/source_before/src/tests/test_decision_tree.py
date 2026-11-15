"""Synthetic-only DT tests; official images/test never used."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from src.decision_tree import (dt_candidates, dt_pipeline, frozen_folds,
    smoke_from_frozen, tune_grouped_dt, save_dt_bundle, load_dt_bundle)
from src.models import grouped_folds, track_groups, BatchScorer, batch_predict, first_best_index
from scripts.train_knn import save_folds
from scripts import train_dt
from scripts.summarize_dt import export


class DecisionTreeTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {'Path': f'Train/{c}/{c:05d}_{t:05d}_{f:05d}.png', 'ClassId': c}
            for c in range(43) for t in range(6) for f in range(2)])
        self.frame.insert(0, 'CSVRow', np.arange(len(self.frame)))
        self.groups = track_groups(self.frame)
        self.folds = grouped_folds(self.frame.ClassId.to_numpy(), self.groups)

    def test_grid_defaults_and_tie_break(self):
        grid = list(ParameterGrid(dt_candidates()))
        self.assertEqual(len(grid), 16)
        self.assertEqual(len(list(ParameterGrid(dt_candidates(True)))), 1)
        self.assertEqual(first_best_index({'mean_test_f1_macro': [.5, .5]}), 0)
        model = dt_pipeline('pixel').named_steps['dt']
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.max_features, None)
        self.assertEqual(model.ccp_alpha, 0)
        self.assertEqual(model.min_samples_leaf, 1)
        with self.assertRaises(ValueError):
            dt_pipeline('invalid')

    def test_exact_saved_folds_and_whole_track_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_folds(root, self.frame, self.groups, self.folds)
            groups, folds = frozen_folds(self.frame, root / 'fold_manifest.csv')
            for expected, actual in zip(self.folds, folds):
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)
            subset, small_groups, small_folds = smoke_from_frozen(self.frame, groups, folds)
            self.assertEqual(len(np.unique(small_groups)), 129)
            for g in np.unique(small_groups):
                self.assertEqual(sum(small_groups == g), sum(groups == g))
            assignment = {self.frame.iloc[i].Path: f for f, (_, valid) in enumerate(folds) for i in valid}
            for f, (train, valid) in enumerate(small_folds):
                self.assertFalse(set(small_groups[train]) & set(small_groups[valid]))
                self.assertTrue(all(assignment[subset.iloc[i].Path] == f for i in valid))
            broken = self.frame.copy()
            broken.loc[0, 'CSVRow'] = 999
            with self.assertRaises(ValueError):
                frozen_folds(broken, root / 'fold_manifest.csv')
            manifest = pd.read_csv(root / 'fold_manifest.csv')
            manifest.loc[0, 'ValidationFold'] = (manifest.loc[0, 'ValidationFold'] + 1) % 3
            manifest.to_csv(root / 'broken.csv', index=False)
            with self.assertRaises(ValueError):
                frozen_folds(self.frame, root / 'broken.csv')

    def test_fold_local_transforms_and_roundtrip_all_methods(self):
        rng = np.random.default_rng(42)
        y = np.repeat([0, 1], 12)
        groups = np.array([f'{c}:{i // 2}' for c in (0, 1) for i in range(12)])
        folds = grouped_folds(y, groups, labels=(0, 1))
        raw = rng.random((24, 8), dtype=np.float32)
        with threadpool_limits(limits=2):
            for method in ('pixel', 'hog', 'hog_pca'):
                observed = []
                original = BatchScorer.__call__
                def inspect(scorer, estimator, x, target):
                    if method != 'pixel':
                        observed.append(estimator.named_steps['scaler'].mean_.copy())
                    return original(scorer, estimator, x, target)
                with patch.object(BatchScorer, '__call__', inspect):
                    grid = tune_grouped_dt(raw, y, groups, folds, method,
                                           smoke=True, labels=(0, 1), verbose=0)
                for mean, (train, _) in zip(observed, folds):
                    np.testing.assert_allclose(mean, raw[train].mean(axis=0), atol=1e-7)
                model = grid.best_estimator_
                if method != 'pixel':
                    before = model.named_steps['scaler'].mean_.copy()
                    model.predict(raw * 1000)
                    np.testing.assert_array_equal(before, model.named_steps['scaler'].mean_)
                    np.testing.assert_allclose(before, raw.mean(axis=0), atol=1e-7)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'dt.joblib'
                    save_dt_bundle(path, model, method, {})
                    loaded = load_dt_bundle(path)
                    np.testing.assert_array_equal(batch_predict(model, raw, 5),
                                                  batch_predict(loaded['estimator'], raw, 7))
                    with self.assertRaises(FileExistsError):
                        save_dt_bundle(path, model, method, {})
        with self.assertRaises(ValueError):
            tune_grouped_dt(raw[:-1], y, groups, folds, 'pixel', labels=(0, 1))

    def test_training_selection_is_json_serializable_and_smoke_only(self):
        rng = np.random.default_rng(42)
        images = rng.random((258, 32, 32, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=2):
            root = Path(tmp)
            output, models = root / 'report', root / 'models'
            output.mkdir()
            models.mkdir()
            with patch.object(train_dt, 'load_train_images',
                              return_value=(images, np.repeat(np.arange(43), 6))), \
                 patch.object(train_dt, 'extract_hog', return_value=images.reshape(258, -1)[:, :8]), \
                 patch.object(train_dt, 'extract_pixel', return_value=images.reshape(258, -1)[:, :8]):
                train_dt.train_run(argparse.Namespace(mode='smoke', batch_size=32, verbose=0),
                    root, self.frame, self.groups, self.folds, output, models,
                    {'train_sha256': 'synthetic', 'fold_manifest_sha256': 'synthetic', 'run_id': 'test'})
            selection = json.loads((output / 'selection.json').read_text())
            self.assertEqual(selection['status'], 'smoke_only')
            self.assertEqual(len(selection['methods']), 3)
            for item in selection['methods']:
                self.assertIsInstance(item['tree_leaves'], int)

    def test_smoke_gate_before_official_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'status': 'smoke_only'}), encoding='utf-8')
            with patch.object(train_dt, 'metadata', side_effect=AssertionError('test read')):
                with self.assertRaises(ValueError):
                    train_dt.test_run(argparse.Namespace(selection=selection), root,
                                      pd.DataFrame(), root, {})

    def test_extraction_contract_drift_rejected_before_test_claim(self):
        # Changing HOG's norm preserves its dimensions but changes its semantics.
        for field, changed in (
            ('hog_params', {**train_dt.HOG_DEFAULTS, 'block_norm': 'L1'}),
            ('image_contract', {'size': [32, 32], 'color': 'RGB', 'roi': False,
                                'scale': 'divide255'}),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = {'train_sha256': 'train', 'fold_manifest_sha256': 'fold',
                          'hog_params': dict(train_dt.HOG_DEFAULTS),
                          'image_contract': {'size': [32, 32], 'color': 'RGB',
                                             'roi': True, 'scale': 'divide255'}}
                old_config = {**config, field: changed}
                methods = [{'method': method, 'params': {}, 'train_images': 39209,
                            'cv_f1_macro': .5, 'model_file': method + '.joblib',
                            'model_sha256': 'model'} for method in train_dt.DT_METHODS]
                selection = root / 'selection.json'
                selection.write_text(json.dumps({
                    'schema_version': 1, 'algorithm': 'decision_tree', 'status': 'frozen',
                    'train_sha256': 'train', 'fold_manifest_sha256': 'fold',
                    'dataset_root': str(root), 'methods': methods, 'winner': 'pixel',
                    'model_dir': str(root)}), encoding='utf-8')
                bundle = {'method': 'pixel', 'metadata': {
                    'config': old_config, 'selection': {'params': {}}}}
                with patch.object(train_dt, 'digest', return_value='model'), \
                     patch.object(train_dt, 'load_dt_bundle', return_value=bundle), \
                     patch.object(train_dt, 'metadata') as metadata_mock, \
                     patch.object(train_dt, 'load_split') as load_mock:
                    with self.assertRaisesRegex(ValueError, 'Bundle provenance mismatch'):
                        train_dt.test_run(argparse.Namespace(selection=selection), root,
                                          pd.DataFrame(index=range(39209)), root, config)
                    metadata_mock.assert_not_called()
                    load_mock.assert_not_called()
                self.assertFalse((root / 'official_test_claim.json').exists())

    def test_new_run_exclusive_and_namespace_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / 'report/knn/full-knn-01/selection.json'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'preserve KNN')
            with patch.object(train_dt, 'ROOT', root):
                _, output, models, logs = train_dt.new_run('smoke', 'one')
                self.assertTrue(models.is_dir())
                self.assertTrue(logs.is_dir())
                self.assertEqual(output, root / 'report/dt/one')
                with self.assertRaises(FileExistsError):
                    train_dt.new_run('smoke', 'one')
                with self.assertRaises(ValueError):
                    train_dt.new_run('smoke', '../bad')
            self.assertEqual(sentinel.read_bytes(), b'preserve KNN')

    def test_export_saved_only_and_no_overwrite(self):
        rng = np.random.default_rng(42)
        raw = rng.random((len(self.frame), 4), dtype=np.float32)
        y = self.frame.ClassId.to_numpy()
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(limits=2):
            root = Path(tmp)
            run = root / 'run'
            run.mkdir()
            selected = []
            for method in ('pixel', 'hog', 'hog_pca'):
                grid = tune_grouped_dt(raw, y, self.groups, self.folds, method, smoke=True, verbose=0)
                pd.DataFrame(grid.cv_results_).to_csv(run / f'cv_{method}.csv', index=False)
                selected.append({'method': method, 'params': grid.best_params_, 'candidate_index': 0,
                                 'cv_f1_macro': float(grid.cv_results_['mean_test_f1_macro'][0]),
                                 'cv_f1_std': float(grid.cv_results_['std_test_f1_macro'][0])})
            (run / 'selection.json').write_text(json.dumps({
                'algorithm': 'decision_tree', 'status': 'smoke_only', 'methods': selected,
                'winner': max(selected, key=lambda r: r['cv_f1_macro'])['method']}), encoding='utf-8')
            (run / 'timings.json').write_text(json.dumps({'methods': {s['method']: {} for s in selected}}))
            with patch('scripts.summarize_dt.plot_comparison') as plot:
                result = export(run, project_root=root)
                self.assertEqual(len(result['candidates']), 3)
                plot.assert_called_once()
                with self.assertRaises(FileExistsError):
                    export(run, project_root=root)
                with self.assertRaises(ValueError):
                    export(run, test_dir=root, project_root=root)


if __name__ == '__main__':
    unittest.main()
