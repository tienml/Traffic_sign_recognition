"""Decision Tree: preflight/smoke/tune never open official Test.csv; test is explicit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.train_knn import (digest, write_json, resolve_dataset_dir, metadata,
                               versions, save_folds, load_train_images)
from src.config import IMAGE_SIZE, USE_ROI, RANDOM_STATE
from src.features import extract_pixel, extract_hog, HOG_DEFAULTS, PCA_VARIANCE
from src.models import batch_predict, classification_metrics
from src.preprocessing import load_split
from src.decision_tree import (DT_METHODS, frozen_folds, smoke_from_frozen,
                               tune_grouped_dt, save_dt_bundle, load_dt_bundle)


def new_run(mode, run_id=None):
    run_id = run_id or f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{mode}-{uuid.uuid4().hex[:8]}'
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Invalid run-id')
    output, models = ROOT / 'report/dt' / run_id, ROOT / 'data/models/dt' / run_id
    logs = ROOT / 'logs/dt' / run_id
    if any(p.exists() for p in (output, models, logs)):
        raise FileExistsError('Run exists; no overwrite permitted')
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    if mode in ('smoke', 'tune'):
        models.mkdir(parents=True)
    return run_id, output, models, logs


def train_run(args, root, train, groups, folds, output, models, config):
    smoke = args.mode == 'smoke'
    if smoke:
        train, groups, folds = smoke_from_frozen(train, groups, folds)
    save_folds(output, train, groups, folds)
    started = time.perf_counter()
    images, y = load_train_images(root, train, smoke)
    timing = {'load_train_seconds': time.perf_counter() - started, 'methods': {}}
    started = time.perf_counter()
    hog = extract_hog(images, **HOG_DEFAULTS)
    timing['extract_hog_seconds'] = time.perf_counter() - started
    selected = []
    for method in DT_METHODS:
        started = time.perf_counter()
        raw = extract_pixel(images) if method == 'pixel' else hog
        feature_time = time.perf_counter() - started if method == 'pixel' else timing['extract_hog_seconds']
        started = time.perf_counter()
        grid = tune_grouped_dt(raw, y, groups, folds, method, smoke=smoke,
                               batch_size=args.batch_size, verbose=args.verbose)
        elapsed = time.perf_counter() - started
        with (output / f'cv_{method}.csv').open('x', newline='', encoding='utf-8') as stream:
            pd.DataFrame(grid.cv_results_).to_csv(stream, index=False)
        best = grid.best_index_
        pca = grid.best_estimator_.named_steps.get('pca')
        tree = grid.best_estimator_.named_steps['dt']
        item = {'method': method, 'candidate_index': int(best), 'params': grid.best_params_,
                'cv_f1_macro': float(grid.cv_results_['mean_test_f1_macro'][best]),
                'cv_f1_std': float(grid.cv_results_['std_test_f1_macro'][best]),
                'cv_accuracy': float(grid.cv_results_['mean_test_accuracy'][best]),
                'pca_refit_dimensions': None if pca is None else int(pca.n_components_),
                'tree_depth': int(tree.get_depth()), 'tree_nodes': int(tree.tree_.node_count),
                'tree_leaves': int(tree.get_n_leaves()), 'train_images': len(y),
                'model_file': f'{method}.joblib'}
        save_dt_bundle(models / item['model_file'], grid.best_estimator_, method,
                       {'config': config, 'selection': item})
        item['model_sha256'] = digest(models / item['model_file'])
        selected.append(item)
        timing['methods'][method] = {'feature_seconds': feature_time,
                                    'search_and_refit_seconds': elapsed,
                                    'refit_seconds': float(grid.refit_time_),
                                    'candidates': len(grid.cv_results_['params']), 'folds': 3}
        print(f'{method}: {elapsed:.3f}s, CV macro-F1={item["cv_f1_macro"]:.6f}', flush=True)
        del grid, raw
    write_json(output / 'timings.json', timing)
    write_json(output / 'selection.json', {
        'schema_version': 1, 'algorithm': 'decision_tree',
        'status': 'smoke_only' if smoke else 'frozen', 'methods': selected,
        'winner': max(selected, key=lambda s: s['cv_f1_macro'])['method'],
        'tie_break': 'first ParameterGrid candidate then pixel/hog/hog_pca',
        'dataset_root': str(root), 'train_sha256': config['train_sha256'],
        'fold_manifest_sha256': config['fold_manifest_sha256'],
        'model_dir': str(models.resolve()), 'run_id': config['run_id'],
        'test_previously_explored': True})


def test_run(args, root, train, output, config):
    if not args.selection:
        raise ValueError('test requires a trusted local full-tune --selection')
    selection_path = Path(args.selection).resolve()
    frozen = json.loads(selection_path.read_text(encoding='utf-8'))
    if (frozen.get('status') != 'frozen' or frozen.get('algorithm') != 'decision_tree'
            or frozen.get('schema_version') != 1
            or frozen.get('train_sha256') != config['train_sha256']
            or frozen.get('fold_manifest_sha256') != config['fold_manifest_sha256']
            or frozen.get('dataset_root') != str(root)
            or [s['method'] for s in frozen.get('methods', [])] != list(DT_METHODS)
            or any(s['train_images'] != 39209 for s in frozen['methods'])
            or len(train) != 39209):
        raise ValueError('Not a compatible full-train frozen DT selection')
    if frozen['winner'] != max(frozen['methods'], key=lambda s: s['cv_f1_macro'])['method']:
        raise ValueError('Winner differs from CV')
    # Check all bundles before touching test; no refit or parameter selection here.
    bundles = {}
    for item in frozen['methods']:
        path = Path(frozen['model_dir']) / item['model_file']
        if digest(path) != item['model_sha256']:
            raise ValueError('Model hash mismatch')
        bundle = load_dt_bundle(path)
        if (bundle['method'] != item['method']
                or bundle['metadata']['selection']['params'] != item['params']
                or bundle['metadata']['config']['train_sha256'] != config['train_sha256']
                or bundle['metadata']['config'].get('hog_params') != config['hog_params']
                or bundle['metadata']['config'].get('image_contract') != config['image_contract']):
            raise ValueError('Bundle provenance mismatch')
        bundles[item['method']] = bundle
    write_json(selection_path.parent / 'official_test_claim.json', {
        'evaluation_run': config['run_id'], 'selection_sha256': digest(selection_path)})
    test = metadata(root, 'Test')  # First official test label access.
    if len(test) != 12630 or set(train.Path) & set(test.Path):
        raise ValueError('Invalid official test size/overlap')
    started = time.perf_counter()
    images, y = load_split(root / 'Test.csv', root, use_roi=USE_ROI)
    if not np.array_equal(y, test.ClassId.to_numpy()):
        raise ValueError('Test row alignment mismatch')
    result = {'selection_sha256': digest(selection_path), 'test_sha256': digest(root / 'Test.csv'),
              'winner_fixed_by_cv': frozen['winner'],
              'load_test_seconds': time.perf_counter() - started, 'methods': {}}
    hog = None
    for method in DT_METHODS:
        started = time.perf_counter()
        if method == 'pixel':
            raw = extract_pixel(images)
        else:
            if hog is None:
                hog = extract_hog(images, **HOG_DEFAULTS)
            raw = hog
        feature_time = time.perf_counter() - started
        started = time.perf_counter()
        predicted = batch_predict(bundles[method]['estimator'], raw, args.batch_size)
        predict_time = time.perf_counter() - started
        if len(predicted) != len(y) or not np.isin(predicted, range(43)).all():
            raise ValueError('Invalid predictions')
        result['methods'][method] = {**classification_metrics(y, predicted),
                                     'feature_seconds': feature_time, 'predict_seconds': predict_time}
        frame = test[['CSVRow', 'Path', 'ClassId']].assign(PredictedClassId=predicted)
        with (output / f'predictions_{method}.csv').open('x', newline='', encoding='utf-8') as stream:
            frame.to_csv(stream, index=False)
    write_json(output / 'test_metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'smoke', 'tune', 'test'))
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--fold-manifest', type=Path, default=ROOT / 'report/knn/full-knn-01/fold_manifest.csv')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--verbose', type=int, default=3)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if IMAGE_SIZE != (32, 32) or USE_ROI is not True or RANDOM_STATE != 42:
        raise ValueError('Image/seed contract changed')
    root = resolve_dataset_dir(args.dataset_dir)
    train = metadata(root, 'Train')
    source_config = args.fold_manifest.parent / 'config.json'
    knn_config = json.loads(source_config.read_text(encoding='utf-8'))
    if (knn_config['train_sha256'] != digest(root / 'Train.csv')
            or knn_config['mode'] != 'tune' or knn_config['seed'] != 42 or len(train) != 39209):
        raise ValueError('Frozen full KNN manifest provenance mismatch')
    groups, folds = frozen_folds(train, args.fold_manifest)
    run_id, output, models, logs = new_run(args.mode, args.run_id)
    config = {'run_id': run_id, 'mode': args.mode, 'algorithm': 'decision_tree',
              'train_sha256': digest(root / 'Train.csv'), 'dataset_root': str(root),
              'fold_manifest': str(args.fold_manifest.resolve()),
              'fold_manifest_sha256': digest(args.fold_manifest),
              'source_config_sha256': digest(source_config), 'versions': versions(),
              'seed': 42, 'n_jobs': 1, 'numeric_threads': 2, 'batch_size': args.batch_size,
              'hog_params': HOG_DEFAULTS, 'pca_variance': PCA_VARIANCE,
              'image_contract': {'size': [32, 32], 'color': 'RGB', 'roi': True, 'scale': 'divide255'},
              'source_sha256': {p: digest(ROOT / p) for p in (
                  'scripts/train_dt.py', 'src/decision_tree.py', 'scripts/train_knn.py',
                  'src/models.py', 'src/features.py', 'src/preprocessing.py', 'src/config.py')}}
    write_json(output / 'config.json', config)
    logger = logging.getLogger('src.preprocessing')
    handler = logging.FileHandler(logs / 'preprocessing.log', mode='x', encoding='utf-8')
    handler.set_name('preprocessing.file')
    logger.addHandler(handler)
    try:
        with (logs / 'console.log').open('x', encoding='utf-8') as console:
            class Tee:
                def write(self, text):
                    sys.__stdout__.write(text)
                    console.write(text)
                    console.flush()
                def flush(self):
                    sys.__stdout__.flush()
                    console.flush()
            from contextlib import redirect_stdout
            with redirect_stdout(Tee()), threadpool_limits(limits=2):
                print(f'Run output: {output}', flush=True)
                if args.mode == 'preflight':
                    save_folds(output, train, groups, folds)
                    write_json(output / 'preflight.json', {
                        'train_images': len(train), 'tracks': len(np.unique(groups)),
                        'folds': 3, 'candidates_per_method': 16, 'cv_fits_full': 144,
                        'official_test_read': False, 'full_train_float32_MiB': 39209 * 3072 * 4 / 2**20})
                elif args.mode == 'test':
                    test_run(args, root, train, output, config)
                else:
                    train_run(args, root, train, groups, folds, output, models, config)
    finally:
        logger.removeHandler(handler)
        handler.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
