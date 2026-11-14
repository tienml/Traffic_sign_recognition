"""Separate K-NN protocol: preflight -> smoke -> tune -> explicit frozen test.

No official test images/labels are read by preflight, smoke or tune.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import IMAGE_SIZE, RANDOM_STATE, USE_ROI
from src.features import HOG_DEFAULTS, PCA_VARIANCE, extract_hog, extract_pixel
from src.models import (KNN_LABELS, KNN_METHODS, batch_predict,
                        classification_metrics, grouped_folds, load_knn_bundle,
                        save_knn_bundle, track_groups, tune_grouped_knn)
from src.preprocessing import CLASS_NAMES, ROI_COLUMNS, load_image, load_split

PACKAGES = ('numpy', 'pandas', 'opencv-python', 'scikit-image', 'scikit-learn',
            'joblib', 'scipy', 'threadpoolctl', 'tqdm')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def resolve_dataset_dir(explicit=None, project_root=ROOT):
    def valid(path):
        return all((path / name).is_file() for name in ('Train.csv', 'Test.csv', 'Meta.csv'))
    if explicit is not None:
        path = Path(explicit).resolve()
        if not valid(path):
            raise ValueError('dataset-dir must directly contain Train.csv/Test.csv/Meta.csv')
        return path
    choices = [p.resolve() for p in (project_root / 'data/archive', project_root / 'data') if valid(p)]
    if len(choices) != 1:
        raise ValueError('Dataset absent or ambiguous; specify --dataset-dir explicitly')
    return choices[0]


def metadata(root, split):
    frame = pd.read_csv(root / f'{split}.csv')
    required = {'Path', 'ClassId', 'Width', 'Height', *ROI_COLUMNS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f'Invalid {split} metadata columns/empty data')
    labels = pd.to_numeric(frame['ClassId'], errors='coerce')
    if (not labels.isin(KNN_LABELS).all() or frame['Path'].isna().any()
            or frame['Path'].duplicated().any()):
        raise ValueError('Invalid labels or missing/duplicate paths')
    frame['ClassId'] = labels.astype(np.int64)
    if not np.array_equal(np.unique(frame['ClassId']), KNN_LABELS):
        raise ValueError('Expected all 43 classes')
    resolved = []
    for relative in frame['Path']:
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing image or path outside dataset: {relative}')
        resolved.append(str(path).casefold())
    if len(set(resolved)) != len(resolved):
        raise ValueError('Aliased image paths')
    frame.insert(0, 'CSVRow', np.arange(len(frame)))
    return frame


def smoke_subset(frame, groups, tracks_per_class=3):
    if tracks_per_class < 3:
        raise ValueError('Smoke needs at least three whole tracks per class')
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []
    for label in KNN_LABELS:
        available = np.unique(groups[frame['ClassId'].to_numpy() == label])
        if len(available) < tracks_per_class:
            raise ValueError('Not enough tracks for smoke')
        selected.extend(rng.choice(available, tracks_per_class, replace=False))
    mask = np.isin(groups, selected)
    return frame.loc[mask].copy(), groups[mask]


def versions():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None  # tqdm is an optional loader/feature dependency.
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': packages}


def new_run(mode, run_id=None):
    run_id = run_id or f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{mode}-{uuid.uuid4().hex[:8]}'
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('run-id may contain only letters, numbers, hyphen and underscore')
    output = ROOT / 'report/knn' / run_id
    model_dir = ROOT / 'data/models/knn' / run_id
    if output.exists() or model_dir.exists():
        raise FileExistsError('Run already exists; no output may be overwritten')
    output.mkdir(parents=True)
    if mode in ('smoke', 'tune'):
        model_dir.mkdir(parents=True)
    return run_id, output, model_dir


def isolate_loader_logging(output):
    # Use the logger's public handler API; do not alter preprocessing globals.
    logger = logging.getLogger('src.preprocessing')
    if not any(h.name == 'preprocessing.file' for h in logger.handlers):
        log_dir = ROOT / 'logs/knn' / output.name
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / 'preprocessing.log', mode='x', encoding='utf-8')
        handler.set_name('preprocessing.file')
        logger.addHandler(handler)


def save_folds(output, frame, groups, folds):
    manifest = frame[['CSVRow', 'Path', 'ClassId']].copy()
    manifest['Group'] = groups
    assignment = np.full(len(frame), -1)
    stats = []
    labels = frame['ClassId'].to_numpy()
    for fold, (train, valid) in enumerate(folds):
        assignment[valid] = fold
        for side, indices in (('train', train), ('validation', valid)):
            stats.append({'fold': fold, 'side': side, 'images': len(indices),
                          'tracks': len(np.unique(groups[indices])),
                          'class_images': np.bincount(labels[indices], minlength=43).tolist(),
                          'class_tracks': [len(np.unique(groups[indices][labels[indices] == c]))
                                           for c in KNN_LABELS]})
    manifest['ValidationFold'] = assignment
    with (output / 'fold_manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        manifest.to_csv(stream, index=False)
    write_json(output / 'fold_statistics.json', stats)


def load_train_images(root, frame, smoke):
    if not smoke:
        images, y = load_split(root / 'Train.csv', root, use_roi=USE_ROI)
    else:
        images = np.empty((len(frame), IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.float32)
        for i, row in enumerate(frame.itertuples(index=False)):
            # Dotted ROI column names are renamed by itertuples: use the frame here.
            images[i] = load_image(root / row.Path,
                                   frame.iloc[i][list(ROI_COLUMNS)].to_numpy(), use_roi=USE_ROI)
        y = frame['ClassId'].to_numpy(copy=True)
    track_groups(frame, y)
    return images, y


def run_training(args, root, train, groups, output, model_dir, config):
    smoke = args.mode == 'smoke'
    if smoke:
        train, groups = smoke_subset(train, groups, args.smoke_tracks_per_class)
    y = train['ClassId'].to_numpy()
    folds = grouped_folds(y, groups)
    save_folds(output, train, groups, folds)
    started = time.perf_counter()
    images, y = load_train_images(root, train, smoke)
    timing = {'load_train_seconds': time.perf_counter() - started, 'methods': {}}
    # HOG is stateless and shared; learned transforms remain inside every fold.
    started = time.perf_counter()
    hog = extract_hog(images, **HOG_DEFAULTS)
    timing['extract_hog_seconds'] = time.perf_counter() - started
    selected = []
    for method in KNN_METHODS:
        print(f'[{method}] {len(y)} images, shared 3-fold track CV', flush=True)
        started = time.perf_counter()
        raw = extract_pixel(images) if method == 'pixel' else hog
        extract_seconds = time.perf_counter() - started if method == 'pixel' else timing['extract_hog_seconds']
        started = time.perf_counter()
        grid = tune_grouped_knn(raw, y, groups, folds, method, smoke=smoke,
                                batch_size=args.batch_size, verbose=args.verbose)
        elapsed = time.perf_counter() - started
        results = pd.DataFrame(grid.cv_results_)
        with (output / f'cv_{method}.csv').open('x', newline='', encoding='utf-8') as stream:
            results.to_csv(stream, index=False)
        best = grid.best_index_
        pca = grid.best_estimator_.named_steps.get('pca')
        selection = {'method': method, 'candidate_index': int(best),
                     'params': grid.best_params_,
                     'cv_f1_macro': float(grid.cv_results_['mean_test_f1_macro'][best]),
                     'cv_f1_std': float(grid.cv_results_['std_test_f1_macro'][best]),
                     'cv_accuracy': float(grid.cv_results_['mean_test_accuracy'][best]),
                     'pca_refit_dimensions': None if pca is None else int(pca.n_components_),
                     'model_file': f'{method}.joblib', 'train_images': len(y)}
        save_knn_bundle(model_dir / selection['model_file'], grid.best_estimator_, method,
                        {'config': config, 'selection': selection, 'versions': config['versions']})
        selection['model_sha256'] = digest(model_dir / selection['model_file'])
        selected.append(selection)
        timing['methods'][method] = {'feature_seconds': extract_seconds,
                                     'search_and_refit_seconds': elapsed,
                                     'refit_seconds': float(grid.refit_time_),
                                     'candidates': len(results), 'folds': len(folds)}
        print(f'[{method}] search+refit {elapsed:.3f}s; CV macro-F1 {selection["cv_f1_macro"]:.6f}', flush=True)
        del grid, raw
    winner = max(range(len(selected)), key=lambda i: selected[i]['cv_f1_macro'])
    frozen = {'schema_version': 1, 'status': 'smoke_only' if smoke else 'frozen',
              'winner': selected[winner]['method'], 'methods': selected,
              'tie_break': 'first ParameterGrid candidate, then pixel/hog/hog_pca',
              'train_sha256': config['train_sha256'], 'dataset_root': str(root),
              'model_dir': str(model_dir.resolve()), 'run_id': config['run_id'],
              'test_previously_explored': True,
              'warning': 'Earlier EDA inspected a test subset; do not describe test as never seen.'}
    write_json(output / 'timings.json', timing)
    write_json(output / 'selection.json', frozen)  # Last: incomplete runs are not frozen.
    print(json.dumps({'selection': str(output / 'selection.json'), 'timings': timing}, indent=2), flush=True)


def run_test(args, root, train, output, config):
    if not args.selection:
        raise ValueError('test requires --selection from a completed full tune run')
    selection_path = Path(args.selection).resolve()
    frozen = json.loads(selection_path.read_text(encoding='utf-8'))
    if (frozen.get('schema_version') != 1 or frozen.get('status') != 'frozen'
            or frozen.get('train_sha256') != digest(root / 'Train.csv')
            or frozen.get('dataset_root') != str(root)
            or [item['method'] for item in frozen.get('methods', [])] != list(KNN_METHODS)
            or any(item['train_images'] != len(train) for item in frozen['methods'])):
        raise ValueError('Selection is not a compatible full-train frozen selection')
    # An exclusive claim prevents accidental repeated official evaluation, even on failure.
    claim = selection_path.parent / 'official_test_claim.json'
    write_json(claim, {'evaluation_run': config['run_id'], 'selection_sha256': digest(selection_path)})
    test = metadata(root, 'Test')  # First access to official test labels.
    if set(train['Path']) & set(test['Path']):
        raise ValueError('Train/test paths overlap')
    started = time.perf_counter()
    images, y = load_split(root / 'Test.csv', root, use_roi=USE_ROI)
    if not np.array_equal(y, test['ClassId'].to_numpy()):
        raise ValueError('Test loader labels do not align with CSV')
    metrics = {'selection_sha256': digest(selection_path), 'winner_fixed_by_cv': frozen['winner'],
               'test_sha256': digest(root / 'Test.csv'),
               'load_test_seconds': time.perf_counter() - started, 'methods': {}}
    hog = None
    for item in frozen['methods']:
        method = item['method']
        model_path = Path(frozen['model_dir']) / item['model_file']
        if digest(model_path) != item['model_sha256']:
            raise ValueError('Model hash does not match frozen selection')
        bundle = load_knn_bundle(model_path)
        if (bundle['method'] != method
                or bundle['metadata']['selection']['params'] != item['params']
                or bundle['metadata']['config']['train_sha256'] != frozen['train_sha256']):
            raise ValueError('Bundle metadata differs from frozen selection')
        started = time.perf_counter()
        if method == 'pixel':
            raw = extract_pixel(images)
        else:
            if hog is None:
                hog = extract_hog(images, **HOG_DEFAULTS)
            raw = hog
        feature_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = batch_predict(bundle['estimator'], raw, args.batch_size)
        if len(predicted) != len(test) or not np.isin(predicted, KNN_LABELS).all():
            raise ValueError('Invalid predictions')
        metrics['methods'][method] = {**classification_metrics(y, predicted),
                                      'feature_seconds': feature_seconds,
                                      'predict_seconds': time.perf_counter() - started}
        predictions = test[['CSVRow', 'Path', 'ClassId']].copy()
        predictions['PredictedClassId'] = predicted
        with (output / f'predictions_{method}.csv').open('x', newline='', encoding='utf-8') as stream:
            predictions.to_csv(stream, index=False)
        del bundle, raw
    write_json(output / 'test_metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'smoke', 'tune', 'test'))
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--smoke-tracks-per-class', type=int, default=3)
    parser.add_argument('--selection', type=Path, help='Trusted local full-tune selection.json')
    parser.add_argument('--verbose', type=int, default=3)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if IMAGE_SIZE != (32, 32) or USE_ROI is not True:
        raise ValueError('Approved contract requires 32x32 RGB with ROI')
    root = resolve_dataset_dir(args.dataset_dir)
    train = metadata(root, 'Train')
    groups = track_groups(train)
    folds = grouped_folds(train['ClassId'].to_numpy(), groups)
    run_id, output, model_dir = new_run(args.mode, args.run_id)
    isolate_loader_logging(output)
    config = {'run_id': run_id, 'mode': args.mode, 'dataset_root': str(root),
              'train_sha256': digest(root / 'Train.csv'), 'seed': RANDOM_STATE,
              'versions': versions(), 'batch_size': args.batch_size,
              'n_jobs': 1, 'numeric_threads': 2, 'hog_params': HOG_DEFAULTS,
              'pca_variance': PCA_VARIANCE, 'class_names': CLASS_NAMES,
              'image_contract': {'size': list(IMAGE_SIZE), 'color': 'RGB', 'roi': USE_ROI,
                                 'scale': 'divide255', 'augmentation': False}}
    write_json(output / 'config.json', config)
    print(f'Run output: {output}', flush=True)
    with threadpool_limits(limits=2):
        if args.mode == 'preflight':
            save_folds(output, train, groups, folds)
            summary = {'train_images': len(train), 'classes': len(KNN_LABELS),
                       'tracks': len(np.unique(groups)), 'folds': 3,
                       'candidates_per_method': 12, 'cv_fits_full': 108,
                       'official_test_read': False, 'versions': config['versions']}
            write_json(output / 'preflight.json', summary)
            print(json.dumps(summary, indent=2), flush=True)
        elif args.mode == 'test':
            run_test(args, root, train, output, config)
        else:
            run_training(args, root, train, groups, output, model_dir, config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
