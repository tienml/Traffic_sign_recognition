"""Additive Decision Tree protocol; shared public grouped-CV/metric contracts."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.config import RANDOM_STATE
from src.features import PCA_VARIANCE, METHODS
from src.models import (KNN_LABELS, BatchScorer, first_best_index, track_groups,
                        validate_folds)

DT_METHODS = METHODS
DT_LABELS = KNN_LABELS


def float32(values):
    return np.asarray(values, dtype=np.float32)


def dt_pipeline(method):
    if method not in DT_METHODS:
        raise ValueError('Unknown representation')
    steps = []
    if method != 'pixel':
        steps.append(('scaler', StandardScaler()))
    if method == 'hog_pca':
        steps.extend([('pca', PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)),
                      ('float32', FunctionTransformer(float32))])
    return Pipeline(steps + [('dt', DecisionTreeClassifier(random_state=RANDOM_STATE))])


def dt_candidates(smoke=False):
    return {'dt__criterion': ['gini'] if smoke else ['gini', 'entropy'],
            'dt__max_depth': [10] if smoke else [None, 10, 20, 30],
            'dt__min_samples_split': [2] if smoke else [2, 5]}


def frozen_folds(frame, manifest_path, labels=DT_LABELS):
    """Consume, not regenerate, the exact KNN row assignments. Fail closed."""
    manifest = pd.read_csv(manifest_path)
    groups = track_groups(frame)
    keys = ['CSVRow', 'Path', 'ClassId']
    if (not set(keys + ['Group', 'ValidationFold']).issubset(manifest)
            or not manifest[keys].equals(frame[keys].reset_index(drop=True))
            or not np.array_equal(manifest.Group.to_numpy(), groups)
            or not manifest.ValidationFold.isin([0, 1, 2]).all()
            or set(manifest.ValidationFold) != {0, 1, 2}):
        raise ValueError('Frozen manifest is not aligned with Train.csv')
    assignment = manifest.ValidationFold.to_numpy()
    folds = [(np.flatnonzero(assignment != i), np.flatnonzero(assignment == i)) for i in range(3)]
    validate_folds(frame.ClassId.to_numpy(), groups, folds, labels)
    return groups, folds


def smoke_from_frozen(frame, groups, folds):
    """One complete track/class/frozen validation fold; retain original assignment."""
    assignment = np.empty(len(frame), dtype=int)
    for i, (_, valid) in enumerate(folds):
        assignment[valid] = i
    chosen = []
    rng = np.random.default_rng(RANDOM_STATE)
    for c in DT_LABELS:
        for i in range(3):
            available = np.unique(groups[(frame.ClassId.to_numpy() == c) & (assignment == i)])
            if not len(available):
                raise ValueError('Smoke requires a track per class per frozen fold')
            chosen.append(rng.choice(available))
    mask = np.isin(groups, chosen)
    subset = frame.loc[mask].copy().reset_index(drop=True)
    subset_assignment = assignment[mask]
    subset_folds = [(np.flatnonzero(subset_assignment != i),
                     np.flatnonzero(subset_assignment == i)) for i in range(3)]
    validate_folds(subset.ClassId.to_numpy(), groups[mask], subset_folds)
    return subset, groups[mask], subset_folds


def tune_grouped_dt(features, y, groups, folds, method, *, smoke=False,
                    batch_size=128, verbose=3, labels=DT_LABELS):
    validate_folds(y, groups, folds, labels)
    if features.ndim != 2 or len(features) != len(y) or not np.isfinite(features).all():
        raise ValueError('Invalid raw feature matrix')
    grid = GridSearchCV(dt_pipeline(method), dt_candidates(smoke), cv=folds,
                        scoring=BatchScorer(batch_size, labels), refit=first_best_index,
                        n_jobs=1, error_score='raise', verbose=verbose)
    with threadpool_limits(limits=2):
        grid.fit(features, y)
    return grid


def save_dt_bundle(path, estimator, method, metadata):
    if method not in DT_METHODS:
        raise ValueError('Unknown representation')
    with Path(path).open('xb') as stream:
        joblib.dump({'schema_version': 1, 'algorithm': 'decision_tree', 'method': method,
                     'estimator': estimator, 'metadata': metadata,
                     'raw_dimensions': int(estimator.n_features_in_),
                     'labels': list(DT_LABELS)}, stream, compress=3)


def load_dt_bundle(path):
    # Trusted local joblib only: deserialization can execute code.
    bundle = joblib.load(path)
    if (bundle.get('schema_version') != 1 or bundle.get('algorithm') != 'decision_tree'
            or bundle.get('method') not in DT_METHODS or bundle.get('labels') != list(DT_LABELS)
            or bundle.get('raw_dimensions') != bundle['estimator'].n_features_in_
            or not isinstance(bundle['estimator'].named_steps.get('dt'), DecisionTreeClassifier)):
        raise ValueError('Incompatible Decision Tree bundle')
    return bundle
