"""Export saved KNN evidence only; never load models/images or fit/predict.

Re-run with the same sources updates only namespaced derived outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('pixel', 'hog', 'hog_pca')
METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_summary(frame):
    required = {'CSVRow', 'Path', 'ClassId', 'PredictedClassId'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Missing/empty prediction columns')
    if (frame.CSVRow.duplicated().any() or frame.Path.duplicated().any()
            or not frame.CSVRow.eq(np.arange(len(frame))).all()
            or not frame.ClassId.isin(range(43)).all()
            or not frame.PredictedClassId.isin(range(43)).all()):
        raise ValueError('Invalid prediction alignment/labels')
    y, predicted = frame.ClassId, frame.PredictedClassId
    p, r, f, support = precision_recall_fscore_support(
        y, predicted, labels=range(43), zero_division=0)
    classes = pd.DataFrame({'class_id': range(43), 'support': support,
                            'precision': p, 'recall': r, 'f1': f})
    classes['correct'] = [int(((y == c) & (predicted == c)).sum()) for c in range(43)]
    metrics = dict(zip(METRICS, [accuracy_score(y, predicted), p.mean(), r.mean(), f.mean()]))
    errors = (frame.loc[y != predicted].groupby(['ClassId', 'PredictedClassId'])
              .size().reset_index(name='count')
              .sort_values(['count', 'ClassId', 'PredictedClassId'], ascending=[False, True, True]))
    return metrics, classes, errors


def plot_comparison(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Two series: fixed blue/orange identity, shape and direct labels as backup.
    surface, ink, grid = '#fcfcfb', '#0b0b0b', '#e1e0d9'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'text.color': ink,
                         'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink}):
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=surface)
        ax.set_facecolor(surface)
        positions = np.arange(len(summary))
        ax.errorbar(summary.cv_f1_macro_mean * 100, positions - .12,
                    xerr=summary.cv_f1_macro_std * 100, fmt='o', markersize=8,
                    color='#2a78d6', markeredgecolor=surface, markeredgewidth=1.5,
                    linewidth=1.5, capsize=4, label='Train grouped CV mean ± SD (3 folds)')
        ax.scatter(summary.test_f1_macro * 100, positions + .12, marker='D', s=64,
                   color='#eb6834', edgecolor=surface, linewidth=1.5,
                   label='Official test (frozen models)')
        for i, row in enumerate(summary.itertuples()):
            ax.text(row.cv_f1_macro_mean * 100 + 4, i - .12,
                    f'{row.cv_f1_macro_mean * 100:.2f}', va='center', fontsize=10)
            ax.text(row.test_f1_macro * 100 + 1.5, i + .12,
                    f'{row.test_f1_macro * 100:.2f}', va='center', fontsize=10)
        ax.set(yticks=positions, yticklabels=['Pixel', 'HOG (CV winner)', 'HOG + PCA'],
               xlim=(0, 100), ylim=(2.6, -.6), xlabel='Macro-F1 (%)',
               title='KNN: representation comparison')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=grid, linewidth=.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.10), frameon=False, fontsize=9)
        fig.text(.02, .02, 'Saved artifacts only • SD is fold variability, not a confidence interval',
                 fontsize=9, color='#52514e')
        fig.tight_layout(rect=(0, .05, 1, .94))
        fig.savefig(path, dpi=160, facecolor=surface)
        plt.close(fig)


def export(tune_dir, test_dir, project_root=ROOT):
    tune_dir, test_dir = Path(tune_dir).resolve(), Path(test_dir).resolve()
    sources = []

    def load(path, csv=False):
        sources.append(path)
        return pd.read_csv(path) if csv else json.loads(path.read_text(encoding='utf-8'))

    selection = load(tune_dir / 'selection.json')
    test = load(test_dir / 'test_metrics.json')
    timing = load(tune_dir / 'timings.json')
    claim = load(tune_dir / 'official_test_claim.json')
    if (selection['status'] != 'frozen' or len(selection['methods']) != 3
            or [s['method'] for s in selection['methods']] != list(METHODS)
            or test['selection_sha256'] != sha256(tune_dir / 'selection.json')
            or claim['selection_sha256'] != test['selection_sha256']
            or claim['evaluation_run'] != test_dir.name
            or test['winner_fixed_by_cv'] != selection['winner']):
        raise ValueError('Frozen selection/test provenance mismatch')
    summary, candidates, times, classes_out, errors_out = [], [], [], [], []
    alignment = None
    for item in selection['methods']:
        method = item['method']
        cv = load(tune_dir / f'cv_{method}.csv', csv=True)
        best = cv.iloc[item['candidate_index']]
        if (not np.isclose(best.mean_test_f1_macro, item['cv_f1_macro'])
                or item['candidate_index'] != int(cv.mean_test_f1_macro.to_numpy().argmax())
                or any(best[f'param_{key}'] != value for key, value in item['params'].items())):
            raise ValueError('Saved CV selection mismatch')
        predictions = load(test_dir / f'predictions_{method}.csv', csv=True)
        keys = predictions[['CSVRow', 'Path', 'ClassId']]
        if alignment is not None and not keys.equals(alignment):
            raise ValueError('Methods do not share the same test rows')
        alignment = keys
        observed, classes, errors = prediction_summary(predictions)
        for key in METRICS:
            if not np.isclose(observed[key], test['methods'][method][key], atol=1e-12, rtol=0):
                raise ValueError(f'Saved metric mismatch: {method}/{key}')
        row = {'method': method, 'train_images': item['train_images'], 'test_images': len(predictions),
               'k': item['params']['knn__n_neighbors'], 'weights': item['params']['knn__weights'],
               'metric': item['params']['knn__metric'],
               'refit_dimensions': item['pca_refit_dimensions'] or (3072 if method == 'pixel' else 324)}
        for key in METRICS:
            row[f'cv_{key}_mean'] = best[f'mean_test_{key}']
            row[f'cv_{key}_std'] = best[f'std_test_{key}']
            row[f'test_{key}'] = observed[key]
        row['test_errors'] = int((predictions.ClassId != predictions.PredictedClassId).sum())
        summary.append(row)
        candidates.append(cv.assign(method=method, candidate_index=np.arange(len(cv))))
        times.append({'method': method, **timing['methods'][method],
                      'test_feature_incremental_seconds': test['methods'][method]['feature_seconds'],
                      'test_predict_seconds': test['methods'][method]['predict_seconds']})
        classes_out.append(classes.assign(method=method))
        errors_out.append(errors.assign(method=method))
    if max(summary, key=lambda r: r['cv_f1_macro_mean'])['method'] != selection['winner']:
        raise ValueError('Winner differs from CV')
    tables, figures, logs = [Path(project_root) / p for p in
                             ('report/tables', 'report/figures', 'logs/knn')]
    for folder in (tables, figures, logs):
        folder.mkdir(parents=True, exist_ok=True)
    outputs = {
        'knn_summary.csv': pd.DataFrame(summary),
        'knn_candidates.csv': pd.concat(candidates, ignore_index=True),
        'knn_timings.csv': pd.DataFrame(times),
        'knn_class_summary.csv': pd.concat(classes_out, ignore_index=True),
        'knn_error_pairs.csv': pd.concat(errors_out, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / name, index=False, encoding='utf-8')
    plot_comparison(outputs['knn_summary.csv'], figures / 'knn_cv_test.png')
    for run in (tune_dir, test_dir):
        source = run / 'preprocessing.log'
        if source.exists():
            sources.append(source)
            shutil.copyfile(source, logs / f'knn_{run.name}_preprocessing.log')
    provenance = {'operation': 'export_saved_artifacts_no_fit_no_predict',
                  'tune_dir': str(tune_dir), 'test_dir': str(test_dir),
                  'sources': {str(p): sha256(p) for p in sources},
                  'shared_load_train_seconds': timing['load_train_seconds'],
                  'shared_extract_hog_train_seconds': timing['extract_hog_seconds'],
                  'shared_load_test_seconds': test['load_test_seconds'],
                  'timing_warning': 'HOG train feature time is shared, not twice. HOG+PCA test reuses HOG. Historical predict_seconds includes batch prediction, pipeline transforms, label validation and metrics computation; excludes decode/ROI/resize, raw feature extraction and bundle loading. It is not pure prediction time.'}
    (logs / 'knn_export_provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'tables': list(outputs), 'figure': str(figures / 'knn_cv_test.png'),
                      'verified_prediction_rows_per_method': len(alignment),
                      'winner_unchanged': selection['winner']}, indent=2))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tune-dir', type=Path, default=ROOT / 'report/knn/full-knn-01')
    parser.add_argument('--test-dir', type=Path,
                        default=ROOT / 'report/knn/20261113T194939Z-test-adf82d0d')
    args = parser.parse_args()
    export(args.tune_dir, args.test_dir)


if __name__ == '__main__':
    main()
