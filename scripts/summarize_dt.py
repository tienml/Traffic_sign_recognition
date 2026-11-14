"""Export DT saved evidence only, with optional official test; never fit/predict."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts.train_knn import digest, write_json
from scripts.summarize_knn import prediction_summary, METRICS
from src.decision_tree import DT_METHODS, dt_candidates
from sklearn.model_selection import ParameterGrid


def plot_comparison(summary, path, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Monochrome: identity encoded with marker + legend, not categorical colors.
    with plt.rc_context({'font.family': 'DejaVu Sans', 'text.color': '#171717'}):
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='#ffffff')
        positions = np.arange(len(summary))
        ax.errorbar(summary.cv_f1_macro * 100, positions - .10,
                    xerr=summary.cv_f1_std * 100, fmt='o', markersize=8,
                    color='#171717', markeredgecolor='white', markeredgewidth=2,
                    capsize=4, label='Train grouped CV mean ± SD')
        for i, row in enumerate(summary.itertuples()):
            ax.annotate(f'{row.cv_f1_macro * 100:.2f}',
                        (row.cv_f1_macro * 100, i - .10), xytext=(0, -18),
                        textcoords='offset points', ha='center', color='#171717')
        if 'test_f1_macro' in summary:
            ax.scatter(summary.test_f1_macro * 100, positions + .12,
                       marker='D', s=64, facecolors='white', edgecolors='#171717',
                       linewidths=2, label='Official test; selection frozen by CV')
        ax.set(xlim=(0, 100), ylim=(2.6, -.6), yticks=positions,
               yticklabels=summary.method, xlabel='Macro-F1 (%)',
               title='Decision Tree — ' + ('TRAIN-ONLY SMOKE (not final)' if smoke else 'frozen CV selection'))
        ax.grid(axis='x', color='#e5e5e5', linewidth=.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.25), frameon=False, fontsize=9)
        fig.text(.02, .02, 'Saved artifacts only; SD is fold variability, not a confidence interval. See CSV for values.', fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .95))
        fig.savefig(path, dpi=160)
        plt.close(fig)


def export(tune_dir, test_dir=None, project_root=ROOT, export_id=None):
    tune_dir = Path(tune_dir).resolve()
    sources = []
    def load(path, csv=False):
        sources.append(path)
        return pd.read_csv(path) if csv else json.loads(path.read_text(encoding='utf-8'))
    selection = load(tune_dir / 'selection.json')
    timing = load(tune_dir / 'timings.json')
    smoke = selection.get('status') == 'smoke_only'
    if (selection.get('algorithm') != 'decision_tree'
            or selection.get('status') not in ('frozen', 'smoke_only')
            or [s['method'] for s in selection['methods']] != list(DT_METHODS)):
        raise ValueError('Invalid DT selection')
    test = None
    if test_dir is not None:
        if smoke:
            raise ValueError('Smoke cannot be paired with official test')
        test_dir = Path(test_dir).resolve()
        test = load(test_dir / 'test_metrics.json')
        claim = load(tune_dir / 'official_test_claim.json')
        if (test['selection_sha256'] != digest(tune_dir / 'selection.json')
                or claim['selection_sha256'] != test['selection_sha256']
                or claim['evaluation_run'] != test_dir.name
                or test['winner_fixed_by_cv'] != selection['winner']):
            raise ValueError('Selection/test provenance mismatch')
    summaries, candidates, times, classes, errors = [], [], [], [], []
    alignment = None
    for item in selection['methods']:
        method = item['method']
        cv = load(tune_dir / f'cv_{method}.csv', csv=True)
        expected = list(ParameterGrid(dt_candidates(smoke)))
        if len(cv) != len(expected) or not np.isfinite(cv.mean_test_f1_macro).all():
            raise ValueError('Incomplete CV grid')
        for i, params in enumerate(expected):
            for key, value in params.items():
                observed = cv.iloc[i][f'param_{key}']
                if not (pd.isna(observed) if value is None else observed == value):
                    raise ValueError('Candidate grid mismatch')
        best_index = int(cv.mean_test_f1_macro.to_numpy().argmax())
        if (item['candidate_index'] != best_index or item['params'] != expected[best_index]
                or not np.isclose(cv.iloc[best_index].mean_test_f1_macro, item['cv_f1_macro'], atol=1e-12, rtol=0)):
            raise ValueError('Selection differs from CV')
        summary = {**item, 'status': selection['status']}
        summary.pop('params')
        summary.update(item['params'])
        candidates.append(cv.assign(method=method, candidate_index=np.arange(len(cv))))
        times.append({'method': method, **timing['methods'][method]})
        if test is not None:
            predictions = load(test_dir / f'predictions_{method}.csv', csv=True)
            keys = predictions[['CSVRow', 'Path', 'ClassId']]
            if len(predictions) != 12630 or (alignment is not None and not keys.equals(alignment)):
                raise ValueError('Invalid official prediction alignment/count')
            alignment = keys
            metrics, per_class, pairs = prediction_summary(predictions)
            for metric in METRICS:
                if not np.isclose(metrics[metric], test['methods'][method][metric], atol=1e-12, rtol=0):
                    raise ValueError('Saved metric mismatch')
                summary[f'test_{metric}'] = metrics[metric]
            classes.append(per_class.assign(method=method))
            errors.append(pairs.assign(method=method))
            times[-1].update({f'test_{k}': v for k, v in test['methods'][method].items() if k.endswith('_seconds')})
        summaries.append(summary)
    if selection['winner'] != max(summaries, key=lambda s: s['cv_f1_macro'])['method']:
        raise ValueError('Winner differs from CV')
    export_id = export_id or tune_dir.name
    if not export_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in export_id):
        raise ValueError('Invalid export-id')
    root = Path(project_root)
    tables, figures = root / 'report/tables', root / 'report/figures'
    logs = root / 'logs/dt'
    outputs = {'summary': pd.DataFrame(summaries), 'candidates': pd.concat(candidates),
               'timings': pd.DataFrame(times)}
    if test is not None:
        outputs.update(class_summary=pd.concat(classes), error_pairs=pd.concat(errors))
    paths = [tables / f'dt_{export_id}_{name}.csv' for name in outputs]
    figure = figures / f'dt_{export_id}_cv_test.png'
    provenance = logs / f'dt_{export_id}_export_provenance.json'
    if any(p.exists() for p in paths + [figure, provenance]):
        raise FileExistsError('Export exists; choose a fresh --export-id')
    for folder in (tables, figures, logs):
        folder.mkdir(parents=True, exist_ok=True)
    for path, frame in zip(paths, outputs.values()):
        with path.open('x', newline='', encoding='utf-8') as stream:
            frame.to_csv(stream, index=False)
    plot_comparison(outputs['summary'], figure, smoke)
    write_json(provenance, {'operation': 'saved_artifacts_no_fit_no_predict',
                           'sources': {str(p): digest(p) for p in sources},
                           'official_test_included': test is not None,
                           'timing_warning': 'HOG train time is shared. Search includes refit. Predict time excludes metrics.'})
    print(json.dumps({'tables': [str(p) for p in paths], 'figure': str(figure)}, indent=2))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tune-dir', required=True, type=Path)
    parser.add_argument('--test-dir', type=Path)
    parser.add_argument('--export-id')
    args = parser.parse_args()
    export(args.tune_dir, args.test_dir, export_id=args.export_id)


if __name__ == '__main__':
    main()
