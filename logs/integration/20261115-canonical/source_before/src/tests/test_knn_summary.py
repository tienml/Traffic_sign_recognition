"""Regression tests for saved-evidence exporter; no model inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.summarize_knn import prediction_summary, export, ROOT
from scripts.train_knn import isolate_loader_logging


class SummaryTests(unittest.TestCase):
    def test_metrics_keep_all_43_labels(self):
        frame = pd.DataFrame({'CSVRow': [0, 1], 'Path': ['a', 'b'],
                              'ClassId': [0, 1], 'PredictedClassId': [0, 0]})
        metrics, classes, errors = prediction_summary(frame)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertAlmostEqual(metrics['recall_macro'], 1 / 43)
        self.assertEqual(len(classes), 43)
        self.assertEqual(int(errors.iloc[0]['count']), 1)
        for column, value in [('CSVRow', 0), ('Path', 'a'), ('ClassId', 43),
                              ('PredictedClassId', -1)]:
            invalid = frame.copy()
            invalid.loc[1, column] = value
            with self.assertRaises(ValueError):
                prediction_summary(invalid)

    def test_existing_artifacts_export_without_mutation(self):
        tune = ROOT / 'report/knn/full-knn-01'
        test = ROOT / 'report/knn/20261113T194939Z-test-adf82d0d'
        if not (tune / 'selection.json').exists():
            self.skipTest('Local saved artifacts unavailable')
        before = (tune / 'selection.json').read_bytes()
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.summarize_knn.plot_comparison'):
            results = export(tune, test, Path(tmp))
            self.assertEqual(len(results['knn_candidates.csv']), 36)
            self.assertEqual(len(results['knn_class_summary.csv']), 129)
            self.assertEqual(results['knn_summary.csv'].test_images.tolist(), [12630] * 3)
            self.assertTrue((Path(tmp) / 'logs/knn/knn_export_provenance.json').is_file())
        self.assertEqual(before, (tune / 'selection.json').read_bytes())

    def test_future_loader_log_is_under_logs(self):
        import logging
        logger = logging.getLogger('src.preprocessing')
        prior = logger.handlers[:]
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.train_knn.ROOT', Path(tmp)):
            logger.handlers = []
            try:
                isolate_loader_logging(Path(tmp) / 'report/knn/new-run')
                self.assertTrue((Path(tmp) / 'logs/knn/new-run/preprocessing.log').is_file())
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = prior


if __name__ == '__main__':
    unittest.main()
