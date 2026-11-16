"""Kiểm thử trích xuất đặc trưng trên dữ liệu tổng hợp nhỏ, không cần GTSRB."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import features as feat
from src.config import RANDOM_STATE


class FeatureExtractionTests(unittest.TestCase):
    """Kiểm tra shape, tính tất định và nguyên tắc chỉ fit trên train."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.train = self.rng.random((12, 32, 32, 3), dtype=np.float32)
        self.test = self.rng.random((5, 32, 32, 3), dtype=np.float32)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.log_path = Path(self.temporary.name) / "logs" / "features.log"
        self.logger = feat.logger
        saved = (self.logger.level, self.logger.propagate, list(self.logger.handlers))
        self.addCleanup(self._restore_logger, saved)
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

    def _restore_logger(self, saved) -> None:
        level, propagate, handlers = saved
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
        self.logger.setLevel(level)
        self.logger.propagate = propagate
        for handler in handlers:
            self.logger.addHandler(handler)

    def test_pixel_matches_flatten_and_has_3072_dims(self) -> None:
        X_train, X_test = feat.build_features(
            self.train, self.test, method="pixel"
        )
        self.assertEqual(X_train.shape, (12, 32 * 32 * 3))
        self.assertEqual(X_test.shape, (5, 32 * 32 * 3))
        np.testing.assert_array_equal(
            X_train[0], self.train[0].reshape(-1)
        )

    def test_hog_shape_and_determinism(self) -> None:
        first, _ = feat.build_features(self.train, self.test, method="hog")
        second, _ = feat.build_features(self.train, self.test, method="hog")
        self.assertEqual(first.shape, (12, 324))
        np.testing.assert_array_equal(first, second)

    def test_test_rows_do_not_depend_on_batch_composition(self) -> None:
        extractor = feat.FeatureExtractor("hog_pca").fit(self.train)
        batched = extractor.transform(self.test)
        single = extractor.transform(self.test[:1])
        np.testing.assert_allclose(single, batched[:1], rtol=1e-5, atol=1e-6)

    def test_scaler_and_pca_are_fitted_on_train_only(self) -> None:
        extractor = feat.FeatureExtractor("hog_pca").fit(self.train)
        train_mean = extractor.scaler_.mean_.copy()
        train_components = extractor.pca_.components_.copy()
        extractor.transform(self.test * 0.0)
        np.testing.assert_array_equal(extractor.scaler_.mean_, train_mean)
        np.testing.assert_array_equal(
            extractor.pca_.components_, train_components
        )

    def test_pca_keeps_requested_variance_and_shrinks_dimensions(self) -> None:
        extractor = feat.FeatureExtractor(
            "hog_pca", pca_variance=0.90
        ).fit(self.train)
        self.assertLess(extractor.n_features_, 324)
        self.assertGreaterEqual(extractor.explained_variance, 0.90)
        self.assertEqual(extractor.n_features_, extractor.n_components_pca)

    def test_hog_pca_reduces_dimensions_versus_hog(self) -> None:
        hog_feat, _ = feat.build_features(self.train, self.test, method="hog")
        pca_feat, _ = feat.build_features(
            self.train, self.test, method="hog_pca"
        )
        self.assertLess(pca_feat.shape[1], hog_feat.shape[1])

    def test_outputs_are_finite_float32(self) -> None:
        for method in feat.METHODS:
            X_train, X_test = feat.build_features(
                self.train, self.test, method=method
            )
            for array in (X_train, X_test):
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(np.isfinite(array).all())

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            feat.build_features(self.train, self.test, method="sift")

    def test_invalid_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            feat.build_features(self.train[0], self.test, method="pixel")
        with self.assertRaises(ValueError):
            feat.build_features(
                self.train[..., :2], self.test, method="pixel"
            )
        with self.assertRaises(ValueError):
            feat.build_features(
                self.train.astype(np.uint8), self.test, method="pixel"
            )

    def test_transform_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            feat.FeatureExtractor("hog").transform(self.test)

    def test_configure_logging_writes_file_without_touching_global_logging(self) -> None:
        with patch.object(feat, "logger", self.logger):
            feat.configure_logging(self.log_path)
            self.assertTrue(any(
                isinstance(h, feat.logging.FileHandler)
                for h in self.logger.handlers
            ))
        self.assertFalse(feat.logging.getLogger().handlers)

    def test_extract_hog_uses_defaults_when_no_params_given(self) -> None:
        """Gọi thẳng extract_hog không tham số phải dùng HOG_DEFAULTS, không crash."""
        features = feat.extract_hog(self.train)
        self.assertEqual(features.shape, (len(self.train), 324))
        explicit = feat.extract_hog(self.train, **feat.HOG_DEFAULTS)
        np.testing.assert_array_equal(features, explicit)

    def test_extract_hog_merges_partial_overrides(self) -> None:
        """Chỉ đổi một tham số vẫn giữ mặc định cho các tham số còn lại."""
        # 32×32 → 4×4 ô; block 2×2 → 3×3 block; 9 block × 4 ô × 12 hướng.
        features = feat.extract_hog(self.train, orientations=12)
        self.assertEqual(features.shape, (len(self.train), 9 * 4 * 12))
        explicit = feat.extract_hog(
            self.train,
            **{**feat.HOG_DEFAULTS, "orientations": 12},
        )
        np.testing.assert_array_equal(features, explicit)

    def test_pca_variance_curve_is_cumulative_and_capped(self) -> None:
        cumulative, pca = feat.pca_variance_curve(
            self.train, max_components=8
        )
        self.assertLessEqual(len(cumulative), 8)
        self.assertTrue(np.all(np.diff(cumulative) >= -1e-12))
        self.assertAlmostEqual(cumulative[-1], pca.explained_variance_ratio_.sum())


if __name__ == "__main__":
    unittest.main()
