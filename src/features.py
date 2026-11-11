"""Biểu diễn đặc trưng ảnh GTSRB: baseline pixel, HOG và HOG giảm chiều bằng PCA.

Nguyên tắc chống rò rỉ: scaler và PCA chỉ được ``fit`` trên tập train; tập test
chỉ đi qua ``transform``. Hai bước đó nằm cùng một đối tượng ``FeatureExtractor``
nên không thể vô tình fit lại trên test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import NUM_CHANNELS, RANDOM_STATE

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - môi trường tối thiểu vẫn chạy được
    tqdm = None

METHODS = ("pixel", "hog", "hog_pca")

# Tham số HOG cho ảnh 32×32: 4×4 ô, 3×3 block, 9 hướng → 324 chiều mỗi ảnh.
HOG_DEFAULTS: dict[str, object] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

PCA_VARIANCE = 0.95
PCA_MIN_COMPONENTS = 2
PCA_MAX_COMPONENTS = 256
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "features.log"

logger = logging.getLogger("src.features")


def configure_logging(
    log_path: str | Path | None = LOG_PATH, level: int = logging.INFO
) -> logging.Logger:
    """Gắn console (và file nếu ghi được) cho logger của mô-đun, đúng một lần."""
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_features_console", False) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        console._features_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_path is not None and not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        except OSError:
            # Không ghi được log không được làm hỏng quá trình trích đặc trưng.
            logger.warning("Không mở được file log %s; chỉ ghi ra console.", log_path)
    return logger


def _as_image_batch(images: np.ndarray, name: str = "images") -> np.ndarray:
    """Kiểm tra đầu vào đúng hợp đồng (N, H, W, 3) float của bước tiền xử lý."""
    batch = np.asarray(images)
    if batch.ndim != 4:
        raise ValueError(
            f"{name} phải có shape (N, H, W, C); nhận được {batch.shape}"
        )
    if batch.shape[-1] != NUM_CHANNELS:
        raise ValueError(
            f"{name} phải có {NUM_CHANNELS} kênh màu; nhận được {batch.shape[-1]}"
        )
    if not np.issubdtype(batch.dtype, np.floating):
        raise ValueError(f"{name} phải là kiểu số thực; nhận được {batch.dtype}")
    if len(batch) == 0:
        raise ValueError(f"{name} rỗng")
    return batch


def extract_pixel(images: np.ndarray) -> np.ndarray:
    """Baseline: duỗi mỗi ảnh đã chuẩn hóa [0, 1] thành vector pixel."""
    batch = _as_image_batch(images)
    return np.ascontiguousarray(batch.reshape(len(batch), -1), dtype=np.float32)


def extract_hog(
    images: np.ndarray,
    *,
    orientations: int | None = None,
    pixels_per_cell: tuple[int, int] | None = None,
    cells_per_block: tuple[int, int] | None = None,
    block_norm: str | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Histogram of Oriented Gradients cho từng ảnh, trả về (N, n_features)."""
    batch = _as_image_batch(images)
    params = {
        "orientations": orientations,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block,
        "block_norm": block_norm,
    }
    params = {
        key: HOG_DEFAULTS[key] if value is None else value
        for key, value in params.items()
    }
    iterable = batch
    if progress and tqdm is not None:
        iterable = tqdm(batch, total=len(batch), desc="HOG", leave=False)
    vectors = [
        hog(
            image,
            orientations=params["orientations"],
            pixels_per_cell=params["pixels_per_cell"],
            cells_per_block=params["cells_per_block"],
            block_norm=params["block_norm"],
            channel_axis=-1,
            feature_vector=True,
        )
        for image in iterable
    ]
    if progress and tqdm is not None:
        iterable.close()
    return np.stack(vectors).astype(np.float32, copy=False)


class FeatureExtractor:
    """Chuyển ảnh thành đặc trưng, fit scaler/PCA trên train và chỉ transform test."""

    def __init__(
        self,
        method: str = "pixel",
        *,
        scale: bool = True,
        pca_variance: float = PCA_VARIANCE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if method not in METHODS:
            raise ValueError(
                f"Phương pháp {method!r} không hợp lệ; chọn một trong {METHODS}"
            )
        if method == "pixel":
            scale = False  # baseline giữ pixel thô sau chuẩn hóa [0, 1]
        if not 0.0 < pca_variance <= 1.0:
            raise ValueError("pca_variance phải nằm trong (0, 1]")
        self.method = method
        self.scale = scale
        self.pca_variance = pca_variance
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.n_features_: int | None = None

    def _fit_features(self, images: np.ndarray) -> np.ndarray:
        batch = _as_image_batch(images)
        if self.method == "pixel":
            features = extract_pixel(batch)
            self.n_features_ = features.shape[1]
            return features

        features = extract_hog(batch, **HOG_DEFAULTS)  # type: ignore[arg-type]
        if self.scale:
            self.scaler_ = StandardScaler().fit(features)
            features = self.scaler_.transform(features)
        if self.method == "hog_pca":
            self.pca_ = PCA(
                n_components=self.pca_variance, random_state=self.random_state
            ).fit(features)
            if not (
                PCA_MIN_COMPONENTS
                <= self.pca_.n_components_
                <= PCA_MAX_COMPONENTS
            ):
                logger.warning(
                    "PCA giữ %d thành phần, ngoài khoảng kỳ vọng [%d, %d].",
                    self.pca_.n_components_,
                    PCA_MIN_COMPONENTS,
                    PCA_MAX_COMPONENTS,
                )
            features = self.pca_.transform(features)
        self.n_features_ = features.shape[1]
        return np.asarray(features, dtype=np.float32)

    def fit(self, images: np.ndarray) -> "FeatureExtractor":
        """Học scaler/PCA từ đúng tập truyền vào (chỉ dùng cho train)."""
        self._fit_features(images)
        return self

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """Fit và trả đặc trưng của cùng một batch, không trích HOG hai lần."""
        return self._fit_features(images)

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.n_features_ is None:
            raise RuntimeError("Phải gọi fit() trên tập train trước transform()")
        features = self._fit_free_features(images)
        expected = self.n_features_
        if features.shape[1] != expected:
            raise ValueError(
                f"Số đặc trưng không khớp lúc fit ({expected}): {features.shape[1]}"
            )
        return features

    def _fit_free_features(self, images: np.ndarray) -> np.ndarray:
        batch = _as_image_batch(images)
        if self.method == "pixel":
            return extract_pixel(batch)
        features = extract_hog(batch, **HOG_DEFAULTS)  # type: ignore[arg-type]
        if self.scaler_ is not None:
            features = self.scaler_.transform(features)
        if self.pca_ is not None:
            features = self.pca_.transform(features)
        return np.asarray(features, dtype=np.float32)

    @property
    def n_components_pca(self) -> int | None:
        """Số thành phần PCA thực tế sau khi fit (None nếu không dùng PCA)."""
        return None if self.pca_ is None else int(self.pca_.n_components_)

    @property
    def explained_variance(self) -> float | None:
        """Tổng phương sai được giữ lại khi dùng PCA."""
        if self.pca_ is None:
            return None
        return float(self.pca_.explained_variance_ratio_.sum())


def build_features(
    X_train_img: np.ndarray,
    X_test_img: np.ndarray,
    method: str = "pixel",
    *,
    scale: bool = True,
    pca_variance: float = PCA_VARIANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Trả đặc trưng train/test cho pipeline chính (pixel, hog, hog_pca).

    Scaler và PCA được fit trên ``X_train_img`` rồi áp nguyên trạng lên
    ``X_test_img``.
    """
    return build_features_with_info(
        X_train_img,
        X_test_img,
        method=method,
        scale=scale,
        pca_variance=pca_variance,
    )[:2]


def build_features_with_info(
    X_train_img: np.ndarray,
    X_test_img: np.ndarray,
    method: str = "pixel",
    *,
    scale: bool = True,
    pca_variance: float = PCA_VARIANCE,
) -> tuple[np.ndarray, np.ndarray, FeatureExtractor]:
    """Như ``build_features`` nhưng trả thêm đối tượng đã fit để báo cáo."""
    extractor = FeatureExtractor(
        method, scale=scale, pca_variance=pca_variance
    )
    X_train_feat = extractor.fit_transform(X_train_img)
    X_test_feat = extractor.transform(X_test_img)
    extra = ""
    if extractor.pca_ is not None:
        extra = (
            f" PCA giữ {extractor.n_components_pca} thành phần"
            f" ({extractor.explained_variance:.4f} phương sai)."
        )
    logger.info(
        "%s: train %s, test %s.%s",
        method,
        X_train_feat.shape,
        X_test_feat.shape,
        extra,
    )
    return X_train_feat, X_test_feat, extractor


def pca_variance_curve(
    images: np.ndarray,
    *,
    scale: bool = True,
    max_components: int = PCA_MAX_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, PCA]:
    """Đường phương sai tích lũy của PCA trên đặc trưng HOG của batch đầu vào."""
    features = extract_hog(_as_image_batch(images), **HOG_DEFAULTS)  # type: ignore[arg-type]
    if scale:
        features = StandardScaler().fit_transform(features)
    n_components = min(len(features), features.shape[1], max_components)
    pca = PCA(n_components=n_components, random_state=random_state).fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return cumulative, pca


def _demo() -> None:
    """Kiểm tra nhanh trên dữ liệu tổng hợp khi chạy trực tiếp mô-đun."""
    configure_logging()
    rng = np.random.default_rng(RANDOM_STATE)
    train = rng.random((50, 32, 32, 3), dtype=np.float32)
    test = rng.random((10, 32, 32, 3), dtype=np.float32)
    for method in METHODS:
        train_feat, test_feat, extractor = build_features_with_info(
            train, test, method=method
        )
        print(
            f"{method:8s} -> train {train_feat.shape}, test {test_feat.shape},"
            f" PCA={extractor.n_components_pca}"
        )


if __name__ == "__main__":
    _demo()
