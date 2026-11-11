"""Thí nghiệm biểu diễn đặc trưng (Thành viên 2) trên mẫu ablation thống nhất.

Dùng đúng manifest mẫu 8.000 train / 3.000 test của bước tiền xử lý để mọi
so sánh trong nhóm dùng chung một tập mẫu. K-NN (k=3, euclidean, brute) chỉ là
đầu dò để đo tác động của cách biểu diễn, không phải mô hình cuối.

Chạy từ gốc dự án:
    python scripts/feature_ablation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, NUM_CHANNELS, RANDOM_STATE  # noqa: E402
from src.features import (  # noqa: E402
    METHODS,
    PCA_VARIANCE,
    FeatureExtractor,
    configure_logging,
    pca_variance_curve,
)

TABLES = ROOT / "report" / "tables"
FIGURES = ROOT / FIGURES_DIR
CACHE = ROOT / "data" / "cache" / "gtsrb_32.npz"
MANIFEST = TABLES / "ablation_sample_manifest.csv"
LOG_PATH = ROOT / "logs" / "features_ablation.log"
THREADS = 2
K_NEIGHBORS = 3

LABELS = {
    "pixel": "Pixel thô (baseline)",
    "hog": "HOG + chuẩn hóa",
    "hog_pca": "HOG + PCA 95%",
}


def load_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lấy đúng các dòng trong manifest từ cache ảnh 32×32 của bước tiền xử lý."""
    for path in (CACHE, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(
                f"Thiếu {path}. Chạy python -m src.preprocessing trước."
            )
    manifest = pd.read_csv(MANIFEST)
    cache = np.load(CACHE, allow_pickle=False)
    arrays: dict[str, np.ndarray] = {}
    for split, key in (("Train", "X_train_img"), ("Test", "X_test_img")):
        rows = manifest.loc[manifest.Split.eq(split)].sort_values("Order")
        index = rows.CSVRow.to_numpy(dtype=np.int64)
        arrays[f"X_{split.lower()}"] = cache[key][index]
        arrays[f"y_{split.lower()}"] = rows.ClassId.to_numpy(dtype=np.int64)
    images = (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_test"],
        arrays["y_test"],
    )
    expected = (8000, 3000)
    if (len(images[0]), len(images[2])) != expected:
        raise ValueError(
            f"Manifest không đúng mẫu {expected}: "
            f"{len(images[0])}/{len(images[2])}"
        )
    if images[0].dtype != np.float32 or images[0].shape[1:] != (
        32, 32, NUM_CHANNELS,
    ):
        raise ValueError("Cache ảnh sai dtype/shape; tạo lại cache rồi chạy lại.")
    return images


def run_ablation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Trích đặc trưng, huấn luyện K-NN đầu dò và ghi nhận thời gian."""
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for method in METHODS:
        extractor = FeatureExtractor(method, pca_variance=PCA_VARIANCE)
        started = time.perf_counter()
        train_features = extractor.fit_transform(x_train)
        test_features = extractor.transform(x_test)
        extraction_seconds = time.perf_counter() - started

        model = KNeighborsClassifier(
            n_neighbors=K_NEIGHBORS,
            metric="euclidean",
            algorithm="brute",
            n_jobs=1,
        )
        with threadpool_limits(limits=THREADS):
            started = time.perf_counter()
            model.fit(train_features, y_train)
            predicted = model.predict(test_features)
            predict_seconds = time.perf_counter() - started

        predictions[method] = predicted
        rows.append(
            {
                "Phương pháp": method,
                "Mô tả": LABELS[method],
                "Số chiều": int(train_features.shape[1]),
                "Trích đặc trưng (s)": round(extraction_seconds, 2),
                "KNN fit+predict (s)": round(predict_seconds, 2),
                "Accuracy": accuracy_score(y_test, predicted),
                "Macro-F1": f1_score(
                    y_test,
                    predicted,
                    labels=np.arange(43),
                    average="macro",
                    zero_division=0,
                ),
                "Thành phần PCA": extractor.n_components_pca,
                "Phương sai giữ": extractor.explained_variance,
            }
        )
        print(
            f"{method:8s} {train_features.shape[1]:5d} chiều "
            f"acc={rows[-1]['Accuracy']:.4f} "
            f"f1={rows[-1]['Macro-F1']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows), predictions


def plot_results(
    results: pd.DataFrame, cumulative: np.ndarray, pca_components: int
) -> None:
    """Vẽ biểu đồ so sánh số chiều/độ chính xác và đường phương sai tích lũy."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    axes[0].bar(
        results["Mô tả"],
        results["Accuracy"],
        color=colors,
        width=0.6,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy (K-NN k=3)")
    axes[0].set_title("Độ chính xác theo cách biểu diễn")
    axes[0].tick_params(axis="x", rotation=12)
    for patch, value in zip(axes[0].patches, results["Accuracy"]):
        axes[0].text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.02,
            f"{value:.3f}\n({int(results.loc[results['Accuracy'].eq(value), 'Số chiều'].iloc[0])} chiều)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    x_axis = np.arange(1, len(cumulative) + 1)
    axes[1].plot(x_axis, cumulative, color="#4C78A8", linewidth=2)
    axes[1].axhline(PCA_VARIANCE, color="#E45756", linestyle="--")
    axes[1].axvline(pca_components, color="#E45756", linestyle=":")
    axes[1].annotate(
        f"{pca_components} thành phần\n≈ {PCA_VARIANCE:.0%} phương sai",
        xy=(pca_components, PCA_VARIANCE),
        xytext=(pca_components + 15, PCA_VARIANCE - 0.18),
        arrowprops={"arrowstyle": "->", "color": "#E45756"},
        fontsize=9,
    )
    axes[1].set_xlabel("Số thành phần chính")
    axes[1].set_ylabel("Phương sai tích lũy")
    axes[1].set_title("PCA trên đặc trưng HOG (tập train 8.000 ảnh)")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "feature_ablation.png", dpi=150)
    plt.close(fig)


def main() -> int:
    logger = configure_logging(LOG_PATH)
    TABLES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x_train, y_train, x_test, y_test = load_sample()
    logger.info(
        "Mẫu ablation: train=%d test=%d seed=%d pca_variance=%.2f",
        len(x_train),
        len(x_test),
        RANDOM_STATE,
        PCA_VARIANCE,
    )

    results, predictions = run_ablation(
        x_train, y_train, x_test, y_test
    )
    results.to_csv(
        TABLES / "feature_ablation.csv", index=False, encoding="utf-8-sig"
    )

    prediction_table = pd.DataFrame({"ClassId": y_test, **predictions})
    prediction_table.to_csv(
        TABLES / "feature_ablation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cumulative, pca = pca_variance_curve(x_train, max_components=200)
    pd.DataFrame(
        {
            "Component": np.arange(1, len(cumulative) + 1),
            "CumulativeVariance": cumulative,
        }
    ).to_csv(
        TABLES / "feature_pca_cumulative.csv",
        index=False,
        encoding="utf-8-sig",
    )
    components = int(np.searchsorted(cumulative, PCA_VARIANCE) + 1)
    if components != results.loc[
        results["Phương pháp"].eq("hog_pca"), "Thành phần PCA"
    ].iloc[0]:
        logger.warning(
            "Số thành phần lúc trích đặc trưng khác đường cong PCA: %d vs %d",
            results.loc[
                results["Phương pháp"].eq("hog_pca"), "Thành phần PCA"
            ].iloc[0],
            components,
        )
    plot_results(results, cumulative, components)

    # Đối chiếu: predictions phải khớp lại đúng metric đã ghi.
    for method in METHODS:
        predicted = predictions[method]
        recorded = results.loc[results["Phương pháp"].eq(method)].iloc[0]
        assert np.isclose(
            accuracy_score(y_test, predicted), recorded["Accuracy"]
        )
        assert np.isclose(
            f1_score(
                y_test,
                predicted,
                labels=np.arange(43),
                average="macro",
                zero_division=0,
            ),
            recorded["Macro-F1"],
        )

    logger.info(
        "Hoàn tất ablation đặc trưng trong %.1fs; PCA 95%% = %d thành phần.",
        time.perf_counter() - started,
        components,
    )
    print("\n" + results.to_string(index=False))
    print(f"\nPCA {PCA_VARIANCE:.0%} giữ {components} thành phần.")
    print(f"Biểu đồ: {(FIGURES / 'feature_ablation.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
