import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.config import RANDOM_STATE

def train_and_tune(X_train_feat: np.ndarray, y_train: np.ndarray, model_type: str = "knn"):
    """
    Input:
        - X_train_feat, y_train: Đặc trưng và nhãn tập train
        - model_type: 'knn' hoặc 'dt' (Decision Tree)
    Output:
        - best_model: Mô hình tốt nhất sau khi tìm tham số
        - cv_results_df: Bảng kết quả thử nghiệm các tham số
    """
    print(f"[Người 3] Đang huấn luyện và tối ưu mô hình: {model_type.upper()}...")

    if model_type == "knn":
        base_model = KNeighborsClassifier()
        param_grid = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    elif model_type == "dt":
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5]
        }
    else:
        raise ValueError(f"Loại mô hình {model_type} không hợp lệ!")

    grid = GridSearchCV(base_model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train_feat, y_train)

    print(f" -> Tham số tốt nhất ({model_type}): {grid.best_params_}")
    cv_results_df = pd.DataFrame(grid.cv_results_)
    return grid.best_estimator_, cv_results_df

# Additive grouped-CV API. The legacy entry point above is intentionally unchanged.
import re
from pathlib import Path, PurePosixPath

import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from src.features import HOG_DEFAULTS, PCA_VARIANCE
from src.preprocessing import CLASS_NAMES

KNN_METHODS = ("pixel", "hog", "hog_pca")
KNN_LABELS = tuple(range(len(CLASS_NAMES)))


def track_groups(metadata, labels=None):
    """Validate CSV order/class/path agreement; never fall back to image CV."""
    y = metadata["ClassId"].to_numpy()
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, KNN_LABELS).all():
        raise ValueError("ClassId must contain integers 0..42")
    if labels is not None and not np.array_equal(y, labels):
        raise ValueError("Loader labels are not aligned with CSV rows")
    if metadata["Path"].duplicated().any():
        raise ValueError("Duplicate image paths")
    groups = []
    for path, label in zip(metadata["Path"], y):
        normalized = PurePosixPath(str(path).replace("\\", "/"))
        match = re.fullmatch(r"(\d+)_(\d+)_(\d+)\.png", normalized.name)
        if (not match or len(normalized.parts) != 3
                or normalized.parts[0] != "Train"
                or normalized.parts[1] != str(int(label))
                or int(match[1]) != label):
            raise ValueError(f"Invalid train track/class path: {path}")
        groups.append(f"{int(label)}:{int(match[2])}")
    return np.asarray(groups)


def validate_folds(y, groups, folds, labels=KNN_LABELS):
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape or not np.array_equal(np.unique(y), labels):
        raise ValueError("Expected aligned labels/groups and all required classes")
    seen = np.zeros(len(y), dtype=int)
    for train, valid in folds:
        train, valid = np.asarray(train), np.asarray(valid)
        if (len(train) == 0 or len(valid) == 0
                or len(np.unique(train)) != len(train)
                or len(np.unique(valid)) != len(valid)
                or np.any(train < 0) or np.any(valid < 0)
                or np.any(train >= len(y)) or np.any(valid >= len(y))):
            raise ValueError("Invalid fold indices")
        if (len(np.intersect1d(train, valid))
                or len(train) + len(valid) != len(y)
                or len(np.intersect1d(groups[train], groups[valid]))):
            raise ValueError("Fold overlap, missing rows or group leakage")
        if any(not np.array_equal(np.unique(y[idx]), labels) for idx in (train, valid)):
            raise ValueError("Every fold side must contain every class")
        seen[valid] += 1
    if not len(folds) or not np.all(seen == 1):
        raise ValueError("Each row must be validation exactly once")


def grouped_folds(y, groups, n_splits=3, labels=KNN_LABELS):
    y, groups = np.asarray(y), np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("Groups and labels are not aligned")
    for label in labels:
        if len(np.unique(groups[y == label])) < n_splits:
            raise ValueError(f"Class {label} needs at least {n_splits} tracks")
    folds = list(StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    ).split(np.zeros(len(y)), y, groups))
    validate_folds(y, groups, folds, labels)
    return folds


def _float32(values):
    return np.asarray(values, dtype=np.float32)


def knn_pipeline(method):
    if method not in KNN_METHODS:
        raise ValueError(f"Unknown representation: {method}")
    steps = []
    if method != "pixel":
        steps.append(("scaler", StandardScaler()))
    if method == "hog_pca":
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)))
        # Match person 2's feature contract after PCA, including dtype.
        steps.append(("float32", FunctionTransformer(_float32)))
    steps.append(("knn", KNeighborsClassifier(algorithm="brute", n_jobs=1)))
    return Pipeline(steps)


def knn_candidates(smoke=False):
    # ParameterGrid's stable order is the declared tie-break, not elapsed time.
    return {"knn__n_neighbors": [3] if smoke else [3, 5, 7],
            "knn__weights": ["uniform"] if smoke else ["uniform", "distance"],
            "knn__metric": ["euclidean"] if smoke else ["euclidean", "manhattan"]}


def batch_predict(estimator, features, batch_size=128):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if len(features) == 0:
        raise ValueError("Empty prediction input")
    return np.concatenate([estimator.predict(features[start:start + batch_size])
                           for start in range(0, len(features), batch_size)])


def classification_metrics(y, predicted, labels=KNN_LABELS):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=list(labels), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1)}


class BatchScorer:
    """One batched prediction per fold/candidate for all scoring metrics."""
    def __init__(self, batch_size=128, labels=KNN_LABELS):
        self.batch_size, self.labels = batch_size, labels

    def __call__(self, estimator, features, y):
        result = classification_metrics(y, batch_predict(
            estimator, features, self.batch_size), self.labels)
        pca = estimator.named_steps.get("pca")
        result["feature_dimensions"] = (float(pca.n_components_) if pca is not None
                                        else float(features.shape[1]))
        return result


def first_best_index(results):
    scores = np.asarray(results["mean_test_f1_macro"], dtype=float)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("CV scores must all be finite")
    return int(np.argmax(scores))


def tune_grouped_knn(features, y, groups, folds, method, *, smoke=False,
                     batch_size=128, n_jobs=1, verbose=3, labels=KNN_LABELS):
    validate_folds(y, groups, folds, labels)
    if len(features) != len(y) or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Invalid raw feature matrix")
    candidates = knn_candidates(smoke)
    if max(candidates["knn__n_neighbors"]) > min(len(t) for t, _ in folds):
        raise ValueError("k exceeds a fold's training size")
    if n_jobs != 1:
        raise ValueError("Use n_jobs=1 to avoid duplicating image/feature arrays")
    grid = GridSearchCV(knn_pipeline(method), candidates, cv=folds,
                        scoring=BatchScorer(batch_size, labels),
                        refit=first_best_index, n_jobs=1, error_score="raise",
                        return_train_score=False, verbose=verbose)
    with threadpool_limits(limits=2):
        grid.fit(features, y)
    return grid


def save_knn_bundle(path, estimator, method, metadata):
    """Exclusive creation; joblib files must only be loaded from trusted sources."""
    if method not in KNN_METHODS:
        raise ValueError("Unknown representation")
    bundle = {"schema_version": 1, "method": method, "estimator": estimator,
              "class_names": CLASS_NAMES.copy(), "labels": list(KNN_LABELS),
              "raw_dimensions": int(estimator.n_features_in_),
              "hog_params": HOG_DEFAULTS.copy(), "seed": RANDOM_STATE,
              "image_contract": {"size": [32, 32], "color": "RGB", "roi": True,
                                 "scale": "divide255", "augmentation": False},
              "metadata": metadata}
    with Path(path).open("xb") as stream:
        joblib.dump(bundle, stream, compress=3)


def load_knn_bundle(path):
    bundle = joblib.load(path)  # Trusted local files only: pickle can execute code.
    if (bundle.get("schema_version") != 1 or bundle.get("class_names") != CLASS_NAMES
            or bundle.get("labels") != list(KNN_LABELS)
            or bundle.get("method") not in KNN_METHODS
            or bundle.get("hog_params") != HOG_DEFAULTS
            or bundle.get("seed") != RANDOM_STATE
            or bundle.get("image_contract") != {
                "size": [32, 32], "color": "RGB", "roi": True,
                "scale": "divide255", "augmentation": False}
            or bundle.get("raw_dimensions") != bundle["estimator"].n_features_in_):
        raise ValueError("Incompatible K-NN bundle")
    return bundle


if __name__ == "__main__":
    X_dummy = np.random.rand(100, 128)
    y_dummy = np.random.randint(0, 5, size=100)
    model, df_log = train_and_tune(X_dummy, y_dummy, model_type="knn")
    print("Test thành công! Số cấu hình đã thử:", len(df_log))
