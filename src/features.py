import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE

def build_features(X_train_img: np.ndarray, X_test_img: np.ndarray, method: str = "pixel"):
    """
    Input:
        - X_train_img, X_test_img: Mảng ảnh shape (N, 32, 32, 3)
        - method: 'pixel' (Baseline), 'hog', hoặc 'hog_pca'
    Output:
        - X_train_feat, X_test_feat: Mảng đặc trưng 2D shape (N, n_features)
    """
    print(f"[Người 2] Đang trích xuất đặc trưng bằng phương pháp: {method}...")

    if method == "pixel":
        X_train_feat = X_train_img.reshape(X_train_img.shape[0], -1)
        X_test_feat = X_test_img.reshape(X_test_img.shape[0], -1)
        return X_train_feat, X_test_feat

    elif method in ["hog", "hog_pca"]:
        def _extract_hog_batch(images):
            features = []
            for img in images:
                feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), channel_axis=-1)
                features.append(feat)
            return np.array(features, dtype=np.float32)

        X_train_hog = _extract_hog_batch(X_train_img)
        X_test_hog = _extract_hog_batch(X_test_img)

        scaler = StandardScaler()
        X_train_hog = scaler.fit_transform(X_train_hog)
        X_test_hog = scaler.transform(X_test_hog)

        if method == "hog_pca":
            pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
            X_train_hog = pca.fit_transform(X_train_hog)
            X_test_hog = pca.transform(X_test_hog)

        return X_train_hog, X_test_hog

    else:
        raise ValueError(f"Phương pháp {method} không hợp lệ!")

if __name__ == "__main__":
    X_tr_dummy = np.random.rand(50, 32, 32, 3).astype(np.float32)
    X_te_dummy = np.random.rand(10, 32, 32, 3).astype(np.float32)
    for m in ["pixel", "hog", "hog_pca"]:
        f_tr, f_te = build_features(X_tr_dummy, X_te_dummy, method=m)
        print(f"Method {m} -> Train shape: {f_tr.shape}, Test shape: {f_te.shape}")
