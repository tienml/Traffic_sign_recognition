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

if __name__ == "__main__":
    X_dummy = np.random.rand(100, 128)
    y_dummy = np.random.randint(0, 5, size=100)
    model, df_log = train_and_tune(X_dummy, y_dummy, model_type="knn")
    print("Test thành công! Số cấu hình đã thử:", len(df_log))
