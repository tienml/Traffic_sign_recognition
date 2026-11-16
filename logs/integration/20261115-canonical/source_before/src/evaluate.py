import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.config import FIGURES_DIR

def evaluate_and_analyze(model, X_test_feat: np.ndarray, X_test_img: np.ndarray,
                         y_test: np.ndarray, class_names: list, exp_name: str = "baseline"):
    """
    Input: Mô hình đã huấn luyện, đặc trưng test, ảnh gốc test, nhãn test, tên lớp, tên thí nghiệm.
    Output: Dictionary chứa các độ đo và lưu biểu đồ vào report/figures/
    """
    print(f"[Người 4] Đang đánh giá và phân tích lỗi cho thí nghiệm: {exp_name}...")
    y_pred = model.predict(X_test_feat)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "experiment": exp_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4)
    }
    print(f" -> Kết quả {exp_name}: {metrics}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {exp_name}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"cm_{exp_name}.png"))
    plt.close()

    return metrics

if __name__ == "__main__":
    from sklearn.dummy import DummyClassifier
    X_f = np.random.rand(40, 20)
    X_img = np.random.rand(40, 32, 32, 3)
    y_t = np.random.randint(0, 4, size=40)
    dummy_clf = DummyClassifier().fit(X_f, y_t)
    evaluate_and_analyze(dummy_clf, X_f, X_img, y_t, ["C0", "C1", "C2", "C3"], "test_run")
