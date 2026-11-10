import pandas as pd
from src.config import DATA_DIR
from src.preprocessing import load_and_preprocess
from src.features import build_features
from src.models import train_and_tune
from src.evaluate import evaluate_and_analyze

def main():
    X_train_img, X_test_img, y_train, y_test, class_names = load_and_preprocess(DATA_DIR)

    all_results = []

    for feat_method in ["pixel", "hog", "hog_pca"]:
        X_train_feat, X_test_feat = build_features(X_train_img, X_test_img, method=feat_method)

        for model_type in ["knn", "dt"]:
            exp_name = f"{feat_method}_{model_type}"
            best_model, _ = train_and_tune(X_train_feat, y_train, model_type=model_type)

            metrics = evaluate_and_analyze(
                best_model, X_test_feat, X_test_img, y_test, class_names, exp_name=exp_name
            )
            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    print("\n=== BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("report/summary_results.csv", index=False)

if __name__ == "__main__":
    main()
