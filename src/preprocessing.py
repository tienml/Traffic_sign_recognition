import numpy as np
from sklearn.model_selection import train_test_split
from src.config import IMAGE_SIZE, NUM_CHANNELS, TEST_SIZE, RANDOM_STATE

def load_and_preprocess(data_dir: str):
    """
    Input: Đường dẫn thư mục dữ liệu.
    Output:
        - X_train_img, X_test_img: Mảng numpy shape (N, 32, 32, 3), giá trị [0.0, 1.0]
        - y_train, y_test: Mảng 1D chứa nhãn lớp (int)
        - class_names: List tên các loại biển báo
    """
    # TODO (Người 1): Thay phần tạo dữ liệu giả dưới đây bằng code đọc ảnh thật bằng cv2,
    # resize về IMAGE_SIZE và chia cho 255.0 để chuẩn hóa về [0, 1].
    print("[Người 1] Đang tải và tiền xử lý ảnh...")
    num_samples = 200
    num_classes = 10
    X_dummy = np.random.rand(num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS).astype(np.float32)
    y_dummy = np.random.randint(0, num_classes, size=num_samples)
    class_names = [f"Class_{i}" for i in range(num_classes)]

    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_dummy, y_dummy, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_dummy
    )
    return X_train_img, X_test_img, y_train, y_test, class_names

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, classes = load_and_preprocess("data")
    print(f"Test thành công! Shape X_train: {X_tr.shape}, X_test: {X_te.shape}")
