import os

RANDOM_STATE = 42
# Giữ để tương thích import; không dùng vì đã có test chính thức GTSRB.
TEST_SIZE = 0.2

IMAGE_SIZE = (32, 32)
NUM_CHANNELS = 3

DATA_DIR = "data"
DATA_ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
USE_ROI = True
CACHE_PATH = os.path.join(DATA_DIR, "cache", "gtsrb_32.npz")
# Đổi thành False để đọc lại ảnh; giữ nguyên chữ ký của hàm pipeline.
USE_CACHE = True
FIGURES_DIR = os.path.join("report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
