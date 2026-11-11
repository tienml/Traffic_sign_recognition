"""Đọc GTSRB theo tập chính thức, crop ROI, resize và đổi thang pixel."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import warnings
import zlib
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from zipfile import BadZipFile

import cv2
import numpy as np
import pandas as pd

from src.config import (
    CACHE_PATH,
    DATA_ARCHIVE_DIR,
    DATA_DIR,
    IMAGE_SIZE,
    NUM_CHANNELS,
    RANDOM_STATE,
    USE_CACHE,
    USE_ROI,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


CLASS_NAMES = [
    "Speed limit 20",
    "Speed limit 30",
    "Speed limit 50",
    "Speed limit 60",
    "Speed limit 70",
    "Speed limit 80",
    "End of speed limit 80",
    "Speed limit 100",
    "Speed limit 120",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons",
]

ROI_COLUMNS = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
CACHE_VERSION = 1
Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "preprocessing.log"
_LOGGER = logging.getLogger(__name__)
_LOG_LOCK = Lock()


class _BestEffortFileHandler(logging.FileHandler):
    """Lỗi ghi log chỉ cảnh báo một lần, không làm hỏng lượt nạp dữ liệu."""

    def __init__(self, path: Path) -> None:
        """Mở file trễ để xử lý cả lỗi tạo thư mục và lỗi mở file."""
        super().__init__(path, encoding="utf-8", delay=True)
        self.failed = False

    def emit(self, record: logging.LogRecord) -> None:
        """Ghi UTF-8; nếu ổ đĩa lỗi thì các thông báo vẫn ra console."""
        if self.failed:
            return
        try:
            if self.stream is None:
                Path(self.baseFilename).parent.mkdir(
                    parents=True, exist_ok=True
                )
            super().emit(record)
        except (OSError, ValueError):
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Không in traceback chứa đường dẫn riêng khi file log bị lỗi."""
        if not self.failed:
            self.failed = True
            try:
                warnings.warn(
                    "Không ghi được log tiền xử lý; tiếp tục với log console.",
                    stacklevel=3,
                )
            except UserWarning:
                # Bộ lọc -Werror cũng không được biến lỗi log thành lỗi dữ liệu.
                pass


def _get_logger() -> logging.Logger:
    """Khởi tạo logger riêng một lần, kể cả khi nhiều luồng cùng nạp."""
    with _LOG_LOCK:
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = False
        existing = {handler.name for handler in _LOGGER.handlers}
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if "preprocessing.console" not in existing:
            console = logging.StreamHandler()
            console.set_name("preprocessing.console")
            console.setFormatter(formatter)
            _LOGGER.addHandler(console)
        if "preprocessing.file" not in existing:
            file_handler = _BestEffortFileHandler(LOG_PATH)
            file_handler.set_name("preprocessing.file")
            file_handler.setFormatter(formatter)
            _LOGGER.addHandler(file_handler)
    return _LOGGER


def _log_path(path: str | Path, base_dir: Path | None = None) -> str:
    """Chỉ ghi đường dẫn tương đối hoặc tên file, không lộ vị trí máy."""
    path = Path(path)
    root = base_dir or Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return path.name


def _log_dataset(
    logger: logging.Logger, dataset: Dataset, started: float
) -> None:
    """Ghi hợp đồng và thống kê của từng lượt thành công, kể cả cache hit."""
    x_train, x_test, y_train, y_test, names = dataset
    for split, images, labels in (
        ("Train", x_train, y_train),
        ("Test", x_test, y_test),
    ):
        logger.info(
            "%s: X shape=%s dtype=%s min=%.6f max=%.6f; "
            "y shape=%s dtype=%s min=%d max=%d classes=%d",
            split,
            images.shape,
            images.dtype,
            images.min(),
            images.max(),
            labels.shape,
            labels.dtype,
            labels.min(),
            labels.max(),
            len(np.unique(labels)),
        )
    logger.info(
        "Hoàn tất load_and_preprocess: Train=%d Test=%d class_names=%d "
        "errors=0 skipped=0 elapsed=%.3fs",
        len(y_train),
        len(y_test),
        len(names),
        time.perf_counter() - started,
    )


def read_rgb(path: str | Path) -> np.ndarray:
    """Đọc ảnh RGB uint8, hỗ trợ cả đường dẫn có dấu trên Windows."""
    path = Path(path)
    # Dùng đường dẫn tương đối khi CWD có dấu để imread không phát hàng
    # nghìn cảnh báo trên Windows. Nếu tên file có dấu, dùng imdecode.
    read_path = str(path)
    if os.name == "nt" and path.is_absolute():
        try:
            read_path = os.path.relpath(path)
        except ValueError:
            pass  # File nằm trên ổ đĩa khác: giữ đường dẫn tuyệt đối.
    # imread trên Windows có thể mở nhầm tên chuyển mã mà không báo lỗi.
    # Với tên Unicode còn lại, đọc bytes bằng Python ngay từ đầu.
    image = None
    if os.name != "nt" or read_path.isascii():
        image = cv2.imread(read_path, cv2.IMREAD_COLOR)
    if image is None and path.is_file():
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            if encoded.size:
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except (OSError, cv2.error):
            image = None
    if image is None:
        raise OSError(f"Không đọc được ảnh (thiếu hoặc hỏng): {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_roi(
    image: np.ndarray, roi: Sequence[float] | None = None
) -> np.ndarray:
    """Crop ROI đóng [X1, Y1, X2, Y2]; ROI sai thì giữ nguyên ảnh.

    Tọa độ GTSRB tính cả pixel ở X2, Y2, còn lát cắt NumPy loại cận
    phải nên cần cộng 1. Kẹp ROI vào biên ảnh trước khi lấy lát cắt.
    """
    if roi is None:
        return image
    try:
        coordinates = np.asarray(roi, dtype=float)
    except (TypeError, ValueError):
        return image
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        return image
    if not np.equal(coordinates, np.floor(coordinates)).all():
        return image
    x1, y1, x2, y2 = (int(value) for value in coordinates)
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 < x1 or y2 < y1:
        return image
    return image[y1 : y2 + 1, x1 : x2 + 1]


def load_image(
    path: str | Path,
    roi: Sequence[float] | None = None,
    use_roi: bool = True,
    *,
    image_size: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Đọc RGB, tùy chọn crop, resize và chia 255 về float32.

    image_size theo thứ tự (rộng, cao) của OpenCV, gồm hai số nguyên
    dương. None lấy IMAGE_SIZE lúc gọi, không đóng băng cấu hình trong
    đối số mặc định. Tham số resize chỉ phục vụ ablation; pipeline vẫn
    dùng IMAGE_SIZE và INTER_AREA. Không học tham số thống kê từ ảnh
    và không áp dụng augmentation trong hàm này.
    """
    size = IMAGE_SIZE if image_size is None else image_size
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 < value <= np.iinfo(np.int32).max
            for value in size
        )
    ):
        raise ValueError(
            "image_size phải là (rộng, cao) gồm hai số nguyên dương"
        )
    size = tuple(int(value) for value in size)
    image = read_rgb(path)
    if use_roi:
        image = crop_roi(image, roi)
    image = cv2.resize(image, size, interpolation=interpolation)
    return image.astype(np.float32) / np.float32(255.0)


def _read_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Kiểm tra cấu trúc CSV và nhãn nguyên trong khoảng 0..42."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV dữ liệu: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"Path", "ClassId", "Width", "Height", *ROI_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} thiếu cột: {sorted(missing)}")
    if frame.empty or frame[["Path", "ClassId"]].isna().any().any():
        raise ValueError(f"CSV rỗng hoặc thiếu Path/ClassId: {csv_path}")
    labels = pd.to_numeric(frame["ClassId"], errors="coerce")
    valid = labels.between(0, len(CLASS_NAMES) - 1) & (labels % 1 == 0)
    if not valid.all():
        raise ValueError(f"ClassId phải là số nguyên 0..42: {csv_path}")
    frame["ClassId"] = labels.astype(np.int64)
    return frame


def load_split(
    csv_path: str | Path, base_dir: str | Path, use_roi: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Đọc toàn bộ một CSV theo đúng thứ tự, không bỏ âm thầm ảnh lỗi."""
    logger = _get_logger()
    started = time.perf_counter()
    split = Path(csv_path).stem
    total, loaded = 0, 0
    current_path = Path(csv_path)
    progress = None
    try:
        frame = _read_metadata(csv_path)
        total = len(frame)
        logger.info(
            "Đọc %s: count=%d seed=%s IMAGE_SIZE=%s USE_ROI=%s",
            split,
            total,
            RANDOM_STATE,
            IMAGE_SIZE,
            use_roi,
        )
        base_dir = Path(base_dir).resolve()
        width, height = IMAGE_SIZE
        images = np.empty((total, height, width, 3), dtype=np.float32)
        labels = frame["ClassId"].to_numpy(dtype=np.int64, copy=True)
        rows = frame[["Path", *ROI_COLUMNS]].itertuples(index=False, name=None)
        if tqdm is not None:
            progress = tqdm(rows, total=total, desc=split)
            rows = progress
        for index, (relative_path, *roi) in enumerate(rows):
            current_path = (base_dir / str(relative_path)).resolve()
            if not current_path.is_relative_to(base_dir):
                raise ValueError(
                    f"Path nằm ngoài thư mục dữ liệu: {relative_path}"
                )
            images[index] = load_image(current_path, roi, use_roi=use_roi)
            loaded = index + 1
            if tqdm is None and loaded % 5000 == 0:
                logger.info("%s: %d/%d ảnh", split, loaded, total)
    except Exception as error:
        # Chỉ ghi loại lỗi/tên file để không chép đường dẫn máy vào log.
        # Giữ nguyên ngoại lệ cho caller; tuyệt đối không bỏ qua ảnh lỗi.
        logger.error(
            "Thất bại %s: count=%d loaded=%d errors=1 skipped=0 "
            "file=%s error=%s elapsed=%.3fs",
            split,
            total,
            loaded,
            _log_path(current_path, Path(base_dir)),
            type(error).__name__,
            time.perf_counter() - started,
        )
        raise
    finally:
        if progress is not None:
            progress.close()
    logger.info(
        "Đã nạp %s: count=%d errors=0 skipped=0 elapsed=%.3fs",
        split,
        loaded,
        time.perf_counter() - started,
    )
    return images, labels


def _check_images(images: np.ndarray) -> None:
    """Kiểm tra hợp đồng ảnh theo từng khối để tránh mảng tạm quá lớn."""
    expected = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    if images.ndim != 4 or images.shape[1:] != expected:
        raise ValueError(f"Shape ảnh phải là (N, {expected}): {images.shape}")
    if images.dtype != np.float32:
        raise ValueError(f"Ảnh phải có dtype float32, nhận {images.dtype}")
    for start in range(0, len(images), 1024):
        batch = images[start : start + 1024]
        if not np.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
            raise ValueError("Pixel phải hữu hạn và nằm trong [0, 1]")


def to_grayscale(images: np.ndarray) -> np.ndarray:
    """Đổi batch RGB float32 sang (N, H, W, 1), chỉ dùng khi so sánh.

    Chỉ nhận kích thước IMAGE_SIZE để giữ kiểm tra hợp đồng pipeline.
    Không gọi trong pipeline mặc định vì HOG đang nhận ba kênh RGB.
    """
    _check_images(images)
    gray = np.empty((*images.shape[:3], 1), dtype=np.float32)
    for index, image in enumerate(images):
        gray[index, ..., 0] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


def augment_image(
    image: np.ndarray,
    enabled: bool = False,
    rng: np.random.Generator | None = None,
    max_rotation: float = 10.0,
    max_shift: float = 0.05,
    brightness: float = 0.1,
    contrast: float = 0.1,
) -> np.ndarray:
    """Minh họa tăng cường nhẹ, mặc định tắt và chỉ dành cho train.

    Xoay tối đa ±10 độ mô phỏng góc máy; dịch tối đa 5% mô phỏng lệch
    tâm. Đổi sáng và tương phản tối đa 10% mô phỏng ánh sáng, không
    đổi màu sang màu khác. Chỉ biến đổi nhẹ để tránh mất nét đặc trưng.
    CẤM lật ngang/dọc: Turn left/right, Keep left/right đổi nghĩa hoặc
    tạo biển không có thật. Không dùng phép đổi phối cảnh mạnh hay cắt
    ngẫu nhiên có thể làm mất chữ số. Không tự nhân bản dữ liệu.

    Nếu không truyền rng, dùng RANDOM_STATE để tái lập. Muốn nhiều ảnh
    khác nhau, truyền cùng một Generator khởi tạo từ RANDOM_STATE.
    """
    _check_images(image[np.newaxis, ...])
    if not enabled:
        return image.copy()
    limits = (
        (max_rotation, 10.0),
        (max_shift, 0.05),
        (brightness, 0.1),
        (contrast, 0.1),
    )
    if any(not 0 <= value <= bound for value, bound in limits):
        raise ValueError("Augmentation vượt giới hạn nhẹ đã cho phép")
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    height, width = image.shape[:2]
    angle = rng.uniform(-max_rotation, max_rotation)
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2, (height - 1) / 2), angle, 1.0
    )
    matrix[:, 2] += rng.uniform(-max_shift, max_shift, 2) * [width, height]
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    alpha = rng.uniform(1.0 - contrast, 1.0 + contrast)
    beta = rng.uniform(-brightness, brightness)
    return np.clip(transformed * alpha + beta, 0, 1).astype(np.float32)


def _source_key(archive_dir: Path) -> str:
    """Gắn cache với thư mục nguồn và nội dung hai CSV (không pickle)."""
    digest = hashlib.sha256(str(archive_dir.resolve()).encode("utf-8"))
    for name in ("Train.csv", "Test.csv"):
        path = archive_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy CSV dữ liệu: {path}")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_cache(cache_path: Path, source_key: str) -> Dataset | None:
    """Chỉ nhận cache có cấu hình, nguồn và hợp đồng dữ liệu phù hợp."""
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            compatible = (
                int(cached["version"].item()) == CACHE_VERSION
                and tuple(cached["image_size"].tolist()) == tuple(IMAGE_SIZE)
                and bool(cached["use_roi"].item()) == USE_ROI
                and str(cached["source_key"].item()) == source_key
            )
            if not compatible:
                _get_logger().info(
                    "cache=miss: nguồn/cấu hình thay đổi, đọc lại ảnh."
                )
                return None
            x_train, x_test = cached["X_train_img"], cached["X_test_img"]
            y_train, y_test = cached["y_train"], cached["y_test"]
            for images, labels in ((x_train, y_train), (x_test, y_test)):
                _check_images(images)
                if (
                    labels.shape != (len(images),)
                    or not np.issubdtype(labels.dtype, np.integer)
                    or len(labels) == 0
                    or labels.min() < 0
                    or labels.max() >= len(CLASS_NAMES)
                ):
                    raise ValueError("Nhãn trong cache sai hợp đồng")
        return x_train, x_test, y_train, y_test, CLASS_NAMES.copy()
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        BadZipFile,
        zlib.error,
    ) as error:
        _get_logger().warning(
            "cache=miss: bỏ cache không hợp lệ file=%s error=%s",
            _log_path(cache_path),
            type(error).__name__,
        )
        warnings.warn(f"Bỏ cache không hợp lệ: {error}", stacklevel=2)
        return None


def load_and_preprocess(data_dir: str) -> Dataset:
    """Trả X_train, X_test float32 RGB [0,1], y nguyên và 43 tên lớp.

    Dùng toàn bộ Train.csv và Test.csv chính thức, không chia ngẫu nhiên
    và không tự tạo validation. Mỗi biển vật lý có khoảng 30 ảnh liên
    tiếp trong một track; chia ảnh ngẫu nhiên khiến cùng biển xuất hiện
    ở train và test, làm accuracy cao giả tạo. Khi cross-validation,
    thành viên 3 cũng nên chia theo track thay vì chia ảnh độc lập.

    Chỉ crop ROI, resize và chia 255: không học mean/std từ train/test.
    Augmentation mặc định tắt; grayscale chỉ dùng cho thí nghiệm EDA.
    use_cache=True tương ứng USE_CACHE=True trong config, giữ nguyên
    chữ ký một tham số mà main.py sử dụng. Cache có khóa IMAGE_SIZE,
    USE_ROI, phiên bản xử lý và nguồn CSV; đổi cấu hình sẽ tính lại.
    Nếu sửa nội dung ảnh nhưng giữ nguyên CSV, đặt USE_CACHE=False
    và xóa cache cũ trước khi bật lại để tránh dùng kết quả lỗi thời.
    """
    logger = _get_logger()
    started = time.perf_counter()
    use_cache = USE_CACHE
    cache_state = "miss" if use_cache else "disabled"
    train_count, test_count = 0, 0
    logger.info(
        "Bắt đầu load_and_preprocess: seed=%s IMAGE_SIZE=%s "
        "USE_ROI=%s USE_CACHE=%s",
        RANDOM_STATE,
        IMAGE_SIZE,
        USE_ROI,
        use_cache,
    )
    if not use_cache:
        logger.info("cache=disabled")
    try:
        if NUM_CHANNELS != 3:
            raise ValueError("Pipeline yêu cầu NUM_CHANNELS=3 (RGB)")
        data_root = Path(data_dir).resolve()
        archive_dir = data_root / Path(DATA_ARCHIVE_DIR).relative_to(DATA_DIR)
        cache_path = data_root / Path(CACHE_PATH).relative_to(DATA_DIR)
        source_key = _source_key(archive_dir)
        if use_cache:
            cached = _read_cache(cache_path, source_key)
            if cached is not None:
                cache_state = "hit"
                logger.info(
                    "cache=hit file=%s", _log_path(cache_path, data_root)
                )
                _log_dataset(logger, cached, started)
                return cached
            logger.info("cache=miss file=%s", _log_path(cache_path, data_root))

        logger.info("Đọc ảnh GTSRB theo Train.csv / Test.csv chính thức.")
        x_train, y_train = load_split(
            archive_dir / "Train.csv", archive_dir, use_roi=USE_ROI
        )
        train_count = len(y_train)
        x_test, y_test = load_split(
            archive_dir / "Test.csv", archive_dir, use_roi=USE_ROI
        )
        test_count = len(y_test)
        if use_cache:
            temporary_path = None
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Mỗi lượt có file tạm riêng cùng ổ đĩa; đóng trước khi replace
                # để tương thích Windows và không xóa file của tiến trình khác.
                with tempfile.NamedTemporaryFile(
                    prefix=f"{cache_path.stem}_",
                    suffix=".tmp.npz",
                    dir=cache_path.parent,
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    np.savez_compressed(
                        stream,
                        X_train_img=x_train,
                        X_test_img=x_test,
                        y_train=y_train,
                        y_test=y_test,
                        image_size=np.asarray(IMAGE_SIZE, dtype=np.int64),
                        use_roi=np.asarray(USE_ROI),
                        source_key=np.asarray(source_key),
                        version=np.asarray(CACHE_VERSION),
                    )
                temporary_path.replace(cache_path)
                logger.info(
                    "Đã lưu cache file=%s", _log_path(cache_path, data_root)
                )
            except OSError as error:
                logger.warning(
                    "Không lưu được cache file=%s error=%s",
                    _log_path(cache_path, data_root),
                    type(error).__name__,
                )
                warnings.warn(f"Không lưu được cache: {error}", stacklevel=2)
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError as error:
                        logger.warning(
                            "Không dọn được cache tạm file=%s error=%s",
                            _log_path(temporary_path, data_root),
                            type(error).__name__,
                        )
                        warnings.warn(
                            f"Không dọn được cache tạm {temporary_path}: "
                            f"{error}",
                            stacklevel=2,
                        )
        result = x_train, x_test, y_train, y_test, CLASS_NAMES.copy()
        _log_dataset(logger, result, started)
        return result
    except Exception as error:
        logger.error(
            "Thất bại load_and_preprocess: cache=%s Train_complete=%d "
            "Test_complete=%d errors=1 skipped=0 error=%s elapsed=%.3fs",
            cache_state,
            train_count,
            test_count,
            type(error).__name__,
            time.perf_counter() - started,
        )
        raise


if __name__ == "__main__":
    x_tr, x_te, y_tr, y_te, names = load_and_preprocess(DATA_DIR)
    for images in (x_tr, x_te):
        _check_images(images)
    # Metadata và thời gian đã được logger ghi cho cả CLI lẫn notebook.
    distribution = pd.DataFrame(
        {
            "ClassId": np.arange(len(names)),
            "Tên lớp": names,
            "Train": np.bincount(y_tr, minlength=len(names)),
            "Test": np.bincount(y_te, minlength=len(names)),
        }
    )
    print(distribution.to_string(index=False))
