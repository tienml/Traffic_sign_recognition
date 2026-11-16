"""Kiểm thử tiền xử lý bằng ảnh tổng hợp nhỏ, không cần tải GTSRB."""

import inspect
import io
import logging
import os
import struct
import tempfile
import unittest
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import BinaryIO
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd

from src import preprocessing as prep
from src.config import RANDOM_STATE


class PreprocessingTests(unittest.TestCase):
    """Kiểm tra RGB, ROI, nhãn, augmentation và các khóa cache."""

    def setUp(self) -> None:
        """Tạo bộ dữ liệu tạm với đường dẫn Unicode và hai tập riêng."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dữ_liệu"
        self.archive = self.root / "archive"
        self.archive.mkdir(parents=True)
        self.image = np.zeros((8, 10, 3), dtype=np.uint8)
        self.image[:] = (255, 0, 0)
        self.image[2:6, 3:8] = (0, 255, 0)
        encoded = cv2.imencode(
            ".png", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        )[1]
        self.path = self.archive / "ảnh.png"
        encoded.tofile(self.path)
        for name, label in (("Train", 1), ("Test", 2)):
            pd.DataFrame(
                [
                    {
                        "Width": 10,
                        "Height": 8,
                        "Roi.X1": 3,
                        "Roi.Y1": 2,
                        "Roi.X2": 7,
                        "Roi.Y2": 5,
                        "ClassId": label,
                        "Path": "ảnh.png",
                    }
                ]
            ).to_csv(self.archive / f"{name}.csv", index=False)
        self.quiet = patch.object(prep, "tqdm", None)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)
        # Logger riêng từng test; không chạm log repo hay root logger.
        self.log_path = (
            Path(self.temporary.name) / "logs" / "preprocessing.log"
        )
        self.logger = logging.Logger("preprocessing.test")
        self.console = io.StringIO()
        patches = (
            patch.object(prep, "LOG_PATH", self.log_path),
            patch.object(prep, "_LOGGER", self.logger),
            patch("sys.stderr", self.console),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self._close_log_handlers)

    def _close_log_handlers(self) -> None:
        """Đóng stream trước khi Windows xóa thư mục tạm."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def _read_log(self) -> str:
        """Đọc bản ghi UTF-8 trong thư mục tạm sau khi flush."""
        for handler in self.logger.handlers:
            handler.flush()
        return self.log_path.read_text(encoding="utf-8")

    def test_rgb_roi_and_dtype(self) -> None:
        """ROI đóng giữ đủ biên, màu RGB không bị đảo thành BGR."""
        np.testing.assert_array_equal(prep.read_rgb(self.path), self.image)
        output = prep.load_image(self.path, (3, 2, 7, 5))
        self.assertEqual(output.shape, (32, 32, 3))
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(output[0, 0], [0, 1, 0])
        full = prep.load_image(self.path, use_roi=False)
        np.testing.assert_array_equal(full[0, 0], [1, 0, 0])
        np.testing.assert_array_equal(
            prep.crop_roi(self.image, (3, 2, 7, 5)), self.image[2:6, 3:8]
        )

    def test_resize_options_and_default_equality(self) -> None:
        """Đổi size/nội suy cho ablation nhưng mặc định khớp từng pixel cũ."""
        roi = (2, 1, 8, 6)
        cropped = prep.crop_roi(self.image, roi)
        expected = cv2.resize(cropped, (32, 32), interpolation=cv2.INTER_AREA)
        expected = expected.astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(
            prep.load_image(self.path, roi), expected
        )
        np.testing.assert_array_equal(
            prep.load_image(
                self.path,
                roi,
                image_size=(32, 32),
                interpolation=cv2.INTER_AREA,
            ),
            expected,
        )
        for size in ((24, 24), (48, 48), (24, 48)):
            for method in (cv2.INTER_AREA, cv2.INTER_LINEAR):
                with self.subTest(size=size, interpolation=method):
                    actual = prep.load_image(
                        self.path, roi, image_size=size, interpolation=method
                    )
                    expected = cv2.resize(cropped, size, interpolation=method)
                    expected = expected.astype(np.float32) / np.float32(255)
                    self.assertEqual(actual.shape, (size[1], size[0], 3))
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(actual, expected)
        parameters = inspect.signature(prep.load_image).parameters
        for name in ("image_size", "interpolation"):
            self.assertEqual(
                parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
            )
        self.assertIsNone(parameters["image_size"].default)

    def test_resize_runtime_default(self) -> None:
        """IMAGE_SIZE lấy lúc gọi, tương thích các test patch cấu hình."""
        with patch.object(prep, "IMAGE_SIZE", (24, 16)):
            actual = prep.load_image(self.path)
            expected = prep.load_image(self.path, image_size=(24, 16))
            self.assertEqual(actual.shape, (16, 24, 3))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(
                prep.load_image(self.path, image_size=(48, 48)).shape,
                (48, 48, 3),
            )

    def test_resize_invalid_size(self) -> None:
        """Từ chối size sai cấu trúc, boolean, phân số, âm hoặc vượt int32."""
        for size in (
            32,
            (32,),
            (32, 32, 3),
            [32, 32],
            (0, 32),
            (-1, 32),
            (32.0, 32),
            (True, 32),
            (np.bool_(True), 32),
            (np.nan, 32),
            (np.inf, 32),
            (2**31, 32),
        ):
            with self.subTest(size=size):
                with patch.object(prep, "read_rgb") as reader:
                    with self.assertRaisesRegex(ValueError, "image_size"):
                        prep.load_image(self.path, image_size=size)
                    reader.assert_not_called()

    def test_clipped_and_invalid_roi(self) -> None:
        """Kẹp ngoài biên và bỏ qua ROI đảo cận, NaN hoặc không giao ảnh."""
        np.testing.assert_array_equal(
            prep.crop_roi(self.image, (-5, -3, 50, 50)), self.image
        )
        for roi in (
            (7, 5, 3, 2),
            (20, 20, 30, 30),
            (np.nan,) * 4,
            (1, 2),
            (0.5, 1, 3, 4),
            ("sai",) * 4,
        ):
            np.testing.assert_array_equal(
                prep.crop_roi(self.image, roi), self.image
            )
        self.assertEqual(
            prep.crop_roi(self.image, (0, 0, 0, 0)).shape, (1, 1, 3)
        )

    def test_missing_and_corrupt_image(self) -> None:
        """Ảnh thiếu/hỏng phải báo lỗi kèm tên file, không bỏ qua mẫu."""
        with self.assertRaisesRegex(OSError, "missing.png"):
            prep.load_image(self.archive / "missing.png")
        broken = self.archive / "broken.png"
        broken.write_bytes(b"not an image")
        with self.assertRaisesRegex(OSError, "broken.png"):
            prep.load_image(broken)

    def test_grayscale_and_augmentation(self) -> None:
        """Grayscale một kênh; augmentation tắt và tái lập đúng seed."""
        image = prep.load_image(self.path)
        gray = prep.to_grayscale(image[None])
        self.assertEqual(gray.shape, (1, 32, 32, 1))
        self.assertEqual(gray.dtype, np.float32)
        np.testing.assert_array_equal(prep.augment_image(image), image)
        first = prep.augment_image(
            image, enabled=True, rng=np.random.default_rng(RANDOM_STATE)
        )
        second = prep.augment_image(
            image, enabled=True, rng=np.random.default_rng(RANDOM_STATE)
        )
        np.testing.assert_array_equal(first, second)
        self.assertGreaterEqual(float(first.min()), 0)
        self.assertLessEqual(float(first.max()), 1)
        with self.assertRaises(ValueError):
            prep.augment_image(image, enabled=True, max_rotation=90)

    def test_split_validation(self) -> None:
        """Giữ nhãn gốc và từ chối nhãn phân số hay Path vượt thư mục."""
        csv_path = self.archive / "Train.csv"
        x_train, y_train = prep.load_split(csv_path, self.archive)
        self.assertEqual(x_train.shape, (1, 32, 32, 3))
        np.testing.assert_array_equal(y_train, [1])
        frame = pd.read_csv(csv_path)
        frame["ClassId"] = 1.5
        frame.to_csv(csv_path, index=False)
        with self.assertRaises(ValueError):
            prep.load_split(csv_path, self.archive)
        frame["ClassId"] = 1
        frame["Path"] = "../outside.png"
        frame.to_csv(csv_path, index=False)
        with self.assertRaisesRegex(ValueError, "ngoài"):
            prep.load_split(csv_path, self.archive)

    def test_official_split_and_cache_hit(self) -> None:
        """Nhãn hai tập không bị chia lại; cache không đọc lại ảnh."""
        self.assertEqual(
            list(inspect.signature(prep.load_and_preprocess).parameters),
            ["data_dir"],
        )
        with patch.object(prep, "USE_CACHE", True):
            first = prep.load_and_preprocess(str(self.root))
            with patch.object(prep, "load_split", side_effect=AssertionError):
                second = prep.load_and_preprocess(str(self.root))
        for actual, expected in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(first[2], [1])
        np.testing.assert_array_equal(first[3], [2])
        self.assertEqual(len(first[4]), 43)

    def test_cache_invalidation(self) -> None:
        """Cấu hình size/ROI, CSV đổi và cache hỏng đều buộc đọc lại."""
        with patch.object(prep, "USE_CACHE", True):
            prep.load_and_preprocess(str(self.root))
            key = prep._source_key(self.archive)
            cache_path = self.root / "cache" / "gtsrb_32.npz"
            with patch.object(prep, "IMAGE_SIZE", (16, 16)):
                self.assertIsNone(prep._read_cache(cache_path, key))
            with patch.object(prep, "USE_ROI", False):
                self.assertIsNone(prep._read_cache(cache_path, key))
            frame = pd.read_csv(self.archive / "Train.csv")
            frame["ClassId"] = 3
            frame.to_csv(self.archive / "Train.csv", index=False)
            self.assertIsNone(
                prep._read_cache(cache_path, prep._source_key(self.archive))
            )
            cache_path.write_bytes(b"broken cache")
            with self.assertWarns(UserWarning):
                self.assertIsNone(prep._read_cache(cache_path, key))

    @unittest.skipUnless(os.name == "nt", "Lỗi chuyển mã riêng Windows")
    def test_unicode_filename_collision(self) -> None:
        """Không đọc nhầm file có tên là bản chuyển mã lỗi của tên Unicode."""
        folder = Path(self.temporary.name)
        expected = np.full((8, 10, 3), (255, 0, 0), dtype=np.uint8)
        decoy = np.full((8, 10, 3), (0, 0, 255), dtype=np.uint8)
        for name, image in (("é.png", expected), ("Ã©.png", decoy)):
            encoded = cv2.imencode(
                ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )[1]
            encoded.tofile(folder / name)
        np.testing.assert_array_equal(
            prep.read_rgb(folder / "é.png"), expected
        )

    def test_cache_malformed_metadata(self) -> None:
        """Cache có IMAGE_SIZE dạng scalar phải bị bỏ thay vì gây TypeError."""
        cache_path = self.root / "malformed.npz"
        np.savez_compressed(
            cache_path, version=prep.CACHE_VERSION, image_size=np.asarray(32)
        )
        with self.assertWarns(UserWarning):
            self.assertIsNone(prep._read_cache(cache_path, "unused"))

    def test_cache_corrupt_deflate(self) -> None:
        """Luồng DEFLATE hỏng trong NPZ phải được bỏ qua và cảnh báo."""
        cache_path = self.root / "corrupt.npz"
        np.savez_compressed(cache_path, version=prep.CACHE_VERSION)
        with ZipFile(cache_path) as archive:
            offset = archive.getinfo("version.npy").header_offset
        payload = bytearray(cache_path.read_bytes())
        name_size, extra_size = struct.unpack_from("<HH", payload, offset + 26)
        compressed_offset = offset + 30 + name_size + extra_size
        # BTYPE=3 là kiểu block DEFLATE cấm; giữ cấu trúc ZIP nguyên vẹn.
        payload[compressed_offset] |= 0b110
        cache_path.write_bytes(payload)
        with ZipFile(cache_path) as archive:
            with self.assertRaises(zlib.error):
                archive.read("version.npy")
        with self.assertWarns(UserWarning):
            self.assertIsNone(prep._read_cache(cache_path, "unused"))

    def test_cache_directory_unavailable(self) -> None:
        """Không tạo được thư mục cache vẫn trả các mảng đã tiền xử lý."""
        (self.root / "cache").write_bytes(b"not a directory")
        with patch.object(prep, "USE_CACHE", True):
            with self.assertWarns(UserWarning):
                result = prep.load_and_preprocess(str(self.root))
        self.assertEqual(result[0].shape, (1, 32, 32, 3))
        np.testing.assert_array_equal(result[2], [1])

    def test_cache_concurrent_writers(self) -> None:
        """Hai lượt ghi đồng thời dùng file tạm riêng, không xóa lẫn nhau."""
        barrier = Barrier(2)
        paths = []
        real_save = np.savez_compressed

        def synchronized_save(
            destination: str | Path | BinaryIO, **arrays: np.ndarray
        ) -> None:
            """Giữ hai lượt ở cùng bước ghi để tái hiện tranh chấp cache."""
            path = (
                destination.name
                if hasattr(destination, "write")
                else destination
            )
            paths.append(Path(path))
            real_save(destination, **arrays)
            barrier.wait(timeout=20)

        with patch.object(prep, "USE_CACHE", True):
            with patch.object(prep, "_read_cache", return_value=None):
                with patch.object(
                    prep.np, "savez_compressed", side_effect=synchronized_save
                ):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        results = list(
                            executor.map(
                                prep.load_and_preprocess, [str(self.root)] * 2
                            )
                        )
        self.assertEqual(len(set(paths)), 2)
        for result in results:
            np.testing.assert_array_equal(result[2], [1])
            self.assertEqual(result[0].shape, (1, 32, 32, 3))
        cache_dir = self.root / "cache"
        self.assertFalse(list(cache_dir.glob("*.tmp.npz")))
        self.assertIsNotNone(
            prep._read_cache(
                cache_dir / "gtsrb_32.npz", prep._source_key(self.archive)
            )
        )

    def test_cache_cleanup_failure(self) -> None:
        """Lỗi ghi và xóa tệp cache tạm chỉ cảnh báo, không mất dữ liệu."""
        with patch.object(prep, "USE_CACHE", True):
            with patch.object(
                prep.np, "savez_compressed", side_effect=OSError("hết chỗ")
            ):
                with patch.object(
                    Path, "unlink", side_effect=PermissionError("file bị khóa")
                ):
                    with self.assertWarns(UserWarning):
                        result = prep.load_and_preprocess(str(self.root))
        self.assertEqual(result[0].shape, (1, 32, 32, 3))

    def test_cache_disabled(self) -> None:
        """Tắt cache không tạo file và vẫn trả đúng hợp đồng."""
        with patch.object(prep, "USE_CACHE", False):
            result = prep.load_and_preprocess(str(self.root))
        self.assertFalse((self.root / "cache").exists())
        self.assertEqual(result[0].dtype, np.float32)

    def test_logging_source_cache_hit_and_disabled(self) -> None:
        """Mỗi lượt có cấu hình, metadata, trạng thái cache và thời gian."""
        with patch.object(prep, "USE_CACHE", True):
            prep.load_and_preprocess(str(self.root))
            with patch.object(prep, "load_split", side_effect=AssertionError):
                prep.load_and_preprocess(str(self.root))
        with patch.object(prep, "USE_CACHE", False):
            prep.load_and_preprocess(str(self.root))
        log = self._read_log()
        for state in ("miss", "hit", "disabled"):
            self.assertIn(f"cache={state}", log)
        self.assertEqual(log.count("Bắt đầu load_and_preprocess"), 3)
        self.assertEqual(log.count("Hoàn tất load_and_preprocess"), 3)
        self.assertEqual(log.count("Train=1 Test=1 class_names=43"), 3)
        self.assertEqual(log.count("elapsed="), 7)
        self.assertIn(f"seed={RANDOM_STATE}", log)
        self.assertIn("IMAGE_SIZE=(32, 32) USE_ROI=True USE_CACHE=True", log)
        self.assertIn("USE_CACHE=False", log)
        for split in ("Train", "Test"):
            self.assertEqual(log.count(f"{split}: X shape=(1, 32, 32, 3)"), 3)
        self.assertIn("dtype=float32 min=0.000000 max=1.000000", log)
        self.assertIn("y shape=(1,) dtype=int64 min=1 max=1 classes=1", log)
        self.assertIn("errors=0 skipped=0", log)
        self.assertRegex(log, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO")
        self.assertNotIn(str(self.root), log)
        self.assertNotIn(self.root.as_posix(), log)
        self.assertEqual(log, self.console.getvalue())

    def test_logging_error_counts_fail_fast(self) -> None:
        """Ảnh lỗi dừng ngay, ghi loaded/errors/skipped, không báo hoàn tất."""
        csv_path = self.archive / "Test.csv"
        frame = pd.read_csv(csv_path)
        frame = pd.concat([frame] * 3, ignore_index=True)
        frame.loc[1, "Path"] = "ảnh_hỏng.png"
        frame.to_csv(csv_path, index=False)
        (self.archive / "ảnh_hỏng.png").write_bytes(b"not an image")
        with patch.object(prep, "USE_CACHE", False):
            with patch.object(
                prep, "load_image", wraps=prep.load_image
            ) as load:
                with self.assertRaisesRegex(OSError, "ảnh_hỏng.png"):
                    prep.load_and_preprocess(str(self.root))
        self.assertEqual(load.call_count, 3)
        log = self._read_log()
        self.assertIn(
            "Thất bại Test: count=3 loaded=1 errors=1 skipped=0", log
        )
        self.assertIn("file=ảnh_hỏng.png error=OSError elapsed=", log)
        self.assertIn(
            "Train_complete=1 Test_complete=0 errors=1 skipped=0", log
        )
        self.assertNotIn("Hoàn tất load_and_preprocess", log)
        self.assertNotIn("Đã nạp Test", log)
        self.assertNotIn(str(self.root), log)
        self.assertNotIn(self.root.as_posix(), log)
        self.assertFalse((self.root / "cache").exists())

    def test_logging_metadata_failure(self) -> None:
        """CSV thiếu cũng có thời gian và trạng thái lỗi của lượt pipeline."""
        (self.archive / "Train.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            prep.load_and_preprocess(str(self.root))
        log = self._read_log()
        self.assertIn("Thất bại load_and_preprocess", log)
        self.assertIn(
            "errors=1 skipped=0 error=FileNotFoundError elapsed=", log
        )
        self.assertNotIn("Hoàn tất", log)

    def test_logging_reuses_handlers_without_root_changes(self) -> None:
        """Gọi lặp hoặc đồng thời không thêm handler, không đổi root logger."""
        root = logging.getLogger()
        root_handlers, root_level = root.handlers[:], root.level
        with ThreadPoolExecutor(max_workers=4) as executor:
            loggers = list(
                executor.map(lambda _: prep._get_logger(), range(16))
            )
        self.assertTrue(all(logger is self.logger for logger in loggers))
        handlers = self.logger.handlers[:]
        self.assertEqual(len(handlers), 2)
        self.assertEqual(
            {handler.name for handler in handlers},
            {"preprocessing.console", "preprocessing.file"},
        )
        self.assertFalse(self.logger.propagate)
        with patch.object(prep, "USE_CACHE", False):
            for _ in range(3):
                prep.load_and_preprocess(str(self.root))
        self.assertEqual(self.logger.handlers, handlers)
        self.assertEqual(root.handlers, root_handlers)
        self.assertEqual(root.level, root_level)
        self.assertEqual(
            self._read_log().count("Hoàn tất load_and_preprocess"), 3
        )

    def test_logging_directory_unavailable(self) -> None:
        """Thư mục log không ghi được chỉ cảnh báo một lần, loader vẫn chạy."""
        self.log_path.parent.write_bytes(b"not a directory")
        with patch.object(prep, "USE_CACHE", False):
            with self.assertWarnsRegex(
                UserWarning, "Không ghi được log"
            ) as caught:
                first = prep.load_and_preprocess(str(self.root))
                second = prep.load_and_preprocess(str(self.root))
        self.assertEqual(len(caught.warnings), 1)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(len(self.logger.handlers), 2)
        self.assertIn("Hoàn tất load_and_preprocess", self.console.getvalue())

    def test_logging_failure_with_warnings_as_errors(self) -> None:
        """Bộ lọc warning thành exception không làm lỗi log phá loader."""
        self.log_path.parent.write_bytes(b"not a directory")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with patch.object(prep, "USE_CACHE", False):
                result = prep.load_and_preprocess(str(self.root))
        self.assertEqual(result[0].shape, (1, 32, 32, 3))
        self.assertIn("Hoàn tất load_and_preprocess", self.console.getvalue())

    def test_logging_write_failure(self) -> None:
        """Lỗi stream sau khi mở file không phá loader hoặc spam cảnh báo."""
        prep._get_logger().info("Mở log trước khi mô phỏng hết dung lượng")
        file_handler = next(
            item
            for item in self.logger.handlers
            if isinstance(item, logging.FileHandler)
        )
        with patch.object(file_handler.stream, "write", side_effect=OSError):
            with patch.object(prep, "USE_CACHE", False):
                with self.assertWarnsRegex(
                    UserWarning, "Không ghi được log"
                ) as caught:
                    result = prep.load_and_preprocess(str(self.root))
        self.assertEqual(len(caught.warnings), 1)
        self.assertEqual(result[0].shape, (1, 32, 32, 3))
        self.assertIn("Hoàn tất load_and_preprocess", self.console.getvalue())


if __name__ == "__main__":
    unittest.main()
