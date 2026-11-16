"""Traffic-sign web app and explicit deployment CLI.

``python main.py`` starts the local web interface, not historical experiments.
Use ``train`` explicitly to refit fixed pipelines using Train.csv only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ROOT / "data/models"
DEFAULT_DATASET = ROOT / "data"


def _positive(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("phải là số nguyên dương") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("phải là số nguyên dương")
    return number


def _port(value):
    number = _positive(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("port phải thuộc 1..65535")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nhận diện biển báo GTSRB: web, refit Train.csv và dự đoán ảnh đơn.",
        epilog="Không có lệnh con: mở web tại http://127.0.0.1:5000. Không tự huấn luyện hay đọc Test.csv.")
    parser.set_defaults(command="serve", host="127.0.0.1", port=5000,
                        model_dir=DEFAULT_MODELS, dataset_dir=DEFAULT_DATASET)
    commands = parser.add_subparsers(dest="command")
    commands.required = False
    serve = commands.add_parser("serve", help="Chạy giao diện web Flask (mặc định)")
    serve.add_argument("--host", default="127.0.0.1", help="Mặc định chỉ truy cập từ máy này")
    serve.add_argument("--port", type=_port, default=5000)
    serve.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS,
                       help="Thư mục joblib cục bộ TIN CẬY; không dùng file tải lên")
    serve.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                       help="Nguồn ảnh ví dụ; tự tìm data/Train.csv hoặc data/archive/Train.csv")
    train = commands.add_parser("train", help="Refit tham số cố định từ Train.csv, không CV/test")
    train.add_argument("--features", choices=("pixel", "hog", "hog_pca", "all"), default="hog")
    train.add_argument("--models", choices=("knn", "dt"), nargs="+", default=["knn", "dt"])
    train.add_argument("--dataset-dir", type=Path, default=None,
                       help="Thư mục có Train.csv hoặc cha archive; tự tìm data/archive khi bỏ trống")
    train.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    train.add_argument("--max-per-class", type=_positive, default=None,
                       help="Chỉ dùng cho DEMO; bỏ trống để huấn luyện đầy đủ")
    predict = commands.add_parser("predict", help="Dự đoán một ảnh PNG/JPEG/WebP, xuất JSON")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model-id", help="ID từ lệnh models; bỏ trống để ưu tiên HOG đầy đủ mới nhất")
    predict.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    predict.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                         help="ROI theo ảnh đã xoay EXIF, tính cả pixel cuối")
    catalog = commands.add_parser("models", help="Liệt kê model sẵn sàng/lỗi, xuất JSON")
    catalog.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    command = arguments.command or "serve"
    arguments.model_dir = arguments.model_dir.expanduser().resolve()
    if arguments.dataset_dir is not None:
        arguments.dataset_dir = arguments.dataset_dir.expanduser().resolve()
    try:
        if command == "serve":
            from web.app import create_app
            app = create_app(model_dir=arguments.model_dir, dataset_dir=arguments.dataset_dir)
            # No auto-reloader, debug console, hidden training or remote binding.
            app.run(host=arguments.host, port=arguments.port, debug=False, use_reloader=False)
        elif command == "train":
            from src.deployment import train_models
            train_models(dataset_dir=arguments.dataset_dir, model_dir=arguments.model_dir,
                         features=arguments.features, models=arguments.models,
                         max_per_class=arguments.max_per_class,
                         progress=lambda message: print(message, flush=True))
        elif command == "models":
            from src.inference import ModelRegistry
            print(json.dumps(ModelRegistry(arguments.model_dir).list_models(),
                             ensure_ascii=False, indent=2, allow_nan=False))
        elif command == "predict":
            from src.inference import MAX_IMAGE_BYTES, ModelRegistry
            with arguments.image.open("rb") as stream:
                image = stream.read(MAX_IMAGE_BYTES + 1)
            result = ModelRegistry(arguments.model_dir).predict(
                image, model_id=arguments.model_id, roi=arguments.roi)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except ImportError:
        print("Thiếu thư viện. Cài requirements.txt và requirements-web.txt trong môi trường đang dùng.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Không hoàn tất lệnh: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Đã dừng theo yêu cầu.", file=sys.stderr)
        return 130
    except Exception as error:
        from src.inference import InferenceError
        if isinstance(error, InferenceError):
            print(error.message, file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    # Keep Vietnamese CLI output readable in Windows terminals and redirected logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
