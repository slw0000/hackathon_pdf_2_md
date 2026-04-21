from __future__ import annotations

import argparse
import os
import sys
import torch
from datetime import datetime
from pathlib import Path
import gc

from ultralytics import YOLO

from core.utils import apply_device_from_argv, patch_cv2_set_num_threads
from core.qwen_reader import QwenBlockReader

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


apply_device_from_argv()
patch_cv2_set_num_threads()

from docling.datamodel.base_models import InputFormat

from core.converter import build_converter, convert_pdf
from core.utils import clear_cuda_cache

def main() -> None:
    parser = argparse.ArgumentParser(description="Converter (Docling + YoLo + Qwen): PDF → Markdown")
    parser.add_argument("--input-dir", type=Path, required=True, help="Директория с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Каталог результатов")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Ограничить число файлов (для отладки)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Отключить OCR (быстрее на PDF с текстовым слоем; сканы будут хуже)",
    )
    parser.add_argument(
        "--no-table-structure",
        action="store_true",
        help="Отключить TableFormer (быстрее; при проблемах с opencv/cv2)",
    )
    parser.add_argument(
        "--full-quality",
        action="store_true",
        default=True,
        help="Полное качество: images_scale=3.0 и точные таблицы (медленнее)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Устройство для моделей Docling (auto = по умолчанию). "
        "Задаётся до импорта через DOCLING_DEVICE.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Не обрабатывать PDF, если в output уже есть одноимённый .md (продолжение после обрыва).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ОШИБКА: {args.input_dir} не является директорией", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files is not None:
        pdf_files = pdf_files[: args.max_files]

    if not pdf_files:
        print(f"ОШИБКА: нет PDF-файлов в {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.skip_existing:
        before = len(pdf_files)
        pdf_files = [p for p in pdf_files if not (args.output_dir / f"{p.stem}.md").is_file()]
        skipped = before - len(pdf_files)
        if skipped:
            print(f"Пропуск (--skip-existing): уже есть {skipped} .md, к обработке {len(pdf_files)} PDF\n")
        if not pdf_files:
            print("Нечего обрабатывать (все .md уже есть).", flush=True)
            sys.exit(0)

    dev = os.environ.get("DOCLING_DEVICE", "auto")
    print("Инициализация моделей (Docling, Qwen2-VL, PaddleOCR)...")

    # 1. Загружаем Qwen
    qwen_reader = QwenBlockReader("weights/qwen")

    # 2. Билдим Docling (для рендеринга страниц)
    converter = build_converter(
        no_ocr=args.no_ocr,
        no_table_structure=args.no_table_structure,
        full_quality=args.full_quality,
    )
    converter.initialize_pipeline(InputFormat.PDF)

    # 3. Загружаем YoLo
    table_yolo = YOLO('weights/layout/yolov8x-doclaynet.pt')

    print(f"Найдено {len(pdf_files)} PDF-файлов\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"================================= {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} [{i}/{len(pdf_files)}] {pdf_path.name}... =================================", end=" ",
              flush=True)
        try:
            # Вызываем новую функцию пайплайна
            convert_pdf(pdf_path, args.output_dir, converter, qwen_reader, table_yolo)
            print("OK")
        except Exception as e:
            print(f"ОШИБКА: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            gc.collect()

            import paddle
            paddle.device.cuda.empty_cache() if paddle.device.is_compiled_with_cuda() else None

    print(f"\nГотово! Результаты в папке: {args.output_dir}")


if __name__ == "__main__":
    main()