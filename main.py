from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Применяем --device до импорта тяжёлых библиотек
from core.utils import apply_device_from_argv, patch_cv2_set_num_threads

apply_device_from_argv()
patch_cv2_set_num_threads()

from docling.datamodel.base_models import InputFormat

from core.converter import build_converter, convert_pdf
from core.qwen_reader import QwenBlockReader, DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH
from core.utils import clear_cuda_cache

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converter (Docling + PaddleOCR + Qwen2-VL): PDF -> Markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Директория с PDF")
    parser.add_argument("--output-dir", type=Path, required=True, help="Каталог результатов")
    parser.add_argument("--max-files", type=int, default=None, help="Ограничить число файлов (для отладки)")
    parser.add_argument("--no-ocr", action="store_true", help="Отключить OCR")
    parser.add_argument("--no-table-structure", action="store_true", help="Отключить TableFormer")
    parser.add_argument("--full-quality", action="store_true", default=True, help="images_scale=2.0, ACCURATE таблицы")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Устройство для моделей (auto = автоопределение)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Пропускать PDF, если .md уже есть")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Путь к весам Qwen")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace ID для скачивания модели")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"[X] {args.input_dir} не является директорией", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.input_dir.glob("*.pdf"))
    if args.max_files is not None:
        pdf_files = pdf_files[: args.max_files]

    if not pdf_files:
        print(f"[X] Нет PDF-файлов в {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.skip_existing:
        before = len(pdf_files)
        pdf_files = [p for p in pdf_files if not (args.output_dir / f"{p.stem}.md").is_file()]
        skipped = before - len(pdf_files)
        if skipped:
            print(f"[=>]  Пропущено (--skip-existing): {skipped}, к обработке: {len(pdf_files)}\n")
        if not pdf_files:
            print("[<3] Нечего обрабатывать — все .md уже есть.")
            sys.exit(0)

    print("[O] Инициализация моделей (Docling, Qwen2-VL, PaddleOCR)...")

    qwen_reader = QwenBlockReader(
        model_path=args.model_path,
        model_id=args.model_id,
    )

    converter = build_converter(
        no_ocr=args.no_ocr,
        no_table_structure=args.no_table_structure,
        full_quality=args.full_quality,
    )
    converter.initialize_pipeline(InputFormat.PDF)

    print(f"[O] Найдено PDF: {len(pdf_files)}\n")

    for i, pdf_path in enumerate(pdf_files, 1):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{i}/{len(pdf_files)}] {ts} — {pdf_path.name}", flush=True)
        try:
            convert_pdf(pdf_path, args.output_dir, converter, qwen_reader)
            print(f"    [<3] OK")
        except Exception as e:
            print(f"    [X] ОШИБКА: {e}")
        finally:
            clear_cuda_cache()

    print(f"\n[<3] Готово! Результаты в: {args.output_dir}")


if __name__ == "__main__":
    main()
