import os
import re
import shutil
import gc
from pathlib import Path
import sys

IMG_LINK_RE = re.compile(
    r"images/(image_\d+_[a-f0-9]+\.(?:png|jpe?g))",
    flags=re.IGNORECASE,
)


def clear_cuda_cache() -> None:
    """Очищает кэш памяти для всех доступных ускорителей: CUDA, MPS, Paddle."""
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass

    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except ImportError:
        pass


def doc_num_from_stem(stem: str) -> int | None:
    """``document_051`` → 51."""
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def move_or_convert_to_png(src: Path, dst: Path) -> None:
    """Сохранить только PNG: JPEG конвертируем, остальное переносим как есть."""
    ext = src.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        from PIL import Image
        with Image.open(src) as im:
            im.save(dst, format="PNG")
        src.unlink()
    else:
        shutil.move(str(src), str(dst))


def normalize_image_names(
    markdown: str,
    work_images_dir: Path,
    out_images_dir: Path,
    doc_num: int,
) -> str:
    """Переименовать ``image_*_*.(png|jpg)`` → ``doc_<n>_image_<k>.png`` и обновить ссылки в тексте."""
    out_images_dir.mkdir(parents=True, exist_ok=True)
    old_to_new: dict[str, str] = {}
    order = 1

    for m in IMG_LINK_RE.finditer(markdown):
        old_name = m.group(1)
        if old_name in old_to_new:
            continue
        src = work_images_dir / old_name
        if not src.is_file():
            continue
        new_name = f"doc_{doc_num}_image_{order}.png"
        old_to_new[old_name] = new_name
        move_or_convert_to_png(src, out_images_dir / new_name)
        order += 1

    result = markdown
    # Сортируем по убыванию длины, чтобы не было частичных замен
    for old_name, new_name in sorted(old_to_new.items(), key=lambda kv: len(kv[0]), reverse=True):
        result = result.replace(f"images/{old_name}", f"images/{new_name}")
    return result


def apply_device_from_argv() -> None:
    """Читает --device из argv и пишет в DOCLING_DEVICE до импорта тяжёлых библиотек."""
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            if val != "auto":
                os.environ["DOCLING_DEVICE"] = val
            return
        if arg.startswith("--device="):
            val = arg.split("=", 1)[1]
            if val != "auto":
                os.environ["DOCLING_DEVICE"] = val
            return


def patch_cv2_set_num_threads() -> None:
    """TableFormer вызывает cv2.setNumThreads; патчим если атрибута нет."""
    try:
        import cv2
    except ImportError:
        return
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda _nthreads: None  # type: ignore[method-assign]
