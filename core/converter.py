import re
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    ThreadedPdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions
import docling_core.types.doc.document as doc_types

from core.noise_tables_ocr import ocr_table_to_markdown


_AI_PATTERNS = [
    r'\[System\].*?(?=\n\n|\Z)',
    r'\[AI\].*?(?=\n\n|\Z)',
    r'\[USER\].*?(?=\n\n|\Z)',
    r'Please note.*?(?=\n\n|\Z)',
    r'The input looks.*?(?=\n\n|\Z)',
    r'This transcription.*?(?=\n\n|\Z)',
    r'The main content.*?(?=\n\n|\Z)',
    r'Please feel free.*?(?=\n\n|\Z)',
]

_WATERMARK_PATTERNS = [
    r'ЧЕРНОВИК',
    r'ДРАФТ',
    r'DRAFT',
    r'PREVIEW',
    r'CONFIDENTIAL',
    r'КОНФИДЕНЦИАЛЬНО',
]

_COMPILED_AI = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _AI_PATTERNS]
_COMPILED_WATERMARKS = [re.compile(p, re.IGNORECASE) for p in _WATERMARK_PATTERNS]

_BULLET_PATTERNS = [
    r'^\s*·\s+',   # · middle dot
    r'^\s*•\s+',   # • bullet
    r'^\s*◦\s+',   # ◦ white bullet
    r'^\s*▪\s+',   # ▪ black small square
    r'^\s*▫\s+',   # ▫ white small square
    r'^\s*−\s+',   # − minus
    r'^\s*–\s+',   # – en dash
]
_NUMBERED_PATTERN = re.compile(r'^\s*(\d+)[\.\)]\s+')


def remove_unwanted_patterns(text: str) -> str:
    """Удаляет AI-артефакты и водяные знаки из текста."""

    if not text or not text.strip():
        return text

    for pattern in _COMPILED_AI:
        match = pattern.search(text)
        if match:
            text = text[:match.start()]
            break

    for pattern in _COMPILED_WATERMARKS:
        text = pattern.sub('', text)

    lines = [line for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)


def normalize_lists(text: str) -> str:
    """Конвертирует различные символы буллетов в Markdown-формат (-)."""

    lines = text.split('\n')
    result = []

    for line in lines:
        matched = False

        for pattern in _BULLET_PATTERNS:
            if re.match(pattern, line):
                result.append(re.sub(pattern, '- ', line))
                matched = True
                break

        if not matched:
            num_match = _NUMBERED_PATTERN.match(line)
            if num_match:
                result.append(_NUMBERED_PATTERN.sub(f'{num_match.group(1)}. ', line))
            else:
                result.append(line)

    return '\n'.join(result)


def clean_text(text: str) -> str:
    """Объединяет normalize_lists и remove_unwanted_patterns."""

    return remove_unwanted_patterns(normalize_lists(text))


# ---------------------------------------------------------------------------
# Построение конвертера
# ---------------------------------------------------------------------------

def build_converter(
    no_ocr: bool = False,
    no_table_structure: bool = False,
    full_quality: bool = True,
) -> DocumentConverter:
    """
    Создаёт и настраивает Docling DocumentConverter.

    Args:
        no_ocr:              Отключить OCR (для нативных PDF)
        no_table_structure:  Отключить структурный анализ таблиц
        full_quality:        True = scale 2.0 (лучше для мелких цифр в шумных таблицах)
    """
    images_scale = 2.0 if full_quality else 1.5

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=not no_ocr,
        do_table_structure=not no_table_structure,
        generate_picture_images=True,
        images_scale=images_scale,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE
        ),
        accelerator_options=AcceleratorOptions(),
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def get_doc_id(stem: str) -> str:
    """ 'document_051' → '51' """

    match = re.search(r'document_0*(\d+)', stem)
    return match.group(1) if match else stem


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    converter: DocumentConverter,
    qwen_reader,
) -> None:
    """
    Конвертирует один PDF в Markdown с изображениями.

    Args:
        pdf_path:    Путь к исходному PDF
        output_dir:  Папка для результатов (*.md + images/)
        converter:   Настроенный Docling DocumentConverter
        qwen_reader: Инициализированный QwenBlockReader
    """
    stem = pdf_path.stem
    doc_id = get_doc_id(stem)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[O] Обработка {pdf_path.name}...")

    try:
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as e:
        print(f"[X] Ошибка Docling на файле {stem}: {e}")
        return

    md_blocks: list[str] = []
    image_order = 1

    for element, level in doc.iterate_items(with_groups=True):

        # --- Заголовки ---
        if isinstance(element, doc_types.SectionHeaderItem):
            if not element.text.strip():
                continue
            text = clean_text(element.text)
            md_blocks.append(f"{'#' * element.level} {text}\n")

        # --- Обычный текст и элементы списков ---
        elif isinstance(element, (doc_types.TextItem, doc_types.ListItem)):
            if not element.text.strip():
                continue
            text = clean_text(element.text)
            md_blocks.append(text + "  ")

        # --- Таблицы ---
        elif isinstance(element, doc_types.TableItem):
            crop = element.get_image(doc)
            if crop:
                print("    [O] Распознаём таблицу через Qwen...")
                table_md = qwen_reader.read_complex_block(crop, prompt_type="table")
            else:
                table_md = element.export_to_markdown(doc=doc)

            md_blocks.append(clean_text(table_md))

        # --- Изображения ---
        elif isinstance(element, doc_types.PictureItem):
            crop = element.get_image(doc)
            if crop is None:
                continue

            decision = qwen_reader.classify_element(crop)

            if any(label in decision for label in ("IMAGE", "FIGURE", "DIAGRAM")):
                img_filename = f"doc_{doc_id}_image_{image_order}.png"
                crop.save(images_dir / img_filename, format="PNG")
                md_blocks.append(f"![{img_filename}](images/{img_filename})")
                image_order += 1

            elif any(label in decision for label in ("TEXT", "HAND")):
                ocr_text = qwen_reader.read_complex_block(crop, prompt_type="text")
                if ocr_text.strip():
                    md_blocks.append(clean_text(ocr_text) + "  ")

            elif "TABLE" in decision:
                table_md = ocr_table_to_markdown(crop)
                md_blocks.append(clean_text(table_md))



    output_path = output_dir / f"{stem}.md"
    output_path.write_text("\n\n".join(md_blocks), encoding="utf-8")
    print(f"[<3] Сохранено: {output_path}")
