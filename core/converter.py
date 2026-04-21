import os
import re
import torch
from pathlib import Path
from PIL import Image

# Импорты Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
    ThreadedPdfPipelineOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions
import docling_core.types.doc.document as doc_types
from docling_core.types.doc import DocItemLabel

from core.noise_tables_ocr import ocr_table_to_markdown

# core/utils.py
import re

AI_PATTERNS = [
    # AI комментарии от Qwen
    r'\[System\].*?(?=\n\n|\Z)',
    r'\[AI\].*?(?=\n\n|\Z)',
    r'\[USER\].*?(?=\n\n|\Z)',
    r'Please note.*?(?=\n\n|\Z)',
    r'The input looks.*?(?=\n\n|\Z)',
    r'This transcription.*?(?=\n\n|\Z)',
    r'The main content.*?(?=\n\n|\Z)',
    r'Please feel free.*?(?=\n\n|\Z)',
]

WATERMARK_PATTERNS = [
    r'ЧЕРНОВИК',
    r'ДРАФТ',
    r'DRAFT',
    r'PREVIEW',
    r'CONFIDENTIAL',
    r'КОНФИДЕНЦИАЛЬНО',
]

# Компилируем паттерны один раз
COMPILED_AI_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in AI_PATTERNS
]

COMPILED_WATERMARK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in WATERMARK_PATTERNS
]


def remove_unwanted_patterns(text: str) -> str:
    """ Удаляет весь контент блока после нежелательного текста """
    if not text or not text.strip():
        return text

    for pattern in COMPILED_AI_PATTERNS:
        match = pattern.search(text)
        if match:

            text = text[:match.start()]
            break

    for pattern in COMPILED_WATERMARK_PATTERNS:
        text = pattern.sub('', text)

    lines = text.split('\n')
    lines = [line for line in lines if line.strip()]

    return '\n'.join(lines)


def normalize_lists(text: str) -> str:
    """
    Конвертирует различные буллеты в Markdown формат (-)
    """
    lines = text.split('\n')
    result = []

    # Паттерны буллетов
    bullet_patterns = [
        r'^\s*·\s+',  # · (middle dot)
        r'^\s*•\s+',  # • (bullet)
        r'^\s*◦\s+',  # ◦ (white bullet)
        r'^\s*▪\s+',  # ▪ (black small square)
        r'^\s*▫\s+',  # ▫ (white small square)
        r'^\s*−\s+',  # − (minus)
        r'^\s*–\s+',  # – (en dash)
    ]

    # Паттерн нумерованных списков
    numbered_pattern = r'^\s*(\d+)[\.\)]\s+'

    in_list = False
    list_indent = 0

    for line in lines:
        # Проверка на буллет
        is_bullet = False
        for pattern in bullet_patterns:
            if re.match(pattern, line):
                # Заменяем на Markdown буллет
                normalized = re.sub(pattern, '- ', line)
                result.append(normalized)
                in_list = True
                is_bullet = True
                break

        # Проверка на нумерованный список
        if not is_bullet:
            match = re.match(numbered_pattern, line)
            if match:
                num = match.group(1)
                normalized = re.sub(numbered_pattern, f'{num}. ', line)
                result.append(normalized)
                in_list = True
                continue

        # Если не список — сбрасываем флаг
        if not is_bullet and not re.match(numbered_pattern, line):
            if line.strip() == '':
                result.append(line)
            else:
                result.append(line)
                in_list = False

    return '\n'.join(result)


def build_converter(no_ocr: bool, no_table_structure: bool, full_quality: bool) -> DocumentConverter:
    """
    Создает и настраивает конвертер.
    full_quality=True критичен для распознавания мелких цифр в таблицах.
    """
    images_scale = 2.0 if full_quality else 1.5  # Scale 3.0 дает лучший OCR на сканах

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=not no_ocr,
        do_table_structure=not no_table_structure,
        generate_picture_images=True,  # ОБЯЗАТЕЛЬНО: без этого element.get_image() вернет None
        images_scale=images_scale,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE  # Используем самую точную модель для таблиц
        ),
        accelerator_options=AcceleratorOptions(),  # Использует GPU если доступно
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def get_doc_id(stem: str) -> str:
    """document_051 -> 51"""
    match = re.search(r'document_0*(\d+)', stem)
    return match.group(1) if match else stem


def convert_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter, qwen_reader, yolo_reader) -> None:
    stem = pdf_path.stem
    doc_id = get_doc_id(stem)

    # Создаем папку images в корне submission
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[🚀] Обработка {pdf_path.name}...")

    try:
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as e:
        print(f"❌ Ошибка Docling на файле {stem}: {e}")
        return

    final_md_blocks = []
    image_order = 1

    for element, level in doc.iterate_items(
            with_groups=True
    ):

        # 1. Header Items
        if isinstance(element, doc_types.SectionHeaderItem):
            if not element.text.strip(): continue

            text = remove_unwanted_patterns(normalize_lists(element.text))
            final_md_blocks.append("#" * element.level + " " + text + "\n")

        # 2. Text Items
        elif isinstance(element, doc_types.TextItem) or isinstance(element, doc_types.ListItem):
            if not element.text.strip(): continue

            text = remove_unwanted_patterns(normalize_lists(element.text))
            final_md_blocks.append(text + "  ")

        # 3. List Items
        elif isinstance(element, doc_types.ListItem):
            if element.text.strip():
                marker = '-'
                indent = "  " * (level - 1)  # Отступ по уровню вложенности

                final_md_blocks.append(f"{indent}{marker} {element.text}\n")

        # 4. Table Items
        elif isinstance(element, doc_types.TableItem):
            table_md = element.export_to_markdown(doc=doc)

            crop = element.get_image(doc)
            if crop:
                print(f"    [📊] Распознаем растровую таблицу через Qwen...")
                table_md = qwen_reader.read_complex_block(crop, prompt_type="table")

            text = remove_unwanted_patterns(normalize_lists(table_md))
            final_md_blocks.append(text)

        # 5. Picture Items
        elif isinstance(element, doc_types.PictureItem):
            crop = element.get_image(doc)
            # crop.save(f"temp/{c}.png")
            # c += 1
            if crop is None: continue

            # Спрашиваем Qwen, нужно ли это сохранять (PHOTO) или это текст (TEXT)
            decision = qwen_reader.classify_element(crop)
            # print(f"РЕШЕНИЕ: ", decision)

            if "IMAGE" in decision or "FIGURE" in decision or "DIAGRAM" in decision:
                # Формат doc_<id>_image_<order>.png строго по ТЗ
                img_filename = f"doc_{doc_id}_image_{image_order}.png"
                img_path = images_dir / img_filename
                crop.save(img_path, format="PNG")

                final_md_blocks.append(f"![{img_filename}](images/{img_filename})")
                image_order += 1

            elif "TEXT" in decision or "HAND" in decision:
                # Для scan_image, handwritten_ru и т.д. по ТЗ сохраняем только текст
                ocr_text = qwen_reader.read_complex_block(crop, prompt_type="text")
                if ocr_text.strip():
                    text = remove_unwanted_patterns(normalize_lists(ocr_text))
                    final_md_blocks.append(text + "  ")

            elif "TABLE" in decision:

                table_md = ocr_table_to_markdown(crop)
                text = remove_unwanted_patterns(normalize_lists(table_md))

                final_md_blocks.append(text)

    md_output_path = output_dir / f"{stem}.md"
    md_output_path.write_text("\n\n".join(final_md_blocks), encoding="utf-8")