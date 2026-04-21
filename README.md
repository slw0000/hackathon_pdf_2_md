# PDF → Markdown Pipeline

## Команда: **pySwag**

Состав команды:
1. Голубев Артём
2. Жохов Даниил
3. Орлов Михаил

---

### Решение хакатона «Парсинг PDF в Markdown формат».

Пайплайн преобразует PDF-документы в структурированный Markdown с сохранением текста, таблиц и изображений. Поддерживает все 19 типов контента из задания: нативный текст, сканы, рукописный текст, растровые таблицы, многоколоночную вёрстку, объединённые ячейки и другие сложные случаи.

---

## Архитектура решения

```
PDF
 │
 ▼
Docling (структурный анализ)
 │
 ├── SectionHeaderItem ──────────────────────► # Заголовок (уровни 1–4)
 │
 ├── TextItem / ListItem ─────────────────────► Параграф / - список
 │
 ├── TableItem
 │    └── нативная таблица ────────────────────► Docling TableFormer → Markdown
 │
 └── PictureItem ──► Qwen2-VL classify_element()
      ├── IMAGE / FIGURE / DIAGRAM ────────────► сохранить PNG в images/
      ├── TEXT / HAND ─────────────────────────► Qwen2-VL OCR → текст
      ├── TABLE ───────────────────────────────► PaddleOCR → Markdown-таблица
      └── TRASH ───────────────────────────────► пропустить
```

Три модели закрывают разные сложные случаи:

- **Docling + TableFormer** — нативные PDF: извлечение структуры, текста, таблиц
- **Qwen2-VL-2B-Instruct** — мультимодальный OCR: растровые блоки, сканы, рукопись, классификация PictureItem
- **PaddleOCR (ru)** — лёгкий OCR для таблиц, отрендеренных как изображение

Колонтитулы, номера страниц и водяные знаки фильтруются через docling и паттерны постобработки.

---

## Структура репозитория

```
.
├── main.py                    # Точка входа, CLI
├── core/
│   ├── __init__.py
│   ├── converter.py           # Основной пайплайн: итерация по блокам PDF
│   ├── qwen_reader.py         # Qwen2-VL: классификация и OCR блоков
│   ├── noise_tables_ocr.py    # PaddleOCR: OCR таблиц из растровых изображений
│   └── utils.py               # Утилиты: очистка кэша, нормализация имён и др.
├── weights/
│   └── qwen/                  # Веса Qwen2-VL (скачиваются автоматически)
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd <repo-dir>
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Установить зависимости

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями.

Установить uv (если ещё нет):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Установить зависимости (виртуальное окружение создаётся автоматически):

```bash
uv sync
```


### 4. Веса модели

**Qwen2-VL** скачивается **автоматически** при первом запуске в `weights/qwen/` (~4 GB).

Для ручной загрузки:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir weights/qwen
```

---

## Запуск

```bash
python main.py --input-dir <папка с PDF> --output-dir <папка для результатов>
```

### Параметры

| Параметр               | По умолчанию                         | Описание                                        |
|------------------------|--------------------------------------|-------------------------------------------------|
| `--input-dir`          | обязательный                         | Папка с PDF-файлами                             |
| `--output-dir`         | обязательный                         | Папка для результатов                           |
| `--device`             | `auto`                               | Устройство: `auto`, `cuda`, `mps`, `cpu`        |
| `--max-files`          | все файлы                            | Ограничить число файлов (для отладки)           |
| `--skip-existing`      | выкл.                                | Пропускать PDF, если `.md` уже есть             |
| `--no-ocr`             | выкл.                                | Отключить OCR (для нативных PDF)                |
| `--no-table-structure` | выкл.                                | Отключить TableFormer                           |
| `--full-quality`       | вкл.                                 | `images_scale=2.0`, режим ACCURATE для таблиц  |
| `--model-path`         | `weights/qwen`                       | Путь к весам Qwen                               |
| `--model-id`           | `Qwen/Qwen2-VL-2B-Instruct`          | HuggingFace ID для автоскачивания               |

### Примеры

```bash
# Стандартный запуск на GPU
python main.py --input-dir data/pdfs/ --output-dir data/output/ --device cuda

# Продолжить прерванную обработку
python main.py --input-dir data/pdfs/ --output-dir data/output/ --skip-existing

# Быстрая проверка на 5 файлах
python main.py --input-dir data/pdfs/ --output-dir data/output/ --max-files 5
```

---

## Формат результатов

Для каждого `document_051.pdf` пайплайн создаёт:

```
output/
├── document_051.md
└── images/
    ├── doc_51_image_1.png
    ├── doc_51_image_2.png
    └── ...
```

Имена изображений строго соответствуют требованиям задания: `doc_<id>_image_<order>.png`, где `<id>` — номер документа без ведущих нулей.

Ссылки в Markdown:

```markdown
![doc_51_image_1.png](images/doc_51_image_1.png)
```

Для блоков `rasterized_pdf`, `scan_image`, `handwritten_ru`, `image_with_table` извлекается только текст через OCR — PNG в `images/` не сохраняется, ссылка в Markdown не вставляется.

---

## Используемые модели

| Модель | Назначение | Лицензия |
|--------|-----------|----------|
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | OCR и классификация блоков | Apache 2.0 |
| [Docling + TableFormer (ACCURATE)](https://github.com/DS4SD/docling) | Структурный анализ PDF и таблиц | MIT |
| [PaddleOCR (ru)](https://github.com/PaddlePaddle/PaddleOCR) | OCR таблиц из растровых изображений | Apache 2.0 |


