import os
import tempfile
from typing import List, Dict, Union

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPS_ENABLE_GRAPH_CAPTURE"] = "0"
os.environ["PADDLE_DEVICE"] = "cpu"

from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np


class TableOCR:
    """Универсальный OCR для таблиц с автоматическим определением структуры."""

    def __init__(
            self,
            lang: str = 'ru',
            y_threshold: float = 20,
            x_gap_threshold: float = 80,
            score_threshold: float = 0.3,
    ):
        self.ocr = PaddleOCR(lang=lang, use_textline_orientation=True)
        self.y_threshold = y_threshold
        self.x_gap_threshold = x_gap_threshold
        self.score_threshold = score_threshold

    def extract_table(
            self,
            image: Union[str, Image.Image],
            add_header_separator: bool = True,
            header_row_index: int = 0,
    ) -> str:
        """
        Извлечение таблицы из изображения в Markdown.

        Args:
            image:               Путь к файлу (str) или PIL Image
            add_header_separator: Добавить разделитель после заголовка
            header_row_index:    Индекс строки заголовка

        Returns:
            Markdown строка с таблицей
        """
        temp_path = None
        if isinstance(image, Image.Image):
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            img_rgb = image.convert('RGB') if image.mode != 'RGB' else image
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, img_cv)
            image_path = temp_path
        else:
            image_path = image

        try:
            result = self.ocr.predict(image_path)
            elements = self._collect_elements(result)

            if not elements:
                return ""

            rows = self._group_by_rows(elements)
            column_centers = self._find_columns_smart(rows)
            num_cols = len(column_centers)

            if num_cols < 1:
                return ""

            markdown_lines = self._build_markdown(
                rows, column_centers, num_cols,
                add_header_separator, header_row_index,
            )
            return "\n" + "\n".join(markdown_lines)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _collect_elements(self, result) -> List[Dict]:
        """ Собирает фрагменты таблицы """

        elements = []
        for item in result:
            rec_texts = item.get('rec_texts', [])
            rec_polys = item.get('rec_polys', [])
            rec_scores = item.get('rec_scores', [1.0] * len(rec_texts))

            for text, poly, score in zip(rec_texts, rec_polys, rec_scores):
                if score < self.score_threshold or not text.strip():
                    continue

                elements.append({
                    'text': text.strip(),
                    'x_left': poly[0][0],
                    'y': (poly[0][1] + poly[2][1]) / 2,
                    'score': score,
                })
        return elements

    def _group_by_rows(self, elements: List[Dict]) -> List[List[Dict]]:
        """ Определяет порядок рядов таблицы в зависимости от координат """

        elements_sorted = sorted(elements, key=lambda k: k['y'])
        if not elements_sorted:
            return []

        rows = []
        current_row = [elements_sorted[0]]
        current_y = elements_sorted[0]['y']

        for elem in elements_sorted[1:]:
            if abs(elem['y'] - current_y) > self.y_threshold:
                rows.append(sorted(current_row, key=lambda k: k['x_left']))
                current_row = []
                current_y = elem['y']
            current_row.append(elem)

        if current_row:
            rows.append(sorted(current_row, key=lambda k: k['x_left']))

        return rows

    def _find_columns_smart(self, rows: List[List[Dict]]) -> List[float]:
        """ Определяет координаты колонок таблицы """

        all_x = sorted(elem['x_left'] for row in rows for elem in row)
        column_centers = []

        if all_x:
            current_cluster = [all_x[0]]
            for x in all_x[1:]:
                if x - current_cluster[-1] > self.x_gap_threshold:
                    column_centers.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = []
                current_cluster.append(x)
            column_centers.append(sum(current_cluster) / len(current_cluster))

        return column_centers

    def _assign_to_column(self, elem: Dict, column_centers: List[float], x_tolerance: float = 150) -> int:
        """ Присваивает элемент к какой-то из колонок """

        distances = [(abs(elem['x_left'] - col_x), i) for i, col_x in enumerate(column_centers)]
        min_dist, col_idx = min(distances)
        return col_idx if min_dist <= x_tolerance else len(column_centers)

    def _build_markdown(
            self,
            rows: List[List[Dict]],
            column_centers: List[float],
            num_cols: int,
            add_header_separator: bool,
            header_row_index: int,
    ) -> List[str]:
        markdown_lines = []

        for row_idx, row in enumerate(rows):
            cells = [''] * num_cols

            for elem in row:
                col_idx = self._assign_to_column(elem, column_centers)
                if col_idx < num_cols:
                    cells[col_idx] = (cells[col_idx] + ' ' + elem['text']).strip()

            # Пустые ячейки → пробел (для корректного рендера Markdown)
            cells = [cell if cell.strip() else ' ' for cell in cells]

            markdown_lines.append("| " + " | ".join(cells) + " |")

            if add_header_separator and row_idx == header_row_index:
                markdown_lines.append("| " + " | ".join(['---'] * num_cols) + " |")

        return markdown_lines


def ocr_table_to_markdown(
        image: Union[str, Image.Image],
        lang: str = 'ru',
        y_threshold: float = 25,
        x_gap_threshold: float = 80,
) -> str:
    """
    Конвертация таблицы из изображения в Markdown.

    Args:
        image:           PIL Image или путь к файлу (str)
        lang:            Язык ('ru', 'en')
        y_threshold:     Чувствительность группировки строк
        x_gap_threshold: Чувствительность определения колонок

    Returns:
        Markdown строка с таблицей
    """
    ocr = TableOCR(lang=lang, y_threshold=y_threshold, x_gap_threshold=x_gap_threshold)
    return ocr.extract_table(image=image)
