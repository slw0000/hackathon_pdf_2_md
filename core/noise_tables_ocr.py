import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPS_ENABLE_GRAPH_CAPTURE"] = "0"
os.environ["PADDLE_DEVICE"] = "cpu"

from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from typing import Optional, List, Dict, Union
from collections import defaultdict


class TableOCR:
    """Универсальный OCR для таблиц с автоматическим определением структуры"""

    def __init__(
            self,
            lang: str = 'ru',
            y_threshold: float = 20,
            x_gap_threshold: float = 80,
            score_threshold: float = 0.3
    ):
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
        )
        self.y_threshold = y_threshold
        self.x_gap_threshold = x_gap_threshold
        self.score_threshold = score_threshold

    def extract_table(
            self,
            image: Union[str, Image.Image],
            add_header_separator: bool = True,
            header_row_index: int = 0
    ) -> str:
        """
        Извлечение таблицы из изображения в Markdown

        Args:
            image: Путь к файлу (str) или PIL Image
            add_header_separator: Добавить разделитель после заголовка
            header_row_index: Индекс строки заголовка

        Returns:
            Markdown строка с таблицей
        """
        # Конвертируем PIL Image во временный файл
        temp_path = None
        if isinstance(image, Image.Image):
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)

            # Конвертируем в формат для PaddleOCR (BGR)
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
                add_header_separator, header_row_index
            )

            return "\n" + "\n".join(markdown_lines)

        finally:
            # Удаляем временный файл
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _collect_elements(self, result) -> List[Dict]:
        elements = []
        for item in result:
            rec_texts = item.get('rec_texts', [])
            rec_polys = item.get('rec_polys', [])
            rec_scores = item.get('rec_scores', [1.0] * len(rec_texts))

            for text, poly, score in zip(rec_texts, rec_polys, rec_scores):
                if score < self.score_threshold or not text.strip():
                    continue

                x_left = poly[0][0]
                x_right = poly[1][0]
                y_center = (poly[0][1] + poly[2][1]) / 2

                elements.append({
                    'text': text.strip(),
                    'x_left': x_left,
                    'y': y_center,
                    'score': score
                })
        return elements

    def _group_by_rows(self, elements: List[Dict]) -> List[List[Dict]]:
        elements_sorted = sorted(elements, key=lambda k: k['y'])
        rows = []

        if not elements_sorted:
            return rows

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
        all_x = []
        for row in rows:
            for elem in row:
                all_x.append(elem['x_left'])

        all_x.sort()
        column_centers = []

        if all_x:
            current_cluster = [all_x[0]]
            for x in all_x[1:]:
                if x - current_cluster[-1] > self.x_gap_threshold:
                    center = sum(current_cluster) / len(current_cluster)
                    column_centers.append(center)
                    current_cluster = []
                current_cluster.append(x)

            if current_cluster:
                center = sum(current_cluster) / len(current_cluster)
                column_centers.append(center)

        return column_centers

    def _assign_to_column(self, elem: Dict, column_centers: List[float], x_tolerance: float = 150) -> int:
        min_dist = float('inf')
        col_idx = -1

        for i, col_x in enumerate(column_centers):
            dist = abs(elem['x_left'] - col_x)
            if dist < min_dist:
                min_dist = dist
                col_idx = i

        return col_idx if min_dist <= x_tolerance else len(column_centers)

    def _build_markdown(
            self, rows: List[List[Dict]], column_centers: List[float],
            num_cols: int, add_header_separator: bool, header_row_index: int
    ) -> List[str]:
        markdown_lines = []

        for row_idx, row in enumerate(rows):
            cells = [''] * num_cols

            for elem in row:
                col_idx = self._assign_to_column(elem, column_centers)
                if col_idx < num_cols:
                    if cells[col_idx]:
                        cells[col_idx] += ' ' + elem['text']
                    else:
                        cells[col_idx] = elem['text']

            for i, cell in enumerate(cells):
                if not cell.strip():
                    cells[i] = ' '

            row_text = " | ".join(cells)
            markdown_lines.append(f"| {row_text} |")

            if add_header_separator and row_idx == header_row_index:
                separator = " | ".join(['---'] * num_cols)
                markdown_lines.append(f"| {separator} |")

        return markdown_lines


# ============================================================================
# 🚀 БЫСТРАЯ ФУНКЦИЯ (принимает PIL Image!)
# ============================================================================

def ocr_table_to_markdown(
        image: Union[str, Image.Image],
        lang: str = 'ru',
        y_threshold: float = 25,
        x_gap_threshold: float = 80
) -> str:
    """
    Конвертация таблицы из изображения в Markdown

    Args:
        image: PIL Image или путь к файлу (str)
        lang: Язык ('ru', 'en')
        y_threshold: Чувствительность группировки строк
        x_gap_threshold: Чувствительность определения колонок

    Returns:
        Markdown строка с таблицей (str)
    """
    ocr = TableOCR(
        lang=lang,
        y_threshold=y_threshold,
        x_gap_threshold=x_gap_threshold,

    )

    return ocr.extract_table(image=image)
