import torch
import re
import os
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MODEL_PATH = "weights/qwen"


def ensure_model_exists(model_path: str, model_id: str = DEFAULT_MODEL_ID) -> str:
    """
    Проверяет наличие модели в указанной папке.
    Если модель отсутствует — скачивает с HuggingFace.

    Args:
        model_path: Локальный путь к папке с моделью
        model_id:   HuggingFace model ID для скачивания

    Returns:
        Путь к модели (локальный или HuggingFace ID, если скачивание не удалось)
    """
    path = Path(model_path)

    # Считаем модель загруженной, если есть config.json и хотя бы один весовой файл
    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))

    if path.exists() and has_config and has_weights:
        print(f"[<3] Модель найдена: {model_path}")
        return model_path

    print(f"[X️] Модель не найдена в '{model_path}'. Скачиваем {model_id} с HuggingFace...")
    print("     Это может занять несколько минут (~4 GB).")

    try:
        from huggingface_hub import snapshot_download

        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(path),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"[<3] Модель успешно скачана в: {model_path}")
        return model_path

    except Exception as e:
        print(f"[X️] Не удалось скачать модель: {e}")
        print(f"     Попытка загрузить напрямую с HuggingFace: {model_id}")
        return model_id


class QwenBlockReader:
    """Мультимодальный ридер на базе Qwen2-VL для OCR и классификации блоков."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        resolved_path = ensure_model_exists(model_path, model_id)

        self.device = self._detect_device()
        print(f"[O]  Используем устройство: {self.device}")

        self.processor = AutoProcessor.from_pretrained(resolved_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            resolved_path,
            torch_dtype=torch.bfloat16 if self.device in ("mps", "cuda") else torch.float32,
            device_map="auto",
        )

    @staticmethod
    def _detect_device() -> str:
        """Определяет лучшее доступное устройство."""
        env_device = os.environ.get("DOCLING_DEVICE", "")
        if env_device and env_device != "auto":
            return env_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def classify_element(self, image_crop) -> str:
        """
        Классифицирует визуальный блок.

        Returns:
            Одно из: 1) IMAGE | FIGURE | DIAGRAM
                     2) TABLE
                     3) TEXT | HAND
                     4) TRASH
        """
        if image_crop is None:
            return "TRASH"

        prompt = (
            "You are an OCR expert. "
            "Analyze this image. Is it a photo/diagram/image or abstract figures (IMAGE), "
            "a table (TABLE), text/handwriting (TEXT), "
            "or a watermark/noise (TRASH)? Answer with ONE word."
        )
        return self._generate_quick(image_crop, prompt)

    def read_complex_block(self, image_crop, prompt_type: str = "text") -> str:
        """
        Извлекает текст или таблицу из изображения.

        Args:
            image_crop:   PIL Image блока
            prompt_type:  'text' или 'table'

        Returns:
            Распознанный текст / Markdown-таблица
        """
        if image_crop is None:
            return ""

        prompts = {
            "text": (
                "SYSTEM: You are a professional OCR tool.\n"
                "INSTRUCTION: Transcribe the text from the image.\n"
                "CONSTRAINTS:\n"
                "1. Output ONLY the plain text found in the image.\n"
                "2. IGNORE background noise, dots, and scanning artifacts.\n"
                "3. If a section contains only noise or is illegible, SKIP it.\n"
                "4. NO explanations, NO 'Here is the result', NO conversational filler.\n"
                "5. NO LaTeX symbols, NO SQL, NO Python code blocks.\n"
                "6. If the output starts to look like gibberish or repeats, STOP.\n"
                "START OCR OUTPUT:"
            ),
            "table": (
                "TASK: Extract table data from image to simple Markdown format.\n"
                "RULES:\n"
                "1. Output ONLY pipe-separated table (| col1 | col2 |)\n"
                "2. NO LaTeX, NO math symbols ($, \\, {}, ^, _)\n"
                "3. NO HTML tags (<table>, <tr>, <td>)\n"
                "4. NO explanations or comments\n"
                "5. If cell is empty/unclear, write empty string between pipes\n"
                "6. Preserve numbers and dates exactly as shown\n"
                "7. Use Cyrillic letters for Russian text\n"
                "8. STOP after the table ends\n"
                "EXAMPLE OUTPUT:\n"
                "| Header1 | Header2 |\n"
                "| --- | --- |\n"
                "| value1 | value2 |\n"
                "START TABLE:"
            ),
        }

        raw = self._generate_full(image_crop, prompts.get(prompt_type, prompts["text"]))
        return self._clean_output(raw)


    def _build_inputs(self, image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

    def _generate_quick(self, image, prompt: str) -> str:
        inputs = self._build_inputs(image, prompt)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                repetition_penalty=1.5,
            )
        result = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
        return result.strip().upper()

    def _generate_full(self, image, prompt: str) -> str:
        inputs = self._build_inputs(image, prompt)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.8,
                no_repeat_ngram_size=3,
                use_cache=True,
            )
        return self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

    @staticmethod
    def _clean_output(text: str) -> str:
        """Убирает галлюцинации координат и бесконечные пайпы."""

        # Координаты вида (12, 34), (56, 78)
        text = re.sub(r'\(?\d+\s*,\s*\d+\)?\s*,\s*\(?\d+\s*,\s*\d+\)?', '', text)

        # Шесть и более пайпов подряд → один
        text = re.sub(r'(\| *){6,}', '|', text)
        return text.strip()
