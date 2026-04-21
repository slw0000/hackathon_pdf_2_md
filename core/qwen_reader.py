import torch
import re
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class QwenBlockReader:
    def __init__(self, model_path="weights/qwen"):
        # Устройство выбираем из окружения (MPS для Mac, CUDA для GPU)
        self.device = os.environ.get("DOCLING_DEVICE", "cpu")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device in ["mps", "cuda"] else torch.float32,
            device_map="auto"
        )

    def classify_element(self, image_crop) -> str:
        """Определяет, нужно ли сохранять блок как картинку или распознать как текст."""
        if image_crop is None: return "TRASH"

        prompt = ("You are an OCR expert."
            "Analyze this image. Is it a photo/diagram/image or some abstract figures (IMAGE), "
            "a table (TABLE), text/handwriting (TEXT), "
            "or a watermark/noise (TRASH)? Answer with ONE word."
        )
        return self._generate_quick(image_crop, prompt)

    def read_complex_block(self, image_crop, prompt_type="text"):
        if image_crop is None: return ""

        prompts = {
            "text": (
                "SYSTEM: You are a professional OCR tool. "
                "INSTRUCTION: Transcribe the text from the image. "
                "CONSTRAINTS: "
                "1. Output ONLY the plain text found in the image. "
                "2. IGNORE background noise, dots, and scanning artifacts. "
                "3. If a section contains only noise or is illegible, SKIP it. "
                "4. NO explanations, NO 'Here is the result', NO conversational filler. "
                "5. NO LaTeX symbols, NO SQL, NO Python code blocks. "
                "6. If the output starts to look like gibberish or repeats, STOP. "
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
            )
        }
        raw = self._generate_full(image_crop, prompts.get(prompt_type, prompts["text"]))
        return self._clean_output(raw)

    def _generate_quick(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                repetition_penalty=1.5
            )
        res = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return res.strip().upper()

    def _generate_full(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.8,  # Жесткая защита от зацикливания |||
                no_repeat_ngram_size=3,
                use_cache=True
            )
        return self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def _clean_output(self, text):
        # Чистим галлюцинации координат (12, 34)
        text = re.sub(r'\(?\d+\s*,\s*\d+\)?\s*,\s*\(?\d+\s*,\s*\d+\)?', '', text)
        # Чистим бесконечные пайпы
        text = re.sub(r'(\| *){6,}', '|', text)
        return text.strip()