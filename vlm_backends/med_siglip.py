from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .base import VLMBackend


MEDSIGLIP_MODEL_ID = "google/medsiglip-400m"


class MedSigLIPBackend(VLMBackend):
    def __init__(self) -> None:
        self._model = None
        self._processor = None

    @staticmethod
    def model_id() -> str:
        return MEDSIGLIP_MODEL_ID

    def load(self, load_4bit: bool = False, revision: Optional[str] = None) -> None:
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            MEDSIGLIP_MODEL_ID,
            revision=revision,
            token=True,
        )
        self._model = AutoModel.from_pretrained(
            MEDSIGLIP_MODEL_ID,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            token=True,
        )
        self._model.eval()

    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        inputs = self._processor(images=image, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            outputs = self._model.get_image_features(**inputs)
        return outputs

    def get_text_embedding(self, text: str) -> torch.Tensor:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        inputs = self._processor(text=text, return_tensors="pt", padding=True).to(
            self._model.device
        )
        with torch.inference_mode():
            outputs = self._model.get_text_features(**inputs)
        return outputs

    def cosine_similarity(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> float:
        img_norm = F.normalize(image_emb, dim=-1)
        txt_norm = F.normalize(text_emb, dim=-1)
        return (img_norm @ txt_norm.T).item()

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        tumor_text = "tumor features, malignant cells, cancerous tissue in lymph node"
        normal_text = "normal lymphoid tissue, benign lymphocytes, reactive follicle"

        tumor_emb = self.get_text_embedding(tumor_text)
        normal_emb = self.get_text_embedding(normal_text)

        results = []
        for img in images:
            img_emb = self.get_image_embedding(img)
            sim_tumor = self.cosine_similarity(img_emb, tumor_emb)
            sim_normal = self.cosine_similarity(img_emb, normal_emb)
            results.append((sim_tumor, sim_normal))

        n_patches = len(results)
        tumor_scores = [r[0] for r in results]
        normal_scores = [r[1] for r in results]

        mean_tumor = sum(tumor_scores) / n_patches
        mean_normal = sum(normal_scores) / n_patches

        if mean_tumor > mean_normal + 0.05:
            return "FINAL ANSWER: A"
        elif mean_normal > mean_tumor + 0.05:
            return "FINAL ANSWER: B"
        else:
            return "FINAL ANSWER: C"
