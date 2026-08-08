from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .base import VLMBackend


MEDSIGLIP_MODEL_ID = "google/medsiglip-448"

TUMOR_TEXTS = [
    "This is a histopathology image of a lymph node showing tumor "
    "cells, malignant lymphocytes, large atypical cells with high "
    "nuclear-to-cytoplasmic ratio, and invasive growth.",
    "This histopathology image shows a lymph node with metastatic carcinoma.",
    "histopathology image of a lymph node containing tumor cells",
    "tumor",
    "malignant",
]

NORMAL_TEXTS = [
    "This is a histopathology image of a lymph node showing normal "
    "reactive lymphoid tissue, small mature lymphocytes, germinal "
    "centers, and no malignant cells.",
    "This histopathology image shows a normal lymph node.",
    "histopathology image of a normal lymph node",
    "normal",
    "benign",
]


class MedSigLIPBackend(VLMBackend):
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._revision = None
        self._diag_printed = 0
        self._last_sims = None

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
            token=True,
        )
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._revision = revision or getattr(self._model.config, '_commit_hash', None)

    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        model_dtype = next(self._model.parameters()).dtype
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {
            k: (v.to(self._model.device, model_dtype) if v.is_floating_point() else v.to(self._model.device))
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            outputs = self._model.get_image_features(**inputs)
        return outputs

    def get_text_embedding(self, text: str) -> torch.Tensor:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        model_dtype = next(self._model.parameters()).dtype
        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        inputs = {
            k: (v.to(self._model.device, model_dtype) if v.is_floating_point() else v.to(self._model.device))
            for k, v in inputs.items()
        }
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
        tumor_embs = [self.get_text_embedding(t) for t in TUMOR_TEXTS]
        normal_embs = [self.get_text_embedding(t) for t in NORMAL_TEXTS]

        results = []
        for img in images:
            img_emb = self.get_image_embedding(img)
            sim_tumor_prompts = [
                self.cosine_similarity(img_emb, te) for te in tumor_embs
            ]
            sim_normal_prompts = [
                self.cosine_similarity(img_emb, ne) for ne in normal_embs
            ]
            sim_tumor = sim_tumor_prompts[0]
            sim_normal = sim_normal_prompts[0]
            sim_tumor_ens = sum(sim_tumor_prompts) / len(sim_tumor_prompts)
            sim_normal_ens = sum(sim_normal_prompts) / len(sim_normal_prompts)
            if self._diag_printed < 5:
                self._diag_printed += 1
                print(
                    f"[med_siglip] sim_tumor={sim_tumor:.4f} sim_normal={sim_normal:.4f} "
                    f"diff={sim_tumor - sim_normal:+.4f}"
                )
            results.append(
                (
                    sim_tumor,
                    sim_normal,
                    sim_tumor_ens,
                    sim_normal_ens,
                    sim_tumor_prompts,
                    sim_normal_prompts,
                )
            )

        n_patches = len(results)
        tumor_scores = [r[0] for r in results]
        normal_scores = [r[1] for r in results]

        mean_tumor = sum(tumor_scores) / n_patches
        mean_normal = sum(normal_scores) / n_patches

        if self._diag_printed < 20:
            print(
                f"[med_siglip] mean_tumor={mean_tumor:.4f} mean_normal={mean_normal:.4f} "
                f"diff={mean_tumor - mean_normal:+.4f}"
            )

        if len(results) == 1:
            r = results[0]
            self._last_sims = {
                "sim_tumor": r[0],
                "sim_normal": r[1],
                "sim_tumor_ens": r[2],
                "sim_normal_ens": r[3],
                "sim_tumor_by_prompt": r[4],
                "sim_normal_by_prompt": r[5],
            }
        else:
            self._last_sims = {
                "sim_tumor": mean_tumor,
                "sim_normal": mean_normal,
                "sim_tumor_ens": sum(r[2] for r in results) / n_patches,
                "sim_normal_ens": sum(r[3] for r in results) / n_patches,
            }

        if mean_tumor >= mean_normal:
            return "FINAL ANSWER: A"
        return "FINAL ANSWER: B"

    def diagnostics(self) -> dict:
        return self._last_sims or {}

    def config_snapshot(self) -> dict:
        import transformers
        return {
            "model_id": self.model_id(),
            "revision": self._revision if hasattr(self, '_revision') else None,
            "quantization": "none",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(self._device) if hasattr(self, '_device') else "unknown",
            "dtype": str(next(self._model.parameters()).dtype) if self._model is not None else "unknown",
        }
