from __future__ import annotations

from .med_gemma import MedGemmaBackend


class Gemma3Backend(MedGemmaBackend):
    @staticmethod
    def model_id() -> str:
        return "google/gemma-3-27b-it"
