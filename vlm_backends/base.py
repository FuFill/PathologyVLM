from __future__ import annotations

import abc
from typing import Optional

from PIL import Image


class VLMBackend(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def model_id() -> str:
        ...

    @abc.abstractmethod
    def load(self, load_4bit: bool = False, revision: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        ...

    def config_snapshot(self) -> dict:
        return {"model_id": self.model_id(), "revision": None, "quantization": None}
