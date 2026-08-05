from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ai_decision_engine.schemas.models import ModelDefinition, RuntimeModelStatus


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    billed_cost: float | None = None


class ModelProvider(ABC):
    @abstractmethod
    async def availability(self, models: list[ModelDefinition]) -> dict[str, RuntimeModelStatus]: ...

    @abstractmethod
    async def generate(self, model: ModelDefinition, prompt: str) -> GenerationResult: ...
