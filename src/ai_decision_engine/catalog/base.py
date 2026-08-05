from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ai_decision_engine.schemas.models import ModelDefinition, RuntimeModelStatus


class CatalogProviderStatus(StrEnum):
    AVAILABLE = "available"
    CATALOG_UNAVAILABLE = "catalog-unavailable"


@dataclass(frozen=True)
class CatalogSnapshot:
    models: tuple[ModelDefinition, ...]
    statuses: dict[str, RuntimeModelStatus]
    provider_statuses: dict[str, CatalogProviderStatus] = field(default_factory=dict)


class ModelCatalog(Protocol):
    async def discover(self) -> CatalogSnapshot: ...
