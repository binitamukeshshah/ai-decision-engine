from pathlib import Path

import pytest

from ai_decision_engine.ledger import UsageLedger
from ai_decision_engine.schemas.models import Capability, ModelDefinition, ProviderType


@pytest.fixture
def ledger(tmp_path: Path) -> UsageLedger:
    return UsageLedger(tmp_path / "ledger.sqlite3", 10.0)


@pytest.fixture
def local_model() -> ModelDefinition:
    return ModelDefinition(
        "ollama/local",
        "Local",
        ProviderType.OLLAMA,
        frozenset({Capability.CHAT, Capability.CODING}),
        4096,
        True,
        quality_score=0.8,
        speed_score=0.8,
    )
