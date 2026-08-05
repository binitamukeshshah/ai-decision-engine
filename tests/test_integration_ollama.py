import os

import pytest

from ai_decision_engine.providers.ollama_provider import OllamaProvider
from ai_decision_engine.registry.model_registry import build_default_registry


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("ADE_RUN_OLLAMA_INTEGRATION") != "1", reason="set ADE_RUN_OLLAMA_INTEGRATION=1")
@pytest.mark.asyncio
async def test_live_ollama_tags() -> None:
    models = build_default_registry().list_models()
    statuses = await OllamaProvider(os.getenv("ADE_OLLAMA_HOST", "http://localhost:11434")).availability(models)
    assert any(status.provider_reachable for status in statuses.values())
