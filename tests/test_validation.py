import json
import math
from pathlib import Path

import pytest

from ai_decision_engine.config import Settings
from ai_decision_engine.exceptions import ValidationError
from ai_decision_engine.schemas.models import BillingMode, Capability, ExecutionRuntime, ModelDefinition, ProviderType
from ai_decision_engine.schemas.requests import TaskRequest
from ai_decision_engine.service import build_engine


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt: str) -> None:
    with pytest.raises(ValidationError):
        TaskRequest(prompt)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_financial_values_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        TaskRequest("hello", max_cloud_cost=value)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, 1.01])
def test_invalid_quality_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        TaskRequest("hello", minimum_quality=value)


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskRequest("hello", estimated_input_tokens=-1)


def test_cloud_model_requires_explicit_valid_pricing() -> None:
    with pytest.raises(ValidationError):
        ModelDefinition(
            "openai/cloud",
            "Cloud",
            ProviderType.OPENAI,
            frozenset({Capability.CHAT}),
            1000,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.API,
        )
    with pytest.raises(ValidationError):
        ModelDefinition(
            "openai/cloud",
            "Cloud",
            ProviderType.OPENAI,
            frozenset({Capability.CHAT}),
            1000,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.API,
            math.nan,
            1.0,
        )


def test_deliberately_free_cloud_model_is_explicit() -> None:
    model = ModelDefinition(
        "openai/free",
        "Free",
        ProviderType.OPENAI,
        frozenset({Capability.CHAT}),
        1000,
        False,
        ExecutionRuntime.OPENCLAW_GATEWAY,
        BillingMode.FREE_TIER,
        deliberately_free=True,
    )
    assert model.deliberately_free


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf])
def test_invalid_timeout_rejected(monkeypatch, timeout: float) -> None:
    monkeypatch.setenv("ADE_REQUEST_TIMEOUT_SECONDS", str(timeout))
    with pytest.raises(ValidationError):
        Settings.from_env()


def test_catalog_provider_list_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("ADE_OPENCLAW_CATALOG_PROVIDERS", "ollama, Anthropic")
    assert Settings.from_env().openclaw_catalog_providers == ("ollama", "anthropic")


@pytest.mark.parametrize("providers", ["", "ollama,--all", "openai,bad provider"])
def test_invalid_catalog_provider_list_is_rejected(monkeypatch, providers: str) -> None:
    monkeypatch.setenv("ADE_OPENCLAW_CATALOG_PROVIDERS", providers)
    with pytest.raises(ValidationError, match="ADE_OPENCLAW_CATALOG_PROVIDERS"):
        Settings.from_env()


def test_unapproved_openclaw_catalog_model_cannot_be_added(tmp_path: Path) -> None:
    override = json.dumps(
        [{"model_ref": "openai/catalog-only", "billing_mode": "subscription", "context_window": 1000}]
    )
    settings = Settings(
        openclaw_enabled=True,
        openclaw_gateway_token="secret",
        openclaw_model_overrides_json=override,
        usage_ledger_path=tmp_path / "ledger.sqlite3",
    )
    with pytest.raises(ValidationError, match="not in the currently approved"):
        build_engine(settings)
