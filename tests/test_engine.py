from __future__ import annotations

from collections.abc import Sequence

import pytest

from ai_decision_engine.catalog.base import CatalogSnapshot
from ai_decision_engine.exceptions import ChargeStatus, ProviderTimeoutError, RequestBudgetExceededError
from ai_decision_engine.providers.base import GenerationResult, ModelProvider
from ai_decision_engine.providers.provider_registry import ProviderRegistry
from ai_decision_engine.registry.model_registry import ModelRegistry
from ai_decision_engine.router.decision_engine import DecisionEngine
from ai_decision_engine.schemas.models import (
    BillingMode,
    Capability,
    ExecutionRuntime,
    ModelDefinition,
    ModelStatus,
    PrivacyLevel,
    ProviderType,
    RuntimeModelStatus,
)
from ai_decision_engine.schemas.requests import TaskRequest


class FakeProvider(ModelProvider):
    def __init__(self, outcomes: Sequence[GenerationResult | Exception]) -> None:
        self.outcomes = iter(outcomes)
        self.calls = 0

    async def availability(self, models: list[ModelDefinition]) -> dict[str, RuntimeModelStatus]:
        return {m.model_id: RuntimeModelStatus(m.model_id, ModelStatus.AVAILABLE, True, True) for m in models}

    async def generate(self, model: ModelDefinition, prompt: str) -> GenerationResult:
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StaticCatalog:
    def __init__(self, models: list[ModelDefinition]) -> None:
        self.models = models

    async def discover(self) -> CatalogSnapshot:
        return CatalogSnapshot(
            tuple(self.models),
            {
                m.model_id: RuntimeModelStatus(m.model_id, ModelStatus.AVAILABLE, True, True, readiness_verified=True)
                for m in self.models
            },
        )


def cloud_models() -> list[ModelDefinition]:
    return [
        ModelDefinition(
            "openai/first",
            "First",
            ProviderType.OPENAI,
            frozenset({Capability.CHAT}),
            4096,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.API,
            1000,
            1000,
            0.9,
            0.9,
        ),
        ModelDefinition(
            "openai/second",
            "Second",
            ProviderType.OPENAI,
            frozenset({Capability.CHAT}),
            4096,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.API,
            1000,
            1000,
            0.8,
            0.8,
        ),
    ]


def engine_for(ledger, outcomes: Sequence[GenerationResult | Exception]) -> tuple[DecisionEngine, FakeProvider]:
    provider = FakeProvider(outcomes)
    providers = ProviderRegistry()
    providers.register(ExecutionRuntime.OPENCLAW_GATEWAY, provider)
    return DecisionEngine(ModelRegistry(cloud_models()), providers, ledger), provider


@pytest.mark.asyncio
async def test_selection_only_performs_no_execution(ledger) -> None:
    provider = FakeProvider([AssertionError("generate must not be called")])
    providers = ProviderRegistry()
    providers.register(ExecutionRuntime.OPENCLAW_GATEWAY, provider)
    cloud = cloud_models()[0]
    engine = DecisionEngine(ModelRegistry(), providers, ledger, catalog=StaticCatalog([cloud]))
    decision = await engine.select(TaskRequest("hello", privacy=PrivacyLevel.CLOUD_ALLOWED))
    assert decision.selected_model == "openai/first"
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_failed_cloud_attempt_is_charged_and_fallback_is_cumulative(ledger) -> None:
    engine, provider = engine_for(
        ledger,
        [
            ProviderTimeoutError("unknown billing", charge_status=ChargeStatus.POSSIBLE_CHARGE),
            GenerationResult("should not execute"),
        ],
    )
    request = TaskRequest(
        "hello",
        privacy=PrivacyLevel.CLOUD_ALLOWED,
        max_cloud_cost=3.0,
        estimated_input_tokens=1000,
        estimated_output_tokens=1000,
    )
    with pytest.raises(RequestBudgetExceededError):
        await engine.run(request)
    assert provider.calls == 1
    assert ledger.status().finalized_spend == 2.0


@pytest.mark.asyncio
async def test_fallback_returns_attempt_level_cost_metadata(ledger) -> None:
    engine, provider = engine_for(
        ledger,
        [ProviderTimeoutError("unknown billing"), GenerationResult("answer", billed_cost=1.5)],
    )
    response = await engine.run(
        TaskRequest(
            "hello",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            max_cloud_cost=4.0,
            estimated_input_tokens=1000,
            estimated_output_tokens=1000,
        )
    )
    assert provider.calls == 2 and response.output == "answer"
    assert response.explicit_cloud_permission
    assert response.fallback_attempts[0].reserved_cost == 2.0
    assert response.fallback_attempts[0].finalized_cost == 2.0
    assert response.budget_status.finalized_spend == 3.5


@pytest.mark.asyncio
async def test_confirmed_no_charge_releases_reservation(ledger) -> None:
    engine, _ = engine_for(
        ledger,
        [ProviderTimeoutError("preflight", charge_status=ChargeStatus.NO_CHARGE), GenerationResult("answer")],
    )
    response = await engine.run(
        TaskRequest(
            "hello",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            max_cloud_cost=2.0,
            estimated_input_tokens=1000,
            estimated_output_tokens=1000,
        )
    )
    assert response.fallback_attempts[0].finalized_cost == 0
    assert response.budget_status.finalized_spend == 2.0
