from __future__ import annotations

from ai_decision_engine.exceptions import RejectionCode
from ai_decision_engine.ledger import UsageLedger
from ai_decision_engine.providers.base import GenerationResult, ModelProvider
from ai_decision_engine.providers.provider_registry import ProviderRegistry
from ai_decision_engine.registry.model_registry import ModelRegistry
from ai_decision_engine.router.decision_engine import DecisionEngine
from ai_decision_engine.schemas.models import (
    BillingMode,
    Capability,
    Complexity,
    ExecutionRuntime,
    ModelDefinition,
    ModelPrivacyClass,
    ModelStatus,
    PrivacyLevel,
    ProviderType,
    QuotaStatus,
    RuntimeModelStatus,
)
from ai_decision_engine.schemas.requests import TaskRequest


class RecordingProvider(ModelProvider):
    def __init__(self, unavailable: set[str] | None = None) -> None:
        self.unavailable = unavailable or set()
        self.generated: list[str] = []

    async def availability(self, models: list[ModelDefinition]) -> dict[str, RuntimeModelStatus]:
        return {
            model.model_id: RuntimeModelStatus(
                model.model_id,
                ModelStatus.UNAVAILABLE if model.model_id in self.unavailable else ModelStatus.AVAILABLE,
                True,
                True,
                "Subscription quota exhausted." if model.model_id in self.unavailable else "Ready.",
                QuotaStatus.EXHAUSTED if model.model_id in self.unavailable else QuotaStatus.AVAILABLE,
                "Subscription quota exhausted." if model.model_id in self.unavailable else "Quota available.",
            )
            for model in models
        }

    async def generate(self, model: ModelDefinition, prompt: str) -> GenerationResult:
        self.generated.append(model.model_id)
        return GenerationResult("answer")


def model(
    reference: str,
    *,
    local: bool,
    reasoning: float,
    coding: float = 0.8,
    latency: float = 3.0,
    context: int = 100_000,
    runtime: ExecutionRuntime = ExecutionRuntime.OPENCLAW_GATEWAY,
) -> ModelDefinition:
    return ModelDefinition(
        reference,
        reference,
        ProviderType.OLLAMA if local else ProviderType.OPENAI,
        frozenset({Capability.CHAT, Capability.REASONING, Capability.CODING, Capability.LONG_CONTEXT}),
        context,
        local,
        runtime,
        BillingMode.LOCAL if local else BillingMode.SUBSCRIPTION,
        quality_score=max(reasoning, coding),
        speed_score=0.8,
        reasoning_score=reasoning,
        coding_score=coding,
        vision_support=False,
        estimated_latency_seconds=latency,
        privacy_class=ModelPrivacyClass.LOCAL_PRIVATE if local else ModelPrivacyClass.CLOUD,
    )


def engine(ledger: UsageLedger, models: list[ModelDefinition], provider: RecordingProvider) -> DecisionEngine:
    providers = ProviderRegistry()
    providers.register(ExecutionRuntime.OPENCLAW_GATEWAY, provider)
    return DecisionEngine(ModelRegistry(models), providers, ledger)


async def test_scores_every_model_and_explains_reasoning_selection(ledger: UsageLedger) -> None:
    local = model("ollama/qwen3:8b", local=True, reasoning=0.68, latency=2.5)
    cloud = model("openai/gpt-5.6-sol", local=False, reasoning=0.93, latency=4.0)
    decision = await engine(ledger, [local, cloud], RecordingProvider()).select(
        TaskRequest(
            "Evaluate these architecture tradeoffs.",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            reasoning_complexity=Complexity.HIGH,
        )
    )

    assert decision.selected_model == cloud.model_id
    assert decision.execution_provider == "openclaw"
    assert decision.estimated_latency == 4.0
    assert decision.confidence >= 0.9
    assert len(decision.candidate_assessments) == 2
    assert {item.model_id for item in decision.candidate_assessments} == {local.model_id, cloud.model_id}
    rejection = next(item for item in decision.rejected_candidates if item.model_id == local.model_id)
    assert rejection.code == RejectionCode.QUALITY_TOO_LOW
    assert "reasoning score" in rejection.reason


async def test_unavailable_subscription_is_scored_but_rejected(ledger: UsageLedger) -> None:
    local = model("ollama/qwen3:8b", local=True, reasoning=0.9)
    unavailable = model("openai/gpt-5.6-sol", local=False, reasoning=0.95)
    decision = await engine(ledger, [local, unavailable], RecordingProvider({unavailable.model_id})).select(
        TaskRequest("Compare options", privacy=PrivacyLevel.CLOUD_ALLOWED)
    )

    assessment = next(item for item in decision.candidate_assessments if item.model_id == unavailable.model_id)
    rejection = next(item for item in decision.rejected_candidates if item.model_id == unavailable.model_id)
    assert not assessment.available and assessment.score > 0
    assert rejection.code == RejectionCode.QUOTA_UNAVAILABLE
    assert rejection.reason == "Subscription quota exhausted."


async def test_comparable_slower_candidate_has_explicit_rejection(ledger: UsageLedger) -> None:
    fast = model("openai/a-fast", local=False, reasoning=0.9, latency=2.0)
    slow = model("openai/b-slow", local=False, reasoning=0.9, latency=8.0)
    decision = await engine(ledger, [slow, fast], RecordingProvider()).select(
        TaskRequest("Compare options", privacy=PrivacyLevel.CLOUD_ALLOWED)
    )

    assert decision.selected_model == fast.model_id
    rejection = next(item for item in decision.rejected_candidates if item.model_id == slow.model_id)
    assert rejection.code == RejectionCode.LOWER_RANKED
    assert "slower" in rejection.reason


async def test_context_length_is_enforced(ledger: UsageLedger) -> None:
    short = model("openai/short", local=False, reasoning=0.9, context=1_000)
    long = model("openai/long", local=False, reasoning=0.9, context=10_000)
    decision = await engine(ledger, [short, long], RecordingProvider()).select(
        TaskRequest(
            "Process this document",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            required_capabilities=frozenset({Capability.LONG_CONTEXT}),
            estimated_input_tokens=2_000,
            estimated_output_tokens=500,
        )
    )

    assert decision.selected_model == long.model_id
    rejection = next(item for item in decision.rejected_candidates if item.model_id == short.model_id)
    assert "exceeds context length" in rejection.reason


async def test_run_always_executes_through_openclaw(ledger: UsageLedger) -> None:
    direct = model(
        "ollama/qwen3:8b",
        local=True,
        reasoning=0.9,
        runtime=ExecutionRuntime.DIRECT_OLLAMA,
    )
    direct_provider = RecordingProvider()
    gateway_provider = RecordingProvider()
    providers = ProviderRegistry()
    providers.register(ExecutionRuntime.DIRECT_OLLAMA, direct_provider)
    providers.register(ExecutionRuntime.OPENCLAW_GATEWAY, gateway_provider)
    decision_engine = DecisionEngine(ModelRegistry([direct]), providers, ledger)

    response = await decision_engine.run(TaskRequest("hello"))

    assert response.output == "answer"
    assert gateway_provider.generated == [direct.model_id]
    assert direct_provider.generated == []
