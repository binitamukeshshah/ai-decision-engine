import pytest

from ai_decision_engine.exceptions import (
    CapabilityUnsupportedError,
    ModelRoutingUnavailableError,
    ProviderRoutingUnavailableError,
    QualityThresholdError,
    RejectionCode,
)
from ai_decision_engine.router.routing_policy import RoutingPolicy
from ai_decision_engine.router.task_analyzer import HeuristicTaskAnalyzer
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


def available(*models: ModelDefinition) -> dict[str, RuntimeModelStatus]:
    return {m.model_id: RuntimeModelStatus(m.model_id, ModelStatus.AVAILABLE, True, True) for m in models}


def cloud(model_id: str, *, input_price: float = 1.0, speed: float = 0.5, quality: float = 0.8) -> ModelDefinition:
    return ModelDefinition(
        f"openai/{model_id}",
        model_id,
        ProviderType.OPENAI,
        frozenset({Capability.CHAT}),
        4096,
        False,
        ExecutionRuntime.OPENCLAW_GATEWAY,
        BillingMode.API,
        input_price,
        1.0,
        quality,
        speed,
    )


def non_api(model_id: str, billing: BillingMode, *, speed: float = 0.5, quality: float = 0.8) -> ModelDefinition:
    return ModelDefinition(
        f"openai/{model_id}",
        model_id,
        ProviderType.OPENAI,
        frozenset({Capability.CHAT}),
        4096,
        False,
        ExecutionRuntime.OPENCLAW_GATEWAY,
        billing,
        quality_score=quality,
        speed_score=speed,
    )


def test_local_is_preferred_when_adequate(ledger, local_model) -> None:
    expensive = cloud("cloud", quality=1.0, speed=1.0)
    request = TaskRequest("hello", privacy=PrivacyLevel.CLOUD_ALLOWED)
    plan = RoutingPolicy(ledger).rank(
        [expensive, local_model], available(expensive, local_model), HeuristicTaskAnalyzer().analyze(request), request
    )
    assert plan.candidates[0].model.is_local


def test_cheapest_adequate_then_latency_quality_and_id(ledger) -> None:
    costly = cloud("costly", input_price=100.0, speed=1.0, quality=1.0)
    cheap_slow = cloud("z-cheap", input_price=1.0, speed=0.4, quality=0.9)
    cheap_fast_low = cloud("b-cheap", input_price=1.0, speed=0.8, quality=0.7)
    cheap_fast_high_b = cloud("b-best", input_price=1.0, speed=0.8, quality=0.9)
    cheap_fast_high_a = cloud("a-best", input_price=1.0, speed=0.8, quality=0.9)
    models = [costly, cheap_slow, cheap_fast_low, cheap_fast_high_b, cheap_fast_high_a]
    request = TaskRequest("hello", privacy=PrivacyLevel.CLOUD_ALLOWED, estimated_input_tokens=1000)
    plan = RoutingPolicy(ledger).rank(models, available(*models), HeuristicTaskAnalyzer().analyze(request), request)
    assert [c.model.model_id for c in plan.candidates] == [
        "openai/a-best",
        "openai/b-best",
        "openai/b-cheap",
        "openai/z-cheap",
        "openai/costly",
    ]


def test_subscription_after_local_inadequacy_and_free_tier_selection(ledger, local_model) -> None:
    subscription = non_api("subscription", BillingMode.SUBSCRIPTION, quality=0.9)
    free_tier = non_api("free", BillingMode.FREE_TIER, quality=0.8)
    request = TaskRequest("hello", privacy=PrivacyLevel.CLOUD_ALLOWED, minimum_quality=0.85)
    plan = RoutingPolicy(ledger).rank(
        [local_model, subscription, free_tier],
        available(local_model, subscription, free_tier),
        HeuristicTaskAnalyzer().analyze(request),
        request,
    )
    assert plan.candidates[0].model.billing_mode == BillingMode.SUBSCRIPTION
    free_request = TaskRequest("hello", privacy=PrivacyLevel.CLOUD_ALLOWED, minimum_quality=0.75)
    free_plan = RoutingPolicy(ledger).rank(
        [free_tier], available(free_tier), HeuristicTaskAnalyzer().analyze(free_request), free_request
    )
    assert free_plan.candidates[0].model.billing_mode == BillingMode.FREE_TIER


def test_quality_and_unsupported_capability_rejections(ledger, local_model) -> None:
    quality = TaskRequest("hello", minimum_quality=0.9)
    with pytest.raises(QualityThresholdError):
        RoutingPolicy(ledger).rank(
            [local_model], available(local_model), HeuristicTaskAnalyzer().analyze(quality), quality
        )
    vision = TaskRequest("describe this", has_image=True)
    with pytest.raises(CapabilityUnsupportedError):
        RoutingPolicy(ledger).rank(
            [local_model], available(local_model), HeuristicTaskAnalyzer().analyze(vision), vision
        )


def test_specific_availability_rejection_reasons(ledger, local_model) -> None:
    request = TaskRequest("hello")
    requirements = HeuristicTaskAnalyzer().analyze(request)
    unreachable = {
        "ollama/local": RuntimeModelStatus("ollama/local", ModelStatus.UNKNOWN, False, None, "provider down")
    }
    with pytest.raises(ProviderRoutingUnavailableError) as provider_error:
        RoutingPolicy(ledger).rank([local_model], unreachable, requirements, request)
    assert provider_error.value.rejections[0].code == RejectionCode.PROVIDER_UNAVAILABLE

    missing = {
        "ollama/local": RuntimeModelStatus("ollama/local", ModelStatus.UNAVAILABLE, True, False, "not installed")
    }
    with pytest.raises(ModelRoutingUnavailableError) as model_error:
        RoutingPolicy(ledger).rank([local_model], missing, requirements, request)
    assert model_error.value.rejections[0].code == RejectionCode.MODEL_UNAVAILABLE
