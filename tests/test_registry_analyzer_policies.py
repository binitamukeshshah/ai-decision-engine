import pytest

from ai_decision_engine.exceptions import PrivacyBlockedError, ValidationError
from ai_decision_engine.registry.model_registry import build_default_registry
from ai_decision_engine.router.routing_policy import RoutingPolicy
from ai_decision_engine.router.task_analyzer import HeuristicTaskAnalyzer
from ai_decision_engine.schemas.models import (
    SERVICE_CAPABILITIES,
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


def test_registry_advertises_only_end_to_end_capabilities() -> None:
    registry = build_default_registry()
    assert {m.model_id for m in registry.list_models()} == {"ollama/qwen3:8b"}
    assert all(model.capabilities <= SERVICE_CAPABILITIES for model in registry.list_models())
    assert registry.get("ollama/qwen3:8b").context_window == 40960


@pytest.mark.parametrize(
    "prompt",
    [
        "Tell me something",
        "My name is Pat and this is personal",
        "Review our proprietary launch plan",
        "Use this password abc123",
        "Summarize this legal dispute",
        "Improve my résumé",
        "Analyze this customer record",
        "Contact me at person@example.com",
    ],
)
def test_missing_privacy_classification_defaults_local_only(prompt: str) -> None:
    assert HeuristicTaskAnalyzer().analyze(TaskRequest(prompt)).privacy_level == PrivacyLevel.LOCAL_ONLY


def test_explicit_cloud_permission_is_visible_for_benign_prompt() -> None:
    result = HeuristicTaskAnalyzer().analyze(TaskRequest("hello", privacy=PrivacyLevel.CLOUD_ALLOWED))
    assert result.privacy_level == PrivacyLevel.CLOUD_ALLOWED
    assert result.privacy_override_applied


def test_heuristic_can_raise_but_not_lower_sensitivity() -> None:
    result = HeuristicTaskAnalyzer().analyze(
        TaskRequest("my confidential customer record", privacy=PrivacyLevel.CLOUD_ALLOWED)
    )
    assert result.privacy_level == PrivacyLevel.LOCAL_ONLY


def test_local_only_rejects_cloud(ledger) -> None:
    cloud = ModelDefinition(
        "openai/cloud",
        "Cloud",
        ProviderType.OPENAI,
        frozenset({Capability.CHAT}),
        4096,
        False,
        ExecutionRuntime.OPENCLAW_GATEWAY,
        BillingMode.API,
        1.0,
        1.0,
    )
    request = TaskRequest("hello")
    status = {"openai/cloud": RuntimeModelStatus("openai/cloud", ModelStatus.AVAILABLE, True, True)}
    with pytest.raises(PrivacyBlockedError):
        RoutingPolicy(ledger).rank([cloud], status, HeuristicTaskAnalyzer().analyze(request), request)


@pytest.mark.parametrize(
    "capability",
    [
        Capability.VISION,
        Capability.EMBEDDINGS,
        Capability.TOOL_USE,
        Capability.STRUCTURED_OUTPUT,
    ],
)
def test_unsupported_service_capability_cannot_be_advertised(capability: Capability) -> None:
    with pytest.raises(ValidationError):
        ModelDefinition("bad", "Bad", ProviderType.OLLAMA, frozenset({capability}), 1000, True)
