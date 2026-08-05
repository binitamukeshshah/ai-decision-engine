from ai_decision_engine.router.task_analyzer import CapabilityTaskAnalyzer
from ai_decision_engine.schemas.models import Capability, Complexity, LatencyPreference
from ai_decision_engine.schemas.requests import TaskRequest


def test_words_do_not_trigger_capabilities() -> None:
    result = CapabilityTaskAnalyzer().analyze(TaskRequest("code analyze image latest JSON"))
    assert result.capabilities == frozenset({Capability.CHAT})


def test_explicit_complexity_drives_capabilities_and_threshold() -> None:
    result = CapabilityTaskAnalyzer().analyze(
        TaskRequest(
            "Evaluate this task",
            reasoning_complexity=Complexity.HIGH,
            coding_complexity=Complexity.MODERATE,
            latency_preference=LatencyPreference.QUALITY,
        )
    )
    assert {Capability.CHAT, Capability.REASONING, Capability.CODING} == result.capabilities
    assert result.minimum_quality_score == 0.85
    assert result.latency_preference == LatencyPreference.QUALITY
    assert result.analysis_confidence == 0.95


def test_multimodal_signal_is_explicit() -> None:
    result = CapabilityTaskAnalyzer().analyze(TaskRequest("Inspect attachment", multimodal_required=True))
    assert Capability.VISION in result.capabilities


def test_prompt_structure_can_raise_complexity_without_keywords() -> None:
    result = CapabilityTaskAnalyzer().analyze(TaskRequest("?\n" * 9))
    assert result.reasoning_complexity == Complexity.HIGH
    assert Capability.REASONING in result.capabilities
    assert result.analysis_confidence == 0.75


def test_brief_factual_and_explanatory_prompts_remain_low_reasoning() -> None:
    analyzer = CapabilityTaskAnalyzer()

    assert analyzer.analyze(TaskRequest("What is photosynthesis?")).reasoning_complexity == Complexity.LOW
    assert analyzer.analyze(TaskRequest("Explain DNS simply.")).reasoning_complexity == Complexity.LOW


def test_multi_criteria_monetization_strategy_is_high_reasoning() -> None:
    prompt = (
        "Evaluate three monetization strategies for an AI productivity platform. Compare risks, pricing power, "
        "and execution complexity, then recommend a staged launch strategy with explicit tradeoffs."
    )

    assert CapabilityTaskAnalyzer().analyze(TaskRequest(prompt)).reasoning_complexity == Complexity.HIGH


def test_deep_multi_part_system_analysis_is_extreme_reasoning() -> None:
    prompt = (
        "Review the architecture and compare three deployment approaches. Assess privacy, security, reliability, "
        "and failure-mode risks; evaluate cost, latency, scalability, and operational tradeoffs; design a phased "
        "migration with rollback criteria; then synthesize the evidence and recommend an implementation strategy."
    )

    assert CapabilityTaskAnalyzer().analyze(TaskRequest(prompt)).reasoning_complexity == Complexity.EXTREME
