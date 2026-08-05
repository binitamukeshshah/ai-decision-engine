from dataclasses import dataclass

from ai_decision_engine.schemas.models import Capability


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prompt: str
    expected_capabilities: frozenset[Capability]
    expected_route_category: str
    expected_model: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    quality_score: float | None
    latency_seconds: float
    cost: float
    passed: bool
