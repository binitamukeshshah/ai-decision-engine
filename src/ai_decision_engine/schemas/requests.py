from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_decision_engine.exceptions import RejectionCode, ValidationError
from ai_decision_engine.schemas.models import Capability, Complexity, LatencyPreference, PrivacyLevel


@dataclass(frozen=True)
class TaskRequest:
    prompt: str
    has_image: bool = False
    privacy: PrivacyLevel | None = None
    required_capabilities: frozenset[Capability] = field(default_factory=frozenset)
    minimum_quality: float | None = None
    max_cloud_cost: float | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int = 500
    reasoning_complexity: Complexity | None = None
    coding_complexity: Complexity | None = None
    multimodal_required: bool = False
    latency_preference: LatencyPreference = LatencyPreference.BALANCED

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValidationError("Prompt must not be empty.")
        token_values: tuple[tuple[str, int | None], ...] = (
            ("estimated_input_tokens", self.estimated_input_tokens),
            ("estimated_output_tokens", self.estimated_output_tokens),
        )
        for name, token_value in token_values:
            if token_value is not None and token_value < 0:
                raise ValidationError(f"{name} must be non-negative.")
        financial_values: tuple[tuple[str, float | None], ...] = (
            ("minimum_quality", self.minimum_quality),
            ("max_cloud_cost", self.max_cloud_cost),
        )
        for name, financial_value in financial_values:
            if financial_value is not None and (not math.isfinite(financial_value) or financial_value < 0):
                raise ValidationError(f"{name} must be finite and non-negative.")
        if self.minimum_quality is not None and self.minimum_quality > 1:
            raise ValidationError("minimum_quality must be between 0 and 1.")


@dataclass(frozen=True)
class CandidateRejection:
    model_id: str
    code: RejectionCode
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class CandidateAssessment:
    model_id: str
    available: bool
    eligible: bool
    score: float
    reasoning_score: float
    coding_score: float
    speed_score: float
    cost_score: float
    estimated_cost: float
    estimated_latency: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FallbackAttempt:
    model_id: str
    provider: str
    error: str
    reserved_cost: float
    finalized_cost: float
    charge_status: str
    latency_seconds: float


@dataclass(frozen=True)
class BudgetStatus:
    monthly_limit: float
    finalized_spend: float
    reserved_spend: float
    remaining: float
    exhausted: bool

    @property
    def spent(self) -> float:
        return self.finalized_spend


@dataclass(frozen=True)
class DecisionResponse:
    output: str
    selected_model: str
    provider: str
    routing_score: float
    routing_reason: str
    capabilities: frozenset[Capability]
    privacy_classification: PrivacyLevel
    explicit_cloud_permission: bool
    is_local: bool
    estimated_cost: float
    latency_seconds: float
    fallback_attempts: tuple[FallbackAttempt, ...]
    candidate_rejections: tuple[CandidateRejection, ...]
    budget_status: BudgetStatus
    execution_runtime: str
    billing_mode: str
    quota_status: str
    confidence: float
    routing_report: str


@dataclass(frozen=True)
class RoutingDecision:
    selected_provider: str
    selected_model: str
    execution_runtime: str
    billing_mode: str
    routing_score: float
    routing_reason: str
    capabilities: frozenset[Capability]
    privacy_classification: PrivacyLevel
    explicit_cloud_permission: bool
    estimated_cost: float
    rejected_candidates: tuple[CandidateRejection, ...]
    fallback_order: tuple[str, ...]
    quota_status: str
    confidence: float
    routing_report: str
    execution_provider: str = "openclaw"
    estimated_latency: float = 0.0
    reasoning: tuple[str, ...] = ()
    candidate_assessments: tuple[CandidateAssessment, ...] = ()
