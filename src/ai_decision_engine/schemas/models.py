from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai_decision_engine.exceptions import ValidationError


class ProviderType(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ExecutionRuntime(StrEnum):
    DIRECT_OLLAMA = "direct_ollama"
    OPENCLAW_GATEWAY = "openclaw_gateway"


class BillingMode(StrEnum):
    LOCAL = "local"
    SUBSCRIPTION = "subscription"
    FREE_TIER = "free_tier"
    API = "api"


class QuotaStatus(StrEnum):
    AVAILABLE = "available"
    EXHAUSTED = "exhausted"
    COOLDOWN = "cooldown"
    UNKNOWN = "unknown"


class Complexity(StrEnum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class LatencyPreference(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class Capability(StrEnum):
    CHAT = "chat"
    REASONING = "reasoning"
    CODING = "coding"
    VISION = "vision"
    TOOL_USE = "tool_use"
    STRUCTURED_OUTPUT = "structured_output"
    LONG_CONTEXT = "long_context"
    EMBEDDINGS = "embeddings"
    CURRENT_INFORMATION = "current_information"


SERVICE_CAPABILITIES = frozenset({Capability.CHAT, Capability.REASONING, Capability.CODING, Capability.LONG_CONTEXT})


class PrivacyLevel(StrEnum):
    LOCAL_ONLY = "local_only"
    CLOUD_ALLOWED = "cloud_allowed"
    PUBLIC = "public"


class ModelPrivacyClass(StrEnum):
    LOCAL_PRIVATE = "local_private"
    CLOUD = "cloud"


class ModelStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValidationError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    display_name: str
    provider: ProviderType
    capabilities: frozenset[Capability]
    context_window: int
    is_local: bool
    execution_runtime: ExecutionRuntime = ExecutionRuntime.DIRECT_OLLAMA
    billing_mode: BillingMode = BillingMode.LOCAL
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    quality_score: float = 0.5
    speed_score: float = 0.5
    deliberately_free: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    capability_quality: dict[Capability, float] = field(default_factory=dict)
    reasoning_score: float | None = None
    coding_score: float | None = None
    vision_support: bool = False
    estimated_latency_seconds: float | None = None
    privacy_class: ModelPrivacyClass | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.display_name.strip():
            raise ValidationError("Model ID and display name must not be empty.")
        if self.context_window <= 0:
            raise ValidationError("Context window must be positive.")
        scores = (("quality_score", self.quality_score), ("speed_score", self.speed_score))
        for name, score_value in scores:
            if not math.isfinite(score_value) or not 0 <= score_value <= 1:
                raise ValidationError(f"{name} must be finite and between 0 and 1.")
        for name, optional_score in (
            ("reasoning_score", self.reasoning_score),
            ("coding_score", self.coding_score),
        ):
            if optional_score is not None and (not math.isfinite(optional_score) or not 0 <= optional_score <= 1):
                raise ValidationError(f"{name} must be finite and between 0 and 1.")
        if self.estimated_latency_seconds is not None and (
            not math.isfinite(self.estimated_latency_seconds) or self.estimated_latency_seconds <= 0
        ):
            raise ValidationError("estimated_latency_seconds must be finite and positive.")
        if not self.capabilities.issubset(SERVICE_CAPABILITIES):
            unsupported = sorted(c.value for c in self.capabilities - SERVICE_CAPABILITIES)
            raise ValidationError(f"Model advertises unsupported service capabilities: {unsupported}.")
        if self.is_local != (self.billing_mode == BillingMode.LOCAL):
            raise ValidationError("Local models must use local billing mode and cloud models must not.")
        expected_privacy = ModelPrivacyClass.LOCAL_PRIVATE if self.is_local else ModelPrivacyClass.CLOUD
        if self.privacy_class is not None and self.privacy_class != expected_privacy:
            raise ValidationError("Model privacy class must agree with its execution location.")
        if self.vision_support != (Capability.VISION in self.capabilities):
            raise ValidationError("vision_support must agree with the declared vision capability.")
        if self.billing_mode == BillingMode.API and not self.deliberately_free:
            if self.input_cost_per_million_tokens is None or self.output_cost_per_million_tokens is None:
                raise ValidationError("Cloud models require explicit input and output pricing.")
        prices: tuple[tuple[str, float | None], ...] = (
            ("input_cost_per_million_tokens", self.input_cost_per_million_tokens),
            ("output_cost_per_million_tokens", self.output_cost_per_million_tokens),
        )
        for name, price_value in prices:
            if price_value is not None:
                _finite_nonnegative(name, price_value)
        for capability, capability_score in self.capability_quality.items():
            if capability not in self.capabilities:
                raise ValidationError(f"Quality configured for unsupported capability: {capability.value}.")
            if not math.isfinite(capability_score) or not 0 <= capability_score <= 1:
                raise ValidationError("Capability quality scores must be finite and between 0 and 1.")

    def supports(self, required: set[Capability] | frozenset[Capability]) -> bool:
        return required.issubset(self.capabilities)

    def quality_for(self, capability: Capability) -> float:
        if capability == Capability.REASONING and self.reasoning_score is not None:
            return self.reasoning_score
        if capability == Capability.CODING and self.coding_score is not None:
            return self.coding_score
        return self.capability_quality.get(capability, self.quality_score)

    @property
    def effective_privacy_class(self) -> ModelPrivacyClass:
        return self.privacy_class or (ModelPrivacyClass.LOCAL_PRIVATE if self.is_local else ModelPrivacyClass.CLOUD)

    @property
    def latency_estimate(self) -> float:
        """Return configured latency, falling back to legacy normalized speed metadata."""
        return self.estimated_latency_seconds or 1.0 / max(self.speed_score, 0.01)

    @property
    def canonical_ref(self) -> str:
        return self.model_id


@dataclass(frozen=True)
class RuntimeModelStatus:
    model_id: str
    status: ModelStatus
    provider_reachable: bool
    installed: bool | None
    detail: str = ""
    quota_status: QuotaStatus = QuotaStatus.UNKNOWN
    quota_detail: str | None = None
    catalog_present: bool = True
    readiness_verified: bool = False
