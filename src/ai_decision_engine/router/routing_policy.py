from __future__ import annotations

from dataclasses import dataclass, replace

from ai_decision_engine.exceptions import (
    CapabilityUnsupportedError,
    ModelRoutingUnavailableError,
    NoEligibleModelError,
    PrivacyBlockedError,
    ProviderRoutingUnavailableError,
    QualityThresholdError,
    RejectionCode,
    RequestBudgetExceededError,
)
from ai_decision_engine.ledger import UsageLedger
from ai_decision_engine.policies.capability import CapabilityPolicy
from ai_decision_engine.policies.cost import CostPolicy
from ai_decision_engine.policies.privacy import PrivacyPolicy
from ai_decision_engine.router.task_analyzer import TaskRequirements
from ai_decision_engine.schemas.models import (
    SERVICE_CAPABILITIES,
    BillingMode,
    Capability,
    Complexity,
    ModelDefinition,
    ModelStatus,
    QuotaStatus,
    RuntimeModelStatus,
)
from ai_decision_engine.schemas.requests import CandidateAssessment, CandidateRejection, TaskRequest


@dataclass(frozen=True)
class RoutingCandidate:
    model: ModelDefinition
    score: float
    maximum_estimated_cost: float
    reason: str
    confidence: float
    report: str
    estimated_latency: float
    reasoning: tuple[str, ...]
    reasoning_tier_rank: int


@dataclass(frozen=True)
class RoutingPlan:
    candidates: tuple[RoutingCandidate, ...]
    rejections: tuple[CandidateRejection, ...]
    assessments: tuple[CandidateAssessment, ...]


class RoutingPolicy:
    def __init__(self, ledger: UsageLedger) -> None:
        self.ledger = ledger
        self.capability, self.privacy, self.cost = CapabilityPolicy(), PrivacyPolicy(), CostPolicy()

    def rank(
        self,
        models: list[ModelDefinition],
        availability: dict[str, RuntimeModelStatus],
        requirements: TaskRequirements,
        request: TaskRequest,
    ) -> RoutingPlan:
        input_tokens = request.estimated_input_tokens or self.cost.estimate_input_tokens(request.prompt)
        candidates: list[RoutingCandidate] = []
        rejections: list[CandidateRejection] = []
        assessments: list[CandidateAssessment] = []
        service_unsupported = requirements.capabilities - SERVICE_CAPABILITIES
        for model in models:
            runtime = availability.get(model.model_id)
            estimated = self.cost.estimate(model, input_tokens, request.estimated_output_tokens)
            reasoning_score = model.quality_for(Capability.REASONING)
            coding_score = model.quality_for(Capability.CODING)
            latency_estimate = model.latency_estimate
            speed_score = 1.0 / (1.0 + latency_estimate)
            cost_score = 1.0 / (1.0 + estimated * 100.0)
            relevant_quality = [model.quality_for(capability) for capability in requirements.capabilities]
            capability_score = sum(relevant_quality) / len(relevant_quality)
            composite_score = (
                capability_score * 0.65
                + speed_score * (0.2 if requirements.latency_preference.value == "fast" else 0.1)
                + cost_score * 0.15
                + (0.1 if model.is_local else 0.0)
            )
            composite_score = min(1.0, composite_score)
            assessment_reasons = [
                f"Reasoning score {reasoning_score:.2f}.",
                f"Coding score {coding_score:.2f}.",
                f"Estimated latency {latency_estimate:.2f}s.",
                f"Incremental cost ${estimated:.6f}.",
                f"Privacy class {model.effective_privacy_class.value}.",
            ]
            base_assessment = CandidateAssessment(
                model.model_id,
                runtime is not None and runtime.provider_reachable and runtime.status == ModelStatus.AVAILABLE,
                True,
                composite_score,
                reasoning_score,
                coding_score,
                speed_score,
                cost_score,
                estimated,
                latency_estimate,
                tuple(assessment_reasons),
            )

            def reject(
                code: RejectionCode,
                reason: str,
                *,
                base: CandidateAssessment = base_assessment,
            ) -> None:
                rejections.append(CandidateRejection(base.model_id, code, reason, base.score))
                assessments.append(replace(base, eligible=False, reasons=(*base.reasons, reason)))

            if runtime is None or not runtime.provider_reachable:
                reject(
                    RejectionCode.PROVIDER_UNAVAILABLE,
                    runtime.detail if runtime else "Provider status is unknown.",
                )
                continue
            if runtime.status != ModelStatus.AVAILABLE:
                code = (
                    RejectionCode.QUOTA_UNAVAILABLE
                    if runtime.quota_status in {QuotaStatus.EXHAUSTED, QuotaStatus.COOLDOWN}
                    else RejectionCode.MODEL_UNAVAILABLE
                )
                detail = runtime.quota_detail if code == RejectionCode.QUOTA_UNAVAILABLE else runtime.detail
                reject(code, detail or "Model is unavailable.")
                continue
            if service_unsupported or not self.capability.allows(model, requirements):
                missing = sorted(
                    c.value for c in service_unsupported | (requirements.capabilities - model.capabilities)
                )
                reject(RejectionCode.CAPABILITY_UNSUPPORTED, f"Required capabilities unavailable: {missing}.")
                continue
            required_tokens = input_tokens + request.estimated_output_tokens
            if required_tokens > model.context_window:
                reject(
                    RejectionCode.CAPABILITY_UNSUPPORTED,
                    f"Estimated request size {required_tokens} tokens exceeds context length {model.context_window}.",
                )
                continue
            if not self.privacy.allows(model, requirements):
                reject(RejectionCode.PRIVACY_BLOCKED, "Explicit cloud permission was not supplied.")
                continue
            adequacy_capabilities = requirements.capabilities
            if len(adequacy_capabilities) > 1:
                adequacy_capabilities = frozenset(
                    capability for capability in adequacy_capabilities if capability != Capability.CHAT
                )
            assessed = [(capability, model.quality_for(capability)) for capability in adequacy_capabilities]
            limiting_capability, adequacy = min(assessed, key=lambda item: item[1])
            tier_rank = self._reasoning_tier_rank(requirements.reasoning_complexity, model.model_id)
            quality_threshold = self._quality_threshold(requirements, model.model_id)
            if adequacy < quality_threshold:
                reject(
                    RejectionCode.QUALITY_TOO_LOW,
                    f"Estimated {limiting_capability.value} score {adequacy:.2f} is below the "
                    f"required threshold {quality_threshold:.2f}.",
                )
                continue
            if not model.is_local and request.max_cloud_cost is not None and estimated > request.max_cloud_cost:
                reject(RejectionCode.REQUEST_BUDGET_EXHAUSTED, "Maximum estimated cost exceeds the request budget.")
                continue
            if model.billing_mode == BillingMode.LOCAL:
                location = "local execution is adequate and has no metered API cost"
            elif model.billing_mode == BillingMode.SUBSCRIPTION:
                location = "local quality was insufficient and subscription quota is available"
            elif model.billing_mode == BillingMode.FREE_TIER:
                location = "it is the cheapest adequate quota-limited free-tier route"
            else:
                location = "it is the cheapest adequate API-billed route within budget"
            factors = [
                f"{requirements.reasoning_complexity.value.capitalize()} reasoning complexity",
                f"{requirements.coding_complexity.value.capitalize()} coding complexity",
                f"{requirements.latency_preference.value} latency preference",
                f"incremental monetary cost ${estimated:.6f}",
                f"privacy policy {requirements.privacy_level.value}",
            ]
            confidence = min(0.99, requirements.analysis_confidence + max(0.0, adequacy - quality_threshold))
            complexity_reasons = []
            if requirements.reasoning_complexity.value != "none":
                complexity_reasons.append(
                    f"{requirements.reasoning_complexity.value.capitalize()} reasoning complexity requested"
                )
            if requirements.coding_complexity.value != "none":
                complexity_reasons.append(
                    f"{requirements.coding_complexity.value.capitalize()} coding complexity requested"
                )
            task_reason = ". ".join(complexity_reasons)
            if task_reason:
                task_reason += ". "
            reason = (
                f"{task_reason}{location.capitalize()}. Adequacy {adequacy:.2f} meets threshold "
                f"{quality_threshold:.2f}."
            )
            selection_reasons = tuple(
                reason_part
                for reason_part in (
                    f"{requirements.reasoning_complexity.value.capitalize()} reasoning complexity requested."
                    if requirements.reasoning_complexity.value != "none"
                    else "",
                    f"{requirements.coding_complexity.value.capitalize()} coding complexity requested."
                    if requirements.coding_complexity.value != "none"
                    else "",
                    f"{model.billing_mode.value.replace('_', ' ').capitalize()} availability confirmed.",
                    f"Capability score {adequacy:.2f} meets threshold {quality_threshold:.2f}.",
                )
                if reason_part
            )
            tier_preference = self._reasoning_tier_preference(requirements.reasoning_complexity, tier_rank)
            if tier_preference:
                selection_reasons = (*selection_reasons, tier_preference)
                reason = f"{reason} {tier_preference}"
            candidates.append(
                RoutingCandidate(
                    model,
                    composite_score,
                    estimated,
                    reason,
                    confidence,
                    "\n".join(factors),
                    latency_estimate,
                    selection_reasons,
                    tier_rank,
                )
            )
            assessments.append(
                CandidateAssessment(
                    model.model_id,
                    True,
                    True,
                    composite_score,
                    reasoning_score,
                    coding_score,
                    speed_score,
                    cost_score,
                    estimated,
                    latency_estimate,
                    tuple((*assessment_reasons, *selection_reasons)),
                )
            )
        if not candidates:
            self._raise_for_rejections(tuple(rejections))

        # Local adequacy is decisive; then monetary cost, latency, quality, and stable model ID.
        def tie_break(candidate: RoutingCandidate) -> tuple[float, float]:
            if requirements.latency_preference.value == "quality":
                return (-candidate.score, candidate.estimated_latency)
            return (candidate.estimated_latency, -candidate.score)

        candidates.sort(
            key=lambda candidate: (
                candidate.reasoning_tier_rank,
                not candidate.model.is_local,
                candidate.maximum_estimated_cost,
                *tie_break(candidate),
                candidate.model.model_id,
            )
        )
        selected = candidates[0]
        local_quality_rejected = any(
            rejection.code == RejectionCode.QUALITY_TOO_LOW
            and next((model.is_local for model in models if model.model_id == rejection.model_id), False)
            for rejection in rejections
        )
        report_lines = [
            f"Selected: {selected.model.model_id}",
            f"Confidence: {selected.confidence:.0%}",
            f"Reason: {selected.reason}",
            selected.report,
        ]
        if local_quality_rejected and not selected.model.is_local:
            report_lines.append("Local quality threshold not met.")
        for candidate in candidates[1:]:
            if candidate.reasoning_tier_rank > selected.reasoning_tier_rank:
                rank_reason = "The explicit reasoning-tier policy ranks the selected model ahead of this fallback."
            elif candidate.maximum_estimated_cost > selected.maximum_estimated_cost:
                rank_reason = "Comparable adequacy but higher incremental monetary cost."
            elif candidate.estimated_latency > selected.estimated_latency:
                rank_reason = "Comparable adequacy but slower for this task."
            elif candidate.score < selected.score:
                rank_reason = "Lower capability score after adequacy constraints."
            else:
                rank_reason = "Deterministic model-reference tie-break ranked another model first."
            report_lines.append(f"Rejected: {candidate.model.model_id}\nReason: {rank_reason}")
        report_lines.extend(f"Rejected: {item.model_id}\nReason: {item.reason}" for item in rejections)
        candidates[0] = RoutingCandidate(
            selected.model,
            selected.score,
            selected.maximum_estimated_cost,
            selected.reason,
            selected.confidence,
            "\n".join(report_lines),
            selected.estimated_latency,
            selected.reasoning,
            selected.reasoning_tier_rank,
        )
        return RoutingPlan(tuple(candidates), tuple(rejections), tuple(assessments))

    @staticmethod
    def _reasoning_tier_rank(complexity: Complexity, model_id: str) -> int:
        preferences = {
            Complexity.HIGH: ("openai/gpt-5.6-sol", "anthropic/claude-opus-4-8"),
            Complexity.EXTREME: ("anthropic/claude-opus-4-8", "openai/gpt-5.6-sol"),
        }
        preferred = preferences.get(complexity)
        if preferred is None:
            return 0
        try:
            return preferred.index(model_id)
        except ValueError:
            return len(preferred)

    @staticmethod
    def _quality_threshold(requirements: TaskRequirements, model_id: str) -> float:
        if requirements.reasoning_complexity == Complexity.EXTREME and model_id == "openai/gpt-5.6-sol":
            return min(requirements.minimum_quality_score, 0.93)
        return requirements.minimum_quality_score

    @staticmethod
    def _reasoning_tier_preference(complexity: Complexity, rank: int) -> str:
        if complexity not in {Complexity.HIGH, Complexity.EXTREME}:
            return ""
        role = "primary" if rank == 0 else "fallback" if rank == 1 else "non-preferred"
        return f"{complexity.value.capitalize()} reasoning tier policy ranks this model as {role}."

    @staticmethod
    def _raise_for_rejections(rejections: tuple[CandidateRejection, ...]) -> None:
        codes = {item.code for item in rejections}
        ordered = (
            (RejectionCode.PRIVACY_BLOCKED, PrivacyBlockedError, "Cloud routing was blocked by privacy policy."),
            (
                RejectionCode.CAPABILITY_UNSUPPORTED,
                CapabilityUnsupportedError,
                "The required capability is unsupported by the service.",
            ),
            (
                RejectionCode.QUALITY_TOO_LOW,
                QualityThresholdError,
                "No available model meets the minimum quality threshold.",
            ),
            (
                RejectionCode.REQUEST_BUDGET_EXHAUSTED,
                RequestBudgetExceededError,
                "The request cloud budget is exhausted.",
            ),
            (
                RejectionCode.PROVIDER_UNAVAILABLE,
                ProviderRoutingUnavailableError,
                "The required provider is unavailable.",
            ),
            (
                RejectionCode.MODEL_UNAVAILABLE,
                ModelRoutingUnavailableError,
                "No approved model is currently available.",
            ),
        )
        for code, error_type, message in ordered:
            if code in codes:
                raise error_type(message, rejections)
        raise NoEligibleModelError("No eligible approved model was found.", rejections)
