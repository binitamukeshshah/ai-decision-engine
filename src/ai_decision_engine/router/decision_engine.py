from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from ai_decision_engine.catalog.base import ModelCatalog
from ai_decision_engine.exceptions import (
    ChargeStatus,
    LedgerUnavailableError,
    MonthlyBudgetExceededError,
    NoEligibleModelError,
    ProviderError,
    RejectionCode,
    RequestBudgetExceededError,
)
from ai_decision_engine.ledger import BudgetReservation, UsageLedger
from ai_decision_engine.observability import RoutingLogger
from ai_decision_engine.providers.provider_registry import ProviderRegistry
from ai_decision_engine.registry.model_registry import ModelRegistry
from ai_decision_engine.registry.runtime_registry import RuntimeAvailabilityRegistry
from ai_decision_engine.router.routing_policy import RoutingCandidate, RoutingPlan, RoutingPolicy
from ai_decision_engine.router.task_analyzer import Analyzer, HeuristicTaskAnalyzer, TaskRequirements
from ai_decision_engine.schemas.models import BillingMode, ExecutionRuntime, RuntimeModelStatus
from ai_decision_engine.schemas.requests import (
    CandidateRejection,
    DecisionResponse,
    FallbackAttempt,
    RoutingDecision,
    TaskRequest,
)


@dataclass(frozen=True)
class PreparedRoute:
    request: TaskRequest
    requirements: TaskRequirements
    plan: RoutingPlan
    statuses: dict[str, RuntimeModelStatus]


class DecisionEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        provider_registry: ProviderRegistry,
        ledger: UsageLedger,
        analyzer: Analyzer | None = None,
        logger: RoutingLogger | None = None,
        catalog: ModelCatalog | None = None,
    ) -> None:
        self.registry, self.providers, self.ledger = registry, provider_registry, ledger
        self.analyzer, self.catalog = analyzer or HeuristicTaskAnalyzer(), catalog
        self.routing_policy, self.logger = RoutingPolicy(ledger), logger or RoutingLogger()

    async def _prepare(self, request: TaskRequest) -> PreparedRoute:
        requirements = self.analyzer.analyze(request)
        models = self.registry.list_models()
        statuses = await RuntimeAvailabilityRegistry(self.registry, self.providers).refresh()
        if self.catalog is not None:
            snapshot = await self.catalog.discover()
            models.extend(snapshot.models)
            statuses.update(snapshot.statuses)
        plan = self.routing_policy.rank(models, statuses, requirements, request)
        return PreparedRoute(request, requirements, plan, statuses)

    async def model_statuses(self) -> tuple[tuple[object, RuntimeModelStatus], ...]:
        models = self.registry.list_models()
        statuses = await RuntimeAvailabilityRegistry(self.registry, self.providers).refresh()
        if self.catalog is not None:
            snapshot = await self.catalog.discover()
            models.extend(snapshot.models)
            statuses.update(snapshot.statuses)
        return tuple((model, statuses[model.model_id]) for model in models)

    async def select(self, request: TaskRequest) -> RoutingDecision:
        prepared = await self._prepare(request)
        candidate = prepared.plan.candidates[0]
        status = prepared.statuses[candidate.model.model_id]
        ranked_alternatives = tuple(
            CandidateRejection(
                item.model.model_id,
                RejectionCode.LOWER_RANKED,
                self._lower_ranked_reason(candidate, item),
                item.score,
            )
            for item in prepared.plan.candidates[1:]
        )
        return RoutingDecision(
            selected_provider=candidate.model.provider.value,
            selected_model=candidate.model.model_id,
            execution_runtime=ExecutionRuntime.OPENCLAW_GATEWAY.value,
            billing_mode=candidate.model.billing_mode.value,
            routing_score=candidate.score,
            routing_reason=candidate.reason,
            capabilities=prepared.requirements.capabilities,
            privacy_classification=prepared.requirements.privacy_level,
            explicit_cloud_permission=prepared.requirements.privacy_override_applied,
            estimated_cost=candidate.maximum_estimated_cost,
            rejected_candidates=prepared.plan.rejections + ranked_alternatives,
            fallback_order=tuple(item.model.model_id for item in prepared.plan.candidates[1:]),
            quota_status=status.quota_status.value,
            confidence=candidate.confidence,
            routing_report=candidate.report,
            execution_provider="openclaw",
            estimated_latency=candidate.estimated_latency,
            reasoning=candidate.reasoning,
            candidate_assessments=prepared.plan.assessments,
        )

    async def run(self, request: TaskRequest) -> DecisionResponse:
        prepared = await self._prepare(request)
        attempts: list[FallbackAttempt] = []
        dynamic_rejections: list[CandidateRejection] = []
        cumulative_api_cost = 0.0
        started = monotonic()
        last_error: ProviderError | None = None
        for candidate in prepared.plan.candidates:
            reservation: BudgetReservation | None = None
            if candidate.model.billing_mode == BillingMode.API:
                if request.max_cloud_cost is not None and candidate.maximum_estimated_cost > max(
                    0.0, request.max_cloud_cost - cumulative_api_cost
                ):
                    dynamic_rejections.append(
                        CandidateRejection(
                            candidate.model.model_id,
                            RejectionCode.REQUEST_BUDGET_EXHAUSTED,
                            "Cumulative API fallback cost would exceed the request budget.",
                        )
                    )
                    continue
                try:
                    reservation = self.ledger.reserve(
                        model_id=candidate.model.model_id,
                        provider=candidate.model.provider.value,
                        maximum_cost=candidate.maximum_estimated_cost,
                    )
                except MonthlyBudgetExceededError:
                    dynamic_rejections.append(
                        CandidateRejection(
                            candidate.model.model_id,
                            RejectionCode.MONTHLY_BUDGET_EXHAUSTED,
                            "The atomic reservation would exceed the monthly API budget.",
                        )
                    )
                    continue
            attempt_started = monotonic()
            try:
                # The Decision Engine selects models; OpenClaw is the sole execution boundary.
                result = await self.providers.get(ExecutionRuntime.OPENCLAW_GATEWAY).generate(
                    candidate.model, request.prompt
                )
            except ProviderError as exc:
                cost = self._reconcile_failure(reservation, candidate.maximum_estimated_cost, exc)
                cumulative_api_cost += cost
                attempts.append(
                    FallbackAttempt(
                        candidate.model.model_id,
                        candidate.model.provider.value,
                        str(exc),
                        reservation.maximum_cost if reservation else 0.0,
                        cost,
                        exc.charge_status.value,
                        monotonic() - attempt_started,
                    )
                )
                last_error = exc
                continue
            final_cost = result.billed_cost if result.billed_cost is not None else candidate.maximum_estimated_cost
            if reservation:
                final_cost = self.ledger.finalize(reservation, final_cost, outcome="success")
                cumulative_api_cost += final_cost
            status = prepared.statuses[candidate.model.model_id]
            latency = monotonic() - started
            self.logger.event(
                "completed",
                model=candidate.model.model_id,
                runtime=ExecutionRuntime.OPENCLAW_GATEWAY.value,
                billing_mode=candidate.model.billing_mode.value,
                explicit_cloud_permission=prepared.requirements.privacy_override_applied,
            )
            return DecisionResponse(
                result.text,
                candidate.model.model_id,
                candidate.model.provider.value,
                candidate.score,
                candidate.reason,
                prepared.requirements.capabilities,
                prepared.requirements.privacy_level,
                prepared.requirements.privacy_override_applied,
                candidate.model.is_local,
                final_cost,
                latency,
                tuple(attempts),
                prepared.plan.rejections + tuple(dynamic_rejections),
                self.ledger.status(),
                ExecutionRuntime.OPENCLAW_GATEWAY.value,
                candidate.model.billing_mode.value,
                status.quota_status.value,
                candidate.confidence,
                candidate.report,
            )
        rejections = prepared.plan.rejections + tuple(dynamic_rejections)
        codes = {item.code for item in dynamic_rejections}
        if RejectionCode.MONTHLY_BUDGET_EXHAUSTED in codes:
            raise MonthlyBudgetExceededError("Monthly API budget is exhausted.", rejections)
        if RejectionCode.REQUEST_BUDGET_EXHAUSTED in codes:
            raise RequestBudgetExceededError("Cumulative API fallback cost exhausted the request budget.", rejections)
        if last_error:
            raise last_error
        raise NoEligibleModelError("No candidate could be executed.", rejections)

    @staticmethod
    def _lower_ranked_reason(selected: RoutingCandidate, candidate: RoutingCandidate) -> str:
        if candidate.reasoning_tier_rank > selected.reasoning_tier_rank:
            return "The explicit reasoning-tier policy ranks the selected model ahead of this fallback."
        if candidate.maximum_estimated_cost > selected.maximum_estimated_cost:
            return "Comparable adequacy but higher incremental monetary cost."
        if candidate.estimated_latency > selected.estimated_latency:
            return "Comparable adequacy but slower for this task."
        if candidate.score < selected.score:
            return "Lower capability score after adequacy constraints."
        return "Deterministic model-reference tie-break ranked another adequate model first."

    def _reconcile_failure(
        self, reservation: BudgetReservation | None, maximum_cost: float, error: ProviderError
    ) -> float:
        if reservation is None:
            return 0.0
        if error.charge_status == ChargeStatus.NO_CHARGE:
            self.ledger.release(reservation, outcome="provider_confirmed_no_charge")
            return 0.0
        if error.charge_status == ChargeStatus.KNOWN_CHARGE:
            if error.billed_cost is None:
                raise LedgerUnavailableError("Provider reported a known charge without its cost.")
            return self.ledger.finalize(reservation, error.billed_cost, outcome="failed_known_charge")
        return self.ledger.finalize(reservation, maximum_cost, outcome="failed_possible_charge")
