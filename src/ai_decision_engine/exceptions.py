from __future__ import annotations

from enum import StrEnum


class DecisionEngineError(Exception):
    """Base domain error safe to present to callers."""


class ValidationError(DecisionEngineError, ValueError):
    pass


class RejectionCode(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    PRIVACY_BLOCKED = "privacy_policy_blocked_cloud"
    QUALITY_TOO_LOW = "minimum_quality_not_met"
    REQUEST_BUDGET_EXHAUSTED = "request_budget_exhausted"
    MONTHLY_BUDGET_EXHAUSTED = "monthly_budget_exhausted"
    LEDGER_UNAVAILABLE = "ledger_unavailable"
    QUOTA_UNAVAILABLE = "quota_unavailable"
    LOWER_RANKED = "lower_ranked"


class NoEligibleModelError(DecisionEngineError):
    def __init__(self, message: str, rejections: tuple[object, ...] = ()) -> None:
        super().__init__(message)
        self.rejections = rejections


class ProviderRoutingUnavailableError(NoEligibleModelError):
    pass


class ModelRoutingUnavailableError(NoEligibleModelError):
    pass


class CapabilityUnsupportedError(NoEligibleModelError):
    pass


class PrivacyBlockedError(NoEligibleModelError):
    pass


class QualityThresholdError(NoEligibleModelError):
    pass


class BudgetExceededError(NoEligibleModelError):
    pass


class RequestBudgetExceededError(BudgetExceededError):
    pass


class MonthlyBudgetExceededError(BudgetExceededError):
    pass


class LedgerError(DecisionEngineError):
    pass


class LedgerUnavailableError(LedgerError):
    pass


class ProviderError(DecisionEngineError):
    """Provider failure with conservative billing information."""

    def __init__(
        self, message: str, *, charge_status: ChargeStatus | None = None, billed_cost: float | None = None
    ) -> None:
        super().__init__(message)
        self.charge_status = charge_status or ChargeStatus.POSSIBLE_CHARGE
        self.billed_cost = billed_cost


class ChargeStatus(StrEnum):
    NO_CHARGE = "no_charge"
    POSSIBLE_CHARGE = "possible_charge"
    KNOWN_CHARGE = "known_charge"


class ProviderUnavailableError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderHTTPError(ProviderError):
    pass


class InvalidProviderResponseError(ProviderError):
    pass


class ModelUnavailableError(ProviderError):
    pass


class OpenClawGatewayUnavailableError(ProviderUnavailableError):
    pass


class GatewayAuthenticationError(ProviderError):
    retryable = False


class ModelNotAllowedError(ProviderError):
    retryable = False


class ProviderNotConfiguredError(ProviderError):
    retryable = False


class ProviderCredentialExpiredError(ProviderError):
    retryable = False


class ProviderRateLimitedError(ProviderError):
    pass


class SubscriptionQuotaExhaustedError(ProviderError):
    retryable = False


class GatewayTimeoutError(ProviderTimeoutError):
    pass


class InvalidGatewayResponseError(InvalidProviderResponseError):
    pass


class ProviderNotRegisteredError(DecisionEngineError):
    pass
