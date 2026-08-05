from ai_decision_engine.exceptions import ValidationError
from ai_decision_engine.schemas.models import BillingMode, ModelDefinition


class CostPolicy:
    @staticmethod
    def estimate(model: ModelDefinition, input_tokens: int, output_tokens: int) -> float:
        if model.billing_mode != BillingMode.API or model.deliberately_free:
            return 0.0
        if model.input_cost_per_million_tokens is None or model.output_cost_per_million_tokens is None:
            raise ValidationError("Cloud model pricing is unavailable; routing is disabled.")
        return (
            input_tokens * model.input_cost_per_million_tokens + output_tokens * model.output_cost_per_million_tokens
        ) / 1_000_000

    @staticmethod
    def estimate_input_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)
