from ai_decision_engine.exceptions import ProviderNotRegisteredError
from ai_decision_engine.providers.base import ModelProvider
from ai_decision_engine.schemas.models import ExecutionRuntime


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[ExecutionRuntime, ModelProvider] = {}

    def register(self, provider_type: ExecutionRuntime, provider: ModelProvider) -> None:
        self._providers[provider_type] = provider

    def get(self, provider_type: ExecutionRuntime) -> ModelProvider:
        try:
            return self._providers[provider_type]
        except KeyError as exc:
            raise ProviderNotRegisteredError(f"No provider registered for {provider_type.value}.") from exc

    def contains(self, provider_type: ExecutionRuntime) -> bool:
        return provider_type in self._providers
