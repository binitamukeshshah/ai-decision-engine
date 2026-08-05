from ai_decision_engine.providers.provider_registry import ProviderRegistry
from ai_decision_engine.registry.model_registry import ModelRegistry
from ai_decision_engine.schemas.models import ModelStatus, RuntimeModelStatus


class RuntimeAvailabilityRegistry:
    def __init__(self, models: ModelRegistry, providers: ProviderRegistry) -> None:
        self.models, self.providers = models, providers

    async def refresh(self) -> dict[str, RuntimeModelStatus]:
        statuses: dict[str, RuntimeModelStatus] = {}
        for model in self.models.list_models():
            if model.model_id in statuses:
                continue
            group = self.models.for_runtime(model.execution_runtime)
            if not self.providers.contains(model.execution_runtime):
                statuses.update(
                    {
                        m.model_id: RuntimeModelStatus(
                            m.model_id, ModelStatus.UNKNOWN, False, None, "provider not registered"
                        )
                        for m in group
                    }
                )
            else:
                statuses.update(await self.providers.get(model.execution_runtime).availability(group))
        return statuses
