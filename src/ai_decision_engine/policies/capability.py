from ai_decision_engine.router.task_analyzer import TaskRequirements
from ai_decision_engine.schemas.models import ModelDefinition


class CapabilityPolicy:
    def allows(self, model: ModelDefinition, requirements: TaskRequirements) -> bool:
        return model.supports(requirements.capabilities)
