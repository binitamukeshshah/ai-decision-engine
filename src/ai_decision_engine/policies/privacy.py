from ai_decision_engine.router.task_analyzer import TaskRequirements
from ai_decision_engine.schemas.models import ModelDefinition, PrivacyLevel


class PrivacyPolicy:
    def allows(self, model: ModelDefinition, requirements: TaskRequirements) -> bool:
        return requirements.privacy_level != PrivacyLevel.LOCAL_ONLY or model.is_local
