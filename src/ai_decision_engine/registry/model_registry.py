from ai_decision_engine.schemas.models import (
    BillingMode,
    Capability,
    ExecutionRuntime,
    ModelDefinition,
    ModelPrivacyClass,
    ProviderType,
)


class ModelRegistry:
    def __init__(self, models: list[ModelDefinition] | None = None) -> None:
        self._models = {model.model_id: model for model in models or []}

    def register(self, model: ModelDefinition) -> None:
        self._models[model.model_id] = model

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {model_id}") from exc

    def list_models(self) -> list[ModelDefinition]:
        return list(self._models.values())

    def for_runtime(self, runtime: ExecutionRuntime) -> list[ModelDefinition]:
        return [m for m in self._models.values() if m.execution_runtime == runtime]


def build_default_registry() -> ModelRegistry:
    # Quality/speed values are provisional configuration defaults, not benchmark claims.
    return ModelRegistry(
        [
            ModelDefinition(
                "ollama/qwen3:8b",
                "Qwen 3 8B",
                ProviderType.OLLAMA,
                frozenset(
                    {
                        Capability.CHAT,
                        Capability.REASONING,
                        Capability.CODING,
                        Capability.LONG_CONTEXT,
                    }
                ),
                40960,
                True,
                ExecutionRuntime.DIRECT_OLLAMA,
                BillingMode.LOCAL,
                quality_score=0.72,
                speed_score=0.78,
                metadata={"quality_source": "provisional", "ollama_capabilities": ["completion", "tools", "thinking"]},
                capability_quality={Capability.CHAT: 0.78, Capability.REASONING: 0.68, Capability.CODING: 0.72},
                reasoning_score=0.68,
                coding_score=0.72,
                vision_support=False,
                estimated_latency_seconds=2.5,
                privacy_class=ModelPrivacyClass.LOCAL_PRIVATE,
            ),
        ]
    )


def build_default_openclaw_approved_models() -> list[ModelDefinition]:
    """Policy-approved routes; discovery and readiness are still required."""
    return [
        ModelDefinition(
            "ollama/qwen3:8b",
            "Qwen 3 8B",
            ProviderType.OLLAMA,
            frozenset({Capability.CHAT, Capability.REASONING, Capability.CODING, Capability.LONG_CONTEXT}),
            40960,
            True,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.LOCAL,
            quality_score=0.72,
            speed_score=0.78,
            metadata={"quality_source": "provisional", "readiness_source": "openclaw_probe"},
            capability_quality={Capability.CHAT: 0.78, Capability.REASONING: 0.68, Capability.CODING: 0.72},
            reasoning_score=0.68,
            coding_score=0.72,
            vision_support=False,
            estimated_latency_seconds=2.5,
            privacy_class=ModelPrivacyClass.LOCAL_PRIVATE,
        ),
        ModelDefinition(
            "openai/gpt-5.6-sol",
            "GPT-5.6 Sol",
            ProviderType.OPENAI,
            frozenset({Capability.CHAT, Capability.REASONING, Capability.CODING, Capability.LONG_CONTEXT}),
            200000,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.SUBSCRIPTION,
            quality_score=0.9,
            speed_score=0.7,
            metadata={"quality_source": "provisional", "authentication": "openai_oauth_via_openclaw"},
            capability_quality={Capability.CHAT: 0.9, Capability.REASONING: 0.93, Capability.CODING: 0.9},
            reasoning_score=0.93,
            coding_score=0.9,
            vision_support=False,
            estimated_latency_seconds=4.0,
            privacy_class=ModelPrivacyClass.CLOUD,
        ),
        ModelDefinition(
            "anthropic/claude-opus-4-8",
            "Claude Opus 4.8",
            ProviderType.ANTHROPIC,
            frozenset({Capability.CHAT, Capability.REASONING, Capability.CODING, Capability.LONG_CONTEXT}),
            1_048_576,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.SUBSCRIPTION,
            quality_score=0.96,
            speed_score=0.45,
            metadata={"quality_source": "provisional", "authentication": "claude_cli_via_openclaw"},
            capability_quality={Capability.CHAT: 0.94, Capability.REASONING: 0.98, Capability.CODING: 0.92},
            reasoning_score=0.98,
            coding_score=0.92,
            vision_support=False,
            estimated_latency_seconds=8.0,
            privacy_class=ModelPrivacyClass.CLOUD,
        ),
        ModelDefinition(
            "anthropic/claude-sonnet-5",
            "Claude Sonnet 5",
            ProviderType.ANTHROPIC,
            frozenset({Capability.CHAT, Capability.REASONING, Capability.CODING, Capability.LONG_CONTEXT}),
            1_000_000,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.SUBSCRIPTION,
            quality_score=0.92,
            speed_score=0.72,
            metadata={"quality_source": "provisional", "authentication": "claude_cli_via_openclaw"},
            capability_quality={Capability.CHAT: 0.94, Capability.REASONING: 0.92, Capability.CODING: 0.94},
            reasoning_score=0.92,
            coding_score=0.94,
            vision_support=False,
            estimated_latency_seconds=3.5,
            privacy_class=ModelPrivacyClass.CLOUD,
        ),
        ModelDefinition(
            "google/gemini-3.1-pro-preview",
            "Gemini 3.1 Pro Preview",
            ProviderType.GOOGLE,
            frozenset(),
            1_000_000,
            False,
            ExecutionRuntime.OPENCLAW_GATEWAY,
            BillingMode.FREE_TIER,
            quality_score=0.9,
            speed_score=0.65,
            metadata={
                "quality_source": "provisional",
                "policy_note": "multimodal candidate; no end-to-end image support",
            },
            reasoning_score=0.9,
            coding_score=0.86,
            vision_support=False,
            estimated_latency_seconds=4.5,
            privacy_class=ModelPrivacyClass.CLOUD,
        ),
    ]
