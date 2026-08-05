from __future__ import annotations

import json
from typing import Any

from ai_decision_engine.catalog.openclaw import OpenClawCliCatalog
from ai_decision_engine.config import Settings
from ai_decision_engine.exceptions import ValidationError
from ai_decision_engine.ledger import UsageLedger
from ai_decision_engine.providers.ollama_provider import OllamaProvider
from ai_decision_engine.providers.openclaw_gateway import OpenClawGatewayProvider
from ai_decision_engine.providers.provider_registry import ProviderRegistry
from ai_decision_engine.registry.model_registry import (
    ModelRegistry,
    build_default_openclaw_approved_models,
    build_default_registry,
)
from ai_decision_engine.router.decision_engine import DecisionEngine
from ai_decision_engine.schemas.models import BillingMode, Capability, ExecutionRuntime, ModelDefinition, ProviderType


def _openclaw_overrides(raw: str) -> list[ModelDefinition]:
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError("ADE_OPENCLAW_MODEL_OVERRIDES_JSON must be valid JSON.") from exc
    if not isinstance(rows, list):
        raise ValidationError("ADE_OPENCLAW_MODEL_OVERRIDES_JSON must be a JSON array.")
    models: list[ModelDefinition] = []
    provider_map = {
        "openai": ProviderType.OPENAI,
        "anthropic": ProviderType.ANTHROPIC,
        "google": ProviderType.GOOGLE,
        "ollama": ProviderType.OLLAMA,
    }
    for value in rows:
        if not isinstance(value, dict):
            raise ValidationError("Every OpenClaw model override must be an object.")
        row: dict[str, Any] = value
        reference = str(row.get("model_ref", ""))
        family = reference.split("/", 1)[0]
        if family not in provider_map:
            raise ValidationError(f"Unsupported OpenClaw provider family: {family}.")
        models.append(
            ModelDefinition(
                model_id=reference,
                display_name=str(row.get("display_name", reference)),
                provider=provider_map[family],
                execution_runtime=ExecutionRuntime.OPENCLAW_GATEWAY,
                billing_mode=BillingMode(str(row["billing_mode"])),
                capabilities=frozenset(Capability(str(item)) for item in row.get("capabilities", ["chat"])),
                context_window=int(row.get("context_window", 1)),
                is_local=family == "ollama",
                input_cost_per_million_tokens=row.get("input_cost_per_million_tokens"),
                output_cost_per_million_tokens=row.get("output_cost_per_million_tokens"),
                quality_score=float(row.get("quality_score", 0.5)),
                speed_score=float(row.get("speed_score", 0.5)),
                deliberately_free=bool(row.get("deliberately_free", False)),
                metadata={"quality_source": "configured_override"},
                capability_quality={
                    Capability(str(key)): float(score) for key, score in dict(row.get("capability_quality", {})).items()
                },
            )
        )
    return models


def build_engine(settings: Settings | None = None) -> DecisionEngine:
    settings = settings or Settings.from_env()
    providers = ProviderRegistry()
    providers.register(
        ExecutionRuntime.DIRECT_OLLAMA, OllamaProvider(settings.ollama_host, settings.request_timeout_seconds)
    )
    catalog = None
    if settings.openclaw_enabled:
        providers.register(
            ExecutionRuntime.OPENCLAW_GATEWAY,
            OpenClawGatewayProvider(
                settings.openclaw_gateway_url,
                settings.openclaw_gateway_token,
                settings.openclaw_agent,
                settings.openclaw_timeout_seconds,
                settings.openclaw_session_key,
            ),
        )
        configured_overrides = _openclaw_overrides(settings.openclaw_model_overrides_json)
        approved = {model.model_id: model for model in build_default_openclaw_approved_models()}
        for model in configured_overrides:
            if model.model_id not in approved:
                raise ValidationError(f"{model.model_id} is not in the currently approved model pool.")
            approved[model.model_id] = model
        catalog = OpenClawCliCatalog(
            settings.openclaw_catalog_command,
            providers=settings.openclaw_catalog_providers,
            readiness_command=settings.openclaw_readiness_command,
            timeout_seconds=settings.openclaw_catalog_timeout_seconds,
            cache_ttl_seconds=settings.openclaw_catalog_cache_seconds,
            overrides=list(approved.values()),
        )
    registry = ModelRegistry() if settings.openclaw_enabled else build_default_registry()
    return DecisionEngine(
        registry,
        providers,
        UsageLedger(settings.usage_ledger_path, settings.monthly_cloud_budget),
        catalog=catalog,
    )
