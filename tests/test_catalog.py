import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai_decision_engine.catalog import CatalogProviderStatus
from ai_decision_engine.catalog.openclaw import CommandResult, OpenClawCliCatalog
from ai_decision_engine.exceptions import NoEligibleModelError, ProviderUnavailableError, RejectionCode
from ai_decision_engine.providers.provider_registry import ProviderRegistry
from ai_decision_engine.registry.model_registry import ModelRegistry, build_default_openclaw_approved_models
from ai_decision_engine.router.decision_engine import DecisionEngine
from ai_decision_engine.schemas.models import Complexity, ModelStatus, PrivacyLevel, QuotaStatus
from ai_decision_engine.schemas.requests import TaskRequest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict[str, object]:
    value = json.loads((FIXTURES / name).read_text())
    assert isinstance(value, dict)
    return value


@pytest.fixture
def catalog_payload() -> dict[str, object]:
    return fixture("openclaw_2026_7_1_catalog.json")


def status_payload(
    *,
    oauth_expired: bool = False,
    openai_quota: float = 63.0,
    anthropic_quota: float = 54.0,
    openai_probe: str = "ok",
    opus_probe: str = "ok",
    sonnet_probe: str = "unverified",
) -> dict[str, object]:
    payload = deepcopy(fixture("openclaw_2026_7_1_status.json"))
    profiles = payload["auth"]["oauth"]  # type: ignore[index]
    openai_profile = next(row for row in profiles if row["provider"] == "openai")  # type: ignore[union-attr]
    if oauth_expired:
        openai_profile.update(status="expired", usable=False, expiresAt="2020-01-01T00:00:00Z")
    providers = payload["auth"]["providers"]  # type: ignore[index]
    next(row for row in providers if row["provider"] == "openai")["windows"][0][  # type: ignore[union-attr,index]
        "remainingPercent"
    ] = openai_quota
    next(row for row in providers if row["provider"] == "anthropic")["windows"][0][  # type: ignore[union-attr,index]
        "remainingPercent"
    ] = anthropic_quota
    probes = payload["auth"]["probes"]["results"]  # type: ignore[index]
    probe_status = {
        ("openai", "gpt-5.6-sol"): openai_probe,
        ("anthropic", "claude-opus-4-8"): opus_probe,
        ("anthropic", "claude-sonnet-5"): sonnet_probe,
    }
    for row in probes:  # type: ignore[union-attr]
        key = (row["provider"], row["model"].split("/", 1)[-1])
        if key in probe_status:
            row["status"] = probe_status[key]
    return payload


def catalog_with(catalog_payload: dict[str, object], readiness: dict[str, object]) -> OpenClawCliCatalog:
    async def runner(command, timeout):
        if "status" in command:
            payload = readiness
        else:
            provider = command[command.index("--provider") + 1]
            rows = [row for row in catalog_payload["models"] if row["key"].startswith(f"{provider}/")]  # type: ignore[index,union-attr]
            payload = {"count": len(rows), "models": rows}
        return CommandResult(0, json.dumps(payload), "")

    return OpenClawCliCatalog(
        readiness_command=("openclaw", "models", "status", "--probe", "--json"),
        runner=runner,
        overrides=build_default_openclaw_approved_models(),
    )


@pytest.mark.asyncio
async def test_live_shape_only_approved_configured_probe_verified_models_enter_pool(catalog_payload) -> None:
    snapshot = await catalog_with(catalog_payload, status_payload()).discover()
    assert {model.model_id for model in snapshot.models} == {
        "ollama/qwen3:8b",
        "openai/gpt-5.6-sol",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-5",
        "google/gemini-3.1-pro-preview",
    }
    assert snapshot.statuses["ollama/qwen3:8b"].status == ModelStatus.AVAILABLE
    assert snapshot.statuses["openai/gpt-5.6-sol"].status == ModelStatus.AVAILABLE
    assert snapshot.statuses["anthropic/claude-opus-4-8"].status == ModelStatus.AVAILABLE
    assert snapshot.statuses["anthropic/claude-sonnet-5"].status == ModelStatus.UNKNOWN
    assert snapshot.statuses["google/gemini-3.1-pro-preview"].status == ModelStatus.UNAVAILABLE
    assert "openai/gpt-5.5" not in snapshot.statuses
    assert snapshot.statuses["openai/gpt-5.6-sol"].quota_status == QuotaStatus.AVAILABLE
    models = {model.model_id: model for model in snapshot.models}
    assert models["ollama/qwen3:8b"].metadata["measured_runtime_latency_ms"] == 412.0
    assert models["openai/gpt-5.6-sol"].estimated_latency_seconds == 0.921
    assert models["anthropic/claude-opus-4-8"].metadata["measured_runtime_latency_ms"] == 1862.0
    assert snapshot.statuses["google/gemini-3.1-pro-preview"].quota_status == QuotaStatus.COOLDOWN
    serialized = json.dumps(
        {
            "metadata": [model.metadata for model in snapshot.models],
            "details": [status.detail for status in snapshot.statuses.values()],
        }
    )
    assert "private@example.com" not in serialized
    assert "anthropic:claude-cli" not in serialized


@pytest.mark.asyncio
async def test_openclaw_2026_7_1_2_allowed_and_nested_auth_mapping(catalog_payload) -> None:
    readiness = fixture("openclaw_2026_7_1_2_status.json")
    snapshot = await catalog_with(catalog_payload, readiness).discover()
    statuses = snapshot.statuses
    models = {model.model_id: model for model in snapshot.models}

    for reference in (
        "ollama/qwen3:8b",
        "openai/gpt-5.6-sol",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-5",
        "google/gemini-3.1-pro-preview",
    ):
        assert models[reference].metadata["configured"] is True

    assert statuses["ollama/qwen3:8b"].readiness_verified
    assert statuses["openai/gpt-5.6-sol"].readiness_verified
    assert models["openai/gpt-5.6-sol"].metadata["oauth_ready"] is True
    assert statuses["openai/gpt-5.6-sol"].quota_status == QuotaStatus.AVAILABLE
    assert statuses["anthropic/claude-opus-4-8"].readiness_verified
    assert models["anthropic/claude-opus-4-8"].metadata["oauth_ready"] is True
    assert statuses["anthropic/claude-opus-4-8"].quota_status == QuotaStatus.AVAILABLE

    sonnet = statuses["anthropic/claude-sonnet-5"]
    assert not sonnet.readiness_verified
    assert sonnet.status == ModelStatus.UNKNOWN
    assert sonnet.detail == "per-model probe missing, quota unavailable"

    gemini = statuses["google/gemini-3.1-pro-preview"]
    assert not gemini.readiness_verified
    assert gemini.status == ModelStatus.UNAVAILABLE
    assert gemini.quota_status == QuotaStatus.COOLDOWN
    assert "rate limit" in gemini.detail

    serialized = json.dumps(
        {
            "metadata": [model.metadata for model in snapshot.models],
            "details": [status.detail for status in statuses.values()],
        }
    )
    assert "anthropic:claude-cli" not in serialized
    assert "private@example.com" not in serialized
    assert "Private Claude profile" not in serialized


@pytest.mark.asyncio
async def test_openai_oauth_ok_and_successful_probe_verify_readiness(catalog_payload) -> None:
    snapshot = await catalog_with(catalog_payload, fixture("openclaw_2026_7_1_2_status.json")).discover()

    model = next(model for model in snapshot.models if model.model_id == "openai/gpt-5.6-sol")
    status = snapshot.statuses[model.model_id]
    assert model.metadata["oauth_ready"] is True
    assert model.metadata["probe_verified"] is True
    assert status.readiness_verified is True
    assert status.status == ModelStatus.AVAILABLE


@pytest.mark.asyncio
async def test_claude_cli_oauth_expiring_before_expiry_is_usable(catalog_payload) -> None:
    snapshot = await catalog_with(catalog_payload, fixture("openclaw_2026_7_1_2_status.json")).discover()

    model = next(model for model in snapshot.models if model.model_id == "anthropic/claude-opus-4-8")
    assert model.metadata["oauth_ready"] is True
    assert snapshot.statuses[model.model_id].readiness_verified is True


@pytest.mark.asyncio
async def test_oauth_expiring_but_actually_expired_is_rejected(catalog_payload) -> None:
    readiness = fixture("openclaw_2026_7_1_2_status.json")
    oauth_providers = readiness["auth"]["oauth"]["providers"]  # type: ignore[index]
    openai = next(row for row in oauth_providers if row["provider"] == "openai")  # type: ignore[union-attr]
    openai.update(status="expiring", expiresAt="2020-01-01T00:00:00Z")

    snapshot = await catalog_with(catalog_payload, readiness).discover()

    model = next(model for model in snapshot.models if model.model_id == "openai/gpt-5.6-sol")
    status = snapshot.statuses[model.model_id]
    assert model.metadata["probe_verified"] is True
    assert model.metadata["oauth_ready"] is False
    assert status.readiness_verified is False
    assert status.status == ModelStatus.UNAVAILABLE
    assert "authentication unavailable or expired" in status.detail


@pytest.mark.parametrize(
    ("readiness", "expected_detail"),
    [
        (status_payload(oauth_expired=True), "authentication"),
        (status_payload(openai_quota=0), "quota"),
        (status_payload(openai_probe="failed"), "probe"),
        (status_payload(openai_probe="authentication_error"), "authentication"),
    ],
)
@pytest.mark.asyncio
async def test_expired_oauth_empty_quota_or_failed_probe_excludes_openai(
    catalog_payload, readiness, expected_detail
) -> None:
    snapshot = await catalog_with(catalog_payload, readiness).discover()
    status = snapshot.statuses["openai/gpt-5.6-sol"]
    assert status.status == ModelStatus.UNAVAILABLE
    assert expected_detail in status.detail


@pytest.mark.asyncio
async def test_catalog_cache_avoids_repeated_commands(catalog_payload) -> None:
    calls = 0

    async def runner(command, timeout):
        nonlocal calls
        calls += 1
        if "status" in command:
            payload = status_payload()
        else:
            provider = command[command.index("--provider") + 1]
            rows = [row for row in catalog_payload["models"] if row["key"].startswith(f"{provider}/")]
            payload = {"count": len(rows), "models": rows}
        return CommandResult(0, json.dumps(payload), "")

    catalog = OpenClawCliCatalog(
        readiness_command=("openclaw", "models", "status", "--probe", "--json"),
        runner=runner,
        overrides=build_default_openclaw_approved_models(),
    )
    assert await catalog.discover() is await catalog.discover()
    assert calls == 5


@pytest.mark.asyncio
async def test_all_provider_catalogs_failing_fails_discovery_without_exposing_stderr() -> None:
    async def failed(command, timeout):
        return CommandResult(1, "", "Bearer secret-provider-token")

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await OpenClawCliCatalog(runner=failed).discover()
    assert "secret-provider-token" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_all_four_provider_catalogs_use_scoped_commands_and_merge_with_readiness(catalog_payload) -> None:
    calls: list[tuple[str, ...]] = []

    async def runner(command, timeout):
        command_tuple = tuple(command)
        calls.append(command_tuple)
        if "status" in command:
            return CommandResult(0, json.dumps(status_payload()), "")
        provider = command[command.index("--provider") + 1]
        rows = [row for row in catalog_payload["models"] if row["key"].startswith(f"{provider}/")]
        return CommandResult(0, json.dumps({"count": len(rows), "models": rows}), "")

    snapshot = await OpenClawCliCatalog(
        readiness_command=("openclaw", "models", "status", "--probe", "--json"),
        runner=runner,
        overrides=build_default_openclaw_approved_models(),
    ).discover()

    catalog_calls = [command for command in calls if "list" in command]
    assert [command[command.index("--provider") + 1] for command in catalog_calls] == [
        "ollama",
        "openai",
        "anthropic",
        "google",
    ]
    assert all("--json" in command for command in catalog_calls)
    assert all(command != ("openclaw", "models", "list", "--json") for command in catalog_calls)
    assert all(status == CatalogProviderStatus.AVAILABLE for status in snapshot.provider_statuses.values())
    assert snapshot.statuses["openai/gpt-5.6-sol"].readiness_verified


@pytest.mark.asyncio
async def test_one_provider_failure_is_recorded_while_other_catalogs_remain_usable(catalog_payload) -> None:
    async def runner(command, timeout):
        if "status" in command:
            return CommandResult(0, json.dumps(status_payload()), "")
        provider = command[command.index("--provider") + 1]
        if provider == "google":
            return CommandResult(1, "", "private diagnostic")
        rows = [row for row in catalog_payload["models"] if row["key"].startswith(f"{provider}/")]
        return CommandResult(0, json.dumps({"models": rows}), "")

    snapshot = await OpenClawCliCatalog(
        readiness_command=("openclaw", "models", "status", "--probe", "--json"),
        runner=runner,
        overrides=build_default_openclaw_approved_models(),
    ).discover()
    assert snapshot.provider_statuses["google"] == CatalogProviderStatus.CATALOG_UNAVAILABLE
    assert snapshot.provider_statuses["ollama"] == CatalogProviderStatus.AVAILABLE
    assert "google/gemini-3.1-pro-preview" not in snapshot.statuses
    assert "ollama/qwen3:8b" in snapshot.statuses


@pytest.mark.asyncio
async def test_malformed_provider_json_does_not_discard_valid_provider_catalogs(catalog_payload) -> None:
    async def runner(command, timeout):
        if "status" in command:
            return CommandResult(0, json.dumps(status_payload()), "")
        provider = command[command.index("--provider") + 1]
        if provider == "anthropic":
            return CommandResult(0, "not-json", "must remain redacted")
        rows = [row for row in catalog_payload["models"] if row["key"].startswith(f"{provider}/")]
        return CommandResult(0, json.dumps({"models": rows}), "")

    snapshot = await OpenClawCliCatalog(
        readiness_command=("openclaw", "models", "status", "--probe", "--json"),
        runner=runner,
        overrides=build_default_openclaw_approved_models(),
    ).discover()
    assert snapshot.provider_statuses["anthropic"] == CatalogProviderStatus.CATALOG_UNAVAILABLE
    assert "openai/gpt-5.6-sol" in snapshot.statuses
    assert "anthropic/claude-opus-4-8" not in snapshot.statuses


@pytest.mark.asyncio
async def test_duplicate_canonical_model_keys_are_deduplicated() -> None:
    qwen = {
        "key": "ollama/qwen3:8b",
        "name": "Qwen 3 8B",
        "input": ["text"],
        "contextWindow": 40960,
    }

    async def runner(command, timeout):
        if "status" in command:
            return CommandResult(0, json.dumps(status_payload()), "")
        provider = command[command.index("--provider") + 1]
        rows = [qwen, {**qwen, "key": "OLLAMA/qwen3:8b"}] if provider == "ollama" else []
        return CommandResult(0, json.dumps({"models": rows}), "")

    snapshot = await OpenClawCliCatalog(
        readiness_command=("openclaw", "models", "status", "--probe", "--json"),
        runner=runner,
        overrides=build_default_openclaw_approved_models(),
    ).discover()
    assert [model.model_id for model in snapshot.models] == ["ollama/qwen3:8b"]


@pytest.mark.asyncio
async def test_live_pool_routing_local_cloud_and_privacy(catalog_payload, ledger) -> None:
    catalog = catalog_with(catalog_payload, status_payload())
    engine = DecisionEngine(ModelRegistry(), ProviderRegistry(), ledger, catalog=catalog)
    local = await engine.select(TaskRequest("Explain this simply"))
    assert local.selected_model == "ollama/qwen3:8b"

    demanding = await engine.select(
        TaskRequest(
            "Evaluate the supplied architecture constraints.",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            reasoning_complexity=Complexity.HIGH,
        )
    )
    assert demanding.selected_model == "openai/gpt-5.6-sol"
    assert demanding.explicit_cloud_permission
    assert demanding.confidence >= 0.9
    assert "Estimated reasoning score" in demanding.routing_report
    assert "ollama/qwen3:8b" in demanding.routing_report

    with pytest.raises(NoEligibleModelError):
        await engine.select(TaskRequest("private task", reasoning_complexity=Complexity.HIGH))


@pytest.mark.asyncio
async def test_inferred_high_strategy_selects_gpt_while_local_only_blocks_cloud(catalog_payload, ledger) -> None:
    prompt = (
        "Evaluate three monetization strategies for an AI productivity platform. Compare risks, pricing power, "
        "and execution complexity, then recommend a staged launch strategy with explicit tradeoffs."
    )
    readiness = status_payload()
    probes = readiness["auth"]["probes"]["results"]  # type: ignore[index]
    for probe in probes:  # type: ignore[union-attr]
        if probe["model"] == "openai/gpt-5.6-sol":
            probe["latencyMs"] = 5000
        elif probe["model"] == "anthropic/claude-opus-4-8":
            probe["latencyMs"] = 1000
    engine = DecisionEngine(
        ModelRegistry(), ProviderRegistry(), ledger, catalog=catalog_with(catalog_payload, readiness)
    )

    decision = await engine.select(TaskRequest(prompt, privacy=PrivacyLevel.CLOUD_ALLOWED))
    assert decision.selected_model == "openai/gpt-5.6-sol"
    assert decision.fallback_order[0] == "anthropic/claude-opus-4-8"
    assert "High reasoning tier policy ranks this model as primary" in decision.routing_reason

    with pytest.raises(NoEligibleModelError) as exc_info:
        await engine.select(TaskRequest(prompt))
    rejections = {item.model_id: item.code for item in exc_info.value.rejections}
    assert rejections["openai/gpt-5.6-sol"] == RejectionCode.PRIVACY_BLOCKED
    assert rejections["anthropic/claude-opus-4-8"] == RejectionCode.PRIVACY_BLOCKED


@pytest.mark.asyncio
async def test_extreme_reasoning_selects_probe_verified_opus(catalog_payload, ledger) -> None:
    engine = DecisionEngine(
        ModelRegistry(), ProviderRegistry(), ledger, catalog=catalog_with(catalog_payload, status_payload())
    )
    decision = await engine.select(
        TaskRequest(
            "Evaluate the complete system at the highest rigor.",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            reasoning_complexity=Complexity.EXTREME,
        )
    )
    assert decision.selected_model == "anthropic/claude-opus-4-8"


@pytest.mark.asyncio
async def test_inferred_extreme_system_analysis_selects_opus(catalog_payload, ledger) -> None:
    prompt = (
        "Review the architecture and compare three deployment approaches. Assess privacy, security, reliability, "
        "and failure-mode risks; evaluate cost, latency, scalability, and operational tradeoffs; design a phased "
        "migration with rollback criteria; then synthesize the evidence and recommend an implementation strategy."
    )
    engine = DecisionEngine(
        ModelRegistry(), ProviderRegistry(), ledger, catalog=catalog_with(catalog_payload, status_payload())
    )

    decision = await engine.select(TaskRequest(prompt, privacy=PrivacyLevel.CLOUD_ALLOWED))
    assert decision.selected_model == "anthropic/claude-opus-4-8"
    assert decision.fallback_order[0] == "openai/gpt-5.6-sol"
    assert "Extreme reasoning tier policy ranks this model as primary" in decision.routing_reason


@pytest.mark.asyncio
async def test_sonnet_and_rate_limited_gemini_remain_ineligible(catalog_payload, ledger) -> None:
    engine = DecisionEngine(
        ModelRegistry(), ProviderRegistry(), ledger, catalog=catalog_with(catalog_payload, status_payload())
    )
    decision = await engine.select(TaskRequest("Write a short note", privacy=PrivacyLevel.CLOUD_ALLOWED))
    rejected = {item.model_id: item for item in decision.rejected_candidates}
    assert rejected["anthropic/claude-sonnet-5"].code == RejectionCode.MODEL_UNAVAILABLE
    assert "probe" in rejected["anthropic/claude-sonnet-5"].reason
    assert rejected["google/gemini-3.1-pro-preview"].code == RejectionCode.QUOTA_UNAVAILABLE
    assert "rate limit" in rejected["google/gemini-3.1-pro-preview"].reason


@pytest.mark.asyncio
async def test_legacy_probe_auth_and_usage_paths_remain_supported(catalog_payload) -> None:
    readiness = status_payload()
    auth = readiness["auth"]  # type: ignore[index]
    auth["profiles"] = auth.pop("oauth")  # type: ignore[union-attr]
    probes = auth.pop("probes")  # type: ignore[union-attr]
    usage = {"providers": auth.pop("providers")}  # type: ignore[union-attr]
    readiness["probes"] = probes
    readiness["usage"] = usage

    snapshot = await catalog_with(catalog_payload, readiness).discover()

    assert snapshot.statuses["ollama/qwen3:8b"].readiness_verified
    assert snapshot.statuses["openai/gpt-5.6-sol"].readiness_verified
    assert snapshot.statuses["anthropic/claude-opus-4-8"].readiness_verified


@pytest.mark.asyncio
async def test_anthropic_quota_expiry_falls_back_from_opus_to_gpt(catalog_payload, ledger) -> None:
    engine = DecisionEngine(
        ModelRegistry(),
        ProviderRegistry(),
        ledger,
        catalog=catalog_with(catalog_payload, status_payload(anthropic_quota=0)),
    )
    decision = await engine.select(
        TaskRequest(
            "Highest-rigor analysis",
            privacy=PrivacyLevel.CLOUD_ALLOWED,
            reasoning_complexity=Complexity.EXTREME,
        )
    )
    rejected = {item.model_id: item for item in decision.rejected_candidates}
    assert decision.selected_model == "openai/gpt-5.6-sol"
    assert rejected["anthropic/claude-opus-4-8"].code == RejectionCode.QUOTA_UNAVAILABLE
