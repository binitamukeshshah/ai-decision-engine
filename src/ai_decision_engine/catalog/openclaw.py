from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from ai_decision_engine.catalog.base import CatalogProviderStatus, CatalogSnapshot
from ai_decision_engine.exceptions import ProviderUnavailableError
from ai_decision_engine.schemas.models import (
    BillingMode,
    ExecutionRuntime,
    ModelDefinition,
    ModelStatus,
    QuotaStatus,
    RuntimeModelStatus,
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ProbeReadiness:
    found: bool = False
    verified: bool = False
    failure: str | None = None
    latency_ms: float | None = None


CommandRunner = Callable[[Sequence[str], float], Awaitable[CommandResult]]


async def run_command(command: Sequence[str], timeout: float) -> CommandResult:
    """Run a configured read-only OpenClaw command without shell expansion."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError as exc:
        raise ProviderUnavailableError("OpenClaw model catalog command timed out.") from exc
    except OSError as exc:
        raise ProviderUnavailableError("OpenClaw model catalog command is unavailable.") from exc
    return CommandResult(process.returncode or 0, stdout.decode(), stderr.decode())


class OpenClawCliCatalog:
    def __init__(
        self,
        command: Sequence[str] = ("openclaw", "models", "list", "--provider", "{provider}", "--json"),
        *,
        providers: Sequence[str] = ("ollama", "openai", "anthropic", "google"),
        readiness_command: Sequence[str] | None = None,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: float = 30.0,
        runner: CommandRunner = run_command,
        overrides: Sequence[ModelDefinition] = (),
    ) -> None:
        self.command = tuple(command)
        self.providers = tuple(providers)
        self.readiness_command = tuple(readiness_command) if readiness_command else None
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.runner = runner
        self.overrides = {model.model_id: model for model in overrides}
        self._cached: CatalogSnapshot | None = None
        self._cached_at = 0.0

    async def discover(self) -> CatalogSnapshot:
        if self._cached is not None and monotonic() - self._cached_at < self.cache_ttl_seconds:
            return self._cached
        rows_by_key: dict[str, dict[str, Any]] = {}
        provider_statuses: dict[str, CatalogProviderStatus] = {}
        for provider in self.providers:
            try:
                provider_rows = await self._discover_provider(provider)
            except ProviderUnavailableError:
                provider_statuses[provider] = CatalogProviderStatus.CATALOG_UNAVAILABLE
                continue
            provider_statuses[provider] = CatalogProviderStatus.AVAILABLE
            for row in provider_rows:
                reference = self._canonical_reference(row, provider)
                if reference is not None:
                    rows_by_key.setdefault(reference, row)
        if not any(status == CatalogProviderStatus.AVAILABLE for status in provider_statuses.values()):
            raise ProviderUnavailableError(
                "All configured OpenClaw provider catalog commands failed; stderr was redacted."
            )
        status_payload: Any = {}
        if self.readiness_command:
            readiness_result = await self.runner(self.readiness_command, self.timeout_seconds)
            if readiness_result.returncode == 0:
                try:
                    status_payload = json.loads(readiness_result.stdout)
                except json.JSONDecodeError:
                    status_payload = {}
        models: list[ModelDefinition] = []
        statuses: dict[str, RuntimeModelStatus] = {}
        for reference in rows_by_key:
            override = self.overrides.get(reference)
            if override is None:
                # Billing and adequacy are policy data; never guess them from catalog presence.
                continue
            configured = self._configured(status_payload, reference)
            probe = self._probe_readiness(status_payload, reference)
            auth_ready = override.provider.value == "ollama" or self._auth_ready(
                status_payload, override.provider.value
            )
            quota, quota_detail = self._quota(status_payload, override.provider.value, override.billing_mode, probe)
            if probe.failure == "rate_limit":
                quota = QuotaStatus.COOLDOWN
                quota_detail = "The live model probe reported a provider rate limit."
            elif probe.failure == "authentication":
                auth_ready = False
            readiness = configured and probe.verified and auth_ready
            if override.billing_mode != BillingMode.LOCAL:
                readiness = readiness and quota == QuotaStatus.AVAILABLE
            metadata = {
                **override.metadata,
                "catalog_discovered": True,
                "explicitly_approved": True,
                "configured": configured,
                "oauth_ready": auth_ready,
                "probe_verified": probe.verified,
                "readiness_verified": readiness,
                "quota_status": quota.value,
            }
            if probe.latency_ms is not None:
                metadata["measured_runtime_latency_ms"] = probe.latency_ms
            model = ModelDefinition(
                model_id=override.model_id,
                display_name=override.display_name,
                provider=override.provider,
                execution_runtime=ExecutionRuntime.OPENCLAW_GATEWAY,
                billing_mode=override.billing_mode,
                capabilities=override.capabilities,
                context_window=override.context_window,
                is_local=override.is_local,
                input_cost_per_million_tokens=override.input_cost_per_million_tokens,
                output_cost_per_million_tokens=override.output_cost_per_million_tokens,
                quality_score=override.quality_score,
                speed_score=override.speed_score,
                deliberately_free=override.deliberately_free,
                metadata=metadata,
                capability_quality=override.capability_quality,
                reasoning_score=override.reasoning_score,
                coding_score=override.coding_score,
                vision_support=override.vision_support,
                estimated_latency_seconds=(
                    probe.latency_ms / 1000 if probe.latency_ms is not None else override.estimated_latency_seconds
                ),
                privacy_class=override.privacy_class,
            )
            models.append(model)
            statuses[reference] = RuntimeModelStatus(
                reference,
                (
                    ModelStatus.AVAILABLE
                    if readiness
                    else ModelStatus.UNAVAILABLE
                    if probe.found
                    else ModelStatus.UNKNOWN
                ),
                True,
                True,
                "All approval and readiness gates passed."
                if readiness
                else self._missing_gate(configured, auth_ready, probe, quota, override.billing_mode),
                quota,
                quota_detail,
                True,
                readiness,
            )
        self._cached = CatalogSnapshot(tuple(models), statuses, provider_statuses)
        self._cached_at = monotonic()
        return self._cached

    async def _discover_provider(self, provider: str) -> list[dict[str, Any]]:
        result = await self.runner(self._provider_command(provider), self.timeout_seconds)
        if result.returncode != 0:
            raise ProviderUnavailableError(
                f"OpenClaw catalog command for provider {provider!r} failed; stderr was redacted."
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderUnavailableError(
                f"OpenClaw catalog command for provider {provider!r} returned malformed JSON."
            ) from exc
        return self._rows(payload)

    def _provider_command(self, provider: str) -> tuple[str, ...]:
        if "{provider}" in self.command:
            return tuple(provider if part == "{provider}" else part for part in self.command)
        command = list(self.command)
        try:
            json_index = command.index("--json")
        except ValueError:
            command.extend(("--provider", provider, "--json"))
        else:
            command[json_index:json_index] = ["--provider", provider]
        return tuple(command)

    @staticmethod
    def _canonical_reference(row: dict[str, Any], scoped_provider: str) -> str | None:
        raw = str(row.get("key") or row.get("ref") or row.get("id") or row.get("model") or "").strip()
        if not raw:
            return None
        if "/" in raw:
            provider, model = raw.split("/", 1)
        else:
            provider, model = scoped_provider, raw
        provider = provider.strip().lower()
        model = model.strip()
        if not provider or not model:
            return None
        return f"{provider}/{model}"

    @staticmethod
    def _rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("models", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        raise ProviderUnavailableError("OpenClaw model catalog JSON has an unsupported shape.")

    @staticmethod
    def _walk(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            found.append(value)
            for child in value.values():
                found.extend(OpenClawCliCatalog._walk(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(OpenClawCliCatalog._walk(child))
        return found

    @classmethod
    def _configured(cls, payload: Any, reference: str) -> bool:
        for row in cls._walk(payload):
            for key in ("allowed", "configuredModels", "allowedModels", "allowlist"):
                values = row.get(key)
                if isinstance(values, list) and reference in values:
                    return True
            candidate = row.get("modelRef") or row.get("model") or row.get("ref")
            if candidate == reference and row.get("configured") is True:
                return True
        return False

    @classmethod
    def _probe_readiness(cls, payload: Any, reference: str) -> ProbeReadiness:
        provider, model = reference.split("/", 1)
        rows = cls._nested_rows(payload, "auth", "probes", "results")
        if rows is None:
            rows = cls._nested_rows(payload, "probes", "results")
        for row in rows or []:
            candidate = row.get("modelRef") or row.get("ref") or row.get("model")
            row_provider = str(row.get("provider") or "").lower()
            matches = candidate == reference or (candidate == model and row_provider == provider)
            if not matches:
                continue
            status = str(row.get("status") or row.get("probeStatus") or "").lower()
            latency = row.get("latencyMs")
            latency_ms = float(latency) if isinstance(latency, int | float) and latency > 0 else None
            if row.get("ok") is True or status in {"ok", "ready", "success"}:
                return ProbeReadiness(True, True, latency_ms=latency_ms)
            if status in {"rate_limit", "rate_limited", "ratelimit", "too_many_requests"}:
                return ProbeReadiness(True, failure="rate_limit", latency_ms=latency_ms)
            if status in {
                "authentication_error",
                "auth_error",
                "authentication_failed",
                "unauthorized",
                "invalid_credentials",
                "forbidden",
            }:
                return ProbeReadiness(True, failure="authentication", latency_ms=latency_ms)
            diagnostic = " ".join(
                str(row.get(key) or "").lower() for key in ("error", "errorCode", "reason", "message")
            )
            if any(token in diagnostic for token in ("auth", "credential", "unauthorized", "forbidden")):
                return ProbeReadiness(True, failure="authentication", latency_ms=latency_ms)
            return ProbeReadiness(True, failure="probe", latency_ms=latency_ms)
        return ProbeReadiness()

    @staticmethod
    def _nested_rows(payload: Any, *path: str) -> list[dict[str, Any]] | None:
        value = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        if not isinstance(value, list):
            return None
        return [row for row in value if isinstance(row, dict)]

    @classmethod
    def _auth_ready(cls, payload: Any, provider: str) -> bool:
        now = datetime.now(UTC)
        paths = (
            (("auth", "providers"), False),
            (("auth", "oauth", "providers"), True),
            (("auth", "oauth"), True),
            (("auth", "profiles"), True),
        )
        rows = [(row, oauth_path) for path, oauth_path in paths for row in (cls._nested_rows(payload, *path) or [])]
        for row, oauth_path in rows:
            row_provider = str(row.get("provider") or "").lower()
            provider_matches = row_provider == provider or (provider == "anthropic" and row_provider == "claude-cli")
            if not provider_matches:
                continue
            auth_type = str(row.get("type") or row.get("kind") or row.get("authType") or "").lower()
            status = str(row.get("status") or "").lower()
            has_auth_signal = (
                oauth_path
                or auth_type in {"oauth", "token", "api_key", "api-key", "apikey", "static"}
                or any(key in row for key in ("authenticated", "usable", "credentialStatus"))
            )
            if not has_auth_signal:
                continue
            credential_status = str(row.get("credentialStatus") or "").lower()
            unavailable_statuses = {"expired", "invalid", "unusable", "authentication_error", "auth_error"}
            if status in unavailable_statuses or credential_status in {
                "expired",
                "invalid",
                "unusable",
            }:
                continue
            expiry = row.get("expiresAt") or row.get("expires_at")
            if expiry and cls._oauth_expired(expiry, now):
                continue
            usable_status = status in {"ok", "valid", "ready", "usable", "expiring"}
            if (
                row.get("usable") is not False
                and row.get("authenticated") is not False
                and (usable_status or not status)
            ):
                return True
        return False

    @staticmethod
    def _oauth_expired(value: Any, now: datetime) -> bool:
        try:
            if isinstance(value, int | float):
                timestamp = float(value)
                if timestamp > 10_000_000_000:
                    timestamp /= 1000
                expires = datetime.fromtimestamp(timestamp, UTC)
            else:
                expires = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
        except (OSError, OverflowError, TypeError, ValueError):
            return True
        return expires <= now

    @classmethod
    def _quota(
        cls, payload: Any, provider: str, billing: BillingMode, probe: ProbeReadiness
    ) -> tuple[QuotaStatus, str | None]:
        if billing == BillingMode.LOCAL:
            return QuotaStatus.AVAILABLE, "Local inference has no subscription quota window."
        rows = cls._nested_rows(payload, "auth", "providers")
        if rows is None:
            rows = cls._nested_rows(payload, "usage", "providers")
        for row in rows or []:
            if row.get("provider") != provider:
                continue
            provider_status = str(row.get("quotaStatus") or row.get("status") or "").lower()
            if provider_status in {"exhausted", "cooldown", "rate_limit", "rate_limited"}:
                status = QuotaStatus.EXHAUSTED if provider_status == "exhausted" else QuotaStatus.COOLDOWN
                return status, "The provider reported unavailable quota."
            windows = row.get("windows")
            if not isinstance(windows, list) or not windows:
                continue
            remaining: list[float] = []
            for window in windows:
                if not isinstance(window, dict):
                    continue
                if isinstance(window.get("remainingPercent"), int | float):
                    remaining.append(float(window["remainingPercent"]))
                elif isinstance(window.get("usedPercent"), int | float):
                    remaining.append(max(0.0, 100.0 - float(window["usedPercent"])))
            if remaining:
                minimum = min(remaining)
                status = QuotaStatus.AVAILABLE if minimum > 0 else QuotaStatus.EXHAUSTED
                return status, f"Rolling usage window minimum remaining: {minimum:.1f}%."
        if probe.verified:
            return QuotaStatus.AVAILABLE, "The successful live model probe reported no quota restriction."
        return QuotaStatus.UNKNOWN, "Rolling subscription quota was not available."

    @staticmethod
    def _missing_gate(
        configured: bool, auth: bool, probe: ProbeReadiness, quota: QuotaStatus, billing: BillingMode
    ) -> str:
        missing: list[str] = []
        if not configured:
            missing.append("not configured")
        if not auth:
            missing.append("authentication unavailable or expired")
        if not probe.found:
            missing.append("per-model probe missing")
        elif probe.failure == "authentication":
            missing.append("per-model probe reported an authentication failure")
        elif probe.failure == "rate_limit":
            missing.append("per-model probe reported a rate limit")
        elif not probe.verified:
            missing.append("per-model probe failed")
        if billing != BillingMode.LOCAL and quota != QuotaStatus.AVAILABLE:
            missing.append("quota unavailable")
        return ", ".join(missing) or "Readiness unverified."
