from __future__ import annotations

from typing import Any

import httpx

from ai_decision_engine.exceptions import (
    ChargeStatus,
    GatewayAuthenticationError,
    GatewayTimeoutError,
    InvalidGatewayResponseError,
    ModelNotAllowedError,
    OpenClawGatewayUnavailableError,
    ProviderCredentialExpiredError,
    ProviderNotConfiguredError,
    ProviderRateLimitedError,
    SubscriptionQuotaExhaustedError,
)
from ai_decision_engine.providers.base import GenerationResult, ModelProvider
from ai_decision_engine.schemas.models import ModelDefinition, ModelStatus, QuotaStatus, RuntimeModelStatus


class OpenClawGatewayProvider(ModelProvider):
    """Executes canonical model references through OpenClaw's operator-authenticated Gateway."""

    def __init__(
        self,
        base_url: str,
        token: str,
        agent: str,
        timeout_seconds: float = 180.0,
        session_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.agent_model = agent if agent.startswith("openclaw/") else f"openclaw/{agent}"
        self.agent_id = self.agent_model.split("/", 1)[1]
        self.timeout_seconds = timeout_seconds
        self.session_key = session_key
        self.transport = transport

    async def availability(self, models: list[ModelDefinition]) -> dict[str, RuntimeModelStatus]:
        # Readiness comes from the catalog/status probe. Avoid paid calls during routing.
        return {
            model.model_id: RuntimeModelStatus(
                model.model_id,
                ModelStatus.AVAILABLE if model.metadata.get("readiness_verified") else ModelStatus.UNKNOWN,
                True,
                True,
                str(model.metadata.get("readiness_detail", "Catalog presence is not execution readiness.")),
                QuotaStatus(str(model.metadata.get("quota_status", QuotaStatus.UNKNOWN.value))),
                str(model.metadata.get("quota_detail")) if model.metadata.get("quota_detail") else None,
                True,
                bool(model.metadata.get("readiness_verified")),
            )
            for model in models
        }

    async def generate(self, model: ModelDefinition, prompt: str) -> GenerationResult:
        if not self._token:
            raise GatewayAuthenticationError(
                "OpenClaw Gateway execution requires an operator bearer token.",
                charge_status=ChargeStatus.NO_CHARGE,
            )
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "x-openclaw-agent-id": self.agent_id,
            "x-openclaw-model": model.canonical_ref,
        }
        if self.session_key:
            headers["x-openclaw-session-key"] = self.session_key
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.post(
                    f"{self.base_url}/v1/responses",
                    headers=headers,
                    json={"model": self.agent_model, "input": prompt, "stream": False},
                )
        except httpx.TimeoutException as exc:
            raise GatewayTimeoutError("OpenClaw Gateway timed out.") from exc
        except httpx.HTTPError as exc:
            raise OpenClawGatewayUnavailableError(
                "OpenClaw Gateway is unavailable.", charge_status=ChargeStatus.NO_CHARGE
            ) from exc
        if response.status_code >= 400:
            self._raise_gateway_error(response)
        try:
            payload = response.json()
            text = self._output_text(payload)
            usage = payload.get("usage", {})
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidGatewayResponseError("OpenClaw Gateway returned an invalid response.") from exc
        if not text.strip():
            raise InvalidGatewayResponseError("OpenClaw Gateway returned empty output.")
        return GenerationResult(text.strip(), usage.get("input_tokens"), usage.get("output_tokens"))

    @staticmethod
    def _output_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        parts: list[str] = []
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    parts.append(str(content.get("text", "")))
        return "".join(parts)

    def _raise_gateway_error(self, response: httpx.Response) -> None:
        try:
            message = str(response.json().get("error", {}).get("message", "Gateway request failed."))
        except (ValueError, AttributeError):
            message = "Gateway request failed."
        lowered = message.lower()
        safe_message = message.replace(self._token, "[REDACTED]")
        if "not allowed" in lowered:
            raise ModelNotAllowedError(safe_message, charge_status=ChargeStatus.NO_CHARGE)
        if "not configured" in lowered:
            raise ProviderNotConfiguredError(safe_message, charge_status=ChargeStatus.NO_CHARGE)
        if "expired" in lowered or "credential" in lowered:
            raise ProviderCredentialExpiredError(safe_message, charge_status=ChargeStatus.NO_CHARGE)
        if response.status_code == 401 or response.status_code == 403:
            raise GatewayAuthenticationError(
                "OpenClaw Gateway authentication failed.", charge_status=ChargeStatus.NO_CHARGE
            )
        if response.status_code == 429:
            if "quota" in lowered or "subscription" in lowered:
                raise SubscriptionQuotaExhaustedError(safe_message, charge_status=ChargeStatus.NO_CHARGE)
            raise ProviderRateLimitedError(safe_message, charge_status=ChargeStatus.NO_CHARGE)
        raise InvalidGatewayResponseError(safe_message)
