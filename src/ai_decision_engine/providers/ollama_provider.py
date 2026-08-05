from __future__ import annotations

from typing import Any

import httpx

from ai_decision_engine.exceptions import (
    ChargeStatus,
    InvalidProviderResponseError,
    ModelUnavailableError,
    ProviderError,
    ProviderHTTPError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ai_decision_engine.providers.base import GenerationResult, ModelProvider
from ai_decision_engine.schemas.models import ModelDefinition, ModelStatus, RuntimeModelStatus


class OllamaProvider(ModelProvider):
    def __init__(
        self, host: str, timeout_seconds: float = 180.0, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.request(method, f"{self.host}{path}", **kwargs)
                response.raise_for_status()
                return response
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Ollama request timed out.", charge_status=ChargeStatus.POSSIBLE_CHARGE) from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Ollama is unreachable at {self.host}.", charge_status=ChargeStatus.NO_CHARGE
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelUnavailableError(
                    "Ollama model or endpoint was not found.", charge_status=ChargeStatus.NO_CHARGE
                ) from exc
            raise ProviderHTTPError(
                f"Ollama returned HTTP {exc.response.status_code}.", charge_status=ChargeStatus.POSSIBLE_CHARGE
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Ollama request failed: {exc}", charge_status=ChargeStatus.NO_CHARGE
            ) from exc

    async def availability(self, models: list[ModelDefinition]) -> dict[str, RuntimeModelStatus]:
        try:
            response = await self._request("GET", "/api/tags")
            payload = response.json()
            entries = payload.get("models")
            if not isinstance(entries, list):
                raise InvalidProviderResponseError("Ollama /api/tags omitted the models list.")
            installed = {f"ollama/{entry.get('name')}" for entry in entries if isinstance(entry, dict)}
        except ProviderError as exc:
            return {
                m.model_id: RuntimeModelStatus(
                    m.model_id,
                    ModelStatus.UNKNOWN,
                    not isinstance(exc, ProviderUnavailableError),
                    None,
                    str(exc),
                )
                for m in models
            }
        except (ValueError, InvalidProviderResponseError) as exc:
            return {
                m.model_id: RuntimeModelStatus(m.model_id, ModelStatus.UNKNOWN, True, None, str(exc)) for m in models
            }
        return {
            m.model_id: RuntimeModelStatus(
                m.model_id,
                ModelStatus.AVAILABLE if m.model_id in installed else ModelStatus.UNAVAILABLE,
                True,
                m.model_id in installed,
                "installed" if m.model_id in installed else "known but not installed",
            )
            for m in models
        }

    async def generate(self, model: ModelDefinition, prompt: str) -> GenerationResult:
        response = await self._request(
            "POST",
            "/api/generate",
            json={"model": model.model_id.removeprefix("ollama/"), "prompt": prompt, "stream": False},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidProviderResponseError(
                "Ollama returned invalid JSON.", charge_status=ChargeStatus.POSSIBLE_CHARGE
            ) from exc
        text = payload.get("response")
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderResponseError(
                f"Ollama returned empty output for {model.model_id}.", charge_status=ChargeStatus.POSSIBLE_CHARGE
            )
        return GenerationResult(text.strip(), payload.get("prompt_eval_count"), payload.get("eval_count"))
