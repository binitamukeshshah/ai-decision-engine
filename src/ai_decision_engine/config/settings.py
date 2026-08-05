from __future__ import annotations

import math
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from ai_decision_engine.exceptions import ValidationError


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://localhost:11434"
    request_timeout_seconds: float = 180.0
    monthly_cloud_budget: float = 10.0
    usage_ledger_path: Path = Path(".data/usage-ledger.sqlite3")
    logging_level: str = "INFO"
    openclaw_enabled: bool = False
    openclaw_gateway_url: str = "http://localhost:18789"
    openclaw_gateway_token: str = ""
    openclaw_agent: str = "openclaw/default"
    openclaw_timeout_seconds: float = 180.0
    openclaw_session_key: str | None = None
    openclaw_catalog_command: tuple[str, ...] = (
        "openclaw",
        "models",
        "list",
        "--provider",
        "{provider}",
        "--json",
    )
    openclaw_catalog_providers: tuple[str, ...] = ("ollama", "openai", "anthropic", "google")
    openclaw_readiness_command: tuple[str, ...] = ("openclaw", "models", "status", "--probe", "--json")
    openclaw_catalog_timeout_seconds: float = 10.0
    openclaw_catalog_cache_seconds: float = 30.0
    openclaw_model_overrides_json: str = "[]"

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        settings = cls(
            ollama_host=os.getenv("ADE_OLLAMA_HOST", cls.ollama_host),
            request_timeout_seconds=float(os.getenv("ADE_REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds)),
            monthly_cloud_budget=float(os.getenv("ADE_MONTHLY_CLOUD_BUDGET", cls.monthly_cloud_budget)),
            usage_ledger_path=Path(os.getenv("ADE_USAGE_LEDGER_PATH", str(cls.usage_ledger_path))),
            logging_level=os.getenv("ADE_LOGGING_LEVEL", cls.logging_level),
            openclaw_enabled=os.getenv("ADE_OPENCLAW_ENABLED", "false").lower() in {"1", "true", "yes"},
            openclaw_gateway_url=os.getenv("ADE_OPENCLAW_GATEWAY_URL", cls.openclaw_gateway_url),
            openclaw_gateway_token=os.getenv("ADE_OPENCLAW_GATEWAY_TOKEN", ""),
            openclaw_agent=os.getenv("ADE_OPENCLAW_AGENT", cls.openclaw_agent),
            openclaw_timeout_seconds=float(os.getenv("ADE_OPENCLAW_TIMEOUT_SECONDS", cls.openclaw_timeout_seconds)),
            openclaw_session_key=os.getenv("ADE_OPENCLAW_SESSION_KEY") or None,
            openclaw_catalog_command=tuple(
                shlex.split(
                    os.getenv(
                        "ADE_OPENCLAW_CATALOG_COMMAND",
                        "openclaw models list --provider {provider} --json",
                    )
                )
            ),
            openclaw_catalog_providers=tuple(
                provider.strip().lower()
                for provider in os.getenv("ADE_OPENCLAW_CATALOG_PROVIDERS", "ollama,openai,anthropic,google").split(",")
                if provider.strip()
            ),
            openclaw_readiness_command=tuple(
                shlex.split(os.getenv("ADE_OPENCLAW_READINESS_COMMAND", "openclaw models status --probe --json"))
            ),
            openclaw_catalog_timeout_seconds=float(
                os.getenv("ADE_OPENCLAW_CATALOG_TIMEOUT_SECONDS", cls.openclaw_catalog_timeout_seconds)
            ),
            openclaw_catalog_cache_seconds=float(
                os.getenv("ADE_OPENCLAW_CATALOG_CACHE_SECONDS", cls.openclaw_catalog_cache_seconds)
            ),
            openclaw_model_overrides_json=os.getenv("ADE_OPENCLAW_MODEL_OVERRIDES_JSON", "[]"),
        )
        if not math.isfinite(settings.request_timeout_seconds) or settings.request_timeout_seconds <= 0:
            raise ValidationError("ADE_REQUEST_TIMEOUT_SECONDS must be finite and positive.")
        if not math.isfinite(settings.monthly_cloud_budget) or settings.monthly_cloud_budget < 0:
            raise ValidationError("ADE_MONTHLY_CLOUD_BUDGET must be finite and non-negative.")
        for name, value in (
            ("OpenClaw timeout", settings.openclaw_timeout_seconds),
            ("catalog timeout", settings.openclaw_catalog_timeout_seconds),
            ("catalog cache TTL", settings.openclaw_catalog_cache_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be finite and positive.")
        if not settings.openclaw_catalog_command:
            raise ValidationError("ADE_OPENCLAW_CATALOG_COMMAND must not be empty.")
        if not settings.openclaw_catalog_providers:
            raise ValidationError("ADE_OPENCLAW_CATALOG_PROVIDERS must contain at least one provider.")
        if any(not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", provider) for provider in settings.openclaw_catalog_providers):
            raise ValidationError("ADE_OPENCLAW_CATALOG_PROVIDERS contains an invalid provider id.")
        return settings
