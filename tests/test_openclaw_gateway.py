import httpx
import pytest

from ai_decision_engine.exceptions import (
    GatewayAuthenticationError,
    ModelNotAllowedError,
    OpenClawGatewayUnavailableError,
    ProviderRateLimitedError,
    SubscriptionQuotaExhaustedError,
)
from ai_decision_engine.providers.openclaw_gateway import OpenClawGatewayProvider
from ai_decision_engine.schemas.models import BillingMode, Capability, ExecutionRuntime, ModelDefinition, ProviderType


def model() -> ModelDefinition:
    return ModelDefinition(
        "openai/example",
        "Example",
        ProviderType.OPENAI,
        frozenset({Capability.CHAT}),
        1000,
        False,
        ExecutionRuntime.OPENCLAW_GATEWAY,
        BillingMode.SUBSCRIPTION,
    )


@pytest.mark.asyncio
async def test_gateway_request_auth_model_override_and_response_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer operator-secret"
        assert request.headers["x-openclaw-model"] == "openai/example"
        assert request.headers["x-openclaw-agent-id"] == "main"
        assert request.headers["x-openclaw-session-key"] == "session"
        assert b'"model":"openclaw/main"' in request.content
        return httpx.Response(
            200,
            json={
                "output": [{"content": [{"type": "output_text", "text": "hello"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    provider = OpenClawGatewayProvider(
        "http://gateway", "operator-secret", "main", session_key="session", transport=httpx.MockTransport(handler)
    )
    result = await provider.generate(model(), "hi")
    assert result.text == "hello" and result.input_tokens == 2


@pytest.mark.parametrize(
    ("status", "message", "error"),
    [
        (401, "bad token", GatewayAuthenticationError),
        (429, "rate limited", ProviderRateLimitedError),
        (429, "subscription quota exhausted", SubscriptionQuotaExhaustedError),
        (400, 'Model "openai/example" is not allowed', ModelNotAllowedError),
    ],
)
@pytest.mark.asyncio
async def test_gateway_typed_errors_and_token_redaction(status: int, message: str, error: type[Exception]) -> None:
    secret = "operator-secret"
    provider = OpenClawGatewayProvider(
        "http://gateway",
        secret,
        "main",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"error": {"message": f"{message} {secret}"}})
        ),
    )
    with pytest.raises(error) as caught:
        await provider.generate(model(), "hi")
    assert secret not in str(caught.value)


@pytest.mark.asyncio
async def test_gateway_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    provider = OpenClawGatewayProvider("http://gateway", "secret", "main", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenClawGatewayUnavailableError):
        await provider.generate(model(), "hi")
