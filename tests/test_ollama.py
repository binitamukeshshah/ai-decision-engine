import httpx
import pytest

from ai_decision_engine.exceptions import InvalidProviderResponseError, ModelUnavailableError
from ai_decision_engine.providers.ollama_provider import OllamaProvider
from ai_decision_engine.schemas.models import ModelStatus


@pytest.mark.asyncio
async def test_availability_comes_from_tags(local_model) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "local"}]})

    statuses = await OllamaProvider("http://test", transport=httpx.MockTransport(handler)).availability([local_model])
    assert statuses["ollama/local"].status == ModelStatus.AVAILABLE


@pytest.mark.asyncio
async def test_generate_parses_response(local_model) -> None:
    provider = OllamaProvider(
        "http://test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"response": " hi ", "eval_count": 2})),
    )
    result = await provider.generate(local_model, "hello")
    assert result.text == "hi" and result.output_tokens == 2


@pytest.mark.asyncio
async def test_generate_maps_missing_and_empty(local_model) -> None:
    missing = OllamaProvider("http://test", transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with pytest.raises(ModelUnavailableError):
        await missing.generate(local_model, "x")
    empty = OllamaProvider(
        "http://test", transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"response": ""}))
    )
    with pytest.raises(InvalidProviderResponseError):
        await empty.generate(local_model, "x")
