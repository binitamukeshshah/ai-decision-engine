import os

import pytest

from ai_decision_engine.config import Settings
from ai_decision_engine.schemas.models import PrivacyLevel
from ai_decision_engine.schemas.requests import TaskRequest
from ai_decision_engine.service import build_engine


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("ADE_RUN_OPENCLAW_INTEGRATION") != "1", reason="set ADE_RUN_OPENCLAW_INTEGRATION=1")
@pytest.mark.asyncio
async def test_live_openclaw_selection() -> None:
    decision = await build_engine(Settings.from_env()).select(
        TaskRequest("Reply briefly", privacy=PrivacyLevel.CLOUD_ALLOWED)
    )
    assert "/" in decision.selected_model
