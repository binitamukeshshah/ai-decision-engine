from __future__ import annotations

import importlib.util
from pathlib import Path

from ai_decision_engine.router.task_analyzer import CapabilityTaskAnalyzer
from ai_decision_engine.schemas.models import Complexity, PrivacyLevel


def load_runner():
    path = Path(__file__).resolve().parents[1] / "evals" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_v1_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_complexity_case_is_constructed_without_execution() -> None:
    runner = load_runner()
    case = {
        "prompt": "Analyze the decision.",
        "request": {"privacy": "cloud_allowed", "reasoning_complexity": "high"},
    }
    request = runner.request_from_case(case)
    requirements = CapabilityTaskAnalyzer().analyze(request)

    assert request.privacy == PrivacyLevel.CLOUD_ALLOWED
    assert request.reasoning_complexity == Complexity.HIGH
    assert requirements.reasoning_complexity == Complexity.HIGH


def test_sensitive_prompt_remains_local_even_with_cloud_permission() -> None:
    runner = load_runner()
    case = {
        "prompt": "Summarize this confidential customer incident report.",
        "request": {"privacy": "cloud_allowed"},
    }
    request = runner.request_from_case(case)
    requirements = CapabilityTaskAnalyzer().analyze(request)

    assert requirements.privacy_level == PrivacyLevel.LOCAL_ONLY
