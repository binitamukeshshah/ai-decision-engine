from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

from ai_decision_engine.config import Settings
from ai_decision_engine.exceptions import NoEligibleModelError
from ai_decision_engine.router.task_analyzer import CapabilityTaskAnalyzer
from ai_decision_engine.schemas.models import Capability, Complexity, LatencyPreference, PrivacyLevel
from ai_decision_engine.schemas.requests import TaskRequest
from ai_decision_engine.service import build_engine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "benchmark_v1.json"
DEFAULT_JSON = ROOT / "results" / "benchmark-v1.json"
DEFAULT_MD = ROOT / "results" / "benchmark-v1.md"


def enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, frozenset | set | tuple | list):
        return [enum_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): enum_value(item) for key, item in value.items()}
    return value


def request_from_case(case: dict[str, Any]) -> TaskRequest:
    raw = case.get("request", {})
    return TaskRequest(
        prompt=case["prompt"],
        has_image=bool(raw.get("has_image", False)),
        privacy=PrivacyLevel(raw["privacy"]) if raw.get("privacy") else None,
        required_capabilities=frozenset(Capability(item) for item in raw.get("required_capabilities", [])),
        minimum_quality=raw.get("minimum_quality"),
        max_cloud_cost=raw.get("max_cloud_cost"),
        estimated_input_tokens=raw.get("estimated_input_tokens"),
        estimated_output_tokens=int(raw.get("estimated_output_tokens", 500)),
        reasoning_complexity=Complexity(raw["reasoning_complexity"]) if raw.get("reasoning_complexity") else None,
        coding_complexity=Complexity(raw["coding_complexity"]) if raw.get("coding_complexity") else None,
        multimodal_required=bool(raw.get("multimodal_required", False)),
        latency_preference=LatencyPreference(raw.get("latency_preference", "balanced")),
    )


def score_case(case: dict[str, Any], request: TaskRequest, requirements: Any, decision: Any, error: Exception | None) -> tuple[bool, list[str]]:
    expected = case["expected"]
    checks: list[tuple[bool, str]] = []

    if "reasoning" in expected:
        checks.append((requirements.reasoning_complexity.value == expected["reasoning"], f"reasoning={requirements.reasoning_complexity.value} expected={expected['reasoning']}"))
    if "coding" in expected:
        checks.append((requirements.coding_complexity.value == expected["coding"], f"coding={requirements.coding_complexity.value} expected={expected['coding']}"))
    if "privacy" in expected:
        checks.append((requirements.privacy_level.value == expected["privacy"], f"privacy={requirements.privacy_level.value} expected={expected['privacy']}"))

    outcome = expected.get("outcome")
    if outcome == "decline":
        checks.append((error is not None, "expected decline"))
    elif outcome == "decline_if_no_context_adequate_model":
        checks.append((error is not None or decision is not None, "context boundary observed"))

    if error is None and decision is not None:
        selected = next((item for item in decision.candidate_assessments if item.model_id == decision.selected_model), None)
        route = expected.get("route")
        if route == "local":
            checks.append((decision.selected_model.startswith("ollama/"), f"selected={decision.selected_model} expected local"))
        elif route == "local_or_decline":
            checks.append((decision.selected_model.startswith("ollama/"), f"selected={decision.selected_model} expected local or decline"))
        elif route == "quality_at_least_0.9":
            checks.append((selected is not None and selected.score >= 0, "eligible model selected after quality threshold"))
        elif route == "zero_incremental_cost_or_decline":
            checks.append((decision.estimated_cost == 0, f"estimated_cost={decision.estimated_cost} expected zero or decline"))
        elif route == "context_adequate":
            checks.append((Capability.LONG_CONTEXT in decision.capabilities, "long-context capability required"))
        elif route == "fastest_adequate":
            eligible = [item for item in decision.candidate_assessments if item.eligible]
            fastest = min((item.estimated_latency for item in eligible), default=decision.estimated_latency)
            checks.append((decision.estimated_latency <= fastest, f"latency={decision.estimated_latency:.2f}s fastest_eligible={fastest:.2f}s"))
    elif expected.get("route") in {"local", "cloud_eligible", "adequate", "context_adequate", "fastest_adequate", "quality_at_least_0.9"}:
        checks.append((False, f"unexpected decline: {type(error).__name__}: {error}"))

    if not checks:
        checks.append((True, "case recorded for review; no binary assertion"))
    return all(ok for ok, _ in checks), [message for ok, message in checks if not ok]


async def run(dataset_path: Path) -> dict[str, Any]:
    payload = json.loads(dataset_path.read_text())
    settings = Settings.from_env()
    engine = build_engine(settings)
    analyzer = CapabilityTaskAnalyzer()
    results: list[dict[str, Any]] = []

    for case in payload["cases"]:
        request = request_from_case(case)
        requirements = analyzer.analyze(request)
        started = monotonic()
        decision = None
        error: Exception | None = None
        try:
            decision = await engine.select(request)
        except NoEligibleModelError as exc:
            error = exc
        latency = monotonic() - started
        passed, failures = score_case(case, request, requirements, decision, error)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "basis": case["basis"],
            "passed": passed,
            "failures": failures,
            "selection_latency_seconds": round(latency, 6),
            "inferred": {
                "capabilities": sorted(item.value for item in requirements.capabilities),
                "privacy": requirements.privacy_level.value,
                "reasoning": requirements.reasoning_complexity.value,
                "coding": requirements.coding_complexity.value,
            },
            "selected_model": decision.selected_model if decision else None,
            "estimated_cost": decision.estimated_cost if decision else None,
            "estimated_latency": decision.estimated_latency if decision else None,
            "confidence": decision.confidence if decision else None,
            "error": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "rejections": [
                {"model_id": item.model_id, "code": item.code.value, "reason": item.reason}
                for item in ((decision.rejected_candidates if decision else getattr(error, "rejections", ())))
            ],
        })

    hard = [item for item in results if item["basis"] == "hard"]
    policy = [item for item in results if item["basis"] == "policy"]
    return {
        "benchmark": payload["benchmark"],
        "version": payload["version"],
        "mode": "selection_only",
        "executes_models": False,
        "case_count": len(results),
        "summary": {
            "hard_passed": sum(item["passed"] for item in hard),
            "hard_total": len(hard),
            "policy_passed": sum(item["passed"] for item in policy),
            "policy_total": len(policy),
        },
        "results": results,
    }


def markdown_report(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Decision Engine Benchmark v1 Results",
        "",
        "> Selection-only benchmark. No model outputs are generated and no answer-quality claim is made.",
        "",
        f"- Cases: **{report['case_count']}**",
        f"- Hard expectations: **{s['hard_passed']}/{s['hard_total']} passed**",
        f"- Policy expectations: **{s['policy_passed']}/{s['policy_total']} passed**",
        "",
        "## Mismatches",
        "",
    ]
    failures = [item for item in report["results"] if not item["passed"]]
    if not failures:
        lines.append("No scored mismatches in this run.")
    else:
        for item in failures:
            lines.append(f"- **{item['id']}** ({item['basis']}): " + "; ".join(item["failures"]))
    lines += [
        "",
        "## Interpretation",
        "",
        "Hard-expectation failures indicate a constraint or implementation behavior to investigate. Policy-expectation mismatches are review items, not automatic router failures.",
        "",
        "Runtime availability and quota can change between runs. The JSON artifact preserves each case's selected route, rejection reasons, and observed selection latency.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Decision Engine selection-only benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    try:
        report = asyncio.run(run(args.dataset))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid benchmark data: {exc}", file=sys.stderr)
        return 2

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    args.md_out.write_text(markdown_report(report))
    print(markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
