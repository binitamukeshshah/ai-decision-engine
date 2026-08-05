from __future__ import annotations

import argparse
import asyncio

from ai_decision_engine.exceptions import DecisionEngineError
from ai_decision_engine.schemas.models import ModelDefinition, PrivacyLevel
from ai_decision_engine.schemas.requests import DecisionResponse, RoutingDecision, TaskRequest
from ai_decision_engine.service import build_engine


def _decision(decision: RoutingDecision, show_rejections: bool) -> None:
    print(f"Selected: {decision.selected_model} ({decision.selected_provider})")
    print(
        f"Execution provider: {decision.execution_provider} | Runtime: {decision.execution_runtime} | "
        f"Billing: {decision.billing_mode}"
    )
    print(f"Cloud permission: {'explicitly allowed' if decision.explicit_cloud_permission else 'not granted'}")
    print(f"Reason: {decision.routing_reason}")
    print(f"Confidence: {decision.confidence:.0%}")
    print(
        f"Estimated latency: {decision.estimated_latency:.2f}s | Quota: {decision.quota_status} | "
        f"Estimated API cost: ${decision.estimated_cost:.6f}"
    )
    if decision.fallback_order:
        print(f"Fallback order: {', '.join(decision.fallback_order)}")
    if show_rejections:
        for rejection in decision.rejected_candidates:
            print(f"Rejected {rejection.model_id}: {rejection.code.value} — {rejection.reason}")


def _response(response: DecisionResponse, show_rejections: bool) -> None:
    print(f"Selected: {response.selected_model} ({response.provider})")
    print(f"Runtime: {response.execution_runtime} | Billing: {response.billing_mode}")
    print(f"Cloud permission: {'explicitly allowed' if response.explicit_cloud_permission else 'not granted'}")
    print(f"Reason: {response.routing_reason}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Latency: {response.latency_seconds:.2f}s | API cost: ${response.estimated_cost:.6f}")
    print(f"API budget remaining: ${response.budget_status.remaining:.2f} | Quota: {response.quota_status}")
    if show_rejections:
        for rejection in response.candidate_rejections:
            print(f"Rejected {rejection.model_id}: {rejection.code.value} — {rejection.reason}")
    print(f"\n{response.output}\n")


async def _main_async(args: argparse.Namespace) -> int:
    engine = build_engine()
    if args.command in {"providers", "models"}:
        statuses = await engine.model_statuses()
        if args.command == "providers":
            providers = sorted({model.provider.value for model, _ in statuses if isinstance(model, ModelDefinition)})
            print("\n".join(providers) if providers else "No configured providers.")
        else:
            for model, status in statuses:
                assert isinstance(model, ModelDefinition)
                print(
                    f"{model.model_id} | {model.execution_runtime.value} | {model.billing_mode.value} | "
                    f"{status.status.value} | quota={status.quota_status.value}"
                )
        return 0
    request = TaskRequest(" ".join(args.prompt), privacy=PrivacyLevel.CLOUD_ALLOWED if args.allow_cloud else None)
    if args.select_only:
        _decision(await engine.select(request), args.show_rejections)
    else:
        _response(await engine.run(request), args.show_rejections)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Select or execute the cheapest adequate configured AI model.")
    parser.add_argument("command", nargs="?", help="Prompt, or 'providers'/'models'")
    parser.add_argument("prompt", nargs="*", help="Remaining prompt words")
    parser.add_argument("--allow-cloud", action="store_true", help="Explicitly permit cloud processing")
    parser.add_argument("--select-only", action="store_true", help="Select without model execution")
    parser.add_argument("--show-rejections", action="store_true", help="Show excluded candidates")
    args = parser.parse_args()
    if args.command not in {"providers", "models"}:
        args.prompt = ([args.command] if args.command else []) + args.prompt
        args.command = "task"
    if args.command == "task" and not args.prompt:
        parser.error("a task prompt is required")
    try:
        return asyncio.run(_main_async(args))
    except DecisionEngineError as exc:
        print(f"Routing failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
