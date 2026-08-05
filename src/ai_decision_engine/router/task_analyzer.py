from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ai_decision_engine.schemas.models import Capability, Complexity, LatencyPreference, PrivacyLevel
from ai_decision_engine.schemas.requests import TaskRequest


@dataclass(frozen=True)
class TaskRequirements:
    capabilities: frozenset[Capability]
    privacy_level: PrivacyLevel
    minimum_quality_score: float
    reasoning_complexity: Complexity
    coding_complexity: Complexity
    multimodal_required: bool
    latency_preference: LatencyPreference
    analysis_confidence: float
    prefer_local: bool = True
    requires_current_information: bool = False
    privacy_override_applied: bool = False


class Analyzer(Protocol):
    def analyze(self, request: TaskRequest) -> TaskRequirements: ...


class CapabilityTaskAnalyzer:
    """Classifies capabilities from caller signals and deterministic prompt structure."""

    SENSITIVE_CONTENT = re.compile(
        r"\b(confidential|credential|password|secret|medical|patient|legal|proprietary|customer|personal)\b",
        re.IGNORECASE,
    )
    REASONING_SIGNALS: tuple[tuple[re.Pattern[str], int], ...] = (
        (re.compile(r"\b(compare|contrast|weigh)\b|\bversus\b|\bvs\.?\b", re.IGNORECASE), 2),
        (
            re.compile(
                r"\b(multiple|several|three|four|five)\s+(options|alternatives|strategies|approaches)\b", re.IGNORECASE
            ),
            2,
        ),
        (re.compile(r"\btrade[ -]?offs?\b|\badvantages?\s+and\s+disadvantages?\b", re.IGNORECASE), 2),
        (re.compile(r"\b(risks?|failure modes?)\b", re.IGNORECASE), 1),
        (re.compile(r"\brecommend(?:ation|ed|s)?\b", re.IGNORECASE), 2),
        (re.compile(r"\bstrateg(?:y|ies|ic)\b", re.IGNORECASE), 1),
        (re.compile(r"\b(staged|phased|sequenced?)\b|\brollout plan\b", re.IGNORECASE), 2),
        (
            re.compile(
                r"\b(?:review|evaluate|assess)\b.{0,40}\barchitecture\b|\barchitecture\b.{0,40}\breview\b",
                re.IGNORECASE,
            ),
            7,
        ),
        (re.compile(r"\b(synthesize|integrate)\b|\bacross (?:multiple|several|these)\b", re.IGNORECASE), 2),
    )
    ANALYSIS_ACTIONS: tuple[re.Pattern[str], ...] = tuple(
        re.compile(rf"\b{action}\b", re.IGNORECASE)
        for action in (
            "analyze",
            "assess",
            "compare",
            "contrast",
            "design",
            "evaluate",
            "recommend",
            "review",
            "synthesize",
        )
    )
    ANALYSIS_DIMENSIONS: tuple[re.Pattern[str], ...] = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\brisks?\b",
            r"\b(cost|pricing|revenue|monetization|economics?)\b",
            r"\b(execution|implementation|migration|delivery)\b",
            r"\barchitecture\b",
            r"\b(privacy|security|compliance)\b",
            r"\b(failure|reliability|resilience|rollback)\b",
            r"\b(performance|latency|scalability)\b",
            r"\b(operations?|operability|maintenance)\b",
            r"\b(options?|alternatives?|strategies|approaches)\b",
        )
    )

    def analyze(self, request: TaskRequest) -> TaskRequirements:
        reasoning = request.reasoning_complexity or self._structural_reasoning(request.prompt)
        coding = request.coding_complexity or self._structural_coding(request.prompt, request.required_capabilities)
        multimodal = request.multimodal_required or request.has_image
        capabilities = {Capability.CHAT, *request.required_capabilities}
        if reasoning in {Complexity.MODERATE, Complexity.HIGH, Complexity.EXTREME}:
            capabilities.add(Capability.REASONING)
        if coding != Complexity.NONE:
            capabilities.add(Capability.CODING)
        if multimodal:
            capabilities.add(Capability.VISION)
        privacy = request.privacy or PrivacyLevel.LOCAL_ONLY
        if self.SENSITIVE_CONTENT.search(request.prompt):
            privacy = PrivacyLevel.LOCAL_ONLY
        threshold = (
            request.minimum_quality if request.minimum_quality is not None else self._threshold(reasoning, coding)
        )
        explicit = request.reasoning_complexity is not None or request.coding_complexity is not None
        return TaskRequirements(
            frozenset(capabilities),
            privacy,
            threshold,
            reasoning,
            coding,
            multimodal,
            request.latency_preference,
            0.95 if explicit else 0.75,
            True,
            Capability.CURRENT_INFORMATION in capabilities,
            request.privacy in {PrivacyLevel.CLOUD_ALLOWED, PrivacyLevel.PUBLIC},
        )

    @classmethod
    def _structural_reasoning(cls, prompt: str) -> Complexity:
        words = len(prompt.split())
        structural_markers = prompt.count("\n") + prompt.count("?") + prompt.count(";")
        reasoning_score = sum(weight for pattern, weight in cls.REASONING_SIGNALS if pattern.search(prompt))
        action_count = sum(pattern.search(prompt) is not None for pattern in cls.ANALYSIS_ACTIONS)
        dimension_count = sum(pattern.search(prompt) is not None for pattern in cls.ANALYSIS_DIMENSIONS)

        # Multi-criteria and multi-step requests compound rather than merely adding isolated words.
        if dimension_count >= 3:
            reasoning_score += 2
        if action_count >= 3:
            reasoning_score += 2

        # Extreme requires both a high score and broad, deeply multi-part analysis.
        if reasoning_score >= 15 and action_count >= 4 and dimension_count >= 5:
            return Complexity.EXTREME
        if reasoning_score >= 7:
            return Complexity.HIGH
        if words >= 250 or structural_markers >= 24:
            return Complexity.EXTREME
        if words >= 120 or structural_markers >= 8:
            return Complexity.HIGH
        if words >= 40 or structural_markers >= 4:
            return Complexity.MODERATE
        return Complexity.LOW

    @staticmethod
    def _structural_coding(prompt: str, declared: frozenset[Capability]) -> Complexity:
        if Capability.CODING in declared:
            return Complexity.MODERATE
        code_blocks = prompt.count("```") // 2
        code_lines = sum(1 for line in prompt.splitlines() if line.startswith(("    ", "\t")))
        if code_blocks >= 2 or code_lines >= 20:
            return Complexity.HIGH
        if code_blocks or code_lines >= 3:
            return Complexity.MODERATE
        return Complexity.NONE

    @staticmethod
    def _threshold(reasoning: Complexity, coding: Complexity) -> float:
        levels = {
            Complexity.NONE: 0.5,
            Complexity.LOW: 0.5,
            Complexity.MODERATE: 0.72,
            Complexity.HIGH: 0.85,
            Complexity.EXTREME: 0.96,
        }
        return max(levels[reasoning], levels[coding])


HeuristicTaskAnalyzer = CapabilityTaskAnalyzer
TaskAnalyzer = CapabilityTaskAnalyzer
