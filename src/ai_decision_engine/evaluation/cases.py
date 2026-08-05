from ai_decision_engine.evaluation.models import EvaluationCase
from ai_decision_engine.schemas.models import Capability

SYNTHETIC_CASES = (
    EvaluationCase("simple-chat", "Explain why leaves are green.", frozenset({Capability.CHAT}), "local"),
    EvaluationCase(
        "simple-code",
        "Write a Python function to add numbers.",
        frozenset({Capability.CHAT, Capability.CODING}),
        "local",
    ),
    EvaluationCase(
        "complex-code",
        "Analyze and refactor this complex Python API.",
        frozenset({Capability.CHAT, Capability.CODING, Capability.REASONING}),
        "local",
    ),
    EvaluationCase("private", "Summarize this confidential medical note.", frozenset({Capability.CHAT}), "local_only"),
    EvaluationCase(
        "reasoning",
        "Compare these strategies and tradeoffs.",
        frozenset({Capability.CHAT, Capability.REASONING}),
        "local",
    ),
    EvaluationCase(
        "current",
        "What is today's latest news?",
        frozenset({Capability.CHAT, Capability.CURRENT_INFORMATION}),
        "cloud_or_decline",
    ),
    EvaluationCase(
        "image", "Describe the attached image.", frozenset({Capability.CHAT, Capability.VISION}), "cloud_or_decline"
    ),
    EvaluationCase("unavailable", "Hello", frozenset({Capability.CHAT}), "fallback_or_decline"),
    EvaluationCase("budget", "Use a cloud model.", frozenset({Capability.CHAT}), "decline"),
)
