from ai_decision_engine.cli import main
from ai_decision_engine.schemas.models import Capability, PrivacyLevel
from ai_decision_engine.schemas.requests import RoutingDecision


def test_cli_help_starts(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["ai-decision-engine", "--help"])
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_select_only_smoke_performs_selection(monkeypatch, capsys) -> None:
    class Engine:
        async def select(self, request):
            return RoutingDecision(
                "ollama",
                "ollama/qwen3:8b",
                "direct_ollama",
                "local",
                0.8,
                "local adequate",
                frozenset({Capability.CHAT}),
                PrivacyLevel.LOCAL_ONLY,
                False,
                0.0,
                (),
                (),
                "available",
                0.9,
                "Selected local model.",
            )

    monkeypatch.setattr("ai_decision_engine.cli.build_engine", lambda: Engine())
    monkeypatch.setattr("sys.argv", ["ai-decision-engine", "hello", "--select-only"])
    assert main() == 0
    assert "ollama/qwen3:8b" in capsys.readouterr().out
