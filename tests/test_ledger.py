import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_decision_engine.exceptions import LedgerUnavailableError, MonthlyBudgetExceededError
from ai_decision_engine.ledger import UsageLedger


def test_atomic_concurrent_reservations_and_exact_boundary(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledgers = [UsageLedger(path, 10.0) for _ in range(4)]

    def reserve(index: int) -> bool:
        try:
            ledgers[index].reserve(model_id=str(index), provider="cloud", maximum_cost=3.0)
            return True
        except MonthlyBudgetExceededError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert sum(results) == 3
    status = ledgers[0].status()
    assert status.reserved_spend == 9.0 and status.remaining == 1.0
    exact = ledgers[0].reserve(model_id="exact", provider="cloud", maximum_cost=1.0)
    assert ledgers[0].status().remaining == 0
    with pytest.raises(MonthlyBudgetExceededError):
        ledgers[0].reserve(model_id="over", provider="cloud", maximum_cost=0.01)
    ledgers[0].finalize(exact, 1.0, outcome="success")


def test_finalize_release_and_month_rollover(tmp_path: Path) -> None:
    current = [datetime(2026, 8, 31, tzinfo=UTC)]
    ledger = UsageLedger(tmp_path / "ledger.sqlite3", 10.0, clock=lambda: current[0])
    first = ledger.reserve(model_id="a", provider="cloud", maximum_cost=4.0)
    ledger.finalize(first, 2.0, outcome="success")
    released = ledger.reserve(model_id="b", provider="cloud", maximum_cost=3.0)
    ledger.release(released, outcome="no_charge")
    assert ledger.status().finalized_spend == 2.0
    current[0] = datetime(2026, 9, 1, tzinfo=UTC)
    assert ledger.status().remaining == 10.0


def test_corrupt_or_unavailable_ledger_fails_closed(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not sqlite")
    with pytest.raises(LedgerUnavailableError):
        UsageLedger(corrupt, 10.0)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(LedgerUnavailableError):
        UsageLedger(directory, 10.0)


def test_database_damage_after_startup_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = UsageLedger(path, 10.0)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE cloud_usage")
    with pytest.raises(LedgerUnavailableError):
        ledger.reserve(model_id="a", provider="cloud", maximum_cost=1.0)
