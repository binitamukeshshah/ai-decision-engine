from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ai_decision_engine.exceptions import LedgerUnavailableError, MonthlyBudgetExceededError, ValidationError
from ai_decision_engine.schemas.requests import BudgetStatus

BILLING_UNITS_PER_DOLLAR = 1_000_000_000


def _to_units(cost: float) -> int:
    """Round up to nanodollars so enforcement is conservative and deterministic."""
    return math.ceil(cost * BILLING_UNITS_PER_DOLLAR)


def _to_dollars(units: int) -> float:
    return units / BILLING_UNITS_PER_DOLLAR


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    maximum_cost: float
    month: str


class UsageLedger:
    """Transactional SQLite API-spend ledger safe across processes."""

    def __init__(
        self,
        path: Path,
        monthly_limit: float,
        *,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not math.isfinite(monthly_limit) or monthly_limit < 0:
            raise ValidationError("Monthly cloud budget must be finite and non-negative.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValidationError("Ledger timeout must be finite and positive.")
        self.path = path
        self.monthly_limit = monthly_limit
        self.clock = clock or (lambda: datetime.now(UTC))
        self.timeout_seconds = timeout_seconds
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.path, timeout=self.timeout_seconds, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            return connection
        except (sqlite3.Error, OSError) as exc:
            raise LedgerUnavailableError(f"Cloud budget ledger is unavailable: {exc}") from exc

    def _initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                integrity = connection.execute("PRAGMA quick_check").fetchone()
                if integrity is None or integrity[0] != "ok":
                    raise LedgerUnavailableError("Cloud budget ledger failed its integrity check.")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cloud_usage (
                        reservation_id TEXT PRIMARY KEY,
                        month TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        reserved_units INTEGER NOT NULL CHECK(reserved_units >= 0),
                        finalized_units INTEGER CHECK(finalized_units >= 0),
                        state TEXT NOT NULL CHECK(state IN ('reserved', 'finalized', 'released')),
                        outcome TEXT
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS cloud_usage_month ON cloud_usage(month, state)")
        except LedgerUnavailableError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise LedgerUnavailableError(f"Cloud budget ledger cannot be initialized: {exc}") from exc

    def _month(self) -> str:
        return self.clock().astimezone(UTC).strftime("%Y-%m")

    @staticmethod
    def _totals(connection: sqlite3.Connection, month: str) -> tuple[int, int]:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN state = 'finalized' THEN finalized_units ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN state = 'reserved' THEN reserved_units ELSE 0 END), 0)
            FROM cloud_usage WHERE month = ?
            """,
            (month,),
        ).fetchone()
        if row is None:
            raise LedgerUnavailableError("Cloud budget ledger returned no totals.")
        return int(row[0]), int(row[1])

    def status(self) -> BudgetStatus:
        try:
            with self._connect() as connection:
                finalized, reserved = self._totals(connection, self._month())
        except LedgerUnavailableError:
            raise
        except sqlite3.Error as exc:
            raise LedgerUnavailableError(f"Cloud budget ledger cannot be read: {exc}") from exc
        remaining = max(0, _to_units(self.monthly_limit) - finalized - reserved)
        return BudgetStatus(
            self.monthly_limit,
            _to_dollars(finalized),
            _to_dollars(reserved),
            _to_dollars(remaining),
            remaining <= 0,
        )

    def reserve(self, *, model_id: str, provider: str, maximum_cost: float) -> BudgetReservation:
        if not math.isfinite(maximum_cost) or maximum_cost < 0:
            raise ValidationError("Reservation cost must be finite and non-negative.")
        maximum_units = _to_units(maximum_cost)
        reservation = BudgetReservation(uuid.uuid4().hex, _to_dollars(maximum_units), self._month())
        now = self.clock().astimezone(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                finalized, reserved = self._totals(connection, reservation.month)
                if finalized + reserved + maximum_units > _to_units(self.monthly_limit):
                    connection.rollback()
                    raise MonthlyBudgetExceededError("Monthly cloud budget is exhausted.")
                connection.execute(
                    """INSERT INTO cloud_usage
                    (reservation_id, month, created_at, updated_at, model_id, provider, reserved_units, state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved')""",
                    (reservation.reservation_id, reservation.month, now, now, model_id, provider, maximum_units),
                )
                connection.commit()
            return reservation
        except MonthlyBudgetExceededError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise LedgerUnavailableError(f"Cloud budget reservation failed: {exc}") from exc

    def finalize(self, reservation: BudgetReservation, cost: float, *, outcome: str) -> float:
        if not math.isfinite(cost) or cost < 0:
            raise ValidationError("Final cost must be finite and non-negative.")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT state, reserved_units FROM cloud_usage WHERE reservation_id = ?",
                    (reservation.reservation_id,),
                ).fetchone()
                if row is None or row[0] != "reserved":
                    raise LedgerUnavailableError("Budget reservation is missing or no longer active.")
                finalized, reserved = self._totals(connection, reservation.month)
                final_units = _to_units(cost)
                other_reserved = reserved - int(row[1])
                if finalized + other_reserved + final_units > _to_units(self.monthly_limit):
                    connection.rollback()
                    raise LedgerUnavailableError("Final cost exceeds the trusted monthly reservation envelope.")
                connection.execute(
                    """UPDATE cloud_usage SET finalized_units = ?, state = 'finalized', outcome = ?, updated_at = ?
                    WHERE reservation_id = ?""",
                    (final_units, outcome, self.clock().astimezone(UTC).isoformat(), reservation.reservation_id),
                )
                connection.commit()
            return _to_dollars(final_units)
        except LedgerUnavailableError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise LedgerUnavailableError(f"Cloud budget reconciliation failed: {exc}") from exc

    def release(self, reservation: BudgetReservation, *, outcome: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    """UPDATE cloud_usage SET state = 'released', outcome = ?, updated_at = ?
                    WHERE reservation_id = ? AND state = 'reserved'""",
                    (outcome, self.clock().astimezone(UTC).isoformat(), reservation.reservation_id),
                )
                if cursor.rowcount != 1:
                    raise LedgerUnavailableError("Budget reservation is missing or no longer active.")
                connection.commit()
        except LedgerUnavailableError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise LedgerUnavailableError(f"Cloud budget release failed: {exc}") from exc
