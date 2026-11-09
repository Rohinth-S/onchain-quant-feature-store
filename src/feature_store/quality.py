"""Data quality contracts for event ingestion and feature materialization."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .core import REQUIRED_COLUMNS


@dataclass(frozen=True)
class QualityReport:
    rows: int
    invalid_rows: int
    duplicate_rows: int
    missing_columns: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.missing_columns and self.invalid_rows == 0 and self.duplicate_rows == 0 and not self.issues

    def raise_if_invalid(self) -> None:
        if not self.valid:
            problems = list(self.missing_columns) + list(self.issues)
            if self.duplicate_rows:
                problems.append(f"{self.duplicate_rows} duplicate rows")
            raise ValueError("invalid event batch: " + "; ".join(problems))


def inspect_events(events: pd.DataFrame) -> QualityReport:
    """Return a non-throwing report suitable for ingestion dashboards."""
    missing = tuple(sorted(REQUIRED_COLUMNS.difference(events.columns)))
    if missing:
        return QualityReport(len(events), len(events), 0, missing, ("required columns missing",))
    issues: list[str] = []
    timestamp = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    numeric = events[["mid_price", "volume", "gas_gwei", "liquidity_usd"]].apply(pd.to_numeric, errors="coerce")
    invalid_mask = timestamp.isna() | numeric.isna().any(axis=1)
    if (numeric <= 0).any().any():
        issues.append("numeric fields must be positive")
        invalid_mask |= (numeric <= 0).any(axis=1)
    if not events["side"].isin({"buy", "sell"}).all():
        issues.append("side must contain only buy or sell")
        invalid_mask |= ~events["side"].isin({"buy", "sell"})
    if not timestamp.is_monotonic_increasing:
        issues.append("timestamps must be sorted")
    duplicate_rows = int(events.duplicated(subset=["timestamp", "pool", "mid_price", "volume"]).sum())
    return QualityReport(len(events), int(invalid_mask.sum()), duplicate_rows, (), tuple(dict.fromkeys(issues)))
