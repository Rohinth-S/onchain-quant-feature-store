"""Data-quality contracts for event ingestion and feature materialization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import NUMERIC_COLUMNS, OPTIONAL_NUMERIC_COLUMNS, REQUIRED_COLUMNS, chain_positions, event_timestamps, invalid_numerics


@dataclass(frozen=True)
class QualityReport:
    rows: int
    invalid_rows: int
    duplicate_rows: int
    missing_columns: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (self.missing_columns or self.invalid_rows or self.duplicate_rows or self.issues)

    def raise_if_invalid(self) -> None:
        if self.valid:
            return
        problems = [*self.missing_columns, *self.issues]
        if self.duplicate_rows:
            problems.append(f"{self.duplicate_rows} duplicate rows")
        raise ValueError("invalid event batch: " + ", ".join(problems))


def inspect_events(events: pd.DataFrame) -> QualityReport:
    missing = tuple(sorted(REQUIRED_COLUMNS.difference(events.columns)))
    if missing:
        return QualityReport(len(events), len(events), 0, missing, ("required columns missing",))
    if events.empty:
        return QualityReport(0, 0, 0, (), ("event batch is empty",))

    issues = []
    frame = events.copy()
    frame["pool"] = frame["pool"].astype("string").str.strip()
    timestamp = event_timestamps(frame["timestamp"])
    frame["timestamp"] = timestamp
    columns = sorted(NUMERIC_COLUMNS | OPTIONAL_NUMERIC_COLUMNS.intersection(frame.columns))
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    checks = (
        (invalid_numerics(frame[columns]), "trade numeric fields must be finite and positive, gas nonnegative"),
        (timestamp.isna(), "timestamps must be valid"),
        (~frame["side"].isin({"buy", "sell"}), "side must contain only buy or sell"),
        (frame["pool"].isna() | frame["pool"].astype(str).str.strip().eq(""), "pool identifiers cannot be empty"),
    )
    invalid = np.zeros(len(frame), dtype=bool)
    for mask, message in checks:
        if mask.any():
            issues.append(message)
            invalid |= np.asarray(mask)
    identity = ["pool", "event_id"] if "event_id" in frame else ["timestamp", "pool", "mid_price", "volume", "side"]
    if "event_id" in frame:
        frame["event_id"] = frame["event_id"].astype("string").str.strip()
        missing_id = frame["event_id"].isna() | frame["event_id"].eq("").fillna(False)
        if missing_id.any():
            issues.append("event_id cannot be empty when supplied")
            invalid |= np.asarray(missing_id)
    duplicates = int(frame.duplicated(subset=identity).sum())
    try:
        positions = chain_positions(frame)
    except ValueError as exc:
        issues.append(str(exc))
    else:
        if len(positions.columns):
            ordered = positions.assign(pool=frame["pool"])
            duplicates = max(duplicates, int(ordered.duplicated().sum()))
    return QualityReport(len(frame), int(invalid.sum()), duplicates, (), tuple(issues))
