"""Streaming-safe aggregation helpers for feature store consumers.

The functions in this module only use observations available at or before each
row.  That makes them safe to reuse in replay jobs and online feature workers.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    """Time window configuration for a pool-level feature stream."""

    minutes: int = 15

    def __post_init__(self) -> None:
        if self.minutes < 1:
            raise ValueError("minutes must be positive")


def assert_no_future_data(events: pd.DataFrame, *, timestamp: str = "timestamp") -> None:
    """Reject unsorted or future-dated rows before an online aggregation."""

    if timestamp not in events:
        raise ValueError(f"missing timestamp column: {timestamp}")
    ts = pd.to_datetime(events[timestamp], utc=True)
    if ts.isna().any():
        raise ValueError("timestamps must be valid")
    if not ts.is_monotonic_increasing:
        raise ValueError("events must be sorted by timestamp")


def pool_window_features(events: pd.DataFrame, spec: WindowSpec = WindowSpec()) -> pd.DataFrame:
    """Compute causal per-pool count, volume, VWAP, and imbalance windows."""

    required = {"timestamp", "pool", "mid_price", "volume", "side"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["timestamp", "pool"]).reset_index(drop=True)
    assert_no_future_data(frame)
    frame["signed_volume"] = frame["volume"].where(frame["side"].eq("buy"), -frame["volume"])
    frame["notional"] = frame["mid_price"] * frame["volume"]
    window = f"{spec.minutes}min"

    def causal_roll(column: str, aggregation: str = "sum") -> pd.Series:
        # Grouped rolling places each pool contiguously in its result, while
        # ``frame`` is timestamp ordered across pools.  Write by original row
        # index so values cannot silently be assigned to the wrong pool.
        result = pd.Series(index=frame.index, dtype="float64")
        for _, indices in frame.groupby("pool", sort=False).groups.items():
            group = frame.loc[indices].sort_values("timestamp")
            rolling = group.set_index("timestamp")[column].rolling(window, closed="both")
            values = getattr(rolling, aggregation)().to_numpy()
            result.loc[group.index] = values
        return result

    frame["window_volume"] = causal_roll("volume")
    frame["window_signed_volume"] = causal_roll("signed_volume")
    frame["window_notional"] = causal_roll("notional")
    frame["window_trades"] = causal_roll("volume", "count")
    frame["window_vwap"] = frame["window_notional"] / frame["window_volume"]
    frame["window_imbalance"] = frame["window_signed_volume"] / frame["window_volume"]
    return frame.drop(columns=["signed_volume", "notional"])
