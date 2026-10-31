"""Deterministic feature materialization and date partitioning."""
from __future__ import annotations

import pandas as pd

from .core import FeatureConfig, build_features
from .quality import inspect_events


def materialize_features(events: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Validate an event batch, build causal features, and attach lineage columns."""
    report = inspect_events(events)
    report.raise_if_invalid()
    features = build_features(events, cfg)
    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    features.insert(0, "feature_date", timestamps.dt.strftime("%Y-%m-%d"))
    features["feature_version"] = f"h{cfg.horizon}-v{cfg.vol_window}"
    return features


def materialize_partitions(events: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> dict[str, pd.DataFrame]:
    """Return independently writable UTC-date partitions keyed by YYYY-MM-DD."""
    features = materialize_features(events, cfg)
    return {date: partition.reset_index(drop=True) for date, partition in features.groupby("feature_date", sort=True)}
