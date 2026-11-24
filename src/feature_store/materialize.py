"""Deterministic feature materialization and UTC-date partitioning."""

import pandas as pd

from .core import FeatureConfig, build_features
from .quality import inspect_events


def materialize_features(
    events: pd.DataFrame, cfg: FeatureConfig = FeatureConfig(), *, include_unlabeled: bool = False,
) -> pd.DataFrame:
    inspect_events(events).raise_if_invalid()
    features = build_features(events, cfg, include_unlabeled=include_unlabeled)
    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    features.insert(0, "feature_date", timestamps.dt.strftime("%Y-%m-%d"))
    features["feature_version"] = f"h{cfg.horizon}-v{cfg.vol_window}"
    return features


def materialize_partitions(events: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> dict[str, pd.DataFrame]:
    features = materialize_features(events, cfg)
    return {date: rows.reset_index(drop=True) for date, rows in features.groupby("feature_date", sort=True)}
