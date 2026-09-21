"""Input adapters and strict JSON report serialization."""

import json
from pathlib import Path

from .artifacts import write_text_atomic

import pandas as pd
import polars as pl

from .core import FeatureConfig, score_predictions, validate_events, walk_forward_predictions


def load_events(path: str | Path, *, check: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pl.read_csv(path, schema_overrides={"pool": pl.String, "event_id": pl.String})
    elif path.suffix.lower() == ".parquet":
        frame = pl.read_parquet(path)
    else:
        raise ValueError("event input must be CSV or Parquet")
    events = frame.to_pandas()
    if check:
        validate_events(events)
    return events


def load_events_csv(path: str | Path) -> pd.DataFrame:
    return load_events(path)


def write_json(value: dict, destination: str | Path) -> Path:
    path = Path(destination)
    write_text_atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def make_report(features: pd.DataFrame, train_size: int | None = None) -> tuple[dict, pd.DataFrame]:
    size = train_size if train_size is not None else min(80, max(8, len(features) // 2))
    predictions = walk_forward_predictions(features, size)
    score = score_predictions(predictions)
    if score["observations"]:
        score["rmse_bps"] = score["execution_cost"]["rmse"]
    score["train_size"] = size
    measures = features[["realized_vol", "order_flow_imbalance", "slippage_bps", "gas_cost_usd"]]
    summary = json.loads(measures.describe().round(6).to_json())
    return {
        "rows": len(features), "columns": list(features.columns), "walk_forward": score,
        "feature_summary": summary,
    }, predictions


def write_report(features: pd.DataFrame, path: str | Path, cfg: FeatureConfig) -> Path:
    report, _ = make_report(features)
    report["config"] = {"horizon": cfg.horizon, "vol_window": cfg.vol_window}
    return write_json(report, path)
