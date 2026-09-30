"""Input adapters and strict JSON report serialization."""

import json
from io import BytesIO
from pathlib import Path

from .artifacts import write_text_atomic

import pandas as pd
import polars as pl

from .core import FeatureConfig, score_predictions, validate_events, walk_forward_predictions


def load_events(path: str | Path, *, check: bool = True) -> pd.DataFrame:
    path = Path(path)
    return decode_events(path.read_bytes(), path.suffix, check=check)


def decode_events(raw: bytes, suffix: str, *, check: bool = True) -> pd.DataFrame:
    suffix = suffix.lower()
    capture = BytesIO(raw)
    try:
        if suffix == ".csv":
            frame = pl.read_csv(capture, schema_overrides={"pool": pl.String, "event_id": pl.String})
        elif suffix == ".parquet":
            frame = pl.read_parquet(capture)
        else:
            raise ValueError("event input must be CSV or Parquet")
        events = frame.to_pandas()
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"cannot decode {suffix} events: {exc}") from exc
    if check:
        validate_events(events)
    return events


def load_events_csv(path: str | Path) -> pd.DataFrame:
    return load_events(path)


def write_json(value: dict, destination: str | Path) -> Path:
    path = Path(destination)
    write_text_atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def make_report(
    features: pd.DataFrame, train_size: int | None = None, *, fold_size: int = 32,
) -> tuple[dict, pd.DataFrame]:
    size = train_size if train_size is not None else min(80, max(8, len(features) // 2))
    predictions = walk_forward_predictions(features, size, fold_size=fold_size)
    score = score_predictions(predictions)
    if score["observations"]:
        score["rmse_bps"] = score["execution_cost"]["rmse"]
    score["train_size"] = size
    score["fold_size"] = fold_size
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
