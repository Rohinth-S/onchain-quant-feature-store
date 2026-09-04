from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .core import validate_events, build_features, FeatureConfig, walk_forward_rmse

def load_events_csv(path: str | Path) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["timestamp"])
    validate_events(events)
    return events

def write_report(features: pd.DataFrame, path: str | Path, cfg: FeatureConfig) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {"rows": len(features), "columns": list(features.columns), "walk_forward": walk_forward_rmse(features, train_size=min(80, max(8, len(features) // 2))), "feature_summary": features[["realized_vol", "order_flow_imbalance", "slippage_bps", "gas_cost_usd"]].describe().round(6).to_dict()}
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return destination
