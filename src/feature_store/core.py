"""Causal market features and expanding-window baselines."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "pool", "mid_price", "volume", "side", "gas_gwei", "liquidity_usd"}
NUMERIC_COLUMNS = {"mid_price", "volume", "gas_gwei", "liquidity_usd"}
OPTIONAL_NUMERIC_COLUMNS = {"gas_units", "native_price_usd"}
POSITION_COLUMNS = ("block_number", "log_index")
FEATURE_COLUMNS = (
    "realized_vol", "order_flow_imbalance", "liquidity_concentration",
    "price_impact_bps", "slippage_bps", "gas_cost_usd",
)


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 3
    vol_window: int = 8

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.vol_window < 3:
            raise ValueError("horizon must be >= 1 and vol_window must be >= 3")


def chain_positions(events: pd.DataFrame) -> pd.DataFrame:
    present = [key for key in POSITION_COLUMNS if key in events]
    if not present:
        return pd.DataFrame(index=events.index)
    if len(present) != len(POSITION_COLUMNS):
        raise ValueError("block_number and log_index must be supplied together")
    positions = events[present].astype("string")
    if any(not positions[key].str.fullmatch(r"[0-9]+").fillna(False).all() for key in present):
        raise ValueError("block_number and log_index must be nonnegative integers")
    try:
        return positions.astype("uint64")
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("block_number and log_index must fit in uint64") from exc


def invalid_numerics(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    invalid = ~np.isfinite(values).all(axis=1)
    for index, name in enumerate(frame.columns):
        floor = values[:, index] < 0 if name in {"gas_gwei", "gas_units"} else values[:, index] <= 0
        invalid |= floor
    return invalid


def event_timestamps(data: pd.Series) -> pd.Series:
    converted = pd.to_datetime(data, utc=True, format="mixed", errors="coerce")
    return converted.mask(data.map(lambda value: isinstance(value, Real)))


def validate_events(events: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("events must contain at least one row")
    if event_timestamps(events["timestamp"]).isna().any():
        raise ValueError("timestamps must be valid UTC-compatible datetime values")
    if events["pool"].isna().any() or events["pool"].astype(str).str.strip().eq("").any():
        raise ValueError("pool identifiers cannot be empty")
    columns = sorted(NUMERIC_COLUMNS | OPTIONAL_NUMERIC_COLUMNS.intersection(events.columns))
    numeric = events[columns].apply(pd.to_numeric, errors="coerce")
    if invalid_numerics(numeric).any():
        raise ValueError("prices, volume, and liquidity must be finite and positive, gas nonnegative")
    if not events["side"].isin({"buy", "sell"}).all():
        raise ValueError("side must contain only buy or sell")


def synthetic_events(n: int = 240, seed: int = 7) -> pd.DataFrame:
    if n < 4:
        raise ValueError("n must be at least 4")
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    volume = rng.lognormal(7.0, 0.35, n)
    buy = rng.binomial(1, 0.52, n)
    gas = rng.lognormal(3.5, 0.25, n)
    events = pd.DataFrame({
        "timestamp": t, "pool": "WETH-USDC", "mid_price": mid, "volume": volume,
        "side": np.where(buy, "buy", "sell"), "gas_gwei": gas,
        "liquidity_usd": rng.lognormal(14, 0.18, n), "tick": np.floor(np.log(mid) * 100).astype(int),
    })
    validate_events(events)
    return events


def _pool_features(pool: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    pool = pool.copy()
    log_return = np.log(pool["mid_price"]).diff()
    pool["log_return"] = log_return
    pool["realized_vol"] = log_return.pow(2).rolling(cfg.vol_window, min_periods=3).mean().pow(0.5)
    pool["signed_volume"] = pool["volume"].where(pool["side"].eq("buy"), -pool["volume"])
    volume_window = pool["volume"].rolling(cfg.vol_window, min_periods=3).sum()
    pool["order_flow_imbalance"] = pool["signed_volume"].rolling(cfg.vol_window, min_periods=3).sum() / volume_window
    pool["liquidity_concentration"] = pool["volume"] / volume_window
    pool["price_impact_bps"] = pool["mid_price"].pct_change().abs() * 10_000
    pool["slippage_bps"] = pool["price_impact_bps"] + 10_000 * pool["volume"] / pool["liquidity_usd"]
    pool["gas_cost_usd"] = (
        pool["gas_gwei"] * pool.get("gas_units", 210_000) * 1e-9
        * pool.get("native_price_usd", pool["mid_price"])
    )
    pool["label_end"] = pool["timestamp"].shift(-cfg.horizon)
    pool["target_cost_bps"] = pool["slippage_bps"].shift(-cfg.horizon)
    pool["target_vol"] = log_return.pow(2).rolling(cfg.horizon).mean().shift(-cfg.horizon).pow(0.5)
    return pool


def build_features(
    events: pd.DataFrame, cfg: FeatureConfig = FeatureConfig(), *, include_unlabeled: bool = False,
) -> pd.DataFrame:
    """Compute trailing features and forward labels within each pool's event order."""
    validate_events(events)
    frame = events.copy()
    frame["pool"] = frame["pool"].astype("string").str.strip()
    frame["timestamp"] = event_timestamps(frame["timestamp"])
    columns = sorted(NUMERIC_COLUMNS | OPTIONAL_NUMERIC_COLUMNS.intersection(events.columns))
    frame[columns] = frame[columns].astype(float)
    positions = chain_positions(frame)
    for key in positions:
        frame[key] = positions[key]
    frame = frame.sort_values(["timestamp", *positions], kind="stable").reset_index(drop=True)
    frame["event_sequence"] = frame.index
    pools = [_pool_features(pool, cfg) for _, pool in frame.groupby("pool", sort=False)]
    result = pd.concat(pools).sort_values("event_sequence").dropna(subset=list(FEATURE_COLUMNS))
    if not include_unlabeled:
        result = result.dropna(subset=["target_cost_bps", "target_vol"])
    return result.reset_index(drop=True)


def walk_forward_predictions(
    features: pd.DataFrame, train_size: int = 80, *, fold_size: int = 32,
) -> pd.DataFrame:
    """Refit on labels settled strictly before the first decision of each fold."""
    if train_size < 8 or fold_size < 1:
        raise ValueError("train_size must be >= 8 and fold_size must be positive")
    frame = features.dropna(subset=["label_end", "target_cost_bps", "target_vol"])
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    inputs = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    targets = frame[["target_cost_bps", "target_vol"]].to_numpy(dtype=float)
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("walk-forward inputs must be finite")
    outputs = []
    start = train_size
    while start < len(frame):
        cutoff = frame.loc[start, "timestamp"]
        train = (frame["label_end"] < cutoff).to_numpy()
        if train.sum() < train_size:
            start += 1
            continue
        stop = start + fold_size
        while stop < len(frame) and frame.loc[stop, "timestamp"] == frame.loc[stop - 1, "timestamp"]:
            stop += 1
        x_train = inputs[train]
        mean = x_train.mean(axis=0)
        deviation = x_train.std(axis=0)
        scale = np.where(deviation > 0, deviation, 1.0)
        matrix = np.column_stack((np.ones(len(x_train)), (x_train - mean) / scale))
        coefficients = np.linalg.lstsq(matrix, targets[train], rcond=None)[0]
        batch = frame.iloc[start:stop][["event_sequence", "pool", "timestamp", "label_end", "target_cost_bps", "target_vol"]].copy()
        test_matrix = np.column_stack((np.ones(len(batch)), (inputs[start:stop] - mean) / scale))
        batch[["pred_cost_bps", "pred_vol"]] = np.maximum(test_matrix @ coefficients, 0)
        batch["baseline_cost_bps"] = frame["slippage_bps"].iloc[start:stop].to_numpy()
        batch["baseline_vol"] = frame["realized_vol"].iloc[start:stop].to_numpy()
        batch["training_rows"] = int(train.sum())
        batch["training_last_label_end"] = frame.loc[train, "label_end"].max()
        outputs.append(batch)
        start = stop
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def score_predictions(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {"observations": 0, "status": "insufficient_history"}
    result = {"observations": len(predictions), "status": "evaluated"}
    for target, suffix in (("execution_cost", "cost_bps"), ("volatility", "vol")):
        actual = predictions[f"target_{suffix}"].to_numpy()
        errors = predictions[[f"pred_{suffix}", f"baseline_{suffix}"]].to_numpy() - actual[:, None]
        result[target] = {
            "rmse": float(np.sqrt((errors[:, 0] ** 2).mean())),
            "mae": float(np.abs(errors[:, 0]).mean()),
            "baseline_rmse": float(np.sqrt((errors[:, 1] ** 2).mean())),
        }
    return result


def walk_forward_rmse(features: pd.DataFrame, train_size: int = 80) -> dict:
    """Return the compatibility summary for the execution-cost model."""
    scores = score_predictions(walk_forward_predictions(features, train_size))
    if not scores["observations"]:
        raise ValueError("insufficient settled history for the requested training size")
    return {
        "rmse_bps": scores["execution_cost"]["rmse"],
        "observations": scores["observations"],
        "train_size": train_size,
    }
