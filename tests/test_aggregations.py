import pandas as pd
import pytest

from feature_store.aggregations import WindowSpec, assert_no_future_data, pool_window_features


def _events():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
        "pool": ["a", "a", "a"],
        "mid_price": [100.0, 101.0, 102.0],
        "volume": [2.0, 3.0, 5.0],
        "side": ["buy", "sell", "buy"],
    })


def test_pool_window_features_are_causal_and_aggregate():
    rows = pool_window_features(_events(), WindowSpec(minutes=2))
    assert rows.loc[0, "window_volume"] == 2.0
    assert rows.loc[2, "window_volume"] == 10.0
    assert rows.loc[2, "window_trades"] == 3.0
    assert rows.loc[2, "window_imbalance"] == pytest.approx(0.4)


def test_future_or_unsorted_rows_are_rejected():
    with pytest.raises(ValueError, match="sorted"):
        assert_no_future_data(_events().iloc[::-1])


def test_interleaved_pools_keep_their_own_windows():
    events = pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00Z", "pool": "a", "mid_price": 10, "volume": 1, "side": "buy"},
        {"timestamp": "2026-01-01T00:01:00Z", "pool": "b", "mid_price": 20, "volume": 4, "side": "sell"},
        {"timestamp": "2026-01-01T00:02:00Z", "pool": "a", "mid_price": 11, "volume": 2, "side": "sell"},
        {"timestamp": "2026-01-01T00:03:00Z", "pool": "b", "mid_price": 21, "volume": 5, "side": "buy"},
    ])
    result = pool_window_features(events, WindowSpec(minutes=15))
    a = result[result.pool == "a"].sort_values("timestamp")
    b = result[result.pool == "b"].sort_values("timestamp")
    assert a.window_volume.tolist() == [1, 3]
    assert b.window_volume.tolist() == [4, 9]
