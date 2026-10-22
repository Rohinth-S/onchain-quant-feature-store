import pandas as pd
import pytest

from feature_store.core import build_features, synthetic_events
from feature_store.quality import inspect_events


def test_equal_times_use_the_explicit_chain_position():
    data = synthetic_events(12, seed=5).assign(block_number=10, log_index=list(range(12)), gas_gwei=0)
    data["timestamp"] = pd.Timestamp("2026-04-01", tz="UTC")
    expected = build_features(data, include_unlabeled=True)
    observed = build_features(data.iloc[::-1], include_unlabeled=True)
    pd.testing.assert_frame_equal(expected, observed)
    assert (observed["gas_cost_usd"] == 0).all()


def test_event_identity_is_canonical_and_numeric_times_are_invalid():
    data = synthetic_events(4).assign(pool=["A", " A ", "A", "A"], event_id=["x", " x ", "y", "z"])
    assert inspect_events(data).duplicate_rows == 1
    data["timestamp"] = [1, 2, 3, 4]
    assert inspect_events(data).invalid_rows == 4
    assert not inspect_events(data.assign(block_number=1)).valid
