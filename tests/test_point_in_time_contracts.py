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

def test_report_exposes_settled_training_boundaries_and_rolling_caps():
    from feature_store.io import make_report

    features = build_features(synthetic_events(100), include_unlabeled=True)
    report, prediction = make_report(features, 8, fold_size=5, max_train_rows=11)
    assert prediction.training_rows.max() == 11
    assert (prediction.training_last_label_end < prediction.timestamp).all()
    assert len(report["walk_forward"]["folds"]) == prediction.fold.nunique()
    assert sum(value["observations"] for value in report["pool_scores"].values()) == len(prediction)
    bad = features.copy()
    bad.loc[0, "label_end"] = bad.loc[0, "timestamp"] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="labels must end"):
        make_report(bad, 8)

def test_short_capture_still_has_typed_warehouse_schema(tmp_path):
    import duckdb
    from feature_store.io import make_report
    from feature_store.materialize import materialize_features
    from feature_store.warehouse import publish_snapshot

    features = materialize_features(synthetic_events(8), include_unlabeled=True)
    report, predictions = make_report(features)
    target = publish_snapshot(features, predictions, report, tmp_path / "snapshot")
    with duckdb.connect(str(target / "features.duckdb")) as conn:
        assert conn.execute("SELECT count(*) FROM predictions").fetchone() == (0,)
        assert conn.execute("SELECT typeof(feature_date) FROM latest_features").fetchone() == ("DATE",)

def test_csv_and_parquet_capture_parity(tmp_path):
    from feature_store.io import load_events

    source = synthetic_events(20)
    csv_path = tmp_path / "events.csv"
    parquet_path = tmp_path / "events.parquet"
    source.to_csv(csv_path, index=False)
    source.to_parquet(parquet_path, index=False)
    left = build_features(load_events(csv_path))
    right = build_features(load_events(parquet_path))
    pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-10)
