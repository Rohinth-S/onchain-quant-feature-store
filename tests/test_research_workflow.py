import json
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from feature_store.core import FEATURE_COLUMNS, FeatureConfig, build_features, synthetic_events, walk_forward_predictions
from feature_store.io import make_report
from feature_store.materialize import materialize_features


def test_features_are_pool_isolated_and_prefix_stable():
    left = synthetic_events(50)
    right = synthetic_events(50, seed=11).assign(pool="ALT-USDC", mid_price=9_000)
    combined = build_features(pd.concat([right, left]), include_unlabeled=True)
    expected = build_features(left.iloc[:25], include_unlabeled=True)
    observed = combined[combined.pool.eq("WETH-USDC")].iloc[:len(expected)]
    pd.testing.assert_frame_equal(
        observed[list(FEATURE_COLUMNS)].reset_index(drop=True),
        expected[list(FEATURE_COLUMNS)],
    )
    assert pd.isna(combined.groupby("pool")["label_end"].tail(3)).all()


def test_refits_use_only_settled_labels_and_do_not_split_timestamp_ties():
    data = synthetic_events(60)
    features = build_features(pd.concat([data, data.assign(pool="ALT-USDC")]), FeatureConfig(horizon=5))
    predictions = walk_forward_predictions(features, train_size=8, fold_size=7)
    assert (predictions["training_last_label_end"] < predictions["timestamp"]).all()
    assert predictions.groupby("timestamp")["training_rows"].nunique().eq(1).all()
    assert predictions.groupby("timestamp").size().eq(2).all()


def test_small_batch_can_materialize_without_a_model():
    features = materialize_features(synthetic_events(7), include_unlabeled=True)
    report, predictions = make_report(features)
    assert report["rows"] == 4 and report["walk_forward"]["status"] == "insufficient_history"
    assert predictions.empty


def test_cli_publishes_queryable_labels_separately(tmp_path):
    source = tmp_path / "events.csv"
    snapshot = tmp_path / "snapshot"
    synthetic_events(85).to_csv(source, index=False)
    command = [
        sys.executable, "-m", "feature_store.cli", "--input", str(source),
        "--dataset-dir", str(snapshot), "--output", str(tmp_path / "report.json"),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((snapshot / "report.json").read_text(encoding="utf-8"))
    assert report["source"]["rows"] == 85 and report["walk_forward"]["observations"] > 0
    with duckdb.connect(str(snapshot / "features.duckdb"), read_only=True) as db:
        rows = db.execute("SELECT count(*) FROM features").fetchone()[0]
        labels = db.execute("SELECT count(*) FROM labels").fetchone()[0]
        assert rows == 82 and rows - labels == 3
        assert "target_cost_bps" not in {column[0] for column in db.execute("DESCRIBE features").fetchall()}
        parquet = str(snapshot / "parquet" / "*" / "*.parquet")
        assert db.execute("SELECT count(*) FROM read_parquet(?)", [parquet]).fetchone()[0] == rows
    duplicate = subprocess.run(command, capture_output=True, text=True)
    assert duplicate.returncode != 0 and "snapshot already exists" in duplicate.stderr


def test_event_quality_records_nonfinite_values_and_duplicate_ids():
    events = synthetic_events(8).assign(event_id="same")
    events.loc[0, "gas_gwei"] = float("inf")
    with pytest.raises(ValueError, match="finite and positive.*duplicate"):
        materialize_features(events)
