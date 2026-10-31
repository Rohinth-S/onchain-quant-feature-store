import pandas as pd
import pytest

from feature_store.core import synthetic_events
from feature_store.materialize import materialize_features, materialize_partitions
from feature_store.quality import inspect_events


def test_quality_report_and_materialized_lineage():
    events = synthetic_events(80, seed=3)
    report = inspect_events(events)
    assert report.valid and report.rows == 80
    features = materialize_features(events)
    assert {"feature_date", "feature_version"}.issubset(features.columns)
    assert features["feature_version"].iloc[0] == "h3-v8"


def test_partitions_are_sorted_and_invalid_batches_fail():
    events = synthetic_events(90).assign(timestamp=lambda frame: pd.to_datetime(frame.timestamp, utc=True))
    partitions = materialize_partitions(events)
    assert list(partitions) == sorted(partitions)
    broken = events.copy()
    broken.loc[0, "volume"] = -1
    with pytest.raises(ValueError, match="positive"):
        materialize_features(broken)
