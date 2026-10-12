"""Publish immutable DuckDB and Parquet snapshots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pandas as pd

from .io import write_json

LABEL_COLUMNS = ("label_end", "target_cost_bps", "target_vol")


def publish_snapshot(
    features: pd.DataFrame, predictions: pd.DataFrame, report: dict, destination: str | Path,
) -> Path:
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"snapshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".feature-stage-", dir=target.parent) as temp:
        stage = Path(temp) / "snapshot"
        stage.mkdir()
        feature_rows = features.drop(columns=list(LABEL_COLUMNS))
        label_rows = features[["event_sequence", *LABEL_COLUMNS]].dropna()
        if not feature_rows["event_sequence"].is_unique:
            raise ValueError("feature event_sequence keys must be unique")
        if not predictions["event_sequence"].is_unique or not predictions["event_sequence"].isin(label_rows["event_sequence"]).all():
            raise ValueError("each scored event must reference exactly one completed label")
        with duckdb.connect(str(stage / "features.duckdb"), config={"threads": 1}) as db:
            db.execute("SET TimeZone = 'UTC'")
            for name, frame in (("features", feature_rows), ("labels", label_rows)):
                db.register("incoming", frame)
                projection = "* REPLACE (CAST(feature_date AS DATE) AS feature_date)" if name == "features" else "*"
                db.execute(f"CREATE TABLE {name} AS SELECT {projection} FROM incoming ORDER BY event_sequence")
                db.unregister("incoming")
            db.register("incoming", predictions)
            db.execute("CREATE TABLE predictions AS SELECT * FROM incoming")
            db.unregister("incoming")
            db.execute("CREATE TABLE manifest (payload JSON)")
            db.execute("""
                CREATE VIEW latest_features AS
                SELECT * FROM features
                QUALIFY row_number() OVER (PARTITION BY pool ORDER BY event_sequence DESC) = 1
            """)
            db.execute("INSERT INTO manifest VALUES (?)", [json.dumps(report, allow_nan=False)])
            parquet = (stage / "parquet").as_posix().replace("'", "''")
            if feature_rows.empty:
                (stage / "parquet").mkdir()
                db.execute(f"COPY features TO '{parquet}/empty.parquet' (FORMAT PARQUET)")
            else:
                db.execute(f"COPY features TO '{parquet}' (FORMAT PARQUET, PARTITION_BY (feature_date))")
        write_json(report, stage / "report.json")
        stage.rename(target)
    return target
