"""Materialize and score a reproducible event batch."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .core import FeatureConfig, synthetic_events
from .io import load_events, make_report, write_json
from .materialize import materialize_features
from .warehouse import publish_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate point-in-time on-chain features")
    parser.add_argument("--input", type=Path, help="normalized CSV or Parquet (default: seeded demo)")
    parser.add_argument("--output", type=Path, default=Path("reports/feature-report.json"))
    parser.add_argument("--dataset-dir", type=Path, help="publish a new DuckDB and Parquet snapshot directory")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        config = FeatureConfig(args.horizon, args.window)
        if args.input:
            with args.input.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            events = load_events(args.input)
            provenance = {"kind": args.input.suffix.lower().lstrip("."), "sha256": digest}
        else:
            events = synthetic_events(seed=args.seed)
            provenance = {"kind": "synthetic", "seed": args.seed}
        features = materialize_features(events, config, include_unlabeled=True)
        report, predictions = make_report(features, args.train_size)
        report.update({
            "source": {**provenance, "rows": len(events)},
            "config": asdict(config),
            "pools": {str(pool): int(count) for pool, count in features["pool"].value_counts().items()},
        })
        if args.dataset_dir:
            publish_snapshot(features, predictions, report, args.dataset_dir)
        write_json(report, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.output), "rows": len(features), "evaluation": report["walk_forward"]}))


if __name__ == "__main__":
    main()
