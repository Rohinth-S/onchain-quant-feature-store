"""Materialize and score a reproducible event batch."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from . import __version__
from .artifacts import require_distinct_paths

from .core import FeatureConfig, synthetic_events
from .io import decode_events, make_report, write_json
from .materialize import materialize_features
from .quality import inspect_events
from .warehouse import publish_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate point-in-time on-chain features")
    parser.add_argument("--input", type=Path, help="normalized CSV or Parquet (default: seeded demo)")
    parser.add_argument("--output", type=Path, default=Path("reports/feature-report.json"))
    parser.add_argument("--dataset-dir", type=Path, help="publish a new DuckDB and Parquet snapshot directory")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--fold-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--quality-only", action="store_true", help="inspect raw input before modeling")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    try:
        require_distinct_paths(
            input_file=args.input,
            report=args.output,
            dataset=args.dataset_dir,
        )
        if args.dataset_dir and args.output.resolve().is_relative_to(args.dataset_dir.resolve()):
            raise ValueError("standalone report must be outside the immutable dataset directory")
        config = FeatureConfig(args.horizon, args.window)
        if args.input:
            raw = args.input.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            events = decode_events(raw, args.input.suffix, check=not args.quality_only)
            provenance = {"kind": args.input.suffix.lower().lstrip("."), "sha256": digest}
        else:
            events = synthetic_events(seed=args.seed)
            provenance = {"kind": "synthetic", "seed": args.seed}
        if args.quality_only:
            if args.dataset_dir:
                raise ValueError("quality-only inspection does not create a dataset")
            quality = inspect_events(events)
            value = {"source": provenance, "quality": {**asdict(quality), "valid": quality.valid}}
            write_json(value, args.output)
            print(json.dumps(value, allow_nan=False))
            return
        features = materialize_features(events, config, include_unlabeled=True)
        report, predictions = make_report(features, args.train_size, fold_size=args.fold_size, max_train_rows=args.max_train_rows)
        report.update({
            "source": {**provenance, "rows": len(events)},
            "config": asdict(config),
            "pools": {str(pool): int(count) for pool, count in features["pool"].value_counts().items()},
        })
        report["producer"] = {"name": "onchain-feature-store", "version": __version__, "format_version": 1}
        identity = {
            "source": provenance, "features": asdict(config), "train_size": args.train_size,
            "fold_size": args.fold_size, "max_train_rows": args.max_train_rows, "software": __version__,
        }
        report["dataset_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        if args.dataset_dir:
            publish_snapshot(features, predictions, report, args.dataset_dir)
        write_json(report, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.output), "rows": len(features), "evaluation": report["walk_forward"]}))


if __name__ == "__main__":
    main()
