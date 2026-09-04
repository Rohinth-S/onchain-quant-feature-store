from __future__ import annotations
import argparse
from .core import synthetic_events, build_features, FeatureConfig, walk_forward_rmse
from .io import load_events_csv, write_report

def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe on-chain features")
    parser.add_argument("--input", help="optional normalized event CSV")
    parser.add_argument("--output", default="reports/feature-report.json")
    args = parser.parse_args()
    events = load_events_csv(args.input) if args.input else synthetic_events()
    cfg = FeatureConfig()
    features = build_features(events, cfg)
    report = write_report(features, args.output, cfg)
    print({"rows": len(features), "walk_forward": walk_forward_rmse(features), "report": str(report)})

if __name__ == "__main__":
    main()
