"""python -m ieri_risk {build,train,verify,report,example}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IERI synthetic financial-stress research engine")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", type=Path, default=Path("risk_data/v1"))
    build.add_argument("--businesses", type=int, default=500)
    build.add_argument("--seed", type=int, default=20261110)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=Path("risk_data/v1"))
    train.add_argument("--registry", type=Path, default=Path("artifacts/ieri"))
    train.add_argument("--seed", type=int, default=20261110)
    train.add_argument("--trials", type=int, default=8)
    train.add_argument("--challenge-businesses", type=int, default=200)
    train.add_argument("--bootstrap", type=int, default=200)
    for name in ("verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("--registry", type=Path, default=Path("artifacts/ieri"))
    example = commands.add_parser("example")
    example.add_argument("--output", type=Path, default=Path("risk_examples/score.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            from .dataset import build_dataset
            path = build_dataset(args.output, args.businesses, args.seed)
            print(f"Built and validated separate feature/outcome datasets: {path}")
        elif args.command == "train":
            from .training import train as run_training
            result = run_training(args.data, args.registry, args.seed, args.trials, args.challenge_businesses, args.bootstrap)
            print(json.dumps(result, indent=2))
        elif args.command in ("verify", "report"):
            from .service import RiskService
            service = RiskService(args.registry)
            if args.command == "verify":
                print(f"Verified artifact checksums, schema and dependencies: {service.metadata['model_id']}")
            else:
                print(f"Champion selected on validation: {service.metadata['champion']}")
                print("model             ROC-AUC      AP     Brier     F1")
                for name, r in service.report["final_test"].items():
                    m = r["calibrated"]
                    print(f"{name:16}  {m['roc_auc']:.4f}  {m['average_precision']:.4f}  {m['brier']:.4f}  {m['f1']:.4f}")
                print(service.metadata["notice"])
                for alert in service.report["shortcut_audit"]["alerts"]:
                    print("SIMULATION AUDIT:", alert)
        else:
            from .api import demo
            payload = demo(0)
            write_json(args.output, payload["score_request"])
            write_json(args.output.with_name("history.json"), payload["history_request"])
            print(f"Wrote {args.output} and {args.output.with_name('history.json')}")
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"IERI: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
