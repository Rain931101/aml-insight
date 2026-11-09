"""python -m aml_insight.evaluation --help"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Member B reproducible AML experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split", help="Export manifest for member C before model training")
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("tune", help="Train/select using validation only; never evaluate test")
    p.add_argument("config", type=Path)
    p = sub.add_parser("finalize", help="Exclusive one-time test metrics for a locked run")
    p.add_argument("run", type=Path)
    p = sub.add_parser("explain", help="TreeSHAP and training-median masking check")
    p.add_argument("run", type=Path)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--background", type=int, default=100)
    p = sub.add_parser("aggregate", help="Source-backed research summaries; test runs excluded")
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--include-test", action="store_true")
    p = sub.add_parser("suite", help="Generate configs without training or evaluating test")
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "split":
            from aml_insight.evaluation.runner import prepare_manifest

            print(prepare_manifest(args.config, args.output))
        elif args.command == "tune":
            from aml_insight.evaluation.runner import tune

            print(tune(args.config))
        elif args.command == "finalize":
            from aml_insight.evaluation.runner import finalize

            finalize(args.run)
            print(args.run / "metrics.json")
        elif args.command == "explain":
            from aml_insight.explain.shap import explain_run

            print(explain_run(args.run, args.samples, args.background))
        elif args.command == "aggregate":
            from aml_insight.evaluation.report import aggregate

            aggregate(args.root, args.output, args.include_test)
            print(args.output / "results.md")
        elif args.command == "suite":
            from aml_insight.evaluation.suite import generate_suite

            print(generate_suite(args.config, args.output))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
