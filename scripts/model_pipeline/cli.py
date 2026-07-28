# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""RiverSort baseline model lifecycle CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riversort.data_pipeline import resolve_data_root
from riversort.model_pipeline import (
    ModelPipelineError,
    evaluate,
    preflight,
    promote,
    select_threshold,
    train,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--data-root", help="Overrides RIVERSORT_DATA_ROOT")
    root.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/training_yolo11n_baseline.yaml"),
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("train-smoke")
    baseline = commands.add_parser("train-baseline")
    baseline.add_argument("--batch", type=int, choices=(4, 8))
    threshold = commands.add_parser("select-threshold")
    threshold.add_argument("--batch", type=int, choices=(1, 2, 4, 8))
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--batch", type=int, choices=(1, 2, 4, 8))
    promotion = commands.add_parser("promote")
    promotion.add_argument("--model-version", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    data_root = resolve_data_root(args.data_root)
    try:
        if args.command == "preflight":
            result = preflight(data_root, args.contract)
        elif args.command == "train-smoke":
            result = train(data_root, args.contract, "smoke")
        elif args.command == "train-baseline":
            result = train(data_root, args.contract, "baseline", args.batch)
        elif args.command == "select-threshold":
            result = select_threshold(data_root, args.contract, args.batch)
        elif args.command == "evaluate":
            result = evaluate(data_root, args.contract, args.batch)
        elif args.command == "promote":
            result = promote(data_root, args.contract, args.model_version)
        else:
            raise AssertionError(args.command)
    except (ModelPipelineError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "result": result}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
