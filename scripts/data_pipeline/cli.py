# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Single offline CLI for RiverSort dataset readiness."""

from __future__ import annotations

import argparse
import json

from riversort.data_pipeline import (
    DataPipelineError,
    build_audit_queue,
    build_contact_sheets,
    build_groups,
    extract_sources,
    inventory,
    promote_snapshot,
    record_contact_sheet_review,
    resolve_data_root,
    split_groups,
    stage_fpwr,
    validate_snapshot,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--data-root", help="Overrides RIVERSORT_DATA_ROOT")
    root.add_argument("--dry-run", action="store_true", help="Validate without writing")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("inventory", "extract", "stage-fpwr", "build-audit-queue", "build-contact-sheets", "record-contact-sheet-review", "build-groups", "split", "promote", "validate"):
        sub = commands.add_parser(name)
        if name in {"build-audit-queue", "split"}:
            sub.add_argument("--seed", type=int, default=42)
        if name in {"promote", "validate"}:
            sub.add_argument("--snapshot-id")
        if name == "record-contact-sheet-review":
            sub.add_argument("--reviewer", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    data_root = resolve_data_root(args.data_root)
    try:
        if args.command == "inventory":
            result = inventory(data_root)
        elif args.command == "extract":
            result = extract_sources(data_root, args.dry_run)
        elif args.command == "stage-fpwr":
            result = stage_fpwr(data_root, args.dry_run)
        elif args.command == "build-audit-queue":
            result = build_audit_queue(data_root, args.seed, args.dry_run)
        elif args.command == "build-contact-sheets":
            if args.dry_run:
                raise DataPipelineError("build-contact-sheets has no write-free operation")
            result = build_contact_sheets(data_root)
        elif args.command == "record-contact-sheet-review":
            result = record_contact_sheet_review(data_root, args.reviewer, args.dry_run)
        elif args.command == "build-groups":
            result = build_groups(data_root, args.dry_run)
        elif args.command == "split":
            result = split_groups(data_root, args.seed, args.dry_run)
        elif args.command == "promote":
            result = promote_snapshot(data_root, args.snapshot_id, args.dry_run)
        elif args.command == "validate":
            result = validate_snapshot(data_root, args.snapshot_id)
        else:
            raise AssertionError(args.command)
    except DataPipelineError as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "result": result}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
