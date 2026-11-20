# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Aggregate RoboCasa365 target50 completed episodes by their frozen task groups."""

import argparse
import json
from collections import Counter
from pathlib import Path

from cosmos_framework.scripts.eval_robocasa import SUITES, load_cases


def case_results(cases, results, composite_results, archive, checkpoint, protocol, allow_partial):
    expected = Counter(case["match"]["suite"] for case in cases.values())
    if expected != SUITES:
        raise ValueError(f"Expected the frozen target50 case counts: {SUITES}, found {dict(expected)}")
    valid = Counter()
    successes = Counter()
    excluded = []
    for case_id, case in cases.items():
        suite = case["match"]["suite"]
        directory = results if suite == "atomic_seen" or composite_results is None else composite_results
        path = directory / case_id / "summary.json"
        if not path.is_file():
            excluded.append(case_id)
            continue
        row = json.loads(path.read_text())
        if row.get("case_id") != case_id or row.get("checkpoint") != checkpoint or row.get("weight_variant") != "ema":
            raise ValueError(f"{path}: wrong case identity or checkpoint")
        if row.get("suite", suite) != suite:
            raise ValueError(f"{path}: incorrect task group")
        shift = 8.0 if protocol == "mixed-shift" and suite != "atomic_seen" else 5.0
        if row.get("shift", archive.get("shift") if directory == results else None) != shift:
            raise ValueError(f"{path}: expected shift {shift}")
        if (
            row.get("status") != "completed"
            or not (row.get("terminated") or row.get("truncated"))
            or type(row.get("success")) is not bool
        ):
            excluded.append(case_id)
            continue
        valid[suite] += 1
        successes[suite] += row["success"]
    if excluded and not allow_partial:
        raise ValueError(f"{len(excluded)} case(s) have no completed outcome. First exclusions: {excluded[:5]}")
    return {
        "protocol": protocol,
        "planned": len(cases),
        "valid": valid.total(),
        "successes": successes.total(),
        "success_rate": 100 * successes.total() / valid.total() if valid.total() else None,
        "groups": {
            suite: {
                "planned": planned,
                "valid": valid[suite],
                "successes": successes[suite],
                "success_rate": 100 * successes[suite] / valid[suite] if valid[suite] else None,
            }
            for suite, planned in SUITES.items()
        },
        "excluded": excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--composite-results", type=Path, help="Frozen composite-case directory for a mixed archive")
    parser.add_argument(
        "--archive-manifest", type=Path, help="Original run receipt for summaries without a shift field"
    )
    parser.add_argument("--checkpoint", required=True, help="Exact recorded checkpoint path from the result receipts")
    parser.add_argument("--protocol", choices=("mixed-shift", "uniform-shift5"), required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.composite_results is not None and args.protocol != "mixed-shift":
        parser.error("--composite-results requires --protocol mixed-shift")
    archive = json.loads(args.archive_manifest.read_text()) if args.archive_manifest else {}
    if archive and (archive.get("checkpoint"), archive.get("weight_variant")) != (args.checkpoint, "ema"):
        parser.error("--archive-manifest must describe the selected EMA checkpoint")
    report = case_results(
        load_cases(args.cases),
        args.results,
        args.composite_results,
        archive,
        args.checkpoint,
        args.protocol,
        args.allow_partial,
    )
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
