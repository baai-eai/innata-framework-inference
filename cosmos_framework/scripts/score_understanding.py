# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Compare Innata-0-Understanding evaluation output with the paper or the extended archive."""

import argparse
import json
import math
from pathlib import Path

from cosmos_framework.scripts.eval_understanding import EXTENDED_ONLY, PROFILES

SCORES = PROFILES.with_name("understanding_scores.json")


def compare(phases, extended):
    reports = []
    for reference in json.loads(SCORES.read_text()):
        phase, runner, metric = (reference[key] for key in ("phase", "runner", "metric"))
        if not extended and runner in EXTENDED_ONLY:
            continue
        folder = phases[phase]
        path = folder / runner / "aggregate_scores.json" if folder is not None else None
        row = {key: reference[key] for key in ("name", "phase", "runner", "metric", "reference")}
        if path is None or not path.is_file():
            reports.append({**row, "status": "missing"})
            continue
        aggregate = json.loads(path.read_text())[runner]
        result = aggregate["metrics"][metric]
        value = result["value"] * reference["scale"]
        ok = (
            result["valid_count"] == reference["count"]
            and result.get("missing_count", 0) == 0
            and aggregate.get("error_count", 0) == 0
        )
        reports.append(
            {
                **row,
                "status": "ok" if ok else "incomplete",
                "actual": value,
                "difference": value - reference["reference"],
                "matches_reference": ok and math.isclose(value, reference["reference"], rel_tol=0, abs_tol=1e-9),
                "valid_count": result["valid_count"],
                "reference_count": reference["count"],
            }
        )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (args.standard or args.video):
        parser.error("provide --standard and --video, or one result root with --allow-partial")
    metrics = compare({"standard": args.standard, "video": args.video}, args.extended)
    comparable = sum(row["status"] == "ok" for row in metrics)
    matching = sum(row.get("matches_reference", False) for row in metrics)
    report = {"required": len(metrics), "comparable": comparable, "matching": matching, "metrics": metrics}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
    if comparable != len(metrics) and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
