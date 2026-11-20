# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Connect the frozen RoboDojo simulator harness and score its native episode results."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCORER_SHA256 = "9ec361af587e1788d83d3cf8670e05b817ff252d33c33829a6337dc16ae370d2"
UNSCORED = {
    "NATIVE_INVALID_INITIAL_LAYOUT",
    "NATIVE_UNSTABLE_UNSCORED",
    "NATIVE_PARALLEL_INVALID_INITIAL_LAYOUT",
    "NATIVE_PARALLEL_UNSTABLE_UNSCORED",
}


def load_json(path):
    return json.loads(path.read_text())


def rebase_layouts(cases_path, assets, output):
    manifest = load_json(cases_path)
    rows = manifest["cases"]
    if len(rows) != 2100 or len({row["case_id"] for row in rows}) != 2100:
        raise ValueError("Expected the frozen 2,100-case manifest")
    failures = []
    for case in rows:
        parts = Path(case["layout_path"]).parts
        relative = Path(*parts[parts.index("Eval_Layout") :])
        candidate = (assets / relative).resolve()
        if not candidate.is_relative_to(assets) or not candidate.is_file():
            failures.append(case["case_id"])
        elif hashlib.sha256(candidate.read_bytes()).hexdigest() != case["layout_sha256"]:
            failures.append(case["case_id"])
        else:
            case["layout_path"] = str(candidate)
    if failures:
        raise ValueError(f"{len(failures)} layouts missing or changed. First cases: {failures[:5]}")
    with output.open("x") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def score(cases_path, result_root, unscored_path, scorer, allow_partial):
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != SCORER_SHA256:
        raise ValueError("Use the official RoboDojo scoring rules from the frozen evaluation")
    dimensions = runpy.run_path(str(scorer))["DIMENSIONS"]
    cases = load_json(cases_path)["cases"]
    index = {row["case_id"]: row for row in cases}
    if len(cases) != 2100 or len(index) != len(cases) or {row["seed"] for row in cases} != {0}:
        raise ValueError("Expected the frozen 2,100 RoboDojo cases at official seed 0")
    skipped = {}
    for row in load_json(unscored_path):
        if row["case_id"] in skipped or row.get("native_has_score_for_case") or row["reason"] not in UNSCORED:
            raise ValueError(f"Invalid unscored case record: {row['case_id']}")
        skipped[row["case_id"]] = row["reason"]
    if not skipped.keys() <= index.keys():
        raise ValueError("Unscored result contains a case not in the manifest")

    scores = {}
    for path in sorted(result_root.glob("*/OFFICIAL_RESULT.json")):
        case_id = path.parent.name
        if case_id in skipped or case_id not in index:
            raise ValueError(f"Unexpected official result for {case_id}")
        record = load_json(path)
        details = list(record["details"].values())
        if len(details) != 1:
            raise ValueError(f"Expected one official episode in {path}")
        row = details[0]
        case = index[case_id]
        if record["seed"] != case["seed"] or int(row["layout_id"]) != case["layout_id"]:
            raise ValueError(f"Simulator config changed for {case_id}")
        if not 0 <= row["score"] <= 1 or type(row["success"]) is not bool:
            raise ValueError(f"Invalid native score for {case_id}")
        scores[case_id] = (row["score"], row["success"])
    missing = index.keys() - skipped.keys() - scores.keys()
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} episode results")
    if not scores:
        raise ValueError("No valid native results")

    entries = defaultdict(list)
    for case_id, value in scores.items():
        case = index[case_id]
        entries[(case["merged_task"], case["variant"])].append(value)

    families = {}
    for family, tasks in dimensions.items():
        by_task = []
        for task in tasks:
            variants = ("standard", "random") if family == "Generalization" else ("standard",)
            cells = [entries[(task, variant)] for variant in variants]
            if all(cells):
                by_task.append([mean(mean(item[channel] for item in cell) for cell in cells) for channel in (0, 1)])
        families[family] = {
            "tasks_scored": len(by_task),
            "tasks_planned": len(tasks),
            "score": 100 * mean(pair[0] for pair in by_task) if by_task else None,
            "success_rate": 100 * mean(pair[1] for pair in by_task) if by_task else None,
        }
    return {
        "planned": len(cases),
        "valid": len(scores),
        "unscored": len(skipped),
        "missing": len(missing),
        "successes": sum(success for _, success in scores.values()),
        "episode_success_rate": 100 * mean(success for _, success in scores.values()),
        "five_family_score": mean(row["score"] for row in families.values())
        if all(row["score"] is not None for row in families.values())
        else None,
        "five_family_success_rate": mean(row["success_rate"] for row in families.values())
        if all(row["success_rate"] is not None for row in families.values())
        else None,
        "families": families,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    layouts = phases.add_parser("layouts", help="Verify the frozen layouts against a local asset checkout")
    layouts.add_argument("--cases", type=Path, required=True)
    layouts.add_argument("--assets", type=Path, required=True, help="RoboDojo Assets directory containing Eval_Layout")
    layouts.add_argument("--output", type=Path, required=True)
    gateway = phases.add_parser("gateway", help="Connect the frozen RoboDojo harness to the HF WAM service")
    gateway.add_argument("--harness", type=Path, required=True, help="Frozen Plan v9 harness source directory")
    gateway.add_argument("--model-url", action="append", required=True, help="Each running HF WAM server")
    gateway.add_argument("--checkpoint", type=Path, required=True)
    gateway.add_argument("--request-dir", type=Path, required=True, help="New directory shared with the policy server")
    gateway.add_argument("--run-id", required=True)
    gateway.add_argument("--host", default="localhost")
    gateway.add_argument("--port", type=int, required=True)

    scorer = phases.add_parser("score")
    scorer.add_argument("--cases", type=Path, required=True)
    scorer.add_argument(
        "--results", type=Path, required=True, help="Root containing case directories with OFFICIAL_RESULT.json"
    )
    scorer.add_argument(
        "--unscored", type=Path, required=True, help="JSON array of documented native simulator exceptions"
    )
    scorer.add_argument("--official-scorer", type=Path, required=True)
    scorer.add_argument("--allow-partial", action="store_true")
    scorer.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.phase == "layouts":
        rebase_layouts(args.cases, args.assets.resolve(strict=True), args.output)
        return

    if args.phase == "score":
        report = score(args.cases, args.results, args.unscored, args.official_scorer, args.allow_partial)
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(output)
        print(output, end="")
        return

    checkpoint = str(args.checkpoint.resolve(strict=True))
    harness = args.harness.resolve(strict=True)
    sys.path.insert(0, os.fspath(harness))

    import httpx
    import uvicorn
    from harness.wam_client import CONTENT_TYPE, MAGIC, MAX_IMAGE, MAX_META, unpack

    from cosmos_framework.inference.robodojo_eval_gateway import create_gateway

    if len(set(url.rstrip("/") for url in args.model_url)) != len(args.model_url):
        parser.error("--model-url must identify distinct WAM servers")

    with httpx.Client(trust_env=False) as client:
        for url in args.model_url:
            response = client.get(f"{url.rstrip('/')}/info")
            response.raise_for_status()
            info = response.json()
            if (info["checkpoint_path"], info["sft_profile"], info["weight_variant"]) != (
                checkpoint,
                "robodojo",
                "ema",
            ):
                parser.error(f"unexpected WAM behind {url}")
    app = create_gateway(
        args.request_dir.resolve(),
        args.run_id,
        checkpoint,
        args.model_url,
        unpack,
        CONTENT_TYPE,
        len(MAGIC) + 4 + MAX_META + MAX_IMAGE,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
