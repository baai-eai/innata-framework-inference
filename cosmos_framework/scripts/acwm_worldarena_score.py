# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Aggregate the 15 Innata-0 ACWM WorldArena Track 1 metrics."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

VIDEO_METRICS = (
    "aesthetic_quality",
    "background_consistency",
    "depth_accuracy",
    "dynamic_degree",
    "flow_score",
    "image_quality",
    "motion_smoothness",
    "photometric_smoothness",
    "semantic_alignment",
    "subject_consistency",
    "trajectory_accuracy",
)
MOTION_CAP = {"dynamic_degree", "flow_score", "motion_smoothness"}
VLM_METRICS = ("Instruction_Following", "Interaction_Quality", "Perspectivity")
EPISODES = {f"episode{i}" for i in range(1, 1001)}


def episode_name(path):
    matches = re.findall(r"(?:^|/)(episode[0-9]+)(?=/|\.mp4$|$)", path)
    if len(set(matches)) != 1:
        raise ValueError(f"Expected an unambiguous episode identifier in {path!r}")
    return matches[0]


def check_coverage(names, label):
    names = list(names)
    if len(names) != len(EPISODES) or set(names) != EPISODES:
        raise ValueError(f"{label}: expected exactly 1,000 unique episodes, got {len(names)} rows")


def normalized(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid normalized score {value!r}")
    return float(value)


def read_metric(root, dimension):
    file = root / dimension / "generated_results.json"
    records = json.loads(file.read_text())[dimension][1]
    names = [episode_name(row["video_path"]) for row in records]
    check_coverage(names, dimension)
    return dict(zip(names, (normalized(row["video_results_normalized"]) for row in records), strict=True))


def aggregate(video_results, gt_results, vlm_results, jepa_result, jepa_pairs):
    scores = {}
    for dimension in VIDEO_METRICS:
        generated = read_metric(video_results, dimension)
        if dimension in MOTION_CAP:
            reference = read_metric(gt_results, dimension)
            scores[dimension] = mean(min(generated[name], reference[name]) for name in generated)
        else:
            scores[dimension] = mean(generated.values())

    vlm = []
    for path in sorted(vlm_results.rglob("*.json")):
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a WorldArena VLM per-episode results array")
        vlm.extend(rows)
    check_coverage((episode_name(row["video"]) for row in vlm), "VLM")
    for dimension in VLM_METRICS:
        values = []
        for row in vlm:
            if row.get("error"):
                raise ValueError(f"VLM scoring failed for {row['video']}")
            result = row["metrics"][dimension]
            if not isinstance(result.get("reason"), str) or not result["reason"].strip():
                raise ValueError(f"Missing VLM rationale for {row['video']} / {dimension}")
            values.append(normalized(result["score_normalized"]))
        scores[dimension] = mean(values)

    check_coverage(json.loads(jepa_pairs.read_text()), "JEPA paired episodes")
    scores["jepa_similarity"] = normalized(json.loads(jepa_result.read_text())["score"])
    return {
        "episodes": len(EPISODES),
        "metrics": {name: value * 100 for name, value in scores.items()},
        "mean15_capped": 100 * mean(scores.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-results", type=Path, required=True, help="Generated-video perdim/ directory")
    parser.add_argument(
        "--gt-results", type=Path, required=True, help="Ground-truth perdim/ for the three motion metrics"
    )
    parser.add_argument("--vlm-results", type=Path, required=True, help="VLM per-episode results directory")
    parser.add_argument("--jepa-result", type=Path, required=True, help="JEPA results.json")
    parser.add_argument("--jepa-pairs", type=Path, required=True, help="JEPA actual matched episode list")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = aggregate(args.video_results, args.gt_results, args.vlm_results, args.jepa_result, args.jepa_pairs)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
