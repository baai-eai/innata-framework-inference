# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Run the WorldArena Track 1 scorers on an Innata-0-ACWM rollout."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cosmos_framework.inference.release_models import ACWM_CHECKPOINT_NAME
from cosmos_framework.scripts.acwm_worldarena_score import MOTION_CAP, VIDEO_METRICS, check_coverage


def link(source, target):
    source = source.resolve(strict=True)
    if not target.is_symlink() or target.resolve() != source:
        target.symlink_to(source)


def prepare(inputs, generated, ground_truth, first_frames, work, episodes=()):
    rows = [json.loads(line) for line in (inputs / "eval-prompts-1000.jsonl").read_text().splitlines()]
    check_coverage((row["episode"] for row in rows), "Evaluation prompts")
    if episodes:
        selected = set(episodes)
        rows = [row for row in rows if row["episode"] in selected]
        if {row["episode"] for row in rows} != selected:
            raise ValueError("Unknown WorldArena episode")
    else:
        check_coverage((path.stem for path in generated.glob("*.mp4")), "Generated videos")
        check_coverage((path.stem for path in ground_truth.glob("*.mp4")), "Ground-truth videos")

    standard = work / "standard"
    reference = work / "reference"
    labels = work / "labels"
    for directory in (standard, reference, labels):
        directory.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows:
        episode = row["episode"]
        video = f"{episode}.mp4"
        # The official preprocessor reads the task name from the fifth
        # component from the end of the ground-truth video path.
        task = row["task"]
        if not isinstance(task, str) or Path(task).name != task or task in ("", ".", ".."):
            raise ValueError(f"{episode}: invalid task name")
        target = labels / task / "robotwin" / "video" / "observation" / video
        target.parent.mkdir(parents=True, exist_ok=True)
        first_frame = first_frames / f"{episode}.png"
        if not first_frame.is_file():
            raise FileNotFoundError(first_frame)
        link(ground_truth / video, target)
        link(generated / video, standard / f"{task}_{video}")
        link(ground_truth / video, reference / f"{task}_{video}")
        summary.append({"gt_path": str(target), "image": str(first_frame), "prompt": row["prompt"]})
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "video", "vlm", "jepa"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--gt-videos", type=Path)
    parser.add_argument("--gt-first-frames", type=Path, help="Original WorldArena first frames for scoring")
    parser.add_argument("--episode", action="append", help="Prepare individual smoke cases")
    parser.add_argument("--metric", action="append", choices=VIDEO_METRICS, help="Run a subset of video metrics")
    parser.add_argument("--evaluator", type=Path, help="WorldArena video_quality_ood checkout")
    parser.add_argument("--config", type=Path, help="WorldArena config.yaml with local scorer weights")
    args = parser.parse_args()
    if args.episode and args.phase != "prepare":
        parser.error("--episode applies to prepare")
    if args.metric and args.phase != "video":
        parser.error("--metric applies to video")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    def required(flag):
        value = vars(args)[flag]
        if value is None:
            parser.error(f"{args.phase} requires --{flag.replace('_', '-')}")
        return value.resolve(strict=True)

    if args.phase == "prepare":
        prepare(
            required("inputs"),
            required("videos"),
            required("gt_videos"),
            required("gt_first_frames"),
            work,
            args.episode,
        )
        return

    evaluator = required("evaluator")

    def run(*command, cwd=evaluator):
        subprocess.run([sys.executable, *(str(arg) for arg in command)], check=True, cwd=cwd)

    if args.phase == "jepa":
        run(
            "batch.py",
            "--real_dir",
            required("gt_videos"),
            "--gen_dir",
            required("videos"),
            "--save_intersection",
            work / "intersection_names.json",
            "--output_root",
            work / "jepa",
            cwd=evaluator / "JEDi",
        )
        return

    config_path = required("config")
    summary = work / "summary.json"
    if not summary.is_file():
        parser.error("Run prepare before scoring")
    if args.phase == "vlm":
        output = work / "vlm"
        model_name = ACWM_CHECKPOINT_NAME
        # Reuse the model name of a scoring run started before the release naming change.
        legacy_name = "innata-acwm-a115"
        if (output / legacy_name).is_dir():
            model_name = legacy_name
        run(
            "VLM_judge.py",
            "--model_name",
            model_name,
            "--video_dir",
            required("videos"),
            "--summary_json",
            summary,
            "--metrics",
            "all",
            "--num_frames",
            16,
            "--output_root",
            output,
            "--tmp_root",
            work / "vlm-tmp",
            "--config_path",
            config_path,
        )
        return

    import yaml

    base = yaml.safe_load(config_path.read_text())
    selected = [metric for metric in VIDEO_METRICS if not args.metric or metric in args.metric]
    for view, metrics, result_dir in (
        ("standard", selected, "perdim"),
        ("reference", [metric for metric in selected if metric in MOTION_CAP], "gt-perdim"),
    ):
        if not metrics:
            continue
        data = work / f"{view}-data"
        config = dict(base)
        config["data"] = {"gt_path": str(data / "gt_dataset"), "val_base": str(data / "generated_dataset")}
        local_config = work / f"{view}.yaml"

        local_config.write_text(yaml.safe_dump(config, sort_keys=False))
        run("preprocess_datasets.py", "--summary_json", summary, "--gen_video_dir", work / view, "--output_base", data)
        run("processing/video_resize.py", "--config_path", local_config)
        if "trajectory_accuracy" in metrics:
            run("processing/detection_tracking.py", "--config_path", local_config)
        for metric in metrics:
            destination = work / result_dir / metric
            destination.mkdir(parents=True, exist_ok=True)
            config["save_path"] = str(destination)
            local_config.write_text(yaml.safe_dump(config, sort_keys=False))
            run("evaluate.py", "--dimension", metric, "--config", local_config, "--overwrite")


if __name__ == "__main__":
    main()
