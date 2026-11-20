# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Run the Innata-0-Understanding paper or extended evaluation with the pinned wm_evaluation adapter."""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from cosmos_framework.inference.release_models import UNDERSTANDING_CHECKPOINT_NAME

PROFILES = Path(__file__).resolve().parents[1] / "inference/defaults/release_eval/understanding.json"
EXTENDED_ONLY = {"intphys2", "refspatial", "robospatial", "sharerobot", "where2place"}
GENERATION = {"do_sample": False, "temperature": 0.0, "top_p": 1.0}


def free_port():
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        return listener.getsockname()[1]


def materialize_data(config, root):
    if isinstance(config, dict):
        return {
            key: str(root / value) if key == "data_dir" else materialize_data(value, root)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [materialize_data(value, root) for value in config]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("standard", "video"))
    parser.add_argument("--evaluator", type=Path, required=True, help="wm_evaluation checkout, with its models adapter")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help=f"Local {UNDERSTANDING_CHECKPOINT_NAME} Hugging Face directory"
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root of the prepared benchmark datasets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=UNDERSTANDING_CHECKPOINT_NAME)
    parser.add_argument("--suite", choices=("paper", "extended"), default="paper")
    parser.add_argument("--runner", action="append", help="Select a runner, using repeated flags to select several")
    parser.add_argument("--gpu-id", type=int, action="append", help="One TP=1 service per physical CUDA device")
    parser.add_argument("--vllm", default="vllm", help="vLLM executable in the pinned server environment")
    parser.add_argument("--mmsi-metadata", type=Path, help="Override the MMSI-Video metadata directory")
    parser.add_argument("--limit", type=int, help="Optional sample limit for a smoke run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    if json.loads((checkpoint / "config.json").read_text()).get("model_type") != "qwen3_vl":
        parser.error(f"--checkpoint must be the {UNDERSTANDING_CHECKPOINT_NAME} Qwen3-VL Hugging Face export")
    evaluator = args.evaluator.resolve(strict=True)
    if not (evaluator / "models/cosmos_qwen3_vl/model.py").is_file():
        parser.error("--evaluator must include the matching cosmos_qwen3_vl adapter in models/")
    data = args.data_root.resolve(strict=True)
    config = json.loads(PROFILES.read_text())[args.phase]
    runners = [name for name in config["run"]["benchmarks"] if args.suite == "extended" or name not in EXTENDED_ONLY]
    if args.runner:
        if not set(args.runner) <= set(runners):
            parser.error("--runner must be registered for the selected phase and suite")
        runners = [name for name in runners if name in args.runner]
    config["run"]["benchmarks"] = runners
    config["benchmarks"] = materialize_data({name: config["benchmarks"][name] for name in runners}, data)
    for name, settings in config["benchmarks"].items():
        if not Path(settings["data_dir"]).is_dir():
            parser.error(f"missing prepared dataset for {name}: {settings['data_dir']}")
    if args.phase == "video":
        metadata = args.mmsi_metadata or evaluator / "benchmarks/mmsi_video/meta_data"
        config["benchmarks"]["mmsi_video"]["metadata_dir"] = str(metadata.resolve(strict=True))
    gpu_ids = args.gpu_id or [0]
    if min(gpu_ids) < 0 or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpu-id values must be non-negative and distinct")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    binary = shutil.which(args.vllm)
    if binary is None and not args.dry_run:
        parser.error(f"vLLM executable not found: {args.vllm}")
    output = args.output_dir.resolve()
    config["run"].update(output_dir=str(output), show_progress=False)
    config["parallel"]["adaptive-batch"]["num_services"] = len(gpu_ids)
    config["model_overrides"].update(
        vllm_executable=binary or args.vllm,
        vllm_host="localhost",
        vllm_port=free_port(),
        vllm_model_path=str(checkpoint),
        media_root=str(data),
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    profile = output / f"{args.run_name}-{args.phase}-config.json"
    if profile.exists():
        parser.error("choose a new --run-name for this run")
    with tempfile.TemporaryDirectory(prefix="innata-eval-") as temporary:
        runtime = Path(temporary)
        (runtime / "generation_config.json").write_text(json.dumps(GENERATION) + "\n")
        config["model_overrides"].update(vllm_runtime_dir=str(runtime), vllm_generation_config=str(runtime))
        profile.write_text(json.dumps(config, indent=2) + "\n")
        command = [
            sys.executable,
            str(evaluator / "run_registered_model.py"),
            "--config",
            str(profile),
            "--run-name",
            f"{args.run_name}-{args.phase}",
            "--gpu-ids",
            *map(str, gpu_ids),
        ]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        subprocess.run(command, cwd=evaluator, check=True)


if __name__ == "__main__":
    main()
