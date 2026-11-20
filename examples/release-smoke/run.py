# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Run a bundled Innata HF inference request and check the output contract."""

import argparse
import json
import subprocess
import sys
import tempfile
from math import isfinite
from pathlib import Path

CASES = {
    "understanding": ("model_type", "qwen3_vl"),
    "acwm": ("innata_acwm_profile", "a115"),
    "robodojo": ("innata_wam_profile", "robodojo"),
    "robocasa": ("innata_wam_profile", "robocasa"),
}


def verify(case, directory):
    result = json.loads((directory / "sample_outputs.json").read_text())
    if result["status"] != "success" or not result["outputs"]:
        raise ValueError(f"Inference returned {result['status']!r}")
    if case == "acwm":
        import av

        with av.open(str(directory / "vision.mp4")) as container:
            stream = container.streams.video[0]
            shape = (stream.width, stream.height, stream.frames, float(stream.average_rate))
        if shape != (736, 544, 17, 16.0):
            raise ValueError(f"Unexpected video metadata: {shape}")
        return "736×544 MP4, 17 frames at 16 fps"

    content = result["outputs"][0]["content"]
    if case == "understanding":
        if not content["text"].strip():
            raise ValueError("Empty Understanding response")
        return "nonempty text response"

    width = 14 if case == "robodojo" else 12
    for actions in (content["action"], content["action128"]["normalized_action"]):
        if len(actions) != 16 or any(
            len(row) != width or not all(isfinite(value) for value in row) for row in actions
        ):
            raise ValueError(f"Expected finite 16×{width} actions")
    return f"16×{width} native and normalized actions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local Hugging Face weight directory")
    parser.add_argument("--vae", type=Path, help="Local Wan2.2 VAE for ACWM and WAM")
    parser.add_argument("--output", type=Path, default=Path("outputs/release-smoke"), help="Parent output directory")
    args = parser.parse_args()

    key, expected = CASES[args.case]
    try:
        config = json.loads((args.checkpoint / "config.json").read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read checkpoint config: {exc}")
    if config.get(key) != expected:
        parser.error(f"{args.case} requires a checkpoint with {key}={expected!r}")
    if args.case != "understanding" and (args.vae is None or not args.vae.is_file()):
        parser.error("ACWM and WAM require --vae pointing to Wan2.2_VAE.pth")

    args.output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{args.case}-", dir=args.output)).resolve()
    request = Path(__file__).resolve().with_name(f"{args.case}.json")
    command = [
        sys.executable, "-m", "cosmos_framework.scripts.inference",
        "-i", str(request), "-o", str(run),
        "--checkpoint-path", str(args.checkpoint.resolve()), "--no-guardrails",
    ]
    if args.case != "understanding":
        command.extend(["--vae-path", str(args.vae.resolve())])
    if args.case == "acwm":
        command.append("--use-torch-compile")
    elif args.case in {"robodojo", "robocasa"}:
        command.extend(["--use-torch-compile", "--compiled-region", "language", "--no-compile-dynamic"])
    subprocess.run(command, check=True)
    print(f"PASS {args.case}: {verify(args.case, run / args.case)}\nOutputs: {run}")


if __name__ == "__main__":
    main()
