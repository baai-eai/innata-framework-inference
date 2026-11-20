# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Run the Innata RoboCasa365 policy on the frozen target50 simulator cases."""

import argparse
import io
import json
import math
import sys
import time
import uuid
from pathlib import Path

CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]
LAYOUT = [
    ["main", "top", 640, 480],
    ["left_wrist", "bottom_left", 320, 240],
    ["auxiliary_0", "bottom_right", 320, 240],
]
SUITES = {"atomic_seen": 900, "composite_seen": 800, "composite_unseen": 800}


def load_cases(path):
    cases = json.loads(path.read_text())["cases"]
    indexed = {}
    for row in cases:
        case = row["case_id"]
        if Path(case).name != case or not case.startswith("target50-") or case in indexed:
            raise ValueError(f"Invalid or duplicate RoboCasa case: {case!r}")
        indexed[case] = row
    return indexed


def make_request(observation, case, path, seed, shift):
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn.functional import interpolate

    meta, arrays = observation.metadata, observation.arrays
    match = case["match"]
    if meta["task_name"] != match["task_name"] or meta["case_id"] != case["case_id"]:
        raise ValueError("Simulator returned the wrong task or case")
    if meta["camera_names"] != CAMERAS or meta["control_frequency_hz"] != 20:
        raise ValueError("Unexpected RoboCasa camera order or control frequency")
    if meta["horizon"] != match["official_horizon"]:
        raise ValueError("Simulator changed the official case horizon")
    if meta.get("success_check_interval") != 1:
        raise ValueError("The reference simulator must check success at every control step")
    step = meta["step"]
    if arrays["history_steps"][-1] != step:
        raise ValueError("Expected the current synchronized observation")
    rgb, state = arrays["history_rgb"][-1], arrays["state_history"][-1]
    if rgb.shape != (3, 256, 256, 3) or rgb.dtype != np.uint8 or state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Expected three RGB cameras and finite native State16")

    canvas = torch.zeros((3, 720, 640), dtype=torch.uint8)
    regions = ((0, 0, 640, 480), (0, 480, 320, 240), (320, 480, 320, 240))
    for image, (x, y, width, height) in zip(rgb, regions, strict=True):
        with io.BytesIO() as buffer:
            Image.fromarray(image).save(
                buffer, format="JPEG", quality=95, subsampling=2, optimize=False, progressive=False
            )
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                pixels = torch.from_numpy(np.array(decoded.convert("RGB"))).permute(2, 0, 1)[None].float()
        h, w = pixels.shape[-2:]
        scale = max(width / w, height / h)
        h, w = math.ceil(h * scale), math.ceil(w * scale)
        pixels = interpolate(pixels, (h, w), mode="bilinear", align_corners=False).round_().clamp_(0, 255)
        dx, dy = (w - width) // 2, (h - height) // 2
        canvas[:, y : y + height, x : x + width] = pixels[0, :, dy : dy + height, dx : dx + width]
    canvas[:, 478:482] = 0
    canvas[:, 480:, 318:322] = 0
    Image.fromarray(canvas.permute(1, 2, 0).numpy()).save(path)

    caption = {
        "command": case["command"],
        "embodiment": "panda_omron_mobile__parallel_gripper",
        "duration": "0.8s",
        "fps": 20.0,
        "action_frequency_hz": 20,
        "resolution": {"W": 640, "H": 720},
        "aspect_ratio": "8:9",
        "camera_layout": LAYOUT,
    }
    return {
        "name": f"{case['case_id']}_step{step:05d}",
        "model_mode": "wam",
        "prompt": json.dumps(caption, ensure_ascii=False),
        "vision_path": str(path),
        "fps": 20,
        "action_chunk_size": 16,
        "num_frames": 17,
        "condition_frame_indexes_vision": [0],
        "num_steps": 30,
        "guidance": 2.0,
        "shift": shift,
        "seed": seed,
        "extra": {
            "action128": {
                "action_domain": "panda_omron_mobile/native_control12/robocasa365_source_controller_v1",
                "action_dim_mask": [True] * 12,
                "action_fps": 20,
                "camera_layout": LAYOUT,
                "selected_roles": [role[0] for role in LAYOUT],
                "return_video": False,
            },
            "sft": {
                "state_history": [state.tolist()],
                "state_mask_history": [[True] * 16],
                "state_domain": "panda_omron_mobile/state_eef20_base9/current_canonical_tcp_world_v1",
            },
        },
    }


def policy_actions(http, url, request, checkpoint):
    import numpy as np

    deadline = time.monotonic() + 1200
    while True:
        response = http.post(f"{url}/generate", json=request, timeout=1200)
        if response.status_code != 503 or "Retry-After" not in response.headers or time.monotonic() >= deadline:
            break
        time.sleep(1)
    response.raise_for_status()
    result = response.json()
    if result["status"] != "success":
        raise RuntimeError("WAM inference failed")
    content = result["outputs"][0]["content"]
    meta = content["action128"]
    if (
        meta["checkpoint_path"] != checkpoint
        or meta["weight_variant"] != "ema"
        or meta["action_schema"]["version"] != "robocasa-sft-eval-action-v1"
    ):
        raise ValueError("Unexpected RoboCasa WAM checkpoint or action contract")
    actions = np.array(content["action"], dtype=np.float32)
    if actions.shape != (16, 12) or not np.isfinite(actions).all():
        raise ValueError("Expected finite native Control12 [16,12]")
    return actions


def run_episode(case, directory, environment, http, args):
    import numpy as np
    from robocasa_env_client.client import RemoteError

    case_id = case["case_id"]
    suite = case["match"]["suite"]
    shift = 8.0 if args.protocol == "mixed-shift" and suite != "atomic_seen" else 5.0
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary["status"] == "completed":
            if (summary["case_id"], summary["checkpoint"], summary["shift"]) != (case_id, args.checkpoint, shift):
                raise ValueError(f"{case_id}: an existing result uses another policy or protocol")
            return
    request_id = uuid.uuid4().hex
    directory = directory / request_id
    directory.mkdir(parents=True, exist_ok=False)
    summary = {
        "case_id": case_id,
        "suite": suite,
        "shift": shift,
        "checkpoint": args.checkpoint,
        "weight_variant": "ema",
    }
    observation = None
    try:
        deadline = time.monotonic() + 1800
        while True:
            try:
                observation = environment.create(case_id=case_id, request_id=f"{request_id}-create")
                break
            except RemoteError as exc:
                if exc.status != 429 or time.monotonic() >= deadline:
                    raise
                time.sleep(5)
        chunk = 0
        while not observation.done:
            step = observation.metadata["step"]
            if args.max_control_steps and step >= args.max_control_steps:
                break
            path = directory / f"step_{step:05d}.png"
            request = make_request(observation, case, path, seed=42 + chunk, shift=shift)
            commands = policy_actions(http, args.model_url, request, args.checkpoint)
            remaining = observation.metadata["horizon"] - step
            if args.max_control_steps:
                remaining = min(remaining, args.max_control_steps - step)
            commands = commands[:remaining]
            observation = environment.step(
                observation, commands, request_id=f"{request_id}-step-{step}", return_transitions=True
            )
            applied = observation.arrays["applied_actions"]
            np.testing.assert_array_equal(applied, commands[: len(applied)])
            if len(applied) < len(commands) and not observation.done:
                raise RuntimeError("Unexpected partial control chunk")
            if args.keep_steps:
                np.save(path.with_suffix(".npy"), applied)
                path.with_suffix(".json").write_text(json.dumps(request, indent=2) + "\n")
            else:
                path.unlink()
            chunk += 1
        meta = observation.metadata
        summary.update(
            status="completed" if observation.done else "partial",
            success=meta["success"],
            terminated=meta["terminated"],
            truncated=meta["truncated"],
            steps=meta["step"],
            chunks=chunk,
        )
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        try:
            if observation is not None:
                environment.close(observation.environment_id)
        finally:
            record = directory / "summary.json"
            record.write_text(json.dumps(summary, indent=2) + "\n")
            record.replace(summary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Frozen target50 case and detailed command manifest")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path reported by the WAM server")
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--sdk-zip", type=Path, required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("mixed-shift", "uniform-shift5"), default="mixed-shift")
    parser.add_argument("--case", action="append", help="Run a selected case. Repeat to select several.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-control-steps", type=int, default=0, help="Optional short-run limit, zero uses the full horizon"
    )
    parser.add_argument("--keep-steps", action="store_true")
    args = parser.parse_args()
    if args.max_control_steps < 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid control step limit or shard")
    cases = load_cases(args.cases)
    if args.case and (args.num_shards != 1 or not set(args.case) <= cases.keys()):
        parser.error("--case requires existing case IDs and a single shard")
    selected = [key for key in sorted(cases) if not args.case or key in args.case]
    args.model_url = args.model_url.rstrip("/")
    args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    output = args.output_dir.resolve()
    sys.path.insert(0, str(args.sdk_zip.resolve(strict=True)))

    import requests
    import torch
    from robocasa_env_client.client import Client

    torch.set_num_threads(1)
    environment = Client(args.env_url, token_file=args.token_file, timeout=600)
    environment.health()
    with requests.Session() as http:
        http.trust_env = False
        identity = http.get(f"{args.model_url}/info", timeout=30)
        identity.raise_for_status()
        info = identity.json()
        if (info["checkpoint_path"], info["weight_variant"], info["sft_profile"]) != (
            args.checkpoint,
            "ema",
            "robocasa",
        ):
            raise ValueError("Model server does not match the selected RoboCasa checkpoint")
        for name in selected[args.shard_index :: args.num_shards]:
            run_episode(cases[name], output / name, environment, http, args)


if __name__ == "__main__":
    main()
