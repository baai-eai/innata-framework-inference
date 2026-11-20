# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Generate the Innata-0-ACWM WorldArena Track 1 evaluation videos."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from cosmos_framework.inference.release_models import ACWM_CHECKPOINT_NAME


def load_episodes(path):
    episodes = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        episode = row["episode"]
        if not isinstance(episode, str) or episode in episodes:
            raise ValueError(f"Invalid or duplicate episode: {episode!r}")
        episodes[episode] = row
    expected = {f"episode{i}" for i in range(1, 1001)}
    if episodes.keys() != expected:
        raise ValueError("WorldArena manifest must cover episode1 through episode1000 exactly")
    return [episodes[f"episode{i}"] for i in range(1, 1001)]


def run_episode(row, data, output, pipe, setup, overwrite):
    import av
    import numpy as np
    from PIL import Image

    from cosmos_framework.inference.args import OmniSampleOverrides

    name = row["episode"]
    destination = output / f"{name}.mp4"
    count = row["job_num_frames"]
    actions = np.load(data / "actions" / f"{name}.npy", allow_pickle=False)
    if not isinstance(count, int) or count < 4 or actions.ndim != 2 or actions.shape[1] != 20 or len(actions) < count:
        raise ValueError(f"{name}: expected {count} RoboTwin absolute actions with 20 values per frame")
    actions = actions[:count].astype(np.float32, copy=False)
    if not np.isfinite(actions).all():
        raise ValueError(f"{name}: actions must be finite")
    generated = count - count % 4
    if destination.is_file() and not overwrite:
        with av.open(destination) as video:
            stream = video.streams.video[0]
            if (stream.width, stream.height, stream.average_rate, stream.frames) != (736, 544, 30, generated + 1):
                raise ValueError(
                    f"Existing {destination} has unexpected video metadata. Use --overwrite to replace it."
                )
        return

    first_frame = data / "first_frames" / f"{name}.png"
    with Image.open(first_frame) as image:
        condition = np.asarray(image.convert("RGB"))
    if condition.shape != (544, 736, 3):
        raise ValueError(f"{name}: expected a prepared 736x544 first frame")

    # The reference rollout conditions each chunk on the previous encoded
    # chunk's final RGB frame. Keep the intermediate H.264 round trip.
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output) as directory:
        scratch = Path(directory)
        encoded = scratch / destination.name
        with av.open(encoded, "w") as video:
            stream = video.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 736, 544, "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            video.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(condition), format="rgb24")))
            for start in range(0, generated, 300):
                block = actions[start : min(start + 300, generated)]
                chunk_name = f"{name}__c{start // 300:03d}"
                observation = scratch / f"{chunk_name}.png"
                action_path = scratch / f"{chunk_name}.json"
                Image.fromarray(condition).save(observation)
                action_path.write_text(json.dumps(block.tolist()))
                request = OmniSampleOverrides(
                    name=chunk_name,
                    model_mode="forward_dynamics",
                    prompt=row["prompt"],
                    vision_path=str(observation),
                    action_path=str(action_path),
                    action_chunk_size=len(block),
                    domain_name="robotwin_ee_abs",
                    view_point="third_person_view",
                    fps=16,
                    image_size=480,
                    num_steps=30,
                    guidance=1.0,
                    shift=5.0,
                    seed=0,
                    video_save_quality=8,
                )
                request.output_dir = setup.output_dir / chunk_name
                sample = request.build_sample(model_config=pipe.model_config)
                outputs = pipe.generate([sample])
                if not outputs or outputs[0].status != "success":
                    raise RuntimeError(f"{chunk_name}: generation failed")
                with av.open(request.output_dir / "vision.mp4") as chunk:
                    frames = iter(chunk.decode(video=0))
                    next(frames)
                    frames_written = 0
                    for frame in frames:
                        condition = frame.to_ndarray(format="rgb24")
                        if condition.shape != (544, 736, 3):
                            raise ValueError(f"{chunk_name}: the decoded chunk must use the benchmark canvas")
                        video.mux(
                            stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(condition), format="rgb24"))
                        )
                        frames_written += 1
                    if frames_written != len(block):
                        raise ValueError(f"{chunk_name}: got {frames_written} generated frames, expected {len(block)}")
                shutil.rmtree(request.output_dir)
            video.mux(stream.encode())
        os.replace(encoded, destination)
    print(f"{destination.name}: {generated + 1} frames at 30 fps", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path", type=Path, required=True, help=f"Local {ACWM_CHECKPOINT_NAME} Hugging Face directory"
    )
    parser.add_argument("--vae-path", type=Path, required=True, help="Local Wan2.2_VAE.pth")
    parser.add_argument("--inputs", type=Path, required=True, help="Prepared first_frames/, actions/ and prompts JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", action="append", help="Run only this episode. Repeat for multiple smoke cases.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be between 0 and num-shards - 1")
    episodes = load_episodes(args.inputs / "prompts-1000.jsonl")
    if args.episode:
        if args.num_shards != 1:
            parser.error("--episode cannot be combined with sharding")
        selected = set(args.episode)
        episodes = [row for row in episodes if row["episode"] in selected]
        if {row["episode"] for row in episodes} != selected:
            parser.error("Unknown WorldArena episode")
    else:
        episodes = episodes[args.shard_index :: args.num_shards]
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run one standalone process per GPU and distribute episodes with --shard-index")

    from cosmos_framework.inference.common.init import init_script

    init_script(env={"COSMOS_TRAINING": "0", "COSMOS_INTERNAL": "0"})

    from cosmos_framework.inference.args import OmniSetupOverrides

    checkpoint = args.checkpoint_path.resolve(strict=True)
    vae = args.vae_path.resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("innata_acwm_profile") != "a115" or not vae.is_file():
        parser.error(f"Use the released {ACWM_CHECKPOINT_NAME} checkpoint and a local Wan2.2 VAE checkpoint")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".chunks-", dir=output) as stage:
        setup = OmniSetupOverrides(
            checkpoint_path=str(checkpoint),
            output_dir=Path(stage),
            guardrails=False,
            use_torch_compile=True,
            experiment_overrides=[
                "model.config.tokenizer.bucket_name=''",
                f"model.config.tokenizer.vae_path={json.dumps(str(vae))}",
            ],
        ).build_setup()
        pipe = setup.get_inference_cls().create(setup)
        for episode in episodes:
            run_episode(episode, args.inputs, output, pipe, setup, args.overwrite)


if __name__ == "__main__":
    main()
