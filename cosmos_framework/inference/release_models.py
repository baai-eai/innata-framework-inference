# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Inference adapters for the published Innata HF checkpoints."""

import json
import os
from pathlib import Path

UNDERSTANDING_CHECKPOINT_NAME = "Innata-0-Understanding"
ACWM_CHECKPOINT_NAME = "Innata-0-ACWM"
WAM_CHECKPOINT_NAMES = {
    "robodojo": "Innata-0-WAM-RoboDojo",
    "robocasa": "Innata-0-WAM-RoboCasa365",
}


def read_hf_config(checkpoint) -> dict:
    if not checkpoint:
        return {}
    config = Path(checkpoint).expanduser() / "config.json"
    return json.loads(config.read_text()) if config.is_file() else {}


def generate_understanding(checkpoint, samples, output_dir: Path) -> None:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs

    requested = os.environ.get("COSMOS_DEVICE")
    device = torch.device(requested) if requested else torch.accelerator.current_accelerator(check_available=True)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Understanding inference requires a single process")
    if device is None or not torch.get_device_module(device).is_available():
        raise ValueError("Understanding inference requires an available accelerator or COSMOS_DEVICE=cpu")
    if not samples or any(s.sample_meta.model_mode.value != "reasoner" for s in samples):
        raise ValueError(f"{UNDERSTANDING_CHECKPOINT_NAME} accepts reasoner requests")

    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    for sample in samples:
        text_content = {"type": "text", "text": sample.prompt}
        if sample.image_prompt_prefix is not None and sample.image_paths is None:
            raise ValueError("'image_prompt_prefix' requires 'image_paths'")
        if sample.image_paths is not None:
            if not sample.image_paths or sample.vision_path:
                raise ValueError("Specify a non-empty 'image_paths' list or 'vision_path'")
            content = [{"type": "image", "image": str(path)} for path in sample.image_paths]
            if sample.image_prompt_prefix:
                content.insert(0, {"type": "text", "text": sample.image_prompt_prefix})
            content.append(text_content)
        elif sample.vision_path:
            path = str(sample.vision_path)
            media_type = "video" if Path(path).suffix.lower() in {".mp4", ".mov", ".webm", ".avi", ".mkv"} else "image"
            content = [{"type": media_type, media_type: path}, text_content]
        else:
            content = [text_content]
        messages = [{"role": "user", "content": content}]
        if content[0]["type"] == "video":
            from transformers.video_utils import load_video

            if sample.num_frames is not None and sample.video_fps is not None:
                raise ValueError("Specify num_frames or video_fps for a video request, not both")
            frames, metadata = load_video(
                content[0]["video"],
                backend="pyav",
                sample_indices_fn=lambda metadata, **_: processor.video_processor.sample_frames(
                    metadata, num_frames=sample.num_frames, fps=sample.video_fps
                ),
            )
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[prompt], videos=[frames], video_metadata=[metadata], do_sample_frames=False, return_tensors="pt"
            ).to(device)
        else:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=sample.max_new_tokens or 512)
        text = processor.batch_decode(generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
        target = output_dir / sample.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "sample_outputs.json").write_text(
            SampleOutputs(
                args=sample.model_dump(mode="json"), outputs=[SampleOutput(content={"text": text})]
            ).model_dump_json()
            + "\n"
        )


def generate_wam(profile, pipe, samples, output_dir: Path, checkpoint: str) -> None:
    from cosmos_framework.inference.action_server import DistributedActionService

    info = {"checkpoint_path": checkpoint, "output_dir": str(output_dir), "weight_variant": "ema"}
    if profile == "robocasa":
        from cosmos_framework.inference.robocasa_serving import RoboCasaService

        service = RoboCasaService(pipe, None, info)
    elif profile == "robodojo":
        service = DistributedActionService(pipe, None, {**info, "sft_profile": profile})
    else:
        raise ValueError(f"Unknown release WAM profile: {profile!r}")
    for sample in samples:
        request, batch = service.prepare(sample, output_dir)
        result = service.execute(request, batch)
        if result is None:
            continue
        target = output_dir / sample.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "sample_outputs.json").write_text(result.model_dump_json() + "\n")
