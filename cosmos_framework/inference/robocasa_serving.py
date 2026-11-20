# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""H1 RoboCasa State16 → Control12 serving on the current inference backend.

The checkpoint's role registry is authoritative; no weights are rewritten.
Image stitching/JPEG are the client's responsibility and are not repeated here.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from cosmos_framework.inference.action_server import DistributedActionService
from cosmos_framework.inference.action_serving import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
    prepare_wam_request,
)
from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs
from cosmos_framework.inference.robocasa_geometry import STATE_KEY, state128
from cosmos_framework.utils import misc

ACTION_DOMAIN = "panda_omron_mobile/native_control12/robocasa365_source_controller_v1"
STATE_DOMAIN = "panda_omron_mobile/state_eef20_base9/current_canonical_tcp_world_v1"


class StateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state_history: Annotated[list[list[float]], Field(min_length=1, max_length=1)]
    state_mask_history: list[list[bool]] | None = None
    state_domain: str = STATE_DOMAIN


def projection_heads(spec):
    registry = spec["registry"]
    heads = tuple(
        registry["domain_projection_ids"][registry["domains"].index(d)] for d in (ACTION_DOMAIN, STATE_DOMAIN)
    )
    if heads[0] == heads[1] or spec["projection_routing"]["keys"][STATE_KEY] != heads[1]:
        raise ValueError("Distinct saved RoboCasa State/Action heads are required")
    return heads


def action_schema():
    names = [
        "eef_dx_command",
        "eef_dy_command",
        "eef_dz_command",
        "eef_drx_command",
        "eef_dry_command",
        "eef_drz_command",
        "gripper_signed_command",
        "base_x_command",
        "base_y_command",
        "base_yaw_command",
        "torso_command",
        "hybrid_mode_command",
    ]
    meanings = [f"OSC translation input {axis}; source controller frame" for axis in "xyz"]
    meanings += [f"OSC scaled rotation-vector input {axis}; not RPY or rot6d" for axis in "xyz"]
    meanings += [
        "-1 open, +1 close, 0 hold accumulated driver target",
        "base x velocity-controller input; not SI velocity",
        "base y velocity-controller input; not SI velocity",
        "base yaw velocity-controller input; not observed displacement",
        "native torso JOINT_POSITION input; not observed height",
        "native HYBRID_MOBILE_BASE mode input; no thresholding",
    ]
    return {
        "version": "robocasa-sft-eval-action-v1",
        "dimension": 12,
        "index_base": 0,
        "model_slots": list(range(12)),
        "dimensions": [
            {"index": i, "model_slot": i, "name": n, "semantics": m, "unit": "native_controller_input"}
            for i, (n, m) in enumerate(zip(names, meanings, strict=True))
        ],
    }


def prepare_request(overrides, spec, model_config, output_dir, video_key):
    head, state_head = projection_heads(spec)
    extra = dict(overrides.extra or {})
    inputs = StateInput.model_validate(extra.get("sft", {}))
    values = np.asarray(inputs.state_history, dtype=np.float32)
    masks = (
        np.ones_like(values, dtype=bool)
        if inputs.state_mask_history is None
        else np.asarray(inputs.state_mask_history, bool)
    )
    if values.shape != (1, 16) or masks.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("RoboCasa H1 requires finite State16 [1,16]")
    action = dict(extra.get("action128", {}))
    mask = action.get("action_dim_mask")
    if action.get("action_domain") != ACTION_DOMAIN or inputs.state_domain != STATE_DOMAIN:
        raise ValueError("Use the saved RoboCasa Control12 and State domains")
    if not isinstance(mask, list) or len(mask) != 12 or not all(isinstance(v, bool) for v in mask) or not any(mask):
        raise ValueError("Supply 12 native-order action mask booleans")
    if overrides.fps not in (None, 20) or action.get("action_fps", 20) != 20:
        raise ValueError("RoboCasa requires 20Hz video/actions")
    if overrides.action_chunk_size not in (None, 16) or overrides.num_frames not in (None, 17):
        raise ValueError("H1 requires 16 future actions and 17 video frames")
    if overrides.raw_action_dim not in (None, 12):
        raise ValueError("RoboCasa external controls have 12 dimensions")
    states, valid = state128(values, masks)
    if inputs.state_mask_history is None and not valid[:, list(range(10)) + list(range(96, 105))].all():
        raise ValueError("Invalid quaternion or gripper geometry")
    transform = spec["transforms"][spec["registry"]["domains"].index(STATE_DOMAIN)]
    state = torch.as_tensor(states).clone()
    active = torch.tensor(transform["normalization_mask"], dtype=torch.bool)
    scale, bias = (torch.tensor(transform[k], dtype=torch.float32) for k in ("scale", "bias"))
    state[:, active] = state[:, active] * scale[active] + bias[active]
    if transform.get("forward_clamp", True):
        state[:, active] = state[:, active].clamp(transform["clip_min"], transform["clip_max"])
    state = state.masked_fill(~torch.tensor(valid), 0)
    base = overrides.model_copy(deep=True)
    action.update(action_dim_mask=mask + [False] * 116, action_fps=20)
    base.extra = {**extra, "action128": action}
    base.raw_action_dim = None
    base.fps, base.action_chunk_size, base.num_frames = 20, 16, 17
    prepared, batch = prepare_wam_request(base, spec, model_config, output_dir, video_key)
    from cosmos_framework.inference.vision import read_media_frames

    frames, _ = read_media_frames(Path(prepared.sample_args["vision_path"]), max_frames=10)
    if frames.shape != (3, 1, 720, 640):
        raise ValueError("Supply exactly one training-size 640x720 camera canvas")
    # Preserve explicit JSON language fields; generic serving adds task_title.
    original = json.loads(overrides.prompt)
    if not isinstance(original, dict) or not isinstance(original.get("command"), str):
        raise ValueError("Supply a JSON caption containing the complete command")
    caption = json.loads(batch["ai_caption"][0])
    if "task_title" not in original:
        caption.pop("task_title", None)
    batch["ai_caption"] = [json.dumps(caption, ensure_ascii=False)]
    targets = torch.zeros(17, 128)
    targets[:1] = state
    dims = torch.zeros(17, 128, dtype=torch.bool)
    dims[:1] = torch.tensor(valid)
    dims[1:] = torch.tensor(prepared.action_dim_mask)
    ids = torch.full((17,), head, dtype=torch.long)
    ids[0] = state_head
    batch["action"], batch["action_dim_mask"], batch["action_token_domain_ids"] = [[targets]], [dims], [ids]
    # State has valid base coordinates at 96:105. Current inference intersects
    # the slot mask with raw_action_dim, so keep the INTERNAL width at 128;
    # only the HTTP output boundary selects native Control12.
    batch.update(
        make_batched_action_processing_fields(ActionProcessingRecord(raw_action_dim=128, action_normalizer=None), 1)
    )
    plan = batch["sequence_plan"][0]
    plan.condition_frame_indexes_action = [0]
    plan.robot_state_frame_indexes = [0]
    plan.action_timestamps_seconds = [0.0, *(torch.arange(16, dtype=torch.float64) / 20).tolist()]
    prepared.sample_args["raw_action_dim"] = 128
    prepared.sample_args["extra"]["action128"]["action_dim_mask"] = mask
    return prepared, batch


class RoboCasaService(DistributedActionService):
    def __init__(self, pipe, group, info):
        super().__init__(pipe, group, info)
        projection_heads(self.spec)
        info.update(sft_profile="robocasa", action_schema=action_schema(), source_file=str(Path(__file__).resolve()))
        if info.get("weight_variant"):
            probes = []
            for name, parameter in self.model.net.named_parameters():
                if 64 <= parameter.numel() <= 65536:
                    value = parameter.detach()
                    if hasattr(value, "full_tensor"):
                        value = value.full_tensor()
                    value = value.float().cpu().contiguous()
                    probes.append(
                        {
                            "name": name,
                            "shape": list(value.shape),
                            "dtype": str(parameter.dtype),
                            "fp32_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                        }
                    )
                    if len(probes) == 3:
                        break
            info["loaded_parameter_probes"] = probes
        if info.get("compile"):
            misc.set_torch_compile_options(recompile_limit=256, use_duck_shape=False)

    def prepare(self, request, output_dir):
        return prepare_request(request, self.spec, self.model.config, output_dir, self.video_key)

    def execute(self, request, batch):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        started = time.monotonic()
        batch = misc.to(batch, device="cuda")
        sampling = {k: request.sample_args[k] for k in ("guidance_interval", "normalize_cfg", "sigma_max")}
        sampling["has_negative_prompt"] = "neg_ai_caption" in batch
        for key in ("reuse_small_action_und", "reuse_small_action_metadata"):
            if self.info.get(key):
                sampling[key] = True
        with torch.inference_mode():
            samples = self.model.generate_samples_from_batch(
                batch,
                seed=[request.seed],
                num_steps=request.num_steps,
                guidance=request.guidance,
                shift=request.shift,
                **sampling,
            )
            if self.rank != 0:
                return None
            # First token is clean State. Native Control12 is already in controller units.
            physical = samples["action"][0].float()[1:, :12]
            physical = physical.masked_fill(~torch.tensor(request.action_dim_mask[:12], device=physical.device), 0)
            if physical.shape != (16, 12) or not torch.isfinite(physical).all():
                raise RuntimeError("Expected finite native Control12 [16,12]")
        content = {
            "normalized_action": physical.cpu().tolist(),
            "action_units": "native_controller_input",
            "action_domain": request.action_domain,
            "action_dim_mask": request.action_dim_mask[:12],
            "action_fps": request.action_fps,
            "action_timestamps_seconds": batch["sequence_plan"][0].action_timestamps_seconds[1:],
            "state_history_length": 1,
            "seed": request.seed,
            "checkpoint_path": self.info["checkpoint_path"],
            "weight_variant": self.info.get("weight_variant", "online"),
            "conditioning_caption_sha256": hashlib.sha256(batch["ai_caption"][0].encode()).hexdigest(),
            "action_schema": action_schema(),
            "control_range_check": {
                "expected": [-1, 1],
                "outside_count": int((physical.abs() > 1).sum().item()),
                "max_abs": float(physical.abs().max().item()),
                "clipped": False,
            },
        }
        directory = Path(request.sample_args["output_dir"])
        files = []
        if request.return_video:
            from cosmos_framework.tools.visualize.video import save_img_or_video

            with torch.inference_mode():
                video = ((self.model.decode(samples["vision"][0]) + 1) / 2).clamp(0, 1).squeeze(0)
            save_img_or_video(
                video,
                str(directory / "vision"),
                fps=request.sample_args["fps"],
                quality=request.sample_args["video_save_quality"],
            )
            files.append((directory / "vision.mp4").relative_to(Path(self.info["output_dir"])))
        content["elapsed_seconds"] = time.monotonic() - started
        result = SampleOutputs(
            args=request.sample_args,
            outputs=[SampleOutput(content={"action": physical.cpu().tolist(), "action128": content}, files=files)],
        )
        (directory / "sample_args.json").write_text(json.dumps(request.sample_args))
        (directory / "sample_outputs.json").write_text(result.model_dump_json())
        return result
