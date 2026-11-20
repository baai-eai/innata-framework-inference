# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Shared State/history preparation for RoboCasa, RoboDojo and LIBERO SFT.

No dataset reads or checkpoint-time parameter transformations. The loaded model
contract is authoritative for both role-specific normalization and projection IDs.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from cosmos_framework.data.generator.action.robocasa_state import STATE_KEY, state128
from cosmos_framework.inference.action_serving import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
    prepare_wam_request,
)
from cosmos_framework.inference.vision import read_media_frames

ROBODOJO_CAPTION_VERSION = "train_field_order_v1"
ROBODOJO_CAPTION_FIELDS = (
    "command",
    "embodiment",
    "duration",
    "fps",
    "action_frequency_hz",
    "resolution",
    "aspect_ratio",
    "camera_layout",
    "plan_history",
    "current_plan",
)


def sft_action_schema(profile):
    """Versioned evaluator order; indices are zero-based, never inferred from masks."""
    if profile in {"libero", "libero_plus"}:
        return {
            "version": "libero-osc-action7-v1",
            "dimension": 7,
            "model_control_dimension": 10,
            "index_base": 0,
            "model_slots": list(range(10)),
            "dimensions": [
                {"index": i, "name": name, "semantics": meaning, "unit": unit}
                for i, (name, meaning, unit) in enumerate(
                    [
                        *[
                            (
                                f"eef_d{axis}_command",
                                "native OSC translation input; no additional gain",
                                "controller_input",
                            )
                            for axis in "xyz"
                        ],
                        *[
                            (f"eef_dr{axis}_command", "native OSC rotation-vector input; decoded from rot6d", "rotvec")
                            for axis in "xyz"
                        ],
                        ("gripper_command", "-1 open, +1 close; threshold model openness at >0.5", "signed_command"),
                    ]
                )
            ],
        }
    if profile == "robodojo":
        slots = [0, 1, 2, 3, 4, 5, 14, 7, 8, 9, 10, 11, 12, 15]
        names = [
            *[f"left_joint_{i}_absolute_target" for i in range(1, 7)],
            "left_gripper_openness",
            *[f"right_joint_{i}_absolute_target" for i in range(1, 7)],
            "right_gripper_openness",
        ]
        semantics = ["next sampled absolute joint-position target"] * 6 + ["absolute openness: 0 closed, 1 open"]
        semantics *= 2
        units = (["rad"] * 6 + ["openness_0_1"]) * 2
    elif profile == "robocasa":
        slots = list(range(12))
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
        semantics = [f"OSC translation input {axis}; source controller frame" for axis in ("x", "y", "z")]
        semantics += [f"OSC scaled rotation-vector input {axis}; not RPY or rot6d" for axis in ("x", "y", "z")]
        semantics += [
            "-1 open, +1 close, 0 hold accumulated driver target",
            "base x velocity-controller input; not SI velocity",
            "base y velocity-controller input; not SI velocity",
            "base yaw velocity-controller input; not observed displacement",
            "native torso JOINT_POSITION input; not observed height",
            "native HYBRID_MOBILE_BASE mode input; no thresholding",
        ]
        units = ["native_controller_input"] * 12
    else:
        raise ValueError(f"Unknown SFT output profile: {profile}")
    return {
        "version": f"{profile}-sft-eval-action-v1",
        "dimension": len(slots),
        "index_base": 0,
        "model_slots": slots,
        "dimensions": [
            {"index": i, "model_slot": slot, "name": name, "semantics": meaning, "unit": unit}
            for i, (slot, name, meaning, unit) in enumerate(zip(slots, names, semantics, units, strict=True))
        ],
    }


class SFTInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state_history: Annotated[list[list[float]], Field(min_length=1, max_length=5)]
    state_mask_history: list[list[bool]] | None = None
    state_domain: str | None = None


def validate_profile(spec, profile):
    registry = spec["registry"]
    if profile in {"libero", "libero_plus"}:
        from cosmos_framework.inference.libero_sft import ACTION_DOMAIN, ACTION_KEY, STATE_DOMAIN, STATE_KEY

        routing = spec.get("projection_routing", {}).get("keys", {})
        if (
            ACTION_DOMAIN not in registry["domains"]
            or STATE_DOMAIN not in registry["domains"]
            or routing.get(ACTION_KEY) != 34
            or routing.get(STATE_KEY) != 35
            or registry["domain_projection_ids"][registry["domains"].index(ACTION_DOMAIN)] != 34
            or registry["domain_projection_ids"][registry["domains"].index(STATE_DOMAIN)] != 35
        ):
            raise ValueError("LIBERO SFT requires its trained Control10 head34 and canonical State head35")
    elif profile == "robocasa":
        if 32 not in registry["domain_projection_ids"] or 33 not in registry["domain_projection_ids"]:
            raise ValueError("RoboCasa SFT requires Control12 head32 and State head33 in the saved checkpoint")
    elif profile == "robodojo":
        candidates = [
            d
            for d, h in zip(registry["domains"], registry["domain_projection_ids"], strict=True)
            if h == 7 and d.endswith("/state/current_position")
        ]
        if not candidates:
            raise ValueError(
                "RoboDojo SFT requires the saved head7 Joint State normalization domain; not a pretrain checkpoint"
            )
    else:
        raise ValueError(f"Unknown SFT profile: {profile}")


def role_transform(values, mask, spec, domain):
    """Apply exactly the fixed saved affine transform once, independently per role."""
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    value = torch.as_tensor(values, dtype=torch.float32).clone()
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if not torch.isfinite(value[mask]).all():
        raise ValueError("State history contains nonfinite valid values")
    active = torch.tensor(transform["normalization_mask"], dtype=torch.bool)
    scale, bias = (torch.tensor(transform[k], dtype=torch.float32) for k in ("scale", "bias"))
    value[:, active] = value[:, active] * scale[active] + bias[active]
    if transform.get("forward_clamp", True):
        value[:, active] = value[:, active].clamp(transform["clip_min"], transform["clip_max"])
    return value.masked_fill(~mask, 0), mask


def prepare_sft_wam_request(overrides, spec, model_config, output_dir, video_key, profile):
    validate_profile(spec, profile)
    extra = dict(overrides.extra or {})
    inputs = SFTInputs.model_validate(extra.get("sft", {}))
    action = dict(extra.get("action128", {}))
    domain = action.get("action_domain")
    registry = spec["registry"]
    if domain not in registry["domains"]:
        raise ValueError("Unknown SFT action_domain")
    index = registry["domains"].index(domain)
    head = registry["domain_projection_ids"][index]
    libero = profile in {"libero", "libero_plus"}
    width, rate, expected_head = (9, 20, 34) if libero else ((16, 20, 32) if profile == "robocasa" else (14, 25, 7))
    values = np.asarray(inputs.state_history, dtype=np.float32)
    if values.shape != (len(inputs.state_history), width):
        raise ValueError(f"{profile} requires State history [1,{width}]")
    masks = (
        np.ones_like(values, dtype=bool)
        if inputs.state_mask_history is None
        else np.asarray(inputs.state_mask_history, bool)
    )
    if masks.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Invalid State history values/mask geometry")
    if head != expected_head or domain.endswith("/state/current_position"):
        raise ValueError(f"{profile} must use its Action domain, projection head{expected_head}")
    if libero:
        from cosmos_framework.inference.libero_sft import ACTION_DOMAIN

        if domain != ACTION_DOMAIN:
            raise ValueError("Use the trained LIBERO native Control10 domain, not a shared EEF domain")
    slots = sft_action_schema(profile)["model_slots"]
    native_action_mask = action.get("action_dim_mask")
    if (
        not isinstance(native_action_mask, list)
        or len(native_action_mask) != len(slots)
        or not all(isinstance(v, bool) for v in native_action_mask)
    ):
        raise ValueError(f"{profile} action_dim_mask must contain {len(slots)} native-order booleans")
    if libero and (
        not all(native_action_mask)
        or action.get("selected_roles") != ["auxiliary_0", "left_wrist"]
        or action.get("camera_layout") != [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    ):
        raise ValueError("LIBERO requires all 10 controls and the explicit main-above/wrist-below training layout")
    expanded_mask = [False] * 128
    for slot, active in zip(slots, native_action_mask, strict=True):
        expanded_mask[slot] = active
    action["action_dim_mask"] = expanded_mask
    if overrides.fps not in (None, rate) or action.get("action_fps", rate) != rate:
        raise ValueError(f"{profile} SFT requires {rate}Hz video and actions")
    n = len(values)
    if n not in (1, 5) or (n == 5 and profile != "robodojo"):
        raise ValueError("This run supports RoboDojo history5 or the original history1")
    total = n + 16
    vision_conditions = list(range((n - 1) // 4 + 1))
    if overrides.action_chunk_size not in (None, 16) or overrides.num_frames not in (None, total):
        raise ValueError("SFT horizon is 16 actions and 17 total video frames (1 current +16 future)")
    if overrides.condition_frame_indexes_vision not in (None, vision_conditions):
        raise ValueError("SFT conditions latent frame [0]")
    if libero:
        from cosmos_framework.inference.libero_sft import STATE_DOMAIN, state9_to_state128

        states, state_masks = state9_to_state128(values, masks)
        candidates, state_head = [STATE_DOMAIN], 35
    elif profile == "robocasa":
        states, state_masks = state128(values, masks)
        if inputs.state_mask_history is None and not state_masks[:, list(range(10)) + list(range(96, 105))].all():
            raise ValueError("Invalid RoboCasa State16 quaternion or gripper geometry")
        candidates = [d for d, h in zip(registry["domains"], registry["domain_projection_ids"], strict=True) if h == 33]
        action.update(state16=values[-1].tolist(), state_mask16=masks[-1].tolist())
        state_head = spec["projection_routing"]["keys"][STATE_KEY]
    else:
        # Native [left joints6, left gripper, right joints6, right gripper].
        # Expand before applying the saved 128D State normalization parameters.
        states = np.zeros((len(values), 128), dtype=np.float32)
        state_masks = np.zeros((len(values), 128), dtype=bool)
        states[:, slots], state_masks[:, slots] = values, masks
        candidates = [d for d in registry["domains"] if d == domain + "/state/current_position"]
        state_head = head
    state_domain = inputs.state_domain
    if state_domain is None:
        if len(candidates) != 1:
            raise ValueError("Specify the saved State normalization domain in extra.sft.state_domain")
        state_domain = candidates[0]
    if state_domain not in candidates:
        raise ValueError("State domain does not match the selected SFT embodiment/role")
    states, state_masks = role_transform(states, state_masks, spec, state_domain)
    # Reuse original Cosmos request/default/prompt/media-path handling. Expand
    # the prepared short horizon below without changing the pretrain endpoint.
    base = overrides.model_copy(deep=True)
    if libero:
        if overrides.raw_action_dim not in (None, 10):
            raise ValueError("LIBERO model Control10 raw_action_dim must be 10 (environment action is 7D)")
        # The generic pretrain request uses 128D; specialize after its shared
        # caption/media handling, just as the SFT State prefix is added below.
        base.raw_action_dim = None
    action["action_fps"] = rate
    base.extra = {**extra, "action128": action}
    base.fps, base.action_chunk_size, base.num_frames = rate, 16, None
    base.condition_frame_indexes_vision = [0]
    # Transport is a lossless vertical strip of chronologically ordered canvases.
    # Generic preparation sees one correctly sized image; the existing batch is
    # then expanded to the exact training history and future horizon below.
    if n == 5:
        import uuid

        from PIL import Image

        strip = np.asarray(Image.open(base.vision_path).convert("RGB"))
        if strip.shape != (720 * n, 640, 3):
            raise ValueError("History5 requires five RGB 640x720 canvases")
        frames = torch.from_numpy(strip.reshape(n, 720, 640, 3).copy()).permute(3, 0, 1, 2)
        media_fps = rate
        output_dir.mkdir(parents=True, exist_ok=True)
        first = output_dir / ("history-first-" + uuid.uuid4().hex + ".png")
        Image.fromarray(strip[:720]).save(first)
        base.vision_path = str(first)
    prepared, batch = prepare_wam_request(base, spec, model_config, output_dir, video_key)
    if profile == "robodojo":
        caption = json.loads(batch["ai_caption"][0])
        if not isinstance(caption, dict) or not {"command", "plan_history", "current_plan"} <= caption.keys():
            raise ValueError("RoboDojo requires command, plan_history, and current_plan in its caption JSON")
        ordered = {key: caption[key] for key in ROBODOJO_CAPTION_FIELDS if key in caption}
        ordered.update((key, value) for key, value in caption.items() if key not in ordered)
        batch["ai_caption"][0] = json.dumps(ordered, ensure_ascii=False)
    if n == 1:
        frames, media_fps = read_media_frames(Path(prepared.sample_args["vision_path"]), max_frames=10)
    n = len(values)
    if frames.shape[1] != n:
        raise ValueError("vision_path must contain the same single current frame as state_history")
    if n > 1 and not np.isclose(media_fps, rate):
        raise ValueError(f"History clip must be encoded at {rate} FPS")
    height, target_width = batch[video_key][0][0].shape[-2:]
    if tuple(frames.shape[-2:]) != (height, target_width):
        raise ValueError("Supply history canvas at its training size; SFT does not resize historical media")
    if libero and (height, target_width) != (960, 640):
        raise ValueError("LIBERO SFT requires the 640x960 main-above/wrist-below training canvas")
    canvas = torch.zeros(3, total, height, target_width, dtype=torch.uint8)
    canvas[:, :n] = frames
    targets = torch.zeros(total, 128)
    targets[:n] = states
    dims = torch.zeros(total, 128, dtype=torch.bool)
    dims[:n] = state_masks
    dims[n:] = torch.tensor(prepared.action_dim_mask)
    ids = torch.full((total,), head, dtype=torch.long)
    ids[:n] = state_head
    plan = batch["sequence_plan"][0]
    plan.condition_frame_indexes_vision = vision_conditions
    plan.condition_frame_indexes_action = list(range(n))
    plan.robot_state_frame_indexes = list(range(n))
    target_times = (torch.arange(16, dtype=torch.float64) + n - 1 + (1 if profile == "robodojo" else 0)) / rate
    plan.action_timestamps_seconds = [*(torch.arange(n, dtype=torch.float64) / rate).tolist(), *target_times.tolist()]
    batch[video_key] = [[canvas]]
    batch["action"], batch["action_dim_mask"], batch["action_token_domain_ids"] = [[targets]], [dims], [ids]
    if libero:
        batch.update(
            # Keep model width through Action128 inverse normalization. The
            # serving boundary selects Control10 and converts to native action7
            # afterwards; action_dim_mask still restricts valid target slots.
            make_batched_action_processing_fields(ActionProcessingRecord(raw_action_dim=128, action_normalizer=None), 1)
        )
        prepared.sample_args["raw_action_dim"] = 128
    prepared.state_history_length = n
    prepared.sample_args.update(
        num_frames=total, condition_frame_indexes_vision=vision_conditions, vision_path=str(overrides.vision_path)
    )
    prepared.sample_args["extra"]["action128"]["action_dim_mask"] = native_action_mask
    return prepared, batch
