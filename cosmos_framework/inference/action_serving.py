# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action128 WAM HTTP contracts and checkpoint resolution (no model allocation)."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import random
import re
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from cosmos_framework.data.generator.action.unified_action128 import (
    _copy_resized_camera,
    render_caption,
    stable_hash,
    view_layout,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides
from cosmos_framework.inference.common.args import SampleOutputs
from cosmos_framework.inference.vision import read_media_frames

if TYPE_CHECKING:
    from fastapi import FastAPI


class CameraObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str = Field(pattern=r"^(main|left_wrist|right_wrist|auxiliary_[a-zA-Z0-9_]+)$")
    image_base64: str = Field(min_length=1, max_length=12_000_000)
    fps: float = Field(gt=0, le=240, allow_inf_nan=False)


class PreparedAction128Request(BaseModel):
    """Internal rank-broadcast plan, never an HTTP request schema."""

    model_config = ConfigDict(extra="forbid")
    action_domain: str = Field(min_length=1, max_length=256)
    task_title: str = Field(min_length=1, max_length=512)
    prompt: str = Field(min_length=1, max_length=16000)
    cameras: list[CameraObservation] = Field(default_factory=list)
    action_fps: float = Field(gt=0, le=512, allow_inf_nan=False)
    duration_seconds: float = Field(default=2.0, gt=0, le=2.0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    num_steps: int = Field(default=35, ge=1)
    guidance: float = Field(default=1.5, ge=0, le=20, allow_inf_nan=False)
    shift: float = Field(default=5.0, gt=0, le=100, allow_inf_nan=False)
    sample_args: dict[str, Any] = Field(default_factory=dict)
    return_video: bool = True
    # This is a caller-owned slot mask, not a guess based on normalization_mask.
    action_dim_mask: Annotated[list[bool], Field(min_length=128, max_length=128)]
    state16: Annotated[list[float], Field(min_length=16, max_length=16)] | None = None
    state_mask16: Annotated[list[bool], Field(min_length=16, max_length=16)] | None = None
    state_history_length: int = Field(default=0, ge=0, le=9)


def resolve_checkpoint(root: Path) -> tuple[Path, Path]:
    """Resolve a run/output/checkpoints directory or an explicit completed iter/model.

    Only trainer-published latest pointers or .save_complete markers establish
    completion. Never pick the numerically largest possibly in-flight directory.
    """
    root = root.expanduser().resolve()
    if root.name == "model" and (root / ".metadata").is_file():
        root = root.parent
    if re.fullmatch(r"iter_\d+", root.name):
        checkpoints = root.parent
        pointer = checkpoints / "latest_checkpoint.txt"
        if not (root / ".save_complete").is_file() and (
            not pointer.is_file() or pointer.read_text().strip() != root.name
        ):
            raise ValueError(f"Checkpoint has no completion evidence: {root}")
    else:
        pointers = [root / "latest_checkpoint.txt", root / "checkpoints/latest_checkpoint.txt"]
        matches = [p for p in pointers if p.is_file()]
        if not matches:
            matches = list(root.glob("**/checkpoints/latest_checkpoint.txt"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one training run under {root}, found {len(matches)} latest pointers")
        pointer = matches[0]
        name = pointer.read_text().strip()
        if not re.fullmatch(r"iter_\d+", name):
            raise ValueError(f"Invalid latest checkpoint pointer: {pointer}")
        checkpoints = pointer.parent
        root = checkpoints / name
    model = root / "model"
    config = checkpoints.parent / "config.yaml"
    if not (model / ".metadata").is_file() or not any(model.glob("*.distcp")) or not config.is_file():
        raise ValueError(f"Missing DCP model or saved config for {root}")
    return model, config


def decode_camera(camera: CameraObservation) -> np.ndarray:
    try:
        encoded = base64.b64decode(camera.image_base64, validate=True)
        with Image.open(io.BytesIO(encoded)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Camera image exceeds 16 million pixels")
            return np.asarray(image.convert("RGB")).copy()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Invalid base64 PNG/JPEG for camera {camera.role}") from exc


def build_batch(request: PreparedAction128Request, spec: dict, *, video_key: str = "video") -> dict[str, Any]:
    registry = spec["registry"]
    try:
        domain = registry["domains"].index(request.action_domain)
    except ValueError as exc:
        raise ValueError(f"Unknown action_domain: {request.action_domain}; see GET /domains") from exc
    roles = [camera.role for camera in request.cameras]
    if len(set(roles)) != len(roles):
        raise ValueError("Camera roles must be unique")
    if not any(request.action_dim_mask):
        raise ValueError("action_dim_mask must contain an active slot")
    images = {camera.role: decode_camera(camera) for camera in request.cameras}
    # The dataset's view selection and exact canvas geometries are authoritative.
    selected, boxes, width, height = view_layout(
        {
            "video_views": [
                {
                    "role": c.role,
                    "source_key": c.role,
                    "width": images[c.role].shape[1],
                    "height": images[c.role].shape[0],
                }
                for c in request.cameras
            ],
            "view_mapping": {role: role for role in roles},
        }
    )
    video_fps = min(30.0, *(c.fps for c in request.cameras))
    frames = 4 * math.ceil(math.ceil(request.duration_seconds * video_fps - 1e-8) / 4) + 1
    steps = math.ceil(request.duration_seconds * request.action_fps - 1e-8)
    canvas = torch.zeros(3, frames, height, width, dtype=torch.uint8)
    for view, box in zip(selected, boxes, strict=True):
        x0, y0, x1, y1 = box
        _copy_resized_camera([images[view["role"]]], canvas[:, :1, y0:y1, x0:x1])
    if len(selected) > 1:
        boundary = boxes[0][3]
        canvas[:, :, boundary - 2 : boundary + 2] = 0
    if len(selected) == 3:
        canvas[:, :, boxes[1][1] :, width // 2 - 2 : width // 2 + 2] = 0
    locations = ["top", "bottom"] if len(selected) == 2 else ["top", "bottom_left", "bottom_right"]
    layout = (
        None
        if len(selected) == 1
        else [
            [v["role"], location, b[2] - b[0], b[3] - b[1]]
            for v, location, b in zip(selected, locations, boxes, strict=True)
        ]
    )
    return assemble_batch(request, spec, canvas, layout, [v["role"] for v in selected], video_fps, video_key)


def assemble_batch(request, spec, canvas, layout, roles, video_fps, video_key):
    registry = spec["registry"]
    domain = registry["domains"].index(request.action_domain)
    if registry["domain_projection_ids"][domain] == 33:
        raise ValueError("State domain is conditioning-only; select the native Control12 action domain")
    _, frames, height, width = canvas.shape
    steps = math.ceil(request.duration_seconds * request.action_fps - 1e-8)
    embodiment = registry["embodiments"][registry["domain_embodiments"][domain]]
    caption = render_caption(
        {"task_title": request.task_title, "source_task_description": request.prompt},
        start=0,
        duration=request.duration_seconds,
        fps=float(video_fps),
        action_frequency_hz=request.action_fps,
        width=width,
        height=height,
        embodiment=embodiment,
        camera_layout=layout,
        seed=request.seed,
    )
    batch = {
        video_key: [[canvas]],
        "action": [[torch.zeros(steps, 128)]],
        "action_dim_mask": [[torch.tensor(request.action_dim_mask, dtype=torch.bool)]],
        **make_batched_action_processing_fields(ActionProcessingRecord(raw_action_dim=128, action_normalizer=None), 1),
        "ai_caption": [caption],
        "mode": ["wam"],
        "image_size": torch.tensor([[height, width]]),
        "domain_id": [torch.tensor([registry["domain_projection_ids"][domain]])],
        "normalization_domain_id": [domain],
        "action128_contract_fingerprint": [stable_hash(spec)],
        "conditioning_fps": [torch.tensor(video_fps)],
        "conditioning_fps_action": [torch.tensor(request.action_fps)],
        "decode_metadata": [{"selected_roles": roles, "camera_layout": layout}],
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                supervise_text=False,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=True,
                action_start_frame_offset=0,
                action_timestamps_seconds=(torch.arange(steps, dtype=torch.float64) / request.action_fps).tolist(),
            )
        ],
    }
    if registry["domain_projection_ids"][domain] == 32:
        from cosmos_framework.data.generator.action.robocasa_state import STATE_KEY, state128

        if request.action_fps != 20 or request.state16 is None or request.state_mask16 is None:
            raise ValueError("Control12 requires 20 Hz and current state16/state_mask16")
        if any(request.action_dim_mask[12:]):
            raise ValueError("Control12 has only slots 0:12")
        state, state_mask = state128(request.state16, request.state_mask16)
        action = torch.zeros(steps + 1, 128)
        action[0] = torch.from_numpy(state)
        dims = torch.zeros(steps + 1, 128, dtype=torch.bool)
        dims[0] = torch.from_numpy(state_mask)
        dims[1:] = torch.tensor(request.action_dim_mask)
        ids = torch.full((steps + 1,), registry["domain_projection_ids"][domain], dtype=torch.long)
        ids[0] = spec["projection_routing"]["keys"][STATE_KEY]
        batch["action"] = [[action]]
        batch["action_token_domain_ids"] = [ids]
        batch["action_dim_mask"] = [dims]
        batch.update(
            make_batched_action_processing_fields(ActionProcessingRecord(raw_action_dim=12, action_normalizer=None), 1)
        )
        plan = batch["sequence_plan"][0]
        plan.condition_frame_indexes_action = [0]
        plan.robot_state_frame_indexes = [0]
        plan.action_timestamps_seconds = [0.0, *plan.action_timestamps_seconds]
    elif request.state16 is not None or request.state_mask16 is not None:
        raise ValueError("State input is only supported for native Control12 checkpoints")
    return batch


class Action128Extensions(BaseModel):
    """Only Action128 additions, transported by the original request's extra field."""

    model_config = ConfigDict(extra="forbid")
    action_domain: str = Field(min_length=1)
    action_dim_mask: Annotated[list[bool], Field(min_length=128, max_length=128)]
    action_fps: float | None = Field(default=None, gt=0, le=512, allow_inf_nan=False)
    selected_roles: list[str] = Field(default_factory=lambda: ["main"])
    camera_layout: list[list[Any]] | None = None
    task_title: str | None = None
    return_video: bool = True
    state16: Annotated[list[float], Field(min_length=16, max_length=16)] | None = None
    state_mask16: Annotated[list[bool], Field(min_length=16, max_length=16)] | None = None


def prepare_wam_request(
    overrides: OmniSampleOverrides, spec: dict, model_config: Any, output_dir: Path, video_key: str
) -> tuple[PreparedAction128Request, dict]:
    """Use Cosmos' request/default resolution; construct only the Action128 batch."""
    from cosmos_framework.data.generator.sequence_packing.spatial import action128_seam_position

    if overrides.model_mode != ModelMode.WAM:
        raise ValueError("This Action128 checkpoint serves model_mode='wam'")
    extensions = Action128Extensions.model_validate((overrides.extra or {}).get("action128", {}))
    if extensions.action_domain not in spec["registry"]["domains"]:
        raise ValueError("Unknown extra.action128.action_domain; see GET /domains")
    if not any(extensions.action_dim_mask):
        raise ValueError("action_dim_mask must contain an active slot")
    if (overrides.num_outputs or 1) != 1:
        raise ValueError("Action128 serving currently supports num_outputs=1 per request")
    overrides = overrides.model_copy(deep=True)
    overrides.output_dir = output_dir / f"generate_{uuid.uuid4().hex}"
    overrides.name = overrides.name or "action128_wam"
    overrides.seed = overrides.seed if overrides.seed is not None else random.randint(0, 2**31 - 1)
    overrides.output_dir.mkdir(parents=True)
    overrides.download(overrides.output_dir / "inputs")
    if overrides.vision_path is None:
        raise ValueError("vision_path is required for WAM")
    frames, _ = read_media_frames(Path(overrides.vision_path), max_frames=1)
    canvas = frames[:, :1].contiguous()
    height, width = canvas.shape[-2:]
    metadata = {"selected_roles": extensions.selected_roles, "camera_layout": extensions.camera_layout}
    action128_seam_position(metadata, (height, width), 32)
    target_h = overrides.image_size or 16 * math.ceil(height / 16)
    if target_h % 16:
        raise ValueError("Action128 image_size must be a multiple of 16")
    target_w = max(16, 16 * round(width * target_h / height / 16))
    if (target_h, target_w) != (height, width):
        import torch.nn.functional as F

        canvas = (
            F.interpolate(
                canvas.permute(1, 0, 2, 3).float(), size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 0, 2, 3)
        )
        if extensions.camera_layout is not None:
            layout = extensions.camera_layout
            seam = round(layout[0][3] * target_h / height)
            layout[0][2:] = [target_w, seam]
            if len(layout) == 2:
                layout[1][2:] = [target_w, target_h - seam]
            else:
                left_w = round(layout[1][2] * target_w / width)
                layout[1][2:] = [left_w, target_h - seam]
                layout[2][2:] = [target_w - left_w, target_h - seam]
        height, width = target_h, target_w
        action128_seam_position(metadata, (height, width), 32)
    overrides.image_size = height
    native = (
        spec["registry"]["domain_projection_ids"][spec["registry"]["domains"].index(extensions.action_domain)] == 32
    )
    raw_dim = 12 if native else 128
    if overrides.raw_action_dim not in (None, raw_dim):
        raise ValueError(f"Action raw_action_dim must be {raw_dim}")
    overrides.raw_action_dim = raw_dim
    if overrides.condition_frame_indexes_vision not in (None, [0]):
        raise ValueError("This no-state Action128 WAM checkpoint conditions on vision frame [0] only")
    overrides.condition_frame_indexes_vision = [0]
    fps = overrides.fps or 24
    if fps > 30:
        raise ValueError("Action128 video fps must not exceed the training ceiling of 30 Hz")
    action_fps = extensions.action_fps or fps
    steps = overrides.action_chunk_size or 16
    duration = steps / action_fps
    if duration > 2:
        raise ValueError("Action128 action horizon must not exceed the trained 2-second window")
    target_frames = 4 * math.ceil(math.ceil(duration * fps - 1e-8) / 4) + 1
    if overrides.num_frames is not None and overrides.num_frames != target_frames:
        raise ValueError(f"num_frames must equal {target_frames} for this action horizon and video fps")
    overrides.num_frames = target_frames
    sample = overrides.build_sample(model_config=model_config)
    if sample.enable_sound or sample.native_prompt_upsampling:
        raise ValueError("Action128 WAM serving supports non-autoregressive video/action without sound or upsampling")
    canvas = torch.cat((canvas, torch.zeros(3, target_frames - 1, height, width, dtype=torch.uint8)), dim=1)
    prompt = sample.prompt
    title = extensions.task_title or sample.name
    caption = None
    try:
        caption = json.loads(prompt)
        if isinstance(caption, dict):
            title = caption.get("task_title", title)
            prompt = caption.get("command", caption.get("source_task_description", prompt))
    except json.JSONDecodeError:
        pass
    prepared = PreparedAction128Request(
        action_domain=extensions.action_domain,
        task_title=title,
        prompt=prompt,
        action_fps=action_fps,
        duration_seconds=duration,
        seed=sample.seed,
        num_steps=sample.num_steps,
        guidance=sample.guidance,
        shift=sample.shift,
        action_dim_mask=extensions.action_dim_mask,
        state16=extensions.state16,
        state_mask16=extensions.state_mask16,
        sample_args=sample.model_dump(mode="json"),
        return_video=extensions.return_video,
    )
    batch = assemble_batch(prepared, spec, canvas, extensions.camera_layout, extensions.selected_roles, fps, video_key)
    if isinstance(caption, dict):
        formatted = json.loads(batch["ai_caption"][0])
        preserved = dict(caption)
        for key in (
            "embodiment",
            "duration",
            "fps",
            "action_frequency_hz",
            "resolution",
            "aspect_ratio",
            "camera_layout",
        ):
            if key in formatted:
                preserved[key] = formatted[key]
            else:
                preserved.pop(key, None)
        preserved.setdefault("command", prompt)
        # Explicit JSON captions may intentionally be command-only. Do not
        # introduce the request name as additional language conditioning.
        batch["ai_caption"][0] = json.dumps(preserved, ensure_ascii=False)
    if sample.negative_prompt:
        batch["neg_ai_caption"] = [sample.negative_prompt]
    return prepared, batch


def create_app(service: Any) -> FastAPI:
    """Original Cosmos /generate request and response types, with Action128 extras."""
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Cosmos Action128 WAM serving")
    lock = threading.Lock()
    output_dir = Path(service.info.get("output_dir", "/tmp/cosmos_action128_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")
    model_name = service.info.get("model_name", "action128")

    @app.get("/healthz")
    def healthz():
        return {"status": "ready"}

    @app.get("/models")
    def models():
        return [model_name]

    @app.get("/info")
    def info():
        return {**service.info, "models": [model_name], "output_dir": str(output_dir)}

    @app.get("/domains")
    def domains():
        output_schema = service.info.get("action_schema")
        return {
            "domains": [
                domain
                for domain, head in zip(
                    service.spec["registry"]["domains"], service.spec["registry"]["domain_projection_ids"], strict=True
                )
                if head not in {33, 35} and not domain.endswith("/state/current_position")
            ],
            "action_dim": output_schema["dimension"] if output_schema else 128,
            **({"model_action_dim": 128, "action_schema": output_schema} if output_schema else {}),
            "contract_fingerprint": stable_hash(service.spec),
        }

    @app.post("/generate", response_model=SampleOutputs)
    def generate(request: OmniSampleOverrides):
        if request.model and request.model != model_name:
            raise HTTPException(
                status_code=404, detail=f"Model '{request.model}' not found. Available models: {[model_name]}"
            )
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=503, detail="Model is busy; retry after the current request", headers={"Retry-After": "1"}
            )
        try:
            try:
                prepared, batch = service.prepare(request, output_dir)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return service.predict(prepared, batch)
        finally:
            lock.release()

    return app
