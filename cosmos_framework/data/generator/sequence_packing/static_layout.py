# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Collator-side construction and model-side binding of static packed layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence

import torch

from cosmos_framework.data.generator.offline_vae import VaeLatentRef
from cosmos_framework.data.generator.sequence_packing.packers import (
    pack_input_sequence,
    pack_understanding_sequence,
)
from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence, SequencePlan
from cosmos_framework.data.generator.sequence_packing.spatial import (
    action128_seam_position,
    validate_spatial_position_mode,
)

if TYPE_CHECKING:
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


PACKED_SEQUENCE_KEY = "packed_sequence"


@dataclass(frozen=True)
class StaticPackingConfig:
    """Model geometry needed to plan token positions without loading model weights."""

    spatial_compression_factor: int = 16
    temporal_compression_factor: int = 4
    latent_patch_size: int = 2
    unified_3d_mrope_reset_spatial_ids: bool = True
    unified_3d_mrope_temporal_modality_margin: int = 0
    enable_fps_modulation: bool = False
    base_fps: float = 24.0
    vision_temporal_position_mode: str = "latent_index"
    vision_spatial_position_mode: str = "canvas"
    video_temporal_causal: bool = False
    action_dim: int = 32

    def __post_init__(self) -> None:
        validate_spatial_position_mode(
            self.vision_spatial_position_mode,
            reset_spatial=self.unified_3d_mrope_reset_spatial_ids,
            temporal_causal=self.video_temporal_causal,
        )
        for name in ("spatial_compression_factor", "temporal_compression_factor", "latent_patch_size"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.vision_temporal_position_mode != "latent_index":
            raise ValueError(
                "Collator-side static packing currently requires vision_temporal_position_mode='latent_index'; "
                "nonuniform UniAE positions need an explicit lightweight geometry provider."
            )
        if self.video_temporal_causal:
            raise ValueError(
                "Collator-side static packing is not enabled for video_temporal_causal layouts; "
                "the maintained Nano/Edge training recipes use video_temporal_causal=False."
            )


def _as_plan(value: Any) -> SequencePlan:
    if isinstance(value, SequencePlan):
        return value
    if isinstance(value, dict):
        return SequencePlan(**value)
    raise TypeError(f"sequence_plan must be SequencePlan or dict, got {type(value).__name__}")


def _as_token_ids(value: Any) -> list[int]:
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, torch.Tensor)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        return [int(token) for token in value.reshape(-1).tolist()]
    if isinstance(value, list):
        return [int(token) for token in value]
    raise TypeError(f"text_token_ids must be a tensor or list, got {type(value).__name__}")


def _as_items(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        items = list(value)
        while len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = list(items[0])
        if not all(isinstance(item, torch.Tensor) for item in items):
            raise TypeError("media/action items must all be tensors")
        return items
    raise TypeError(f"media/action value must be tensor or list, got {type(value).__name__}")


def _metadata_int(sample: dict[str, Any], key: str) -> int | None:
    value = sample.get(key)
    if value is None:
        return None
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{key} must be scalar, got shape {tuple(value.shape)}")
        value = value.item()
    return int(value)


def _fps(sample: dict[str, Any], key: str = "conditioning_fps") -> float:
    value = sample.get(key, sample.get("conditioning_fps", 1.0))
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1)[0].item()
    return float(value)


def _planned_spatial_shape(sample: dict[str, Any], media: torch.Tensor) -> tuple[int, int]:
    height, width = int(media.shape[-2]), int(media.shape[-1])
    frame_size = sample.get("image_size")
    if frame_size is None:
        return height, width
    while isinstance(frame_size, (list, tuple)) and len(frame_size) == 1:
        frame_size = frame_size[0]
    if not isinstance(frame_size, torch.Tensor):
        return height, width
    values = frame_size.reshape(-1)
    if values.numel() < 4:
        return height, width
    return int(values[2].item()), int(values[3].item())


def planned_vision_latent_shape(
    sample: dict[str, Any],
    media: torch.Tensor | VaeLatentRef,
    *,
    is_image: bool,
    config: StaticPackingConfig,
) -> tuple[int, int, int]:
    """Latent geometry after the same resize/per-camera policy used by the model."""
    if isinstance(media, VaeLatentRef):
        if is_image:
            raise ValueError("offline VAE latent references are supported only for video generation")
        _, latent_t, latent_h, latent_w = media.latent_shape
        return latent_t, latent_h, latent_w
    if is_image:
        if media.ndim != 3:
            raise ValueError(f"image must have shape [C,H,W], got {tuple(media.shape)}")
        latent_t = 1
    else:
        if media.ndim != 4:
            raise ValueError(f"video must have shape [C,T,H,W], got {tuple(media.shape)}")
        pixel_t = int(media.shape[1])
        num_views = _metadata_int(sample, "sample_n_views") if "enable_per_camera_vae_encoding" in sample else None
        frames_per_view = (
            _metadata_int(sample, "num_video_frames_per_view") if "enable_per_camera_vae_encoding" in sample else None
        )
        if num_views is not None:
            if frames_per_view is None or pixel_t != num_views * frames_per_view:
                raise ValueError("invalid per-camera VAE metadata for static packing")
            latent_t = num_views * (1 + (frames_per_view - 1) // config.temporal_compression_factor)
        else:
            latent_t = 1 + (pixel_t - 1) // config.temporal_compression_factor

    pixel_h, pixel_w = _planned_spatial_shape(sample, media)
    latent_h = max(pixel_h // config.spatial_compression_factor, 1)
    latent_w = max(pixel_w // config.spatial_compression_factor, 1)
    return latent_t, latent_h, latent_w


def _latent_proxy(sample, media, *, is_image, config):
    shape = planned_vision_latent_shape(sample, media, is_image=is_image, config=config)
    # Zero channels make this a shape carrier with no latent-sized allocation.
    return torch.empty((1, 0, *shape), dtype=torch.float32)


def build_static_packed_sequence(
    samples: Sequence[dict[str, Any]],
    *,
    special_tokens: dict[str, int],
    config: StaticPackingConfig,
) -> PackedSequence:
    """Build all shape- and supervision-dependent topology in a DataLoader worker."""

    if not samples:
        raise ValueError("cannot build a packed layout from an empty sample list")
    plans = [_as_plan(sample["sequence_plan"]) for sample in samples]
    text_ids = [_as_token_ids(sample["text_token_ids"]) for sample in samples]
    has_generation = [plan.has_vision or plan.has_action for plan in plans]
    if any(has_generation) and not all(has_generation):
        raise ValueError("one packed sequence cannot mix Understanding-only and Generation/Action samples")
    if not any(has_generation):
        return pack_understanding_sequence(plans, text_ids, special_tokens)

    is_image_batch = all(
        "images" in sample and "video" not in sample and "video_latent_ref" not in sample for sample in samples
    )
    if not is_image_batch and any("images" in sample for sample in samples):
        raise ValueError("one packed sequence cannot mix image and video generation")
    has_offline_latent = any("video_latent_ref" in sample for sample in samples)
    if has_offline_latent and not all("video_latent_ref" in sample for sample in samples):
        raise ValueError("one packed sequence cannot mix online pixel video and offline VAE latent references")

    spatial_seams: list[float | None] = []
    vision_tokens: list[torch.Tensor] = []
    vision_item_counts: list[int] = []
    num_views_per_item: list[int] = []
    action_tokens: list[torch.Tensor] = []
    action_domain_ids: list[torch.Tensor] = []
    raw_action_dims: list[torch.Tensor | None] = []
    fps_values: list[float] = []
    action_fps_values: list[float] = []

    for sample, plan in zip(samples, plans, strict=True):
        fps_values.append(_fps(sample))
        if plan.has_vision:
            media_key = "images" if is_image_batch else "video_latent_ref" if has_offline_latent else "video"
            media_value = sample[media_key]
            items = [media_value] if isinstance(media_value, VaeLatentRef) else _as_items(media_value)
            vision_item_counts.append(len(items))
            num_views = _metadata_int(sample, "sample_n_views") or 1
            if config.vision_spatial_position_mode == "action128_seam":
                if len(items) != 1 or num_views != 1 or has_offline_latent or is_image_batch:
                    raise ValueError("action128_seam requires one online stitched video per sample")
                spatial_seams.append(
                    action128_seam_position(
                        sample.get("decode_metadata"),
                        _planned_spatial_shape(sample, items[0]),
                        config.spatial_compression_factor * config.latent_patch_size,
                    )
                )
            for item in items:
                vision_tokens.append(_latent_proxy(sample, item, is_image=is_image_batch, config=config))
                num_views_per_item.append(num_views)
        else:
            vision_item_counts.append(0)

        if plan.has_action:
            action_fps_values.append(_fps(sample, "conditioning_fps_action"))
            action_items = _as_items(sample["action"])
            if len(action_items) != 1:
                raise ValueError("each Action sample must contain exactly one action tensor")
            action = action_items[0]
            action_tokens.append(torch.empty((int(action.shape[0]), 0), dtype=torch.float32))
            domain_id = sample.get("domain_id", torch.zeros(1, dtype=torch.long))
            while isinstance(domain_id, list) and len(domain_id) == 1:
                domain_id = domain_id[0]
            action_domain_ids.append(torch.as_tensor(domain_id, dtype=torch.long).reshape(-1))
            raw_dim = sample.get("raw_action_dim")
            while isinstance(raw_dim, list) and len(raw_dim) == 1:
                raw_dim = raw_dim[0]
            raw_action_dims.append(torch.as_tensor(raw_dim) if raw_dim is not None else None)

    multi_item_counts = vision_item_counts if any(count != 1 for count in vision_item_counts) else None
    multiview_counts = num_views_per_item if any(count != 1 for count in num_views_per_item) else None
    proxy = SimpleNamespace(
        batch_size=len(samples),
        is_image_batch=is_image_batch,
        x0_tokens_vision=vision_tokens,
        x0_tokens_action=action_tokens,
        x0_tokens_sound=None,
        fps_vision=torch.tensor(fps_values, dtype=torch.float32),
        temporal_positions_vision=None,
        spatial_seams_vision=spatial_seams or None,
        fps_action=torch.tensor(action_fps_values, dtype=torch.float32),
        action_domain_id=action_domain_ids or None,
        raw_action_dim=raw_action_dims or None,
        num_vision_items_per_sample=multi_item_counts,
        num_views_per_vision_item=multiview_counts,
        control_weights=None,
    )
    packed = pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=text_ids,
        gen_data_clean=proxy,
        input_timesteps=torch.zeros(len(samples), dtype=torch.float32),
        special_tokens=special_tokens,
        latent_patch_size=config.latent_patch_size,
        unified_3d_mrope_reset_spatial_ids=config.unified_3d_mrope_reset_spatial_ids,
        unified_3d_mrope_temporal_modality_margin=config.unified_3d_mrope_temporal_modality_margin,
        enable_fps_modulation=config.enable_fps_modulation,
        base_fps=config.base_fps,
        temporal_compression_factor=config.temporal_compression_factor,
        vision_temporal_position_mode=config.vision_temporal_position_mode,
        vision_spatial_position_mode=config.vision_spatial_position_mode,
        video_temporal_causal=config.video_temporal_causal,
        action_dim=config.action_dim,
    )
    # Keep input validity separate from supervision, including rebuilt validation layouts.
    for modality, key, attribute in (
        ("action", "action_loss_mask", "loss_valid_mask"),
        ("vision", "vision_loss_mask", "loss_valid_mask"),
        ("action", "action_dim_mask", "input_valid_mask"),
    ):
        present = [key in sample for sample in samples]
        if any(present):
            if not all(present):
                raise ValueError("cannot mix legacy and Action128 mask contracts in one packed sequence")
            target = getattr(packed, modality)
            if target is None:
                raise ValueError(f"{key} provided without {modality} topology")
            setattr(target, attribute, [sample[key] for sample in samples])
    return packed


def _bind_timesteps(
    packed: PackedSequence,
    input_timesteps: torch.Tensor,
    *,
    vision: bool,
) -> None:
    modality = packed.vision if vision else packed.action
    if modality is None:
        return
    pieces: list[torch.Tensor] = []
    for payload_index, noisy_frames in enumerate(modality.noisy_frame_indexes):
        sample_index = modality.sample_indexes[payload_index]
        sample_timestep = input_timesteps[sample_index].detach().cpu().reshape(-1)
        if vision:
            _, patch_h, patch_w = modality.token_shapes[payload_index]
            repeats = patch_h * patch_w
        else:
            repeats = 1
        if sample_timestep.numel() == 1:
            values = sample_timestep.expand(noisy_frames.numel())
        else:
            values = sample_timestep[noisy_frames.long()]
        pieces.append(values.repeat_interleave(repeats).to(torch.float32))
    modality.timesteps = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def bind_static_packed_sequence(
    packed: PackedSequence,
    gen_data_clean: GenerationDataClean,
    input_timesteps: torch.Tensor,
    *,
    latent_patch_size: int,
    vision_spatial_position_mode: str | None = None,
) -> PackedSequence:
    """Attach model-produced latent/action payloads to a collator-built layout."""

    if vision_spatial_position_mode is not None and packed.vision_spatial_position_mode != vision_spatial_position_mode:
        raise ValueError("static layout spatial position mode differs from model configuration")
    if packed.vision is not None and packed.vision_spatial_position_mode == "action128_seam":
        if packed.spatial_seams_vision != getattr(gen_data_clean, "spatial_seams_vision", None):
            raise ValueError("static layout spatial seams differ from model inputs")
    if packed.vision is not None:
        actual = list(gen_data_clean.x0_tokens_vision or [])
        if len(actual) != len(packed.vision.token_shapes):
            raise ValueError(f"planned {len(packed.vision.token_shapes)} vision items but VAE produced {len(actual)}")
        for index, (token, expected) in enumerate(zip(actual, packed.vision.token_shapes, strict=True)):
            actual_shape = (
                int(token.shape[2]),
                math.ceil(int(token.shape[3]) / latent_patch_size),
                math.ceil(int(token.shape[4]) / latent_patch_size),
            )
            if actual_shape != tuple(expected):
                raise ValueError(
                    f"vision item {index} static layout mismatch: planned {tuple(expected)}, VAE produced {actual_shape}"
                )
        packed.vision.tokens = actual
        _bind_timesteps(packed, input_timesteps, vision=True)

    if packed.action is not None:
        actual_action = list(gen_data_clean.x0_tokens_action or [])
        actual_shapes = [(int(token.shape[0]),) for token in actual_action]
        if actual_shapes != [tuple(shape) for shape in packed.action.token_shapes]:
            raise ValueError(
                f"action static layout mismatch: planned {packed.action.token_shapes}, produced {actual_shapes}"
            )
        packed.action.tokens = actual_action
        packed.action.domain_id = list(gen_data_clean.action_domain_id or [])
        packed.action.raw_action_dim = gen_data_clean.raw_action_dim
        _bind_timesteps(packed, input_timesteps, vision=False)

    packed.is_image_batch = gen_data_clean.is_image_batch
    packed.num_vision_items_per_sample = gen_data_clean.num_vision_items_per_sample
    packed.num_views_per_vision_item = gen_data_clean.num_views_per_vision_item
    packed.control_weights = gen_data_clean.control_weights
    return packed
