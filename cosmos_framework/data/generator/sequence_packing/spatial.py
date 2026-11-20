# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spatial RoPE consumers for the published Action128 stitched canvas contract."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence


SPATIAL_POSITION_MODES = {"canvas", "action128_seam"}


def validate_spatial_position_mode(mode: str, *, reset_spatial: bool, temporal_causal: bool) -> None:
    if mode not in SPATIAL_POSITION_MODES:
        raise ValueError(f"Unsupported vision_spatial_position_mode: {mode!r}")
    if mode == "action128_seam" and (not reset_spatial or temporal_causal):
        raise ValueError("action128_seam requires reset_spatial_ids=True and video_temporal_causal=False")


def action128_seam_position(metadata: dict, canvas_hw: tuple[int, int], pixel_stride: int) -> float | None:
    """Read the actual horizontal seam; never infer a seam from latent dimensions.

    Coordinates retain the existing patch-corner anchor: row r represents pixel
    y=r*pixel_stride. A fractional seam is therefore intentional. Single-camera
    samples return None and keep their original downward-positive coordinates.
    """
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if not isinstance(metadata, dict) or "camera_layout" not in metadata:
        raise ValueError("action128_seam requires decode_metadata.camera_layout")
    roles = metadata.get("selected_roles")
    if not isinstance(roles, list) or len(roles) not in (1, 2, 3):
        raise ValueError("action128_seam requires one to three selected_roles")
    layout = metadata["camera_layout"]
    if len(roles) == 1:
        if layout is not None:
            raise ValueError("single-camera Action128 camera_layout must be None")
        return None
    if not isinstance(layout, list) or len(layout) != len(roles):
        raise ValueError("camera_layout must match selected_roles")
    expected_locations = ["top", "bottom"] if len(roles) == 2 else ["top", "bottom_left", "bottom_right"]
    if any(not isinstance(item, (list, tuple)) or len(item) != 4 for item in layout):
        raise ValueError("camera_layout entries must be [role, location, width, height]")
    if [item[0] for item in layout] != roles or [item[1] for item in layout] != expected_locations:
        raise ValueError("camera_layout roles/locations do not describe an Action128 stitched canvas")
    sizes = [item[2:] for item in layout]
    if any(type(value) is not int or value <= 0 for size in sizes for value in size):
        raise ValueError("camera_layout dimensions must be positive integer pixels")
    height, width = canvas_hw
    seam = sizes[0][1]
    if sizes[0][0] != width or not 0 < seam < height:
        raise ValueError("camera_layout top region does not match canvas geometry")
    if any(size[1] != height - seam for size in sizes[1:]) or sum(size[0] for size in sizes[1:]) != width:
        raise ValueError("camera_layout bottom regions do not match canvas geometry")
    return seam / pixel_stride


def apply_vision_spatial_positions(
    packed: PackedSequence, mode: str, seams: list[float | None] | None
) -> PackedSequence:
    """Change only vision H after packing, preserving all temporal cursors/IDs.

    In particular, the legacy max(T,H,W)+1 cursor must be computed *before*
    spatial translation/reflection, or following text/action/samples can shift.
    """
    packed.vision_spatial_position_mode = mode
    if mode == "canvas" or packed.vision is None:
        return packed
    if seams is None or len(seams) != len(packed.vision.token_shapes):
        raise ValueError("action128_seam requires one spatial_seams_vision entry per vision item")
    packed.spatial_seams_vision = list(seams)
    packed.position_ids = packed.position_ids.float()
    offset = 0
    for shape, seam in zip(packed.vision.token_shapes, seams, strict=True):
        count = math.prod(shape)
        indexes = packed.vision.sequence_indexes[offset : offset + count]
        if seam is not None:
            if not math.isfinite(seam) or seam <= 0:
                raise ValueError("spatial seam must be finite and positive")
            packed.position_ids[1, indexes] = seam - packed.position_ids[1, indexes]
        offset += count
    if offset != packed.vision.sequence_indexes.numel():
        raise ValueError("vision spatial topology does not match token shapes")
    return packed
