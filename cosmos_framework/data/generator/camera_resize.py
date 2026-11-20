# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-exact cover resize shared by action workers and decode helpers."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def copy_resized_camera(frames: list[np.ndarray], destination: torch.Tensor) -> None:
    """Cover-resize into a CTHW uint8 canvas with bounded float intermediates.

    Reuse one float32 frame while preserving the existing bilinear/round/clamp
    pixels. Direct conversion avoids stacking high-resolution uint8 frames.
    """
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("all input arrays must have the same shape")
    height, width = destination.shape[-2:]
    scale = max(width / shape[1], height / shape[0])
    h, w = math.ceil(shape[0] * scale), math.ceil(shape[1] * scale)
    crop_x, crop_y = (w - width) // 2, (h - height) // 2
    # NHWC storage preserves the channels-last interpolation path. A single
    # frame keeps the working set bounded even for high-resolution cameras.
    converted = torch.empty((1, *shape), dtype=torch.float32, device="cpu")
    pixels = converted.permute(0, 3, 1, 2)
    for index, frame in enumerate(frames):
        # np.stack previously normalized these layouts. Copy only when needed
        # so flipped views remain supported and read-only arrays do not warn.
        if not frame.flags.writeable or any(stride < 0 for stride in frame.strides):
            frame = frame.copy(order="C")
        pixels.copy_(torch.from_numpy(frame[None]).permute(0, 3, 1, 2))
        resized = F.interpolate(pixels, (h, w), mode="bilinear", align_corners=False)
        resized.round_().clamp_(0, 255)
        cropped = resized[:, :, crop_y : crop_y + height, crop_x : crop_x + width]
        # copy_ performs the same float-to-byte cast directly into the canvas.
        destination[:, index : index + 1].copy_(cropped.permute(1, 0, 2, 3))
