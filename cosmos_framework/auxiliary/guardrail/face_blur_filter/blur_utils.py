# Copied from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/auxiliary/guardrail/face_blur_filter/blur_utils.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import cv2
import numpy as np


def pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    """
    Pixelate a face region by reducing resolution and then upscaling.

    Args:
        face_img: Face region to pixelate
        blocks: Number of blocks to divide the face into (in each dimension)

    Returns:
        Pixelated face region
    """
    h, w = face_img.shape[:2]
    # Shrink the image and scale back up to create pixelation effect
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated
