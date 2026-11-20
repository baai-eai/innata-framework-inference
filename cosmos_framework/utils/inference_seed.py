# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stable per-modality streams for architecture-invariant inference noise."""

_UINT32_MODULUS = 2**32
_MODALITY_OFFSETS = {"vision": 0, "action": 0x9E3779B9, "sound": 0x3C6EF372}


def modality_noise_seed(seed: int, modality: str) -> int:
    """Keep the video seed stable while separating action/audio random streams."""
    if not 0 <= seed < _UINT32_MODULUS:
        raise ValueError("Inference seed must be a uint32")
    return (seed + _MODALITY_OFFSETS[modality]) % _UINT32_MODULUS
