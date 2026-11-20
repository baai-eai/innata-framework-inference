# Copied from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/model/generator/tokenizers/audio/avae_utils/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
AVAE Tokenizer

This module provides the AVAE tokenizer with spec_convnext encoder,
oobleck decoder, and VAE bottleneck configuration.
"""

from cosmos_framework.model.generator.tokenizers.audio.avae_utils.env import AttrDict
from cosmos_framework.model.generator.tokenizers.audio.avae_utils.models import LatentAutoEncoderV2, load_generator

__all__ = [
    "LatentAutoEncoderV2",
    "AttrDict",
    "load_generator",
]
