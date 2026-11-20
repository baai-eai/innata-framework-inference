# Copied from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/model/attention/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.


"""

from cosmos_framework.model.attention.frontend import (
    attention,
    merge_attentions,
    multi_dimensional_attention,
    multi_dimensional_attention_varlen,
    spatio_temporal_attention,
)

__all__ = [
    "attention",
    "multi_dimensional_attention",
    "multi_dimensional_attention_varlen",
    "spatio_temporal_attention",
    "merge_attentions",
]
