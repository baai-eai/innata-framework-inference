# Copied from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/model/generator/mot/qwen3_vl_unified_mot.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Backward-compatibility shim: this module was renamed to unified_mot.py.
# Existing serialized configs / checkpoints may reference the old module path,
# so we re-export everything from the new location.
from cosmos_framework.model.generator.mot.unified_mot import *  # noqa: F401, F403
from cosmos_framework.model.generator.mot.unified_mot import (  # noqa: F401  # explicit re-exports for type checkers
    LayerTypes,
    MoTDecoderLayer,
    Nemotron3DenseVLTextConfig,
    Nemotron3DenseVLTextForCausalLM,
    Nemotron3DenseVLTextModel,
    PackedAttentionMoT,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeTextForCausalLM,
    Qwen3VLMoeTextModel,
    Qwen3VLTextConfig,
    Qwen3VLTextForCausalLM,
    Qwen3VLTextModel,
    Qwen3VLTextMoTDecoderLayer,
)
