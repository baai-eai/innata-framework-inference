# Mainly adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/data/generator/sequence_packing/__init__.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""High-level entry points for VFM sequence packing."""

from cosmos_framework.data.generator.sequence_packing.modality import ModalityData
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence, pack_understanding_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
)

prepare_runtime_sequence = pack_input_sequence

__all__ = [
    "ModalityData",
    "PackedSequence",
    "SequencePlan",
    "build_sequence_plans_from_data_batch",
    "pack_input_sequence",
    "pack_understanding_sequence",
    "prepare_runtime_sequence",
]
