# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Literal, get_args

JointAttentionBackend = Literal[
    "split_default",
    "split_fa3_fork",
    "unified_fa3_grouped",
    "split_fa4",
    "unified_fa4_mask",
    "split_flex_triton",
    "unified_flex_triton",
]
JOINT_ATTENTION_BACKENDS = get_args(JointAttentionBackend)
