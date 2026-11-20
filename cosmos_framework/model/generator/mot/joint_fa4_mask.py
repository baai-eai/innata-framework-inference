# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CuTe callback, imported only when the direct FA4 backend is selected."""

import cutlass
import cutlass.cute as cute
from flash_attn.cute import utils


@cute.jit
def mixed_sequence_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    causal = utils.scalar_to_ssa(aux_tensors[0][batch[0]], cutlass.Int32)
    offset = utils.scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, cutlass.Int32)
    return (causal == 0) | (n_idx <= m_idx + offset)
