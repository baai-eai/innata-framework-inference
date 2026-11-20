# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opaque FA4 varlen operators for compiled training (first-order gradients).

The CuTe Python interface performs DLPack conversion and cannot execute on Dynamo
FakeTensors. Keep that runtime boundary explicit; use the same FA4 forward and
backward kernels for eager and compiled, split and mixed-sequence calls.
"""

import torch


def _kwargs(cu_q, cu_k, max_q, max_k, causal, scale, flags):
    from cosmos_framework.model.generator.mot.joint_fa4_mask import mixed_sequence_mask

    return dict(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=causal,
        softmax_scale=scale,
        mask_mod=mixed_sequence_mask if flags is not None else None,
        aux_tensors=[flags] if flags is not None else None,
    )


@torch.library.custom_op("cosmos_joint::fa4_fwd", mutates_args=())
def fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    scale: float,
    flags: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd

    out, lse, *_ = _flash_attn_fwd(q, k, v, return_lse=True, **_kwargs(cu_q, cu_k, max_q, max_k, causal, scale, flags))
    return out.contiguous(), lse.contiguous()


@fa4_fwd.register_fake
def _fake_fwd(q, k, v, cu_q, cu_k, max_q, max_k, causal, scale, flags):
    return (
        q.new_empty((q.shape[0], q.shape[1], v.shape[2])),
        q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
    )


@torch.library.custom_op("cosmos_joint::fa4_bwd", mutates_args=())
def fa4_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    scale: float,
    flags: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_bwd

    grads = _flash_attn_bwd(q, k, v, out, dout, lse, **_kwargs(cu_q, cu_k, max_q, max_k, causal, scale, flags))
    return tuple(g.contiguous() for g in grads)


@fa4_bwd.register_fake
def _fake_bwd(q, k, v, out, dout, lse, cu_q, cu_k, max_q, max_k, causal, scale, flags):
    return tuple(t.new_empty(t.shape) for t in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v, cu_q, cu_k, max_q, max_k, causal, scale, flags = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, flags)
    ctx.options = (max_q, max_k, causal, scale)
    ctx.mark_non_differentiable(lse)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, cu_q, cu_k, flags = ctx.saved_tensors
    dq, dk, dv = fa4_bwd(q, k, v, out, dout, lse, cu_q, cu_k, *ctx.options, flags)
    return dq, dk, dv, None, None, None, None, None, None, None


fa4_fwd.register_autograd(_backward, setup_context=_setup_context)


def fa4_varlen(q, k, v, cu_q, cu_k, max_q, max_k, *, causal=False, scale=None, flags=None):
    return fa4_fwd(q, k, v, cu_q, cu_k, max_q, max_k, causal, q.shape[-1] ** -0.5 if scale is None else scale, flags)[0]
