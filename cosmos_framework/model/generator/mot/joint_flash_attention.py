# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional FA3 grouped and FA4 per-sequence mask adapters for two-way packs."""

import os

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq, get_und_seq


def prepare_flash_metadata(pack, backend):
    # Import/validate once before decoder compilation. Never load both FA3 .so's.
    if backend in ("split_fa3_fork", "unified_fa3_grouped"):
        if os.environ.get("COSMOS_FA3_PROVIDER") != "fork":
            raise RuntimeError("Set COSMOS_FA3_PROVIDER=fork before process startup for the grouped FA3 backends")
        from flash_attn_3.flash_attn_grouped import flash_attn_varlen_grouped_tensors_func
        from flash_attn_interface import flash_attn_varlen_func, flash_attn_varlen_grouped_func

        for fn in (flash_attn_varlen_func, flash_attn_varlen_grouped_func, flash_attn_varlen_grouped_tensors_func):
            torch._dynamo.allow_in_graph(fn)
    else:
        from flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401

        from cosmos_framework.model.generator.mot.joint_fa4_op import fa4_varlen  # noqa: F401

    nu, ng = get_und_seq(pack).shape[0], get_gen_seq(pack).shape[0]
    fixed = bool(pack.get("_fixed_sequence_length", False))
    cu_u = pack.get("_causal_seq_offsets_pad_segment", pack["_causal_seq_offsets"])
    cu_g = pack["_full_only_seq_offsets_pad_segment"] if fixed else pack["_full_only_seq_offsets"]
    cu_all = pack["_sample_offsets_pad_segment"] if fixed else pack["sample_offsets"]
    if cu_g.shape != cu_all.shape:
        raise ValueError("GEN segments must match sample KV segments for joint varlen attention")
    cu_q = torch.cat((cu_u[:-1], cu_g + nu))
    cu_k = torch.cat((cu_u[:-1], cu_all + nu))
    flags = torch.cat((torch.ones_like(cu_u[:-1]), torch.zeros_like(cu_g[:-1])))
    metadata = {
        "cu_u": cu_u,
        "cu_g": cu_g,
        "cu_all": cu_all,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "flags": flags,
        "flags_bool": flags.bool(),
        "max_u": nu,
        "max_g": ng,
        "max_all": nu + ng if fixed else int(pack["_causal_indices"].shape[0] + pack["_full_indices"].shape[0]),
    }
    for v in metadata.values():
        if isinstance(v, torch.Tensor) and v.shape[0] > 1:
            torch._dynamo.mark_dynamic(v, 0)
    return metadata


def joint_flash_attention(qs, und_k, und_v, all_k, all_v, m, scale):
    backend = m["backend"]
    if backend in ("split_fa3_fork", "unified_fa3_grouped"):
        from flash_attn_3.flash_attn_grouped import flash_attn_varlen_grouped_tensors_func as grouped_tensors
        from flash_attn_interface import flash_attn_varlen_func as fa3
        from flash_attn_interface import flash_attn_varlen_grouped_func as grouped

        if backend == "split_fa3_fork":
            yu = fa3(
                qs[0], und_k, und_v, m["cu_u"], m["cu_u"], m["max_u"], m["max_u"], causal=True, softmax_scale=scale
            )
            yg = fa3(
                qs[1], all_k, all_v, m["cu_g"], m["cu_all"], m["max_g"], m["max_all"], causal=False, softmax_scale=scale
            )
            return yu, yg
        if not any(t.requires_grad for t in (qs[0], und_k, und_v)):
            # Existing fork's independent-group autograd: one forward, GEN-only
            # backward. Do not infer this contract from frozen weights alone.
            return grouped_tensors(
                qs,
                (und_k, all_k),
                (und_v, all_v),
                (m["cu_u"], m["cu_g"]),
                (m["cu_u"], m["cu_all"]),
                (m["max_u"], m["max_g"]),
                (m["max_u"], m["max_all"]),
                (True, False),
                softmax_scale=scale,
            )
        y = grouped(
            torch.cat(qs),
            torch.cat((und_k, all_k)),
            torch.cat((und_v, all_v)),
            m["cu_q"],
            m["cu_k"],
            max(m["max_u"], m["max_g"]),
            m["max_all"],
            m["flags_bool"],
            softmax_scale=scale,
        )
    else:
        from cosmos_framework.model.generator.mot.joint_fa4_op import fa4_varlen as fa4

        if backend == "split_fa4":
            outputs = []
            for q, k, v, cu_q, cu_k, mq, mk, causal in (
                (qs[0], und_k, und_v, m["cu_u"], m["cu_u"], m["max_u"], m["max_u"], True),
                (qs[1], all_k, all_v, m["cu_g"], m["cu_all"], m["max_g"], m["max_all"], False),
            ):
                out = fa4(
                    q,
                    k,
                    v,
                    cu_q=cu_q,
                    cu_k=cu_k,
                    max_q=mq,
                    max_k=mk,
                    causal=causal,
                    scale=scale,
                )
                outputs.append(out[0] if isinstance(out, tuple) else out)
            return outputs
        out = fa4(
            torch.cat(qs),
            torch.cat((und_k, all_k)),
            torch.cat((und_v, all_v)),
            cu_q=m["cu_q"],
            cu_k=m["cu_k"],
            max_q=max(m["max_u"], m["max_g"]),
            max_k=m["max_all"],
            causal=False,
            flags=m["flags"],
            scale=scale,
        )
        y = out[0] if isinstance(out, tuple) else out
    yu, yg = y.split([qs[0].shape[0], qs[1].shape[0]])
    if not any(t.requires_grad for t in (qs[0], und_k, und_v)):
        yu = yu.detach()
    return yu, yg
