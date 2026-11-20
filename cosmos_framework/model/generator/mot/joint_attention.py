# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in joint UND/GEN attention, shared by two- and three-tower training.

Metadata is prepared outside compiled decoder layers. The Flex mask builder works
on block rectangles, never on a token-by-token dense matrix. UND causal segments
and GEN sample boundaries are distinct: a sample may contain several UND segments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from cosmos_framework.configs.base.defaults.joint_attention import JOINT_ATTENTION_BACKENDS
from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_all_seq,
    get_gen_seq,
    get_und_seq,
)

_compiled_flex = torch.compile(flex_attention, dynamic=False)


def _ordered_blocks(blocks, device):
    indices = np.argsort(~blocks, axis=-1, kind="stable").astype(np.int32)
    counts = blocks.sum(axis=-1, dtype=np.int32)
    return (
        torch.from_numpy(counts[None, None]).to(device),
        torch.from_numpy(indices[None, None]).to(device),
    )


def build_joint_block_mask(pack, *, query_stream="all", block_size=128):
    """Exact causal/sample mask with conservative full-block classification.

    A block crossing a segment boundary remains partial even when its union is
    fully visible. This costs predicate work but cannot overexpose tokens.
    Padding rows see only themselves and never a real key.
    """
    if pack["is_sharded"]:
        raise ValueError("Joint attention requires context_parallel_shard_degree=1")
    if query_stream not in ("all", "und", "gen"):
        raise ValueError(f"Invalid query stream {query_stream!r}")
    nu, ng = get_und_seq(pack).shape[0], get_gen_seq(pack).shape[0]
    device = get_und_seq(pack).device
    u_offsets = pack["_causal_seq_offsets"].cpu().tolist()
    u_sample = pack["_causal_sample_ids"][: u_offsets[-1]].cpu().numpy()
    g_offsets = pack["_full_only_seq_offsets"].cpu().tolist()
    g_sample = pack["_full_only_sample_ids"][: g_offsets[-1]].cpu().numpy()
    n = nu + ng
    q_start = nu if query_stream == "gen" else 0
    q_len = nu if query_stream == "und" else ng if query_stream == "gen" else n
    kv_len = nu if query_stream == "und" else n
    if not q_len or not kv_len:
        raise ValueError("An empty attention stream does not need a BlockMask")
    rounded = ((n + block_size - 1) // block_size) * block_size
    # Extra entries make speculative block-rounded mask loads safe.
    sample = np.full(rounded + block_size, -1, dtype=np.int64)
    segment = np.full_like(sample, -1)
    sample[: len(u_sample)] = u_sample
    sample[nu : nu + len(g_sample)] = g_sample
    for i, (a, b) in enumerate(zip(u_offsets, u_offsets[1:])):
        segment[a:b] = i
    sample_t = torch.from_numpy(sample).to(device)
    segment_t = torch.from_numpy(segment).to(device)

    def allowed(b, h, q, k):
        q = q + q_start
        real = (sample_t[q] >= 0) & (sample_t[q] == sample_t[k])
        causal = (k < nu) & (segment_t[q] == segment_t[k]) & (k <= q)
        return (real & ((q >= nu) | causal)) | ((sample_t[q] < 0) & (q == k))

    nq, nk = (q_len + block_size - 1) // block_size, (kv_len + block_size - 1) // block_size
    present = np.zeros((nq, nk), dtype=bool)
    full = np.zeros_like(present)

    def rectangle(qa, qb, ka, kb, causal=False):
        qa, qb = max(qa, q_start), min(qb, q_start + q_len)
        ka, kb = max(ka, 0), min(kb, kv_len)
        if qa >= qb or ka >= kb:
            return
        qi = np.arange((qa - q_start) // block_size, (qb - q_start - 1) // block_size + 1)
        ki = np.arange(ka // block_size, (kb - 1) // block_size + 1)
        qlo, klo = q_start + qi[:, None] * block_size, ki[None, :] * block_size
        qhi, khi = np.minimum(qlo + block_size, q_start + q_len), np.minimum(klo + block_size, kv_len)
        any_pair = np.ones((len(qi), len(ki)), dtype=bool)
        all_pairs = (qlo >= qa) & (qhi <= qb) & (klo >= ka) & (khi <= kb)
        if causal:
            any_pair &= np.minimum(qhi, qb) - 1 >= np.maximum(klo, ka)
            all_pairs &= qlo >= khi - 1
        ix = np.ix_(qi, ki)
        present[ix] |= any_pair
        full[ix] |= all_pairs

    for a, b in zip(u_offsets, u_offsets[1:]):
        rectangle(a, b, a, b, causal=True)

    # Runs, not token pairs; handles interleaved samples and multiple UND segments.
    def runs(ids, offset):
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        return [(int(ids[a]), a + offset, b + offset) for a, b in zip(boundaries, boundaries[1:]) if a < b]

    u_runs, g_runs = runs(u_sample, 0), runs(g_sample, nu)
    for sid, qa, qb in g_runs:
        for kid, ka, kb in u_runs + g_runs:
            if sid == kid:
                rectangle(qa, qb, ka, kb)
    for a, b in ((len(u_sample), nu), (nu + len(g_sample), n)):
        a, b = max(a, q_start), min(b, q_start + q_len, kv_len)
        if a < b:
            # Only the diagonal is visible; never mark a padding block full.
            ids = np.arange(a, b)
            present[(ids - q_start) // block_size, ids // block_size] = True
    partial_counts, partial_indices = _ordered_blocks(present & ~full, device)
    full_counts, full_indices = _ordered_blocks(full, device)
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=allowed,
        seq_lengths=(q_len, kv_len),
    )


def joint_flex_attention(qs, ks, vs, block_mask, *, scale=None):
    """One Flex call for both streams. Inputs/outputs use [S,H,D]."""
    und_requires_grad = any(t.requires_grad for t in (qs[0], ks[0], vs[0]))
    q, k, v = [torch.cat(parts).transpose(0, 1).unsqueeze(0) for parts in (qs, ks, vs)]
    fn = _compiled_flex if q.is_cuda else flex_attention
    y = fn(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=q.shape[1] != k.shape[1],
        scale=scale,
        kernel_options={"BACKEND": "TRITON"},
    )
    yu, yg = y.squeeze(0).transpose(0, 1).split([qs[0].shape[0], qs[1].shape[0]])
    if not und_requires_grad:
        yu = yu.detach()
    return yu, yg


def _split_flex(qs, ks, vs, metadata, scale):
    fn = _compiled_flex if qs[0].is_cuda else flex_attention
    outputs = []
    for i in range(2):
        q = qs[i].transpose(0, 1).unsqueeze(0)
        k = (ks[0] if i == 0 else torch.cat(ks)).transpose(0, 1).unsqueeze(0)
        v = (vs[0] if i == 0 else torch.cat(vs)).transpose(0, 1).unsqueeze(0)
        if qs[i].shape[0] == 0:
            outputs.append(qs[i])
            continue
        y = fn(
            q,
            k,
            v,
            block_mask=metadata["und_mask" if i == 0 else "gen_mask"],
            enable_gqa=q.shape[1] != k.shape[1],
            scale=scale,
            kernel_options={"BACKEND": "TRITON"},
        )
        outputs.append(y.squeeze(0).transpose(0, 1))
    return outputs


def prepare_joint_attention(pack, backend):
    """Prepare per-microbatch metadata outside Dynamo and checkpoint regions."""
    if backend not in JOINT_ATTENTION_BACKENDS or backend == "split_default":
        raise ValueError(f"Expected explicit joint attention backend, got {backend!r}")
    if pack["is_sharded"]:
        raise ValueError("Joint attention backends require CP=1")
    if pack["_causal_indices"].numel() == 0 or pack["_full_indices"].numel() == 0:
        raise ValueError("Two-tower joint attention requires nonempty real UND and GEN streams")
    metadata = {"backend": backend}
    if backend == "unified_flex_triton":
        metadata["block_mask"] = build_joint_block_mask(pack)
    elif backend == "split_flex_triton":
        for stream in ("und", "gen"):
            metadata[stream + "_mask"] = build_joint_block_mask(pack, query_stream=stream)
    else:
        from cosmos_framework.model.generator.mot.joint_flash_attention import prepare_flash_metadata

        metadata.update(prepare_flash_metadata(pack, backend))
    return metadata


def joint_attention(q_pack, k_pack, v_pack, metadata, *, scale=None):
    """Configured two-tower dispatcher, after projections/norm/RoPE."""
    qs, ks, vs = [(get_und_seq(p), get_gen_seq(p)) for p in (q_pack, k_pack, v_pack)]
    backend = metadata["backend"]
    if backend == "unified_flex_triton":
        yu, yg = joint_flex_attention(qs, ks, vs, metadata["block_mask"], scale=scale)
    elif backend == "split_flex_triton":
        yu, yg = _split_flex(qs, ks, vs, metadata, scale)
    else:
        from cosmos_framework.model.generator.mot.joint_flash_attention import joint_flash_attention

        fixed = bool(q_pack.get("_fixed_sequence_length", False))
        all_k = get_all_seq(k_pack, include_padding=fixed)
        all_v = get_all_seq(v_pack, include_padding=fixed)
        yu, yg = joint_flash_attention(qs, ks[0], vs[0], all_k, all_v, metadata, scale)
    return from_und_gen_splits(yu.flatten(-2), yg.flatten(-2), q_pack)
