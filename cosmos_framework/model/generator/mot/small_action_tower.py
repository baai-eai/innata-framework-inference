# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heterogeneous action experts with simultaneous, bidirectional KV exchange.

Only the pack boundary pads action features for legacy modality scatter/gather.
Inside the decoder, UND, video and action are physically separate tensors.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder


@dataclass(frozen=True)
class ActionShape:
    hidden: int
    intermediate: int
    q_heads: int
    kv_heads: int
    head_dim: int
    layers: tuple[int, ...]


def action_shape(preset: str) -> ActionShape:
    if preset == "depth_1b":
        return ActionShape(4096, 12288, 32, 8, 128, (0, 9, 18, 26, 35))
    if preset == "width_heads_1b":
        return ActionShape(1536, 4608, 12, 4, 128, tuple(range(36)))
    if preset == "width_head_dim_1b":
        return ActionShape(1536, 4608, 32, 8, 48, tuple(range(36)))
    if preset == "width_only_1b":
        return ActionShape(1280, 3840, 32, 8, 128, tuple(range(36)))
    raise ValueError(f"Unknown small action preset: {preset}")


class AxisAdapter(nn.Module):
    """Small, explicit linear map on heads or head coordinates (never hidden)."""

    def __init__(self, source_heads, target_heads, source_dim, target_dim):
        super().__init__()
        self.head_axis = source_heads != target_heads
        if self.head_axis:
            assert source_dim == target_dim
            self.linear = nn.Linear(source_heads, target_heads, bias=False)
        else:
            self.linear = nn.Linear(source_dim, target_dim, bias=False)

    def forward(self, x):
        if self.head_axis:
            return self.linear(x.transpose(-1, -2)).transpose(-1, -2)
        return self.linear(x)


class ActionBlock(nn.Module):
    def __init__(self, shape, norm_type, eps, base_heads=8, base_dim=128):
        super().__init__()
        self.shape = shape
        d, m, h, k, dh = shape.hidden, shape.intermediate, shape.q_heads, shape.kv_heads, shape.head_dim
        self.input_layernorm = norm_type(d, eps=eps)
        self.post_attention_layernorm = norm_type(d, eps=eps)
        self.q_norm = norm_type(dh, eps=eps)
        self.k_norm = norm_type(dh, eps=eps)
        self.q_proj = nn.Linear(d, h * dh, bias=False)
        self.k_proj = nn.Linear(d, k * dh, bias=False)
        self.v_proj = nn.Linear(d, k * dh, bias=False)
        self.o_proj = nn.Linear(h * dh, d, bias=False)
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)
        self.adapters = nn.ModuleDict()
        if (k, dh) != (base_heads, base_dim):
            for direction in ("und_to_action", "video_to_action", "action_to_video"):
                incoming = direction != "action_to_video"
                src, dst = ((base_heads, base_dim), (k, dh)) if incoming else ((k, dh), (base_heads, base_dim))
                for kind in ("k", "v"):
                    self.adapters[direction + "_" + kind] = AxisAdapter(src[0], dst[0], src[1], dst[1])

    def adapt(self, x, direction, kind):
        name = direction + "_" + kind
        return self.adapters[name](x) if name in self.adapters else x

    def mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def install_small_action_tower(net):
    shape = action_shape(net.config.action_tower_preset)
    model = net.language_model.model
    attn = model.layers[0].self_attn
    if (len(model.layers), net.hidden_size, attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim) != (
        36,
        4096,
        32,
        8,
        128,
    ):
        raise ValueError("1B presets require the dense Nano 36-layer 4096/32Q/8KV/128 backbone")
    if net.config.action_tower_init != "checkpoint":
        raise ValueError(
            "Small towers require an explicitly converted checkpoint; use prepare_small_action_checkpoint.py"
        )
    if net.config.action_tower_attention_backend not in ("auto", "sdpa", "flex", "flash3_video", "flash3"):
        raise ValueError("Small towers require separate attention calls (auto, sdpa, flex, flash3_video or flash3)")
    if net.config.action_tower_execution != "native":
        raise ValueError("Small towers require native execution")
    norm_type = type(model.norm_moe_gen)
    eps = model.norm_moe_gen.variance_epsilon
    model.small_action_shape = shape
    model.small_action_backend = (
        net.config.action_tower_attention_backend
        if net.config.action_tower_attention_backend in ("flex", "flash3_video", "flash3")
        else "sdpa"
    )
    model.small_action_fixed_lengths = None
    model.small_action_und_short_tokens = net.config.action_tower_und_short_tokens
    if model.small_action_und_short_tokens and (
        not net.config.fixed_sequence_length_enabled
        or not 0 < model.small_action_und_short_tokens < net.config.fixed_und_tokens
    ):
        raise ValueError("Short UND bucket requires fixed shapes and 0 < short < fixed UND capacity")
    if net.config.fixed_sequence_length_enabled:
        action_tokens = net.config.fixed_action_tokens
        if not 0 < action_tokens < net.config.fixed_gen_tokens:
            raise ValueError("Fixed small action towers require 0 < action_tokens < gen_tokens")
        model.small_action_fixed_lengths = (
            net.config.fixed_und_tokens,
            net.config.fixed_gen_tokens - action_tokens,
            action_tokens,
        )
    for i, layer in enumerate(model.layers):
        layer.small_action_enabled = True
        if i in shape.layers:
            layer.action_block_moe_gen_action = ActionBlock(shape, norm_type, eps)
    model.norm_moe_gen_action = norm_type(shape.hidden, eps=eps)
    net.action_hidden_size = shape.hidden
    # Projections are created by Cosmos3VFMNetwork after the action hidden size is known.
    net.action_modality_embed = nn.Parameter(torch.zeros(shape.hidden))
    net.action_time_embedder = TimestepEmbedder(shape.hidden, bias=net.config.enable_input_bias)


def action_rope(position_ids, dtype, head_dim, theta=5000000.0):
    # Same interleaved Qwen MRoPE convention, with an explicit small-head layout.
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).expand(3, -1)
    section = (9, 8, 7) if head_dim == 48 else (24, 20, 20)
    inv = theta ** (-torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
    freq = position_ids.float().unsqueeze(-1) * inv
    out = freq[0].clone()
    for axis in (1, 2):
        idx = slice(axis, section[axis] * 3, 3)
        out[..., idx] = freq[axis, ..., idx]
    emb = torch.cat((out, out), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate(x, position):
    cos, sin = position
    a, b = x.chunk(2, -1)
    return x * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]


def compact_pack(
    pack, positions, position_ids, shape, fixed_lengths=None, attention_backend="sdpa", und_short_tokens=0
):
    """Called once outside checkpointed layers, after ordinary pack/RoPE creation."""
    out = dict(pack)
    nu, ng = int(pack["_num_causal_tokens"]), int(pack["_num_full_tokens"])
    ai = pack["_action_indices"]
    vi = pack["_video_indices"]
    vi = vi[vi < ng]
    out["_small_original_u_len"] = len(pack["causal_seq"])
    out["_small_original_g_len"] = len(pack["full_only_seq"])
    out["_small_vi"] = vi
    out["causal_seq"] = pack["causal_seq"][:nu]
    out["action_seq"] = pack["full_only_seq"][ai, : shape.hidden].contiguous()
    out["full_only_seq"] = pack["full_only_seq"][vi]
    ids = [pack["_causal_sample_ids"][:nu], pack["_full_only_sample_ids"][vi], pack["_full_only_sample_ids"][ai]]
    real_lengths = (nu, len(vi), len(ai))
    out["_small_real_lengths"] = real_lengths
    targets = fixed_lengths or real_lengths
    if fixed_lengths and und_short_tokens and nu <= und_short_tokens:
        targets = (und_short_tokens, targets[1], targets[2])
    for i, key in enumerate(("causal_seq", "full_only_seq", "action_seq")):
        if targets[i] < real_lengths[i]:
            raise ValueError(f"Small tower {key} has {real_lengths[i]} tokens, exceeding capacity {targets[i]}")
        if fixed_lengths is not None:
            out[key] = F.pad(out[key], (0, 0, 0, targets[i] - real_lengths[i]))
            ids[i] = F.pad(ids[i], (0, targets[i] - real_lengths[i]), value=-1)
    out["_small_ids"] = ids
    # Masks are built once, never per layer. Empty queries are handled explicitly.
    out["_small_masks"] = {}

    def mask_for(query_ids, key_ids, *, causal=False, video=False):
        if (attention_backend == "flash3" or (attention_backend == "flash3_video" and video)) and len(query_ids):
            if attention_backend == "flash3" and fixed_lengths is not None:
                raise ValueError("All-stream flash3 requires unpadded inference sequences")
            from .small_action_attention import video_varlen_mask

            return video_varlen_mask(
                query_ids,
                key_ids,
                len(pack["sample_offsets"]) - 1,
                fixed_capacity=fixed_lengths is not None,
                causal=causal,
            )
        if attention_backend in ("flex", "flash3_video") and len(query_ids) and len(key_ids):
            from .small_action_attention import sample_block_mask

            return sample_block_mask(query_ids, key_ids, causal=causal)
        if fixed_lengths is None:
            real = query_ids[:, None] == key_ids[None, :]
        else:
            real = (query_ids[:, None] >= 0) & (query_ids[:, None] == key_ids[None, :])
            # Padding queries attend one dummy key to keep every backend finite;
            # real queries never attend padding keys. Padded outputs are discarded.
            real = real | ((query_ids[:, None] < 0) & (torch.arange(len(key_ids), device=key_ids.device)[None] == 0))
        if causal:
            n = torch.arange(len(query_ids), device=query_ids.device)
            real = real & (n[:, None] >= n[None, :])
        return real

    for active in (False, True):
        ks = torch.cat(ids if active else ids[:2])
        out["_small_masks"][(active, 1)] = mask_for(ids[1], ks, video=True)
        if active:
            out["_small_masks"][(active, 2)] = mask_for(ids[2], ks)
    out["_small_masks"][(False, 0)] = mask_for(ids[0], ids[0], causal=True)

    # all position tensors still have their original token order.
    out["_small_rope"] = tuple(
        tuple((pair[0]["causal_seq"][:nu], pair[1]["causal_seq"][:nu]))
        if i == 0
        else (pair[0]["full_only_seq"][ix], pair[1]["full_only_seq"][ix])
        for i, ix in enumerate((None, vi, ai))
        for pair in (positions,)
    )
    from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq

    ar = action_rope(position_ids, out["action_seq"].dtype, shape.head_dim)
    small_pos = tuple(from_all_seq(x, pack) for x in ar)
    out["_small_action_rope"] = (
        (small_pos[0]["causal_seq"][:nu], small_pos[1]["causal_seq"][:nu]),
        (small_pos[0]["full_only_seq"][vi], small_pos[1]["full_only_seq"][vi]),
        (small_pos[0]["full_only_seq"][ai], small_pos[1]["full_only_seq"][ai]),
    )
    if fixed_lengths is not None:
        for key in ("_small_rope", "_small_action_rope"):
            out[key] = tuple(
                tuple(F.pad(t, (0, 0, 0, targets[i] - real_lengths[i])) for t in pair)
                for i, pair in enumerate(out[key])
            )
    return out


def small_decoder_inputs(pack):
    """Keep variable gather indices and Python counts outside compiled layers."""
    return {
        key: pack[key]
        for key in ("causal_seq", "full_only_seq", "action_seq", "_small_masks", "_small_rope", "_small_action_rope")
    }


def expand_pack(pack, model):
    out = dict(pack)
    nu, nv, na = pack["_small_real_lengths"]
    u = model.norm(pack["causal_seq"][:nu])
    v = model.norm_moe_gen(pack["full_only_seq"][:nv])
    a = model.norm_moe_gen_action(pack["action_seq"][:na])
    out["causal_seq"] = F.pad(u, (0, 0, 0, pack["_small_original_u_len"] - len(u)))
    g = v.new_zeros((pack["_small_original_g_len"], v.shape[-1]))
    g = g.index_copy(0, pack["_small_vi"], v)
    out["full_only_seq"] = g.index_copy(0, pack["_action_indices"], F.pad(a, (0, v.shape[-1] - a.shape[-1])))
    out.pop("action_seq")
    return out


def attend(q, k, v, mask, backend="sdpa", *, frozen_und=False):
    from .small_action_attention import VideoVarlenMask, video_varlen_attend

    if q.shape[0] == 0:
        return q.flatten(1) + (k.sum() + v.sum()) * 0
    if isinstance(mask, VideoVarlenMask):
        return video_varlen_attend(q, k, v, mask)
    if not isinstance(mask, torch.Tensor):
        from .small_action_attention import sparse_attend

        return sparse_attend(q, k, v, mask)
    # SDPA supplies a correct GQA implementation for all three head layouts,
    # including Dh=48. It selects a supported CUDA kernel rather than assuming FA3.
    # PyTorch2.10 AOT full-AC RNG functionalization expects every recomputable
    # cuDNN SDPA op in both forward and backward. Frozen UND has no backward
    # node, causing KeyError. Keep the fallback local to this short stream;
    # trainable VIDEO/ACTION SDPA and all eager execution retain auto dispatch.
    context = sdpa_kernel(SDPBackend.MATH) if frozen_und and torch.compiler.is_compiling() else nullcontext()
    with context:
        y = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            attn_mask=mask[None, None],
            enable_gqa=True,
            scale=q.shape[-1] ** -0.5,
        )
    return y[0].transpose(0, 1).flatten(1)


def project_qkv(module, x, suffix=""):
    weight = getattr(module, "_inference_qkv_weight" + suffix, None)
    if weight is not None and not module.training and not torch.is_grad_enabled():
        widths = tuple(getattr(module, name + suffix).out_features for name in ("q_proj", "k_proj", "v_proj"))
        return F.linear(x, weight, getattr(module, "_inference_qkv_bias" + suffix)).split(widths, dim=-1)
    return tuple(getattr(module, name + suffix)(x) for name in ("q_proj", "k_proj", "v_proj"))


def small_layer_forward(layer, pack, *, memory_value=None, gen_only=False):
    from .inference_small_action_memory import SmallActionMemoryValue

    if gen_only or (memory_value is not None and not isinstance(memory_value, SmallActionMemoryValue)):
        raise ValueError("Small action towers only support request-local UND inference memory")
    if memory_value is not None and (layer.training or torch.is_grad_enabled()):
        raise ValueError("Small action UND memory requires eval and no_grad")
    cached = memory_value is not None and memory_value.und_key is not None
    attn = layer.self_attn
    u, v, a = pack["causal_seq"], pack["full_only_seq"], pack["action_seq"]
    block = getattr(layer, "action_block_moe_gen_action", None)
    active = block is not None
    qs, ks, vs = [], [], []
    streams = (
        (("_moe_gen", layer.input_layernorm_moe_gen(v)),)
        if cached
        else (("", layer.input_layernorm(u)), ("_moe_gen", layer.input_layernorm_moe_gen(v)))
    )
    if cached:
        qs.append(None)
        ks.append(memory_value.und_key)
        vs.append(memory_value.und_value)
    for suffix, x in streams:
        q, k, value = project_qkv(attn, x, suffix)
        qs.append(getattr(attn, "q_norm" + suffix)(q.view(-1, attn.num_attention_heads, attn.head_dim)))
        ks.append(getattr(attn, "k_norm" + suffix)(k.view(-1, attn.num_key_value_heads, attn.head_dim)))
        vs.append(value.view(-1, attn.num_key_value_heads, attn.head_dim))
    if active:
        an = block.input_layernorm(a)
        s = block.shape
        q, k, value = project_qkv(block, an)
        qs.append(block.q_norm(q.view(-1, s.q_heads, s.head_dim)))
        ks.append(block.k_norm(k.view(-1, s.kv_heads, s.head_dim)))
        vs.append(value.view(-1, s.kv_heads, s.head_dim))
    rope = pack["_small_rope"]
    masks = pack["_small_masks"]
    yu = (
        None
        if cached
        else attend(rotate(qs[0], rope[0]), rotate(ks[0], rope[0]), vs[0], masks[(False, 0)], frozen_und=True)
    )
    vk = [rotate(ks[i], rope[i]) for i in (0, 1)]
    vv = vs[:2]
    if active:
        vk.append(rotate(block.adapt(ks[2], "action_to_video", "k"), rope[2]))
        vv.append(block.adapt(vs[2], "action_to_video", "v"))
    yv = attend(rotate(qs[1], rope[1]), torch.cat(vk), torch.cat(vv), masks[(active, 1)])
    if active:
        ar = pack["_small_action_rope"]
        ak = [
            rotate(block.adapt(ks[i], direction, "k"), ar[i])
            for i, direction in enumerate(("und_to_action", "video_to_action"))
        ]
        av = [block.adapt(vs[i], direction, "v") for i, direction in enumerate(("und_to_action", "video_to_action"))]
        ak.append(rotate(ks[2], ar[2]))
        av.append(vs[2])
        ya = attend(rotate(qs[2], ar[2]), torch.cat(ak), torch.cat(av), masks[(True, 2)])
        a = a + block.o_proj(ya)
        a = a + block.mlp(block.post_attention_layernorm(a))
    if cached:
        u = memory_value.und_output
    else:
        u = u + attn.o_proj(yu)
        u = u + layer.mlp(layer.post_attention_layernorm(u))
    v = v + attn.o_proj_moe_gen(yv)
    v = v + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(v))
    out = dict(pack)
    out.update(causal_seq=u, full_only_seq=v, action_seq=a)
    to_store = (u, u.new_empty(0), ks[0], vs[0]) if memory_value is not None and not cached else None
    return out, {}, to_store
