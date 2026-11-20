# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independent GEN_ACTION parameters with strictly modality-routed computation.

Video/action modules process only their own token rows. Attention can reuse
Nano's two calls or use one joint FlexAttention call; visibility is unchanged.
Redundant full-GEN projection is intentionally absent from production code.
"""

from __future__ import annotations

import copy

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_und_gen_splits,
    get_gen_seq,
    get_real_token_mask,
    get_und_seq,
)
from cosmos_framework.model.generator.mot.joint_attention import build_joint_block_mask, joint_flex_attention


def source_key(key: str) -> str | None:
    """Map each additional parameter to its original Nano source."""
    if "_moe_gen_action" in key:
        return key.replace("_moe_gen_action", "_moe_gen")
    for target, source in (("action_time_embedder.", "time_embedder."),):
        if target in key:
            return key.replace(target, source)
    return None


def defer_copy_initialization(state_dict, checkpoint_keys, *, skipped=()):
    """Remove new destinations from a load request; require complete old weights.

    Returns destination buffers to fill *after* the checkpoint has been loaded.
    This works for regular tensors and FSDP DTensors without materializing a second
    full model. Never use it for resuming an already-trained three-tower checkpoint.
    """
    pending = {}
    for key in list(state_dict):
        source = source_key(key)
        if source is None:
            continue
        if key in checkpoint_keys:
            raise ValueError("copy_gen requires a two-tower checkpoint; use checkpoint initialization to resume")
        # A warm start may deliberately omit EMA. Seed its independent new
        # parameters from the loaded regular source, as the existing EMA loader
        # does for the old parameters. Never use an uninitialized EMA buffer.
        if source.startswith("net_ema.") and (source not in checkpoint_keys or any(p in source for p in skipped)):
            source = "net." + source.removeprefix("net_ema.")
        if source not in checkpoint_keys or any(pattern in source for pattern in skipped):
            raise ValueError(f"Cannot initialize {key}: source {source} missing or skipped")
        pending[key] = (state_dict.pop(key), source)
    if not pending:
        raise ValueError("copy_gen requested but no GEN_ACTION parameters were constructed")
    return pending


def validate_checkpoint_initialization(state_dict, checkpoint_keys, *, skipped=()):
    """Do not silently leave new parameters random on a partial DCP load."""
    for key, value in state_dict.items():
        skip = any(pattern in key for pattern in skipped)
        if source_key(key) is not None and (key not in checkpoint_keys or skip):
            raise ValueError(
                f"Missing three-tower checkpoint parameter {key}; use copy_gen only for an old two-tower checkpoint"
            )
        # Warm starts may explicitly replace legacy action projections with the
        # Action128 heads. The loader also skips these tensors before shape checks.
        # Never exempt the mandatory tower parameters checked above, or a resume
        # (whose caller supplies no skip patterns).
        if not skip and hasattr(checkpoint_keys, "get") and key in checkpoint_keys:
            metadata = checkpoint_keys.get(key)
            shape = metadata.shape if isinstance(metadata, torch.Tensor) else getattr(metadata, "size", None)
            if shape is not None and tuple(shape) != tuple(value.shape):
                raise ValueError(f"Checkpoint architecture mismatch for {key}: {tuple(shape)} != {tuple(value.shape)}")


@torch.no_grad()
def finish_copy_initialization(state_dict, pending):
    for key, (destination, source) in pending.items():
        destination.copy_(state_dict[source])
        state_dict[key] = destination


def install_action_tower(net):
    """Construct independent modules before parallelization/materialization."""
    if getattr(net.config, "action_tower_preset", "full") != "full":
        from .small_action_tower import install_small_action_tower

        return install_small_action_tower(net)
    for layer in net.language_model.model.layers:
        if not hasattr(layer, "mlp_moe_gen") or hasattr(layer.mlp_moe_gen, "experts"):
            raise ValueError("three_tower currently requires the dense Nano MoT decoder")
        for name in ("mlp", "input_layernorm", "post_attention_layernorm"):
            setattr(layer, name + "_moe_gen_action", copy.deepcopy(getattr(layer, name + "_moe_gen")))
        attn = layer.self_attn
        if attn.k_norm_und_for_gen is not None:
            raise ValueError("three_tower requires the same UND K representation for all queries")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            setattr(attn, name + "_moe_gen_action", copy.deepcopy(getattr(attn, name + "_moe_gen")))
    model = net.language_model.model
    model.norm_moe_gen_action = copy.deepcopy(model.norm_moe_gen)
    net.action_time_embedder = copy.deepcopy(net.time_embedder)


@torch.no_grad()
def copy_from_gen(net):
    """Explicit in-memory conversion after loading the complete baseline."""
    state = net.state_dict()
    for key, value in state.items():
        source = source_key(key)
        if source is not None:
            value.copy_(state[source])


def resolve_execution(execution, attention_backend="auto", projection_mode="auto"):
    """Resolve attention presets; production projections always route by modality."""
    if execution not in ("native", "nano_compatible", "joint"):
        raise ValueError(f"Invalid action_tower_execution: {execution}")
    if attention_backend not in ("auto", "nano", "flex", "sdpa", "flash3_video", "flash3"):
        raise ValueError(f"Invalid action_tower_attention_backend: {attention_backend}")
    if projection_mode not in ("auto", "routed"):
        raise ValueError(
            f"Invalid action_tower_projection_mode: {projection_mode}; production requires routed tokens. "
            "Full-GEN computation is available only in standalone diagnostic tools."
        )
    if attention_backend == "auto":
        attention_backend = "flex" if execution == "joint" else "nano"
    return attention_backend, "routed"


def prepare_joint_metadata(
    pack, action_indexes, execution="joint", *, attention_backend="auto", projection_mode="auto"
):
    """Build the shared block mask outside decoder/checkpointed regions."""
    if pack["is_sharded"]:
        raise ValueError("three_tower currently requires context_parallel_shard_degree=1")
    attention_backend, projection_mode = resolve_execution(execution, attention_backend, projection_mode)
    device = get_gen_seq(pack).device
    full_indices = pack["_full_indices"].long()
    is_action = torch.isin(full_indices, action_indexes.to(device))
    pack["_action_indices"] = torch.where(is_action)[0]
    # Padding belongs to video routing and is zeroed after each layer.
    mask = torch.zeros(get_gen_seq(pack).shape[0], dtype=torch.bool, device=device)
    mask[: len(is_action)] = is_action
    pack["_video_indices"] = torch.where(~mask)[0]
    pack["_action_tower_execution"] = execution
    pack["_action_tower_attention_backend"] = attention_backend
    pack["_action_tower_projection_mode"] = projection_mode
    if attention_backend in ("nano", "sdpa"):
        return

    pack["_joint_block_mask"] = build_joint_block_mask(pack)


def routed_gen(video_module, action_module, x, pack):
    vi, ai = pack["_video_indices"], pack["_action_indices"]
    video = video_module(x[vi])
    action = action_module(x[ai])
    out = video.new_zeros((x.shape[0], *video.shape[1:]))
    return out.index_copy(0, vi, video).index_copy(0, ai, action)


def layer_forward(layer, pack, positions, *, attention_mask=None, memory_value=None, gen_only=False):
    if memory_value is not None or gen_only:
        raise ValueError("three_tower currently supports joint diffusion without autoregressive memory")
    attn = layer.self_attn
    u, g = get_und_seq(pack), get_gen_seq(pack)
    un = layer.input_layernorm(u)
    gn = routed_gen(layer.input_layernorm_moe_gen, layer.input_layernorm_moe_gen_action, g, pack)
    qs, ks, vs = [], [], []
    for name, x, get_positions in (("und", un, get_und_seq), ("gen", gn, get_gen_seq)):

        def project(base, value):
            if name == "und":
                return getattr(attn, base)(value)
            return routed_gen(getattr(attn, base + "_moe_gen"), getattr(attn, base + "_moe_gen_action"), value, pack)

        q = project("q_proj", x).view(-1, attn.num_attention_heads, attn.head_dim)
        k = project("k_proj", x).view(-1, attn.num_key_value_heads, attn.head_dim)
        v = project("v_proj", x).view(-1, attn.num_key_value_heads, attn.head_dim)
        q, k = project("q_norm", q), project("k_norm", k)
        q, k = attn._apply_rotary_pos_emb(
            q, k, get_positions(positions[0]), get_positions(positions[1]), unsqueeze_dim=1
        )
        qs.append(q)
        ks.append(k)
        vs.append(v)
    und_attention_requires_grad = any(t.requires_grad for t in (qs[0], ks[0], vs[0]))
    if pack["_action_tower_attention_backend"] == "nano":
        qp, kp, vp = [from_und_gen_splits(*parts, pack) for parts in (qs, ks, vs)]
        yp, _ = attn.dispatch_attention_fn(qp, kp, vp, attention_mask)
        yu, yg = get_und_seq(yp), get_gen_seq(yp)
    else:
        yu, yg = joint_flex_attention(qs, ks, vs, pack["_joint_block_mask"], scale=attn.scaling)
        yu, yg = yu.flatten(1), yg.flatten(1)
    if not und_attention_requires_grad:
        # Joint attention concatenates frozen UND and trainable GEN Q/K/V.
        # Its UND result consequently has a grad_fn even though the mask makes
        # UND independent of GEN. Stop that artificial graph at this boundary;
        # never detach when UND Q/K/V have a real trainable upstream source.
        yu = yu.detach()
    u = u + attn.o_proj(yu)
    g = g + routed_gen(attn.o_proj_moe_gen, attn.o_proj_moe_gen_action, yg, pack)
    u = u + layer.mlp(layer.post_attention_layernorm(u))
    gn = routed_gen(layer.post_attention_layernorm_moe_gen, layer.post_attention_layernorm_moe_gen_action, g, pack)
    g = g + routed_gen(layer.mlp_moe_gen, layer.mlp_moe_gen_action, gn, pack)
    u = torch.where(get_real_token_mask(pack, "und")[:, None], u, 0)
    g = torch.where(get_real_token_mask(pack, "gen")[:, None], g, 0)
    return from_und_gen_splits(u, g, pack), {}, None
