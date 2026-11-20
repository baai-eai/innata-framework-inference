# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional post-load transforms for immutable single-GPU small Action models."""

import torch


def fuse_small_action_qkv(model) -> int:
    """Build nonpersistent QKV buffers after checkpoint loading, before warmup.

    Original parameters/state_dict remain intact. Re-run after any weight update;
    these deployment buffers are only valid while the loaded weights stay fixed.
    Returns the extra buffer bytes (include them in deployment memory budgeting).
    """
    net = model.net
    decoder = net.language_model.model
    if model.training or not hasattr(decoder, "small_action_shape"):
        raise ValueError("QKV fusion requires an eval small Action model")
    if model.parallel_dims is not None and (
        model.parallel_dims.cp_enabled or model.parallel_dims.cfgp_enabled or model.parallel_dims.dp_shard_enabled
    ):
        raise ValueError("QKV fusion currently requires one GPU per replica")
    total = 0
    with torch.no_grad():
        for layer in decoder.layers:
            for module, suffix in (
                (layer.self_attn, ""),
                (layer.self_attn, "_moe_gen"),
                (getattr(layer, "action_block_moe_gen_action", None), ""),
            ):
                if module is None:
                    continue
                projections = [getattr(module, name + suffix) for name in ("q_proj", "k_proj", "v_proj")]
                weight = torch.cat([p.weight.detach() for p in projections], dim=0)
                biases = [p.bias for p in projections]
                if any(b is not None for b in biases):
                    bias = torch.cat(
                        [
                            p.bias.detach() if p.bias is not None else p.weight.new_zeros(p.out_features)
                            for p in projections
                        ]
                    )
                else:
                    bias = None
                module.register_buffer("_inference_qkv_weight" + suffix, weight, persistent=False)
                module.register_buffer("_inference_qkv_bias" + suffix, bias, persistent=False)
                total += weight.numel() * weight.element_size()
                if bias is not None:
                    total += bias.numel() * bias.element_size()
    return total
