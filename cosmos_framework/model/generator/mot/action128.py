# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action128 projections and independent source-domain inverse transforms."""

import hashlib
import json
import math

import torch
from omegaconf import OmegaConf
from torch import nn

from cosmos_framework.data.generator.action.projection_routing import (
    NUM_PROJECTION_SLOTS,
    PROJECTION_HEAD_IDS,
    PROJECTION_ROUTING_VERSION,
    PUBLISHED_PROJECTION_FAMILY_IDS,
)


def action128_contract_digest(spec, vision_spatial_position_mode):
    """Fingerprint the published contract, including the spatial RoPE convention."""
    contract = (
        spec
        if vision_spatial_position_mode == "canvas"
        else {
            "action128_spec": spec,
            "vision_spatial_position_mode": vision_spatial_position_mode,
            "spatial_coordinate_version": "pixel_seam_patch_corner_v1",
        }
    )
    return hashlib.blake2b(json.dumps(contract, sort_keys=True).encode(), digest_size=16).digest()


class Action128DomainLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_domains: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, input_size * output_size)
        self.bias = nn.Embedding(num_domains, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt(self.input_size)
        nn.init.trunc_normal_(self.fc.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x, domain_id):
        if x.shape[-1] != self.input_size:
            raise ValueError("Action128 projection input width mismatch")
        weights = self.fc(domain_id).view(-1, self.input_size, self.output_size)
        bias = self.bias(domain_id)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weights).squeeze(1) + bias
        return torch.bmm(x, weights) + bias[:, None]


class Action128Normalization(nn.Module):
    """No projection parameters: source domains only select published transforms."""

    def __init__(self, spec: dict, vision_spatial_position_mode: str = "canvas"):
        super().__init__()
        self.spec = OmegaConf.to_container(spec, resolve=True) if OmegaConf.is_config(spec) else spec
        n = len(spec["registry"]["domains"])
        routing = self.spec["projection_routing"]
        keys = routing.get("keys", {})
        registered_keys = {**PROJECTION_HEAD_IDS, **PUBLISHED_PROJECTION_FAMILY_IDS}
        # Older saved specs contain only the then-registered prefix. Accept
        # append-only extensions without rewriting their checkpoint digest.
        if (
            routing.get("version") != PROJECTION_ROUTING_VERSION
            or routing.get("num_slots") != NUM_PROJECTION_SLOTS
            or not keys
            or any(registered_keys.get(key) != value for key, value in keys.items())
            or not set(spec["registry"]["domain_projection_ids"]).issubset(keys.values())
        ):
            raise ValueError("Action128 projection registry does not match the fixed head mapping")
        self.register_buffer("normalization_mask", torch.empty(n, 128, dtype=torch.bool))
        self.register_buffer("scale", torch.empty(n, 128, dtype=torch.float32))
        self.register_buffer("bias", torch.empty(n, 128, dtype=torch.float32))
        from cosmos_framework.data.generator.sequence_packing.spatial import SPATIAL_POSITION_MODES

        if vision_spatial_position_mode not in SPATIAL_POSITION_MODES:
            raise ValueError(f"Unsupported vision_spatial_position_mode: {vision_spatial_position_mode!r}")
        self.vision_spatial_position_mode = vision_spatial_position_mode
        digest = action128_contract_digest(self.spec, vision_spatial_position_mode)
        self._expected_digest = list(digest)
        self.register_buffer("contract_digest", torch.empty(16, dtype=torch.uint8))
        self.reset_contract()
        self.register_load_state_dict_post_hook(self._validate_loaded_contract)

    def _validate_loaded_contract(self, module, incompatible_keys):
        self.validate_contract()
        # Older checkpoints stored these buffers in the model's BF16 dtype.
        # Once the full contract is verified, recover the authoritative FP32
        # constants from that exact contract rather than promote rounded values.
        self._reset_transforms()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Model-wide .to(dtype) and .to_empty must never quantize the physical
        # inverse transform. Do not reset the digest here: mismatches must fail.
        self._reset_transforms()
        return self

    def _reset_transforms(self):
        transforms = self.spec["transforms"]
        mask = torch.tensor([item["normalization_mask"] for item in transforms], dtype=torch.bool, device="cpu")
        scale = torch.tensor([item["scale"] for item in transforms], dtype=torch.float32, device="cpu")
        bias = torch.tensor([item["bias"] for item in transforms], dtype=torch.float32, device="cpu")
        if (
            mask.shape != self.normalization_mask.shape
            or scale.shape != self.scale.shape
            or bias.shape != self.bias.shape
        ):
            raise ValueError("Action128 published transforms must match registry x 128")
        if not torch.isfinite(scale).all() or not torch.isfinite(bias).all() or (scale[mask] <= 0).any():
            raise ValueError("invalid Action128 published transform")
        self.normalization_mask = mask.to(self.normalization_mask.device)
        self.scale = scale.to(self.scale.device)
        self.bias = bias.to(self.bias.device)

    def reset_contract(self):
        self._reset_transforms()
        self.contract_digest.copy_(
            torch.tensor(self._expected_digest, dtype=torch.uint8, device=self.contract_digest.device)
        )

    def validate_contract(self):
        expected = torch.tensor(self._expected_digest, dtype=torch.uint8, device=self.contract_digest.device)
        if not torch.equal(self.contract_digest, expected):
            raise ValueError(
                "Action128 checkpoint registry/transforms or spatial RoPE mode differ from the current training contract"
            )

    def denormalize(self, action, normalization_domain_id):
        self.validate_contract()
        ids = torch.as_tensor(normalization_domain_id).reshape(-1)
        if ids.numel() != 1:
            raise ValueError("denormalize expects one source normalization domain per sample")
        domain_id = ids[0]
        result = action.float().clone()
        mask = self.normalization_mask[domain_id]
        result[..., mask] = (result[..., mask] - self.bias[domain_id, mask]) / self.scale[domain_id, mask]
        return result
