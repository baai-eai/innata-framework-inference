# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training-only FP8 configuration. Importing this module does not import TorchAO."""

import re

import attrs

GEN_MLP_PATTERN = r"^language_model\.model\.layers\.\d+\.mlp_moe_gen\.(gate_proj|up_proj|down_proj)$"
FP8_TRAINING_RECIPES = ("rowwise", "rowwise_with_gw_hp")


@attrs.define(slots=False)
class FP8TrainingConfig:
    enabled: bool = False
    backend: str = "torchao"
    recipe: str = "rowwise"
    include_regex: list[str] = attrs.field(factory=lambda: [GEN_MLP_PATTERN])
    exclude_regex: list[str] = attrs.field(factory=list)


def validate_fp8_training_config(config) -> None:
    if not config.enabled:
        return
    if config.backend != "torchao" or config.recipe not in FP8_TRAINING_RECIPES:
        raise ValueError("FP8 training requires torchao with recipe rowwise or rowwise_with_gw_hp")
    if not config.include_regex:
        raise ValueError("FP8 training requires a nonempty include_regex")
    for pattern in [*config.include_regex, *config.exclude_regex]:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid FP8 training regex {pattern!r}: {exc}") from exc


def validate_fp8_training_model_config(config, *, task: str = "vfm") -> None:
    """Validate both TOML and resolved model configs, including CLI overrides."""
    fp8 = getattr(config, "fp8_training", None)
    if fp8 is None or not fp8.enabled:
        return
    validate_fp8_training_config(fp8)
    if task != "vfm":
        raise ValueError("FP8 training is currently supported only for VFM")
    if config.precision != "bfloat16":
        raise ValueError("FP8 training requires model.precision=bfloat16")
    if config.lora_enabled or config.ema.enabled:
        raise ValueError("FP8 training with LoRA or EMA is not yet supported")
    if config.tower_structure != "two_tower" or config.action_gen or getattr(config, "sound_gen", False):
        raise ValueError("FP8 training currently supports only the two-tower vision generation model")
    if not config.vision_gen:
        raise ValueError("FP8 training requires vision_gen=true")
    if config.parallelism.context_parallel_shard_degree != 1 or config.parallelism.cfg_parallel_shard_degree != 1:
        raise ValueError("FP8 training with context or CFG parallelism is not yet supported")
    quantization = getattr(config, "quantization", None)
    if quantization is not None and (quantization.method is not None or quantization.modelopt_fp8_checkpoint_path):
        raise ValueError("FP8 training cannot be combined with inference PTQ or ModelOpt FP8 checkpoints")
