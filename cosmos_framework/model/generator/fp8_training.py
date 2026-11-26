# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FP8 GEMM training with ordinary, checkpoint-compatible master parameters."""

import hashlib
import importlib.metadata
import json
import re

import torch
from torch import nn

from cosmos_framework.configs.base.defaults.fp8_training import (
    GEN_MLP_PATTERN,
    FP8TrainingConfig,
    validate_fp8_training_config,
)
from cosmos_framework.utils import log


def convert_fp8_training(model: nn.Module, config: FP8TrainingConfig) -> dict:
    """Convert before AC/compile/FSDP; do not quantize storage or communications.

    Disabled calls are a no-op and do not import TorchAO or initialize CUDA.
    The returned manifest is checkpoint metadata, not a persistent model buffer.
    """
    if not config.enabled:
        return {"version": 1, "enabled": False}
    validate_fp8_training_config(config)
    include = [re.compile(pattern) for pattern in config.include_regex]
    exclude = [re.compile(pattern) for pattern in config.exclude_regex]
    selected = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(pattern.search(name) for pattern in include)
        and not any(pattern.search(name) for pattern in exclude)
    }
    if not selected:
        raise ValueError("No nn.Linear modules matched the FP8 training selection")
    modules = dict(model.named_modules())
    for name, module in selected.items():
        if not re.fullmatch(GEN_MLP_PATTERN, name) or type(module) is not nn.Linear:
            raise ValueError(f"FP8 training only supports plain generation MLP Linear modules, got {name}")
        if type(modules[name.rsplit(".", 1)[0]]).__name__ != "Qwen3VLTextMLP":
            raise ValueError(f"FP8 training currently supports the Nano dense Qwen3VLTextMLP only: {name}")
        if module.in_features % 16 or module.out_features % 16:
            raise ValueError(f"FP8 training requires Linear dimensions divisible by 16: {name} {module.weight.shape}")
    if not torch.accelerator.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise ValueError("FP8 training requires a Hopper or newer CUDA GPU")

    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    recipe = Float8LinearConfig.from_recipe_name(config.recipe)
    # Do not enable FP8 all-gather: it changes the parameter subclass and is not
    # supported by this rowwise recipe in the pinned TorchAO version.
    assert not recipe.enable_fsdp_float8_all_gather
    parameters = dict(model.named_parameters())
    keys = list(model.state_dict())
    convert_to_float8_training(model, module_filter_fn=lambda module, name: name in selected, config=recipe)
    after = dict(model.named_parameters())
    if list(after) != list(parameters) or any(after[name] is not value for name, value in parameters.items()):
        raise RuntimeError("FP8 conversion changed parameter identity or registration order")
    if list(model.state_dict()) != keys:
        raise RuntimeError("FP8 conversion changed checkpoint keys")

    targets = {name: list(module.weight.shape) for name, module in sorted(selected.items())}
    manifest = {
        "version": 1,
        "enabled": True,
        "backend": config.backend,
        "recipe": config.recipe,
        "targets": targets,
        "target_sha256": hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest(),
        "torch": torch.__version__,
        "torchao": importlib.metadata.version("torchao"),
        "cuda": torch.version.cuda,
        "float8_all_gather": False,
    }
    if torch.distributed.is_initialized():
        manifests = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(manifests, manifest)
        if any(value != manifest for value in manifests):
            raise RuntimeError("FP8 training configuration, targets or versions differ across ranks")
    log.info(
        f"FP8 training: recipe={config.recipe}, linears={len(targets)}, "
        f"parameters={sum(module.weight.numel() for module in selected.values())}, "
        f"conversion_dtypes={sorted({str(module.weight.dtype) for module in selected.values()})}, "
        f"torch={manifest['torch']}, torchao={manifest['torchao']}, cuda={manifest['cuda']}, "
        f"sha256={manifest['target_sha256']}"
    )
    log.info(f"FP8 training targets: {json.dumps(targets, sort_keys=True)}", rank0_only=True)
    return manifest


def validate_fp8_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer, manifest: dict) -> None:
    """Check after the optimizer factory applies the existing freeze policy."""
    if not manifest.get("enabled"):
        return
    optimizers = getattr(optimizer, "optimizers", [optimizer])
    optimizer_ids = {
        id(parameter) for inner in optimizers for group in inner.param_groups for parameter in group["params"]
    }
    canonical = {
        ".".join(part for part in name.split(".") if part not in {"_orig_mod", "_checkpoint_wrapped_module"}): parameter
        for name, parameter in model.named_parameters()
    }
    for name in manifest["targets"]:
        weight = canonical[f"{name}.weight"]
        if not weight.requires_grad or id(weight) not in optimizer_ids:
            raise ValueError(f"FP8 training target is not updated by the optimizer: {name}")
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"FP8 training must preserve BF16 parameter storage: {name} {weight.dtype}")
    log.info(f"FP8 optimizer validation: {len(manifest['targets'])} targets, BF16 persistent parameters")


def validate_precision_resume(current: str, saved: str | None, *, warm_start: bool) -> None:
    """Require the same recipe/targets/software on same-job resume, allow migration."""
    if saved is None:
        if json.loads(current).get("enabled"):
            log.warning("Legacy checkpoint has no training precision metadata; initializing dynamic FP8 scaling")
        return
    if json.loads(current) == json.loads(saved):
        return
    if not warm_start:
        raise ValueError(
            "Training precision changed during same-job resume. Use a new job/output directory with "
            "checkpoint.load_path and load_training_state=true for an explicit precision migration."
        )
    log.warning("Explicit checkpoint warm-start changes the training precision; this is a new training trajectory")
