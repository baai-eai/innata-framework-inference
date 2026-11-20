# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Task-aware fixed sequence-length contract for unified MoT training."""

from typing import Any

import attrs


@attrs.define(slots=False)
class FixedSequenceLengthConfig:
    """Physical decoder token lengths before context-parallel sharding.

    The contract is opt-in. When enabled, packed real tokens are padded so
    every unified MoT decoder layer receives one of two physical profiles:
    pure Understanding uses ``(understanding_tokens, 0)`` while Generation /
    Action uses ``(und_tokens, gen_tokens)``.
    """

    enabled: bool = False
    understanding_tokens: int = 0
    und_tokens: int = 0
    gen_tokens: int = 0
    # Small three-tower models reserve this many GEN rows for the action stream.
    # Remaining GEN rows belong to video. Zero preserves the full-width path.
    action_tokens: int = 0


@attrs.frozen(slots=True)
class FixedSequenceTargets:
    """Resolved physical stream lengths for one task microbatch."""

    causal: int
    full: int
    profile: str


def _get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def fixed_sequence_total(fixed: Any) -> int:
    """Return the largest enabled task profile's physical token total."""

    understanding_tokens = int(_get(fixed, "understanding_tokens", 0))
    und_tokens = int(_get(fixed, "und_tokens", 0))
    gen_tokens = int(_get(fixed, "gen_tokens", 0))
    return max(understanding_tokens, und_tokens + gen_tokens)


def resolve_fixed_sequence_targets(fixed: Any, *, has_generation: bool) -> FixedSequenceTargets | None:
    """Resolve the profile for an actual packed-sequence topology.

    The packed topology is the final authority: pure Understanding must never
    materialize a fake GEN stream merely to satisfy a configured shape.
    """

    if not bool(_get(fixed, "enabled", False)):
        return None
    if has_generation:
        und_tokens = int(_get(fixed, "und_tokens", 0))
        gen_tokens = int(_get(fixed, "gen_tokens", 0))
        if und_tokens <= 1 or gen_tokens <= 1:
            raise ValueError(
                "Fixed Generation/Action batches require both und_tokens and gen_tokens greater than one; "
                f"got und_tokens={und_tokens}, gen_tokens={gen_tokens}."
            )
        return FixedSequenceTargets(causal=und_tokens, full=gen_tokens, profile="generation")

    understanding_tokens = int(_get(fixed, "understanding_tokens", 0))
    if understanding_tokens <= 1:
        raise ValueError(
            "Fixed pure-Understanding batches require understanding_tokens greater than one; "
            f"got understanding_tokens={understanding_tokens}."
        )
    return FixedSequenceTargets(causal=understanding_tokens, full=0, profile="understanding")


def validate_fixed_sequence_length_contract(model: Any) -> None:
    """Fail early when a unified MoT fixed/static configuration is invalid.

    ``model`` may be the top-level LazyCall (with a ``config`` child) or the
    model config itself. Standalone VLM configs intentionally have no
    ``fixed_sequence_length`` field and are therefore outside this contract.
    """

    model_config = _get(model, "config", model)
    fixed = _get(model_config, "fixed_sequence_length")
    if fixed is None:
        return

    compile_config = _get(model_config, "compile")
    fixed_enabled = bool(_get(fixed, "enabled", False))
    compile_enabled = bool(_get(compile_config, "enabled", False))
    compile_dynamic = bool(_get(compile_config, "compile_dynamic", True))

    if compile_enabled and not compile_dynamic and not fixed_enabled:
        raise ValueError(
            "Static decoder compile requires model.fixed_sequence_length.enabled=true; "
            "variable sequence length only supports compile_dynamic=true."
        )

    if not fixed_enabled:
        return

    understanding_tokens = int(_get(fixed, "understanding_tokens", 0))
    und_tokens = int(_get(fixed, "und_tokens", 0))
    gen_tokens = int(_get(fixed, "gen_tokens", 0))
    action_tokens = int(_get(fixed, "action_tokens", 0))
    small_action = _get(model_config, "action_tower_preset", "full") != "full"
    if small_action and not 0 < action_tokens < gen_tokens:
        raise ValueError("Fixed small action towers require 0 < action_tokens < gen_tokens")
    if action_tokens and not small_action:
        raise ValueError("fixed action_tokens is supported only for small action tower presets")
    if (und_tokens == 0) != (gen_tokens == 0):
        raise ValueError(
            "Fixed Generation/Action profile requires und_tokens and gen_tokens to be configured together; "
            f"got und_tokens={und_tokens}, gen_tokens={gen_tokens}."
        )
    configured_targets = (understanding_tokens, und_tokens, gen_tokens)
    if any(target < 0 for target in configured_targets):
        raise ValueError("Fixed sequence targets must be zero (disabled) or greater than one.")
    active_targets = [target for target in configured_targets if target > 0]
    if not active_targets:
        raise ValueError("Fixed sequence length requires an Understanding or Generation/Action profile.")
    if any(target <= 1 for target in active_targets):
        raise ValueError("Every configured fixed sequence target must be greater than one.")

    parallelism = _get(model_config, "parallelism")
    cp_size = int(_get(parallelism, "context_parallel_shard_degree", 1))
    if cp_size <= 0:
        raise ValueError(f"context_parallel_shard_degree must be positive; got {cp_size}.")
    if any(target % cp_size != 0 for target in active_targets):
        raise ValueError(
            "Every configured fixed sequence target must be divisible by context_parallel_shard_degree; "
            f"got understanding_tokens={understanding_tokens}, und_tokens={und_tokens}, "
            f"gen_tokens={gen_tokens}, cp_size={cp_size}."
        )

    fixed_total = fixed_sequence_total(fixed)
    max_tokens = int(_get(model_config, "max_num_tokens_after_packing", fixed_total))
    if max_tokens > 0 and max_tokens != fixed_total:
        raise ValueError(
            "model.max_num_tokens_after_packing must equal the largest enabled fixed profile in fixed mode; "
            f"got max_num_tokens_after_packing={max_tokens}, fixed total={fixed_total}."
        )

    compiled_region = str(_get(compile_config, "compiled_region", "language"))
    if compile_enabled and not compile_dynamic and compiled_region != "language":
        raise ValueError(
            "Static fixed-sequence training only supports compiled_region='language' "
            "(all unified MoT decoder layers); VFM encode/decode helpers remain dynamic."
        )
