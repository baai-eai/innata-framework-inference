# Copyright (c) 2026, BAAI. All rights reserved.
# Adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/data/generator/dataflow/batchers.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in SampleBatcher implementations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Optional

from cosmos_framework.data.generator.dataflow.base import SampleBatcher


@dataclass(frozen=True)
class SequenceTokenCounts:
    """Real unified-MoT tokens contributed by one Generation/Action sample."""

    und: int
    gen: int
    action: int = 0

    @property
    def video(self) -> int:
        return self.gen - self.action

    @property
    def total(self) -> int:
        return self.und + self.gen


class SimpleBatcher(SampleBatcher):
    """Fixed-size batching — stock DataLoader behavior. Never needs sample_size."""

    def __init__(self, batch_size: int, drop_last: bool = False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self.drop_last = drop_last

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        buf: list[dict] = []
        for s in samples:
            buf.append(s)
            if len(buf) == self.batch_size:
                s = None
                yield buf
                buf = []
        if buf and not self.drop_last:
            yield buf


class _Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class PoolPackingBatcher(SampleBatcher):
    """Pool-based greedy bin-packing batcher (re-homed from PackingIterableDataset).

    Buffers ``pool_size`` samples and assembles each batch by greedily selecting
    candidates that fit within the packed token budget, never mixing modalities
    within a batch. ``sample_size`` defaults to ``len(sample["input_ids"])``;
    pass ``size_fn`` to override, or subclass and override the method.
    """

    def __init__(
        self,
        max_tokens: int,
        pool_size: int = 16,
        max_batch_size: int = 1,
        long_threshold: int = 6400,
        apply_long_sample_halving: bool = False,
        size_fn: Optional[Callable[[dict], int]] = None,
        reject_oversized: bool = False,
    ):
        self.max_tokens = max_tokens
        self.pool_size = pool_size
        self.max_batch_size = max_batch_size
        self.long_threshold = long_threshold
        self.apply_long_sample_halving = apply_long_sample_halving
        self._size_fn = size_fn
        self.reject_oversized = reject_oversized

    def sample_size(self, sample: dict) -> int:
        if self._size_fn is not None:
            return self._size_fn(sample)
        # len() == shape[0] for a 1-D tensor and also works for list input_ids.
        tokens = sample.get("text_token_ids", sample.get("input_ids"))
        if tokens is None:
            raise KeyError("PoolPackingBatcher requires text_token_ids or input_ids")
        return len(tokens)

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        pool: deque[dict] = deque()
        src = iter(samples)
        exhausted = False
        while True:
            while not exhausted and len(pool) < self.pool_size:
                try:
                    pool.append(next(src))
                except StopIteration:
                    exhausted = True
            if not pool:
                return
            yield self._best_fit_batch(pool)

    def _max_tokens(self, cur_max: int) -> int:
        if not self.apply_long_sample_halving:
            return self.max_tokens
        if cur_max < 1000:
            return self.max_tokens
        return self.max_tokens // 2

    @staticmethod
    def _get_modality(sample: dict) -> "_Modality":
        if "pixel_values" in sample:
            return _Modality.IMAGE
        elif "pixel_values_videos" in sample:
            return _Modality.VIDEO
        return _Modality.TEXT

    def _best_fit_batch(self, pool: deque) -> list[dict]:
        seed = pool.popleft()
        seed_modality = self._get_modality(seed)
        L0 = self.sample_size(seed)
        if self.reject_oversized and L0 > self.max_tokens:
            sample_id = seed.get("catalog_id", seed.get("sample_id", "unknown"))
            raise ValueError(
                f"PoolPackingBatcher sample {sample_id!r} has {L0} tokens, exceeding fixed capacity {self.max_tokens}."
            )
        if L0 >= self.long_threshold or L0 >= self._max_tokens(L0):
            return [seed]
        chosen = [seed]
        cur_max = L0
        cur_total = L0
        while pool:
            if self.max_batch_size and len(chosen) >= self.max_batch_size:
                break
            candidate_idx = self._find_first_fitting_candidate(pool, cur_max, cur_total, seed_modality)
            if candidate_idx is None:
                break
            cand = self._remove_from_pool(pool, candidate_idx)
            chosen.append(cand)
            candidate_size = self.sample_size(cand)
            cur_max = max(cur_max, candidate_size)
            cur_total += candidate_size
        return chosen

    def _find_first_fitting_candidate(self, pool, cur_max, cur_total, seed_modality):
        for idx, cand in enumerate(pool):
            if self._get_modality(cand) != seed_modality:
                continue
            L = self.sample_size(cand)
            new_tokens = cur_total + L
            if new_tokens <= self._max_tokens(cur_max):
                return idx
        return None

    @staticmethod
    def _remove_from_pool(pool: deque, idx: int) -> dict:
        if idx == 0:
            return pool.popleft()
        elif idx == len(pool) - 1:
            return pool.pop()
        else:
            pool.rotate(-idx)
            item = pool.popleft()
            pool.rotate(idx)
            return item


class SequentialTokenBudgetBatcher(SampleBatcher):
    """Pack consecutive samples against one exact token budget.

    Unlike :class:`PoolPackingBatcher`, this batcher never scans past an
    overflowing row.  That makes its output boundaries representable by the
    map-style resume cursor while still accepting an exact ``target - 1`` fit
    for fixed pure-Understanding batches.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        size_fn: Callable[[dict], int],
        max_samples_per_batch: int | None = None,
        task_type: str = "understanding",
        physical_target: int | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if max_samples_per_batch is not None and max_samples_per_batch < 1:
            raise ValueError("max_samples_per_batch must be positive when configured")
        self.max_tokens = int(max_tokens)
        self.size_fn = size_fn
        self.max_samples_per_batch = max_samples_per_batch
        self.task_type = task_type
        self.physical_target = physical_target if physical_target is not None else max_tokens

    def _sample_size(self, sample: dict) -> int:
        size = int(self.size_fn(sample))
        if size < 1:
            raise ValueError(f"{self.task_type} sample has invalid packed token count {size}")
        if size > self.max_tokens:
            sample_id = sample.get("catalog_id", sample.get("sample_id", "unknown"))
            raise ValueError(
                f"fixed {self.task_type} sample {sample_id!r} exceeds its causal capacity: "
                f"real_und={size}, target_und={self.physical_target}, capacity={self.max_tokens}"
            )
        return size

    def batches(self, samples: Iterator[dict]) -> Iterator[list[dict]]:
        group: list[dict] = []
        real_tokens = 0
        for sample in samples:
            sample_tokens = self._sample_size(sample)
            hits_sample_cap = self.max_samples_per_batch is not None and len(group) >= self.max_samples_per_batch
            if group and (hits_sample_cap or real_tokens + sample_tokens > self.max_tokens):
                yield group
                group = []
                real_tokens = 0
            group.append(sample)
            real_tokens += sample_tokens
        if group:
            sample = None
            yield group


from collections import deque as _deque

from cosmos_framework.utils import log


class SequentialPackingBatcher(SampleBatcher):
    """Order-preserving pull-until-budget packing (port of PackingDataLoader.__iter__).

    Accumulates samples in stream order until `max_sequence_length` (or
    `max_samples_per_batch`); a sample that would overflow a non-empty batch is
    carried to the next batch (bounded by `lookahead_limit`); a sample that alone
    exceeds the budget is discarded with a log. `sample_size` ports the VFM VAE
    token formula (needs the tokenizer compression factors and patch size).
    """

    def __init__(
        self,
        max_sequence_length: Optional[int] = None,
        tokenizer_spatial_compression_factor: int = 16,
        tokenizer_temporal_compression_factor: int = 4,
        patch_spatial: int = 2,
        max_samples_per_batch=None,
        lookahead_limit: int = 10,
        fixed_sequence_length_enabled: bool = False,
        fixed_understanding_tokens: int = 0,
        fixed_und_tokens: int = 0,
        fixed_gen_tokens: int = 0,
        fixed_action_tokens: int = 0,
        special_tokens: dict[str, int] | None = None,
    ):
        self.max_sequence_length = max_sequence_length
        self.tokenizer_spatial_compression_factor = tokenizer_spatial_compression_factor
        self.tokenizer_temporal_compression_factor = tokenizer_temporal_compression_factor
        self.patch_spatial = patch_spatial
        self.max_samples_per_batch = max_samples_per_batch
        self.lookahead_limit = lookahead_limit
        self.fixed_sequence_length_enabled = bool(fixed_sequence_length_enabled)
        self.fixed_understanding_tokens = int(fixed_understanding_tokens)
        self.fixed_und_tokens = int(fixed_und_tokens)
        self.fixed_gen_tokens = int(fixed_gen_tokens)
        self.fixed_action_tokens = int(fixed_action_tokens)
        self.special_tokens = special_tokens
        if lookahead_limit < 1:
            raise ValueError("lookahead_limit must be at least 1")
        if self.fixed_sequence_length_enabled:
            if self.max_sequence_length is None:
                raise ValueError("Fixed sequence length requires max_sequence_length token-budget mode")
            if self.special_tokens is None:
                raise ValueError("Fixed sequence length requires special_tokens for exact text-span admission")
            if (self.fixed_und_tokens == 0) != (self.fixed_gen_tokens == 0):
                raise ValueError(
                    "Fixed UND and GEN targets must be configured together; "
                    f"got UND={self.fixed_und_tokens}, GEN={self.fixed_gen_tokens}"
                )
            if self.fixed_action_tokens and not 0 < self.fixed_action_tokens < self.fixed_gen_tokens:
                raise ValueError("Fixed ACTION capacity must satisfy 0 < ACTION < GEN")
            active_targets = [
                target
                for target in (self.fixed_understanding_tokens, self.fixed_und_tokens, self.fixed_gen_tokens)
                if target > 0
            ]
            if not active_targets or any(target <= 1 for target in active_targets):
                raise ValueError("Every configured fixed sequence target must exceed one token")
            fixed_total = max(
                self.fixed_understanding_tokens,
                self.fixed_und_tokens + self.fixed_gen_tokens,
            )
            if self.max_sequence_length != fixed_total:
                raise ValueError(
                    "max_sequence_length must equal the largest configured fixed profile; "
                    f"got {self.max_sequence_length} != {fixed_total}"
                )
        else:
            assert (self.max_sequence_length is None) != (self.max_samples_per_batch is None), (
                "Exactly one of max_sequence_length or max_samples_per_batch must be set "
                "(token-budget mode vs count-only mode), matching legacy PackingDataLoader."
            )

    @staticmethod
    def _num_text_tokens(sample: dict) -> int:
        import torch as _torch

        if "text_token_ids" not in sample:
            raise ValueError("Fixed sequence length requires text_token_ids for every unified-MoT sample")
        text_token_ids = sample["text_token_ids"]
        if isinstance(text_token_ids, list):
            if not text_token_ids:
                raise ValueError("Fixed sequence length received an empty text_token_ids list")
            if isinstance(text_token_ids[0], _torch.Tensor):
                return int(text_token_ids[0].shape[0])
            return len(text_token_ids)
        return int(text_token_ids.shape[0] if text_token_ids.ndim == 1 else text_token_ids.shape[1])

    @staticmethod
    def _metadata_int(sample: dict, key: str) -> int | None:
        import torch as _torch

        value = sample.get(key)
        if value is None:
            return None
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, _torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{key} must be scalar, got shape {tuple(value.shape)}")
            value = value.item()
        return int(value)

    def _vision_latent_shapes(self, sample: dict, *, planned: bool = False) -> tuple[bool, list[tuple[int, int, int]]]:
        """Return image flag and latent ``(T,H,W)`` shapes without offline tensor I/O."""

        latent_ref = sample.get("video_latent_ref")
        if latent_ref is not None:
            # Import lazily: offline_vae validates catalog rows and therefore
            # depends on catalog.py, while catalog.py imports this batcher.
            from cosmos_framework.data.generator.offline_vae import VaeLatentRef

            if not isinstance(latent_ref, VaeLatentRef):
                raise TypeError(f"video_latent_ref must be VaeLatentRef, got {type(latent_ref).__name__}")
            _, latent_t, latent_h, latent_w = latent_ref.latent_shape
            return False, [(latent_t, latent_h, latent_w)]

        is_image_batch = "images" in sample
        media_key = "images" if is_image_batch else "video"
        if media_key not in sample:
            raise ValueError("Unified-MoT packing requires images, video, or video_latent_ref")
        media_value = sample[media_key]
        media_items = media_value if isinstance(media_value, list) else [media_value]
        if planned:
            from cosmos_framework.data.generator.sequence_packing.static_layout import (
                StaticPackingConfig,
                planned_vision_latent_shape,
            )

            geometry = StaticPackingConfig(
                spatial_compression_factor=self.tokenizer_spatial_compression_factor,
                temporal_compression_factor=self.tokenizer_temporal_compression_factor,
                latent_patch_size=self.patch_spatial,
            )
            return is_image_batch, [
                planned_vision_latent_shape(sample, media, is_image=is_image_batch, config=geometry)
                for media in media_items
            ]
        num_views = self._metadata_int(sample, "sample_n_views")
        frames_per_view = self._metadata_int(sample, "num_video_frames_per_view")
        shapes: list[tuple[int, int, int]] = []
        for media in media_items:
            if is_image_batch:
                if media.ndim != 3:
                    raise ValueError(f"images must have shape [C,H,W], got {tuple(media.shape)}")
                _, height, width = media.shape
                latent_frames = 1
            else:
                if media.ndim != 4:
                    raise ValueError(f"video must have shape [C,T,H,W], got {tuple(media.shape)}")
                _, num_frames, height, width = media.shape
                if num_views is not None:
                    if frames_per_view is None or num_frames != num_views * frames_per_view:
                        raise ValueError("Multiview video length must equal sample_n_views * num_video_frames_per_view")
                    latent_frames = num_views * (
                        1 + (frames_per_view - 1) // self.tokenizer_temporal_compression_factor
                    )
                else:
                    latent_frames = 1 + (num_frames - 1) // self.tokenizer_temporal_compression_factor
            latent_height = max(int(height) // self.tokenizer_spatial_compression_factor, 1)
            latent_width = max(int(width) // self.tokenizer_spatial_compression_factor, 1)
            shapes.append((latent_frames, latent_height, latent_width))
        return is_image_batch, shapes

    def sample_token_counts(self, sample: dict) -> SequenceTokenCounts:
        """Estimate the exact real UND/GEN split used by static packing.

        Generation packing contributes the rendered caption plus EOS/BOV to
        UND. GEN contains patchified VAE latents and optional action tokens.
        One physical row per stream remains reserved for the isolated padding
        segment, so admission checks compare these real counts with target-1.
        Optional small-tower VIDEO/ACTION partitions additionally bound their
        real rows; each partition can use its entire capacity.
        """

        from cosmos_framework.data.generator.sequence_packing.sequence import packed_text_span_length

        if self.fixed_und_tokens <= 1 or self.fixed_gen_tokens <= 1:
            raise ValueError("No fixed Generation/Action profile is configured for this sample")
        plan = sample.get("sequence_plan")
        while isinstance(plan, list) and len(plan) == 1:
            plan = plan[0]
        if plan is None:
            raise ValueError("Fixed sequence length requires sequence_plan for exact text-span admission")
        plan_get = plan.get if isinstance(plan, dict) else lambda key, default=None: getattr(plan, key, default)
        assert self.special_tokens is not None
        und_tokens = packed_text_span_length(
            self._num_text_tokens(sample),
            self.special_tokens,
            has_generation=bool(plan_get("has_vision", False) or plan_get("has_action", False)),
            supervise_text=bool(plan_get("supervise_text", True)),
            text_loss_mask=plan_get("text_loss_mask"),
        )
        gen_tokens = 0
        _, latent_shapes = self._vision_latent_shapes(sample, planned=True)
        for latent_frames, latent_height, latent_width in latent_shapes:
            patch_height = math.ceil(latent_height / self.patch_spatial)
            patch_width = math.ceil(latent_width / self.patch_spatial)
            gen_tokens += latent_frames * patch_height * patch_width

        action_tokens = 0
        action_value = sample.get("action")
        action_items = action_value if isinstance(action_value, list) else [action_value]
        for action in action_items:
            if action is not None:
                action_tokens += int(action.shape[0])
        return SequenceTokenCounts(und=und_tokens, gen=gen_tokens + action_tokens, action=action_tokens)

    def sample_size(self, sample: dict) -> int:
        # PORT of _compute_num_tokens_per_sample (joint_dataloader.py:325-400),
        # operating on a SINGLE sample.
        #
        # In the original batched method:
        #   - text_token_ids is a list of tensors  → num_text_tokens = text_token_ids[0].shape[0]
        #   - text_token_ids is a 2-D tensor [B,S] → num_text_tokens = text_token_ids.shape[1]
        # For a single sample:
        #   - 1-D tensor [S]      → shape[0]  (torch.arange(N) case from tests)
        #   - list of tensors     → text_token_ids[0].shape[0]
        #   - list of ints        → len(text_token_ids)
        #   - 2-D tensor [1,S]    → shape[1]  (mirrors original .shape[1] branch)
        import torch as _torch

        text_token_ids = sample["text_token_ids"]
        if isinstance(text_token_ids, list):
            if len(text_token_ids) > 0 and isinstance(text_token_ids[0], _torch.Tensor):
                num_text_tokens = text_token_ids[0].shape[0]
            else:
                num_text_tokens = len(text_token_ids)
        else:
            # tensor: 1-D → shape[0], 2-D → shape[1]
            if text_token_ids.ndim == 1:
                num_text_tokens = text_token_ids.shape[0]
            else:
                num_text_tokens = text_token_ids.shape[1]

        num_tokens = num_text_tokens + 1

        is_offline = "video_latent_ref" in sample
        _, latent_shapes = self._vision_latent_shapes(sample)
        for latent_t_shape, latent_h_shape, latent_w_shape in latent_shapes:
            if is_offline:
                patch_h = math.ceil(latent_h_shape / self.patch_spatial)
                patch_w = math.ceil(latent_w_shape / self.patch_spatial)
            else:
                # Preserve the legacy variable-length estimator exactly for pixel inputs.
                patch_h = latent_h_shape // self.patch_spatial
                patch_w = latent_w_shape // self.patch_spatial
            num_vision_tokens = patch_h * patch_w * latent_t_shape + 2
            num_tokens += num_vision_tokens

        # Action part — single sample: action is a tensor [T_action, D] or None,
        # not wrapped in a list. Mirror the original: iterate as list for uniform handling.
        if "action" in sample:
            action = sample["action"]
            action_list = action if isinstance(action, list) else [action]
            for act in action_list:
                if act is None:
                    continue
                num_tokens += act.shape[0]

        return num_tokens

    @staticmethod
    def packing_key(sample: dict) -> tuple[str, str | None]:
        """Return the model-routing family that may share one packed sequence.

        Image generation, video generation, and each Action dynamics mode have
        different token topologies/conditioning semantics.  Keep them in
        separate sequences even when a source catalog interleaves their rows.
        TI2V/V2V remain in the video family because target-prefix conditioning
        is represented per sample by ``SequencePlan``.
        """
        mode = sample.get("mode")
        if mode is not None:
            return ("action", str(mode))
        if "images" in sample:
            return ("generation", "image")
        if "video" in sample:
            return ("generation", "video")
        if "video_latent_ref" in sample:
            return ("generation", "video")
        raise KeyError("SequentialPackingBatcher requires images, video, video_latent_ref, or an Action mode")

    def batches(self, samples):
        src = iter(samples)
        carry = _deque()
        exhausted = False
        while True:
            current_len = 0
            current_und_tokens = 0
            current_gen_tokens = 0
            current_action_tokens = 0
            num_samples = 0
            group = []
            group_key = None
            skipped = _deque()
            lookahead = 0

            def _next():
                if carry:
                    return carry.popleft()
                return next(src)

            while True:
                if self.max_samples_per_batch is not None and num_samples >= self.max_samples_per_batch:
                    break
                if group and lookahead >= self.lookahead_limit:
                    break
                try:
                    s = _next()
                except StopIteration:
                    exhausted = True
                    break
                sample_key = self.packing_key(s)
                if group_key is not None and sample_key != group_key:
                    skipped.append(s)
                    lookahead += 1
                    continue
                fixed_counts = self.sample_token_counts(s) if self.fixed_sequence_length_enabled else None
                n = fixed_counts.total if fixed_counts is not None else self.sample_size(s)
                if fixed_counts is not None:
                    exceeds_budget = (
                        current_und_tokens + fixed_counts.und > self.fixed_und_tokens - 1
                        or current_gen_tokens + fixed_counts.gen > self.fixed_gen_tokens - 1
                    )
                    if self.fixed_action_tokens:
                        exceeds_budget |= (
                            current_action_tokens + fixed_counts.action > self.fixed_action_tokens
                            or current_gen_tokens - current_action_tokens + fixed_counts.video
                            > self.fixed_gen_tokens - self.fixed_action_tokens
                        )
                else:
                    exceeds_budget = bool(
                        self.max_sequence_length is not None and current_len + n >= self.max_sequence_length
                    )
                if exceeds_budget:
                    if not group:
                        if fixed_counts is not None:
                            sample_id = s.get("catalog_id", s.get("sample_id", "unknown"))
                            raise ValueError(
                                "SequentialPackingBatcher fixed-sequence sample exceeds an UND/GEN capacity: "
                                f"sample={sample_id!r}, real_und={fixed_counts.und} "
                                f"(capacity={self.fixed_und_tokens - 1}), real_gen={fixed_counts.gen} "
                                f"(capacity={self.fixed_gen_tokens - 1}), real_video={fixed_counts.video}, "
                                f"real_action={fixed_counts.action}, "
                                f"action_capacity={self.fixed_action_tokens or 'unpartitioned'}, "
                                f"video_capacity={self.fixed_gen_tokens - self.fixed_action_tokens}"
                            )
                        log.error(
                            f"SequentialPackingBatcher: discarding oversized sample with {n} "
                            f"tokens (max_sequence_length={self.max_sequence_length})",
                            rank0_only=False,
                        )
                        continue
                    skipped.append(s)
                    lookahead += 1
                    continue
                current_len += n
                if fixed_counts is not None:
                    current_und_tokens += fixed_counts.und
                    current_gen_tokens += fixed_counts.gen
                    current_action_tokens += fixed_counts.action
                num_samples += 1
                group.append(s)
                group_key = sample_key
            for s in reversed(skipped):
                carry.appendleft(s)
            if group:
                s = None
                yield group
            if exhausted and not carry:
                return
