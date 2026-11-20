# Copyright (c) 2026, BAAI. All rights reserved.
# Adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/data/generator/dataflow/loader.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CosmosDataLoader — slim orchestrator that wires the four dataflow roles
(DataDistributor -> RawItemProcessor -> SampleBatcher -> BatchCollator) inside
each DataLoader worker. The canonical training dataloader.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from functools import reduce

import numpy as np
import torch
import torch.utils.data

from cosmos_framework.data.generator.dataflow.base import (
    BatchCollator,
    DataDistributor,
    RawItemProcessor,
    SampleBatcher,
)
from cosmos_framework.data.generator.dataflow.batchers import (
    SequentialPackingBatcher,
    SequentialTokenBudgetBatcher,
    SimpleBatcher,
)
from cosmos_framework.data.generator.dataflow.collators import DefaultBatchCollator
from cosmos_framework.data.generator.drop_sample_contract import (
    DROP_SAMPLE_CONTEXT_KEY,
    DROP_SAMPLE_KEY,
    DROP_SAMPLE_REASON_KEY,
)
from cosmos_framework.utils import log


def data_parallel_coordinates(parallel_dims=None):
    """Resolve the same DP coordinates for synchronous and torch loaders."""
    if parallel_dims is not None:
        return parallel_dims.dp_coord
    if torch.distributed.is_initialized():
        rank, size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        if size > 1:
            log.info(
                "CosmosDataLoader: using global rank for DP sharding. "
                "For FSDP+TP/PP pass parallel_dims= for the correct DP rank.",
                rank0_only=True,
            )
        return rank, size
    return 0, 1


class _DataflowIterableDataset(torch.utils.data.IterableDataset):
    """Wires distributor -> processor -> batcher -> collator inside a worker."""

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher,
        collator: BatchCollator,
        dp_rank: int,
        dp_world_size: int,
        enable_map_style_resume: bool = True,
        map_style_resume_mode: str = "exact",
    ):
        super().__init__()
        if map_style_resume_mode not in {"exact", "at_most_once"}:
            raise ValueError(f"unsupported map_style_resume_mode: {map_style_resume_mode!r}")
        self._distributor = distributor
        self._processor = processor
        self._batcher = batcher
        self._collator = collator
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._enable_map_style_resume = enable_map_style_resume
        self._map_style_resume_mode = map_style_resume_mode

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        return self.iter_stream(worker_id, num_workers)

    def iter_stream(self, worker_id: int, num_workers: int, *, capture_resume_rng: bool = False):
        """Synchronous logical shard; contains no worker pool or output prefetch."""
        raw = self._distributor.stream(self._dp_rank, self._dp_world_size, worker_id, num_workers)

        def _processed():
            consecutive_drops = 0
            for item in raw:
                if isinstance(item, dict):
                    meta = {k: value for k, value in item.items() if k.startswith("_dp_")}
                    # Processors may use deterministic stream context (for
                    # example, epoch-scoped target-prefix sampling).  Keep the
                    # private fields visible while processing; they are still
                    # stripped before user-facing collation below.
                    processor_item = dict(item)
                else:
                    meta = {}
                    processor_item = item
                s = self._processor.process(processor_item)
                if isinstance(s, dict):
                    if self._enable_map_style_resume:
                        if meta:
                            s.update(meta)
                        if capture_resume_rng and meta:
                            from cosmos_framework.data.generator.dataflow.rng import capture_rng

                            # Snapshot at the emitted cursor, before a packer's
                            # lookahead advances processing of the next batch.
                            s["_dp_resume_rng"] = capture_rng()
                    else:
                        s = {key: value for key, value in s.items() if not key.startswith("_dp_")}
                if isinstance(s, dict) and bool(s.get(DROP_SAMPLE_KEY)):
                    consecutive_drops += 1
                    if capture_resume_rng and consecutive_drops >= 1000:
                        raise RuntimeError("Shared data stream dropped 1000 consecutive samples without producing data")
                    log.warning(
                        "CosmosDataLoader: dropped sample "
                        f"reason={s.get(DROP_SAMPLE_REASON_KEY, 'unknown')!r} "
                        f"catalog_id={s.get('catalog_id', 'unknown')!r} "
                        f"context={json.dumps(s.get(DROP_SAMPLE_CONTEXT_KEY, {}), ensure_ascii=False, sort_keys=True)}",
                        rank0_only=False,
                    )
                    continue
                consecutive_drops = 0
                output = [s]
                del s, item, processor_item
                yield output.pop()

        for group in self._batcher.batches(_processed()):
            has_meta = bool(group) and isinstance(group[0], dict) and "_dp_epoch" in group[0]
            if has_meta:
                epochs = [s["_dp_epoch"] for s in group]
                positions = [s["_dp_stream_pos"] for s in group]
                stream_positions = list(zip(epochs, positions, strict=True))
                # Epoch and position form one lexicographic cursor. Taking each
                # maximum independently is wrong for a batch crossing an epoch
                # boundary, e.g. (epoch=0,pos=99), (epoch=1,pos=0).
                max_epoch, max_pos = max(stream_positions)
                # Resume records (max_epoch, max_pos) and fast-forwards to max_pos+1 —
                # bit-for-bit with the legacy collate_batch. Sequential packing
                # preserves stream order, so processor-dropped gaps and epoch crossings
                # are safe. Pool packing may retain an earlier position while emitting a
                # later one; exact mode rejects that, while at_most_once explicitly
                # accepts bounded candidate skipping without replay.
                if len(group) > 1:
                    if isinstance(
                        self._batcher, (SequentialPackingBatcher, SequentialTokenBudgetBatcher, SimpleBatcher)
                    ):
                        cursor_safe = all(
                            right > left for left, right in zip(stream_positions, stream_positions[1:], strict=False)
                        )
                    else:
                        cursor_safe = min(epochs) == max_epoch and sorted(positions) == list(
                            range(min(positions), max_pos + 1)
                        )
                    if not cursor_safe and self._map_style_resume_mode != "at_most_once":
                        raise ValueError(
                            "Map-style resume cannot safely stamp a multi-sample batch whose "
                            "stream cursors are reordered or non-contiguous (reordering "
                            "batcher + batch_size>1). Use max_batch_size=1 with pool packing, a "
                            "sequential (order-preserving) batcher, or an iterable (non-resumable) "
                            "source; choose map_style_resume_mode='at_most_once' only when bounded "
                            "candidate skipping is preferable to replay."
                        )
                clean = [{k: v for k, v in s.items() if not k.startswith("_dp_")} for s in group]
                batch = self._collator.collate(clean)
                batch["sample_worker_id"] = torch.tensor([worker_id] * len(group))
                batch["sample_epoch"] = torch.tensor([max_epoch] * len(group))
                batch["sample_index"] = torch.tensor([max_pos] * len(group))
                if capture_resume_rng:
                    frontier = max(group, key=lambda s: (s["_dp_epoch"], s["_dp_stream_pos"]))
                    batch["_cosmos_resume_rng"] = frontier["_dp_resume_rng"]
            else:
                batch = self._collator.collate(group)
            # The suspended generator must not retain the delivered media.
            # Batchers only use their yielded group until the next request.
            group.clear()
            if has_meta:
                clean.clear()
                if capture_resume_rng:
                    del frontier
            output = [batch]
            del batch, group
            yield output.pop()


class CosmosDataLoader(torch.utils.data.DataLoader):
    """Public entry point: bring any dataset into training via four roles.

    Either pass an explicit ``batcher`` (and optional ``collator``), or pass a
    bare ``batch_size=N`` for stock fixed-size batching — the loader then builds
    ``SimpleBatcher(N)`` + ``DefaultBatchCollator()``. Passing both is an error.

    DP coordinates: ``parallel_dims.dp_coord`` > ``torch.distributed`` > (0, 1).
    """

    def __init__(
        self,
        distributor: DataDistributor,
        processor: RawItemProcessor,
        batcher: SampleBatcher | None = None,
        collator: BatchCollator | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        parallel_dims=None,
        enable_map_style_resume: bool = True,
        map_style_resume_mode: str = "exact",
    ):
        if batch_size is not None and batcher is not None:
            raise ValueError("Pass either batch_size= (sugar) or an explicit batcher=, not both.")
        if batch_size is None and batcher is None:
            raise ValueError("Provide either a batcher= or a batch_size=.")
        if batch_size is not None:
            batcher = SimpleBatcher(batch_size=batch_size)
        if collator is None:
            collator = DefaultBatchCollator()

        dp_rank, dp_world_size = data_parallel_coordinates(parallel_dims)

        dataset = _DataflowIterableDataset(
            distributor=distributor,
            processor=processor,
            batcher=batcher,
            collator=collator,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            enable_map_style_resume=enable_map_style_resume,
            map_style_resume_mode=map_style_resume_mode,
        )

        from cosmos_framework.data.generator.dataflow.distributors import MapDistributor

        requires_persistent_workers = (enable_map_style_resume and isinstance(distributor, MapDistributor)) or bool(
            getattr(distributor, "requires_persistent_workers", False)
        )
        if requires_persistent_workers and num_workers > 0 and not persistent_workers:
            log.info(
                "CosmosDataLoader: resumable distributor requires persistent_workers=True for "
                "correct stateful resume; overriding to True.",
                rank0_only=True,
            )
            persistent_workers = True

        if persistent_workers and num_workers == 0:
            log.info(
                "CosmosDataLoader: persistent_workers=True ignored because num_workers=0.",
                rank0_only=True,
            )
            persistent_workers = False

        loader_kwargs: dict = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        super().__init__(dataset, batch_size=None, **loader_kwargs)
        self._preserve_resume_rng = False

    def preserve_rng_on_resume(self) -> None:
        """Do not consume the restored training RNG for a new iterator base seed."""
        self._preserve_resume_rng = True

    def __iter__(self):
        with torch.random.fork_rng(devices=[]) if self._preserve_resume_rng else nullcontext():
            iterator = super().__iter__()
        self._preserve_resume_rng = False
        return iterator


class JointCosmosDataLoader:
    """Wraps multiple ``CosmosDataLoader`` instances with ratio-based seeded selection.

    One output batch = one inner loader, selected deterministically by ratio at each
    step.  Adds a ``"dataset_name"`` key to every yielded batch so downstream callbacks
    can route state updates to the correct inner loader.

    Parameters
    ----------
    dataloaders:
        ``{name: {"dataloader": CosmosDataLoader, "ratio": int}}`` mapping.
        Entries with ``ratio <= 0`` are silently skipped.
    seed:
        Base seed for per-step dataset selection.  Step ``i`` uses
        ``np.random.RandomState(seed + i)`` — fully reproducible on resume via
        ``set_start_iteration``.
    """

    def __init__(
        self,
        dataloaders: dict,
        seed: int = 42,
        selection_mode: str = "probabilistic",
    ) -> None:
        entries = [
            (name, cfg["dataloader"], cfg["ratio"]) for name, cfg in dataloaders.items() if cfg.get("ratio", 0) > 0
        ]
        if not entries:
            raise ValueError("JointCosmosDataLoader: no dataloaders with ratio > 0")

        self._names: list[str] = [e[0] for e in entries]
        reserved_names = {"global_id", "__schedule__", "__fingerprints__", "__execution__"}
        if reserved_names.intersection(self._names):
            raise ValueError(
                "JointCosmosDataLoader: dataset names "
                f"{sorted(reserved_names)} are reserved by the checkpoint state format"
            )
        self._loaders: list[CosmosDataLoader] = [e[1] for e in entries]
        raw_ratios = [e[2] for e in entries]
        ratios = np.array(raw_ratios, dtype=float)
        self._probs: np.ndarray = ratios / ratios.sum()
        self._seed = seed
        if selection_mode not in {"probabilistic", "exact_periodic"}:
            raise ValueError(f"selection_mode must be 'probabilistic' or 'exact_periodic', got {selection_mode!r}")
        self._selection_mode = selection_mode
        self._exact_period: tuple[int, ...] | None = None
        if selection_mode == "exact_periodic":
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw_ratios):
                raise ValueError("exact_periodic selection requires positive integer ratios")
            divisor = reduce(math.gcd, raw_ratios)
            self._exact_period = tuple(index for index, ratio in enumerate(raw_ratios) for _ in range(ratio // divisor))
        self._global_id = 0
        self._cached_cycle = -1
        self._cached_order: list[int] = []
        # Iterators are created lazily on the first __iter__ call so that
        # DataLoaderStateCallback.load_state_dict can install resume env vars
        # before workers are spawned (for num_workers > 0, iter(DataLoader)
        # forks workers immediately; env vars must be set in the parent first).
        self._iterators: list | None = None
        self._preserve_resume_rng = False

        total = ratios.sum()
        lines = [f"JointCosmosDataLoader: {len(self._names)} streams"]
        for name, ratio in zip(self._names, ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

    @property
    def schedule_identity(self) -> dict:
        return {
            "version": 1,
            "mode": self._selection_mode,
            "seed": self._seed,
            "names": list(self._names),
            "ratios": self._probs.tolist(),
            "period": None if self._exact_period is None else len(self._exact_period),
        }

    @property
    def dataset_fingerprints(self) -> dict[str, str]:
        return self.dataset_fingerprints_for_action128(None)

    def dataset_fingerprints_for_action128(self, spec) -> dict[str, str]:
        """Use the old global registry only for append-only source validation."""
        result: dict[str, str] = {}
        for name, loader in zip(self._names, self._loaders, strict=True):
            distributor = getattr(getattr(loader, "dataset", None), "_distributor", None)
            fingerprint = getattr(distributor, "dataset_fingerprint", None)
            dataset = getattr(distributor, "_dataset", None)
            if spec is not None:
                if hasattr(distributor, "dataset_fingerprint_for_spec"):
                    fingerprint = distributor.dataset_fingerprint_for_spec(spec)
                elif hasattr(dataset, "fingerprint_for_spec"):
                    fingerprint = dataset.fingerprint_for_spec(spec)
            if fingerprint is not None:
                result[name] = str(fingerprint)
        return result

    def _exact_period_index(self, global_id: int) -> int:
        assert self._exact_period is not None
        period_length = len(self._exact_period)
        cycle, offset = divmod(global_id, period_length)
        if cycle == self._cached_cycle:
            return self._cached_order[offset]
        digest = hashlib.blake2b(digest_size=8, person=b"cosmosdl")
        digest.update(f"microbatch\0{self._seed}\0{cycle}".encode())
        cycle_seed = int.from_bytes(digest.digest(), "big")
        order = list(self._exact_period)
        np.random.RandomState(cycle_seed % (2**32)).shuffle(order)
        self._cached_cycle, self._cached_order = cycle, order
        return order[offset]

    def dataset_index(self, global_id: int) -> int:
        """Dataset selection depends only on the global microbatch id."""
        if self._exact_period is not None:
            return self._exact_period_index(global_id)
        return int(np.random.RandomState(self._seed + global_id).choice(len(self._loaders), p=self._probs))

    def set_start_iteration(self, iteration: int) -> None:
        """Restore deterministic selection sequence after checkpoint resume.

        Called by ``JointCosmosDataLoaderStateCallback.load_state_dict`` and by the
        trainer (if present) via ``hasattr`` guard.
        """
        self._global_id = iteration

    def preserve_rng_on_resume(self) -> None:
        """Protect training RNG while rebuilding all inner iterator base seeds.

        Worker-local transform RNG is a separate state; this does not claim
        arbitrary multi-worker augmentation can be resumed exactly.
        """
        self._preserve_resume_rng = True

    def __iter__(self):
        # Lazy init: create iterators here (not in __init__) so that
        # load_state_dict can set resume env vars before workers fork.
        if self._iterators is None:
            # PyTorch draws a CPU base_seed even with num_workers=0. Continuous
            # training already owns these iterators; recreating them on resume
            # must not advance the RNG used by diffusion timestep sampling.
            with torch.random.fork_rng(devices=[]) if self._preserve_resume_rng else nullcontext():
                self._iterators = [iter(loader) for loader in self._loaders]
            self._preserve_resume_rng = False
        while True:
            idx = self.dataset_index(self._global_id)
            try:
                batch = next(self._iterators[idx])
            except StopIteration:
                # Inner CosmosDataLoaders are infinite; this guard handles
                # the unlikely case of a finite IterableDataset inner source.
                self._iterators[idx] = iter(self._loaders[idx])
                batch = next(self._iterators[idx])
            batch["dataset_name"] = self._names[idx]
            self._global_id += 1
            yield batch
