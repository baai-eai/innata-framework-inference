# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lazy readers and distributors for the unified five-column Parquet catalog.

The catalog is deliberately an index, not a media container.  Each row stores
strict JSON in ``data`` and ``provenance`` and points at external media or, for
Action, at a LeRobot root.  Startup reads Parquet footers only; row groups are
materialized on demand inside DataLoader workers.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import torch.utils.data

from cosmos_framework.data.generator.dataflow.base import DataDistributor
from cosmos_framework.data.generator.dataflow.timeout_skip import observe_candidate
from cosmos_framework.data.generator.drop_sample_contract import (
    DROP_SAMPLE_CONTEXT_KEY,
    DROP_SAMPLE_KEY,
    DROP_SAMPLE_REASON_KEY,
)
from cosmos_framework.utils import log

CATALOG_COLUMNS = ("id", "type", "dataset", "data", "provenance")
CATALOG_TASK_TYPES = {"understanding", "visual_generation", "action"}


class CatalogSchemaError(ValueError):
    """The catalog is structurally invalid and must not be retried."""


@dataclass(frozen=True)
class CatalogRowGroup:
    file_index: int
    row_group_index: int
    start: int
    num_rows: int


def _catalog_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if path.suffix != ".parquet":
            raise CatalogSchemaError(f"catalog must be a .parquet file: {path}")
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = tuple(sorted(p for p in path.rglob("*.parquet") if p.is_file()))
    if not files:
        raise CatalogSchemaError(f"catalog directory contains no Parquet parts: {path}")
    return files


class ParquetCatalogDataset(torch.utils.data.Dataset):
    """Map-style footer-indexed reader for unified training catalogs.

    Open ``ParquetFile`` objects and the single materialized row group are
    process-local.  Pickling or a PID change clears them, so fork/spawn workers
    never reuse an inherited Arrow handle.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_type: str | None = None,
        max_row_group_rows: int = 2_048,
        max_row_group_bytes: int = 256 * 1024 * 1024,
        handle_cache_size: int = 8,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.files = _catalog_files(self.path)
        if expected_type is not None and expected_type not in CATALOG_TASK_TYPES:
            raise ValueError(f"unknown catalog task type: {expected_type!r}")
        if max_row_group_rows < 1 or max_row_group_bytes < 1 or handle_cache_size < 1:
            raise ValueError("row-group and handle limits must be positive")
        self.expected_type = expected_type
        self.max_row_group_rows = int(max_row_group_rows)
        self.max_row_group_bytes = int(max_row_group_bytes)
        self.handle_cache_size = int(handle_cache_size)

        groups: list[CatalogRowGroup] = []
        ends: list[int] = []
        fingerprint = hashlib.sha256()
        cursor = 0
        for file_index, file_path in enumerate(self.files):
            parquet = pq.ParquetFile(file_path)
            names = tuple(parquet.schema_arrow.names)
            if names != CATALOG_COLUMNS:
                raise CatalogSchemaError(
                    f"catalog schema must be exactly {CATALOG_COLUMNS}, got {names} in {file_path}"
                )
            relative = file_path.name if self.path.is_file() else str(file_path.relative_to(self.path))
            stat = file_path.stat()
            fingerprint.update(relative.encode("utf-8"))
            fingerprint.update(str(stat.st_size).encode("ascii"))
            fingerprint.update(str(parquet.metadata.num_rows).encode("ascii"))
            footer_size = parquet.metadata.serialized_size + 8
            with file_path.open("rb") as handle:
                handle.seek(-footer_size, os.SEEK_END)
                fingerprint.update(handle.read(footer_size))
            for row_group_index in range(parquet.metadata.num_row_groups):
                metadata = parquet.metadata.row_group(row_group_index)
                if metadata.num_rows > self.max_row_group_rows:
                    raise CatalogSchemaError(
                        f"row group has {metadata.num_rows} rows, limit is {self.max_row_group_rows}: "
                        f"{file_path} row_group={row_group_index}; rewrite it with row_group_size<=2048"
                    )
                if metadata.total_byte_size > self.max_row_group_bytes:
                    raise CatalogSchemaError(
                        f"row group uncompressed size {metadata.total_byte_size} exceeds "
                        f"{self.max_row_group_bytes}: {file_path} row_group={row_group_index}"
                    )
                if metadata.num_rows == 0:
                    continue
                groups.append(CatalogRowGroup(file_index, row_group_index, cursor, metadata.num_rows))
                cursor += metadata.num_rows
                ends.append(cursor)
                fingerprint.update(f"{row_group_index}:{metadata.num_rows}:{metadata.total_byte_size}".encode("ascii"))
        if not groups:
            raise CatalogSchemaError(f"catalog is empty: {self.path}")
        self.row_groups = tuple(groups)
        self._row_group_ends = tuple(ends)
        self.fingerprint = fingerprint.hexdigest()
        self._pid = os.getpid()
        self._handles: OrderedDict[Path, pq.ParquetFile] = OrderedDict()
        self._cached_row_group_key: tuple[int, int] | None = None
        self._cached_row_group: Any = None

    def __len__(self) -> int:
        return self._row_group_ends[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pid"] = None
        state["_handles"] = OrderedDict()
        state["_cached_row_group_key"] = None
        state["_cached_row_group"] = None
        return state

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        self._pid = pid
        self._handles = OrderedDict()
        self._cached_row_group_key = None
        self._cached_row_group = None

    def _parquet_file(self, file_index: int) -> pq.ParquetFile:
        self._ensure_process_local()
        path = self.files[file_index]
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = pq.ParquetFile(path)
        self._handles[path] = handle
        while len(self._handles) > self.handle_cache_size:
            self._handles.popitem(last=False)
        return handle

    def _read_row_group(self, group_index: int):
        group = self.row_groups[group_index]
        key = (group.file_index, group.row_group_index)
        self._ensure_process_local()
        if self._cached_row_group_key != key:
            self._cached_row_group = self._parquet_file(group.file_index).read_row_group(
                group.row_group_index, columns=list(CATALOG_COLUMNS)
            )
            self._cached_row_group_key = key
        return self._cached_row_group

    @staticmethod
    def _parse_json(value: Any, *, field: str, row_id: str) -> dict[str, Any]:
        if not isinstance(value, str):
            raise CatalogSchemaError(f"{field} must be strict JSON text for row {row_id!r}")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise CatalogSchemaError(f"invalid {field} JSON for row {row_id!r}: {error}") from error
        if not isinstance(parsed, dict):
            raise CatalogSchemaError(f"{field} must decode to an object for row {row_id!r}")
        return parsed

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group_index = bisect.bisect_right(self._row_group_ends, index)
        group = self.row_groups[group_index]
        offset = index - group.start
        row = self._read_row_group(group_index).slice(offset, 1).to_pylist()[0]
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id:
            raise CatalogSchemaError(f"catalog id must be a non-empty string at index {index}")
        task_type = row["type"]
        if task_type not in CATALOG_TASK_TYPES:
            raise CatalogSchemaError(f"unsupported catalog type {task_type!r} for row {row_id!r}")
        if self.expected_type is not None and task_type != self.expected_type:
            raise CatalogSchemaError(f"catalog row {row_id!r} has type {task_type!r}, expected {self.expected_type!r}")
        if not isinstance(row["dataset"], str) or not row["dataset"]:
            raise CatalogSchemaError(f"dataset must be a non-empty string for row {row_id!r}")
        return {
            "id": row_id,
            "type": task_type,
            "dataset": row["dataset"],
            "data": self._parse_json(row["data"], field="data", row_id=row_id),
            "provenance": self._parse_json(row["provenance"], field="provenance", row_id=row_id),
            "_catalog_index": index,
            "_catalog_fingerprint": self.fingerprint,
        }


def _stable_seed(namespace: str, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"cosmosdl")
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big")


class ParquetBlockDistributor(DataDistributor):
    """Bounded row-group shuffle with disjoint DP-rank/worker assignment."""

    requires_persistent_workers = True

    def __init__(
        self,
        dataset: ParquetCatalogDataset,
        seed: int = 0,
        shuffle: bool = True,
        name: str = "",
    ) -> None:
        self._dataset = dataset
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._name = name
        self.dataset_fingerprint = dataset.fingerprint

    def __len__(self) -> int:
        return len(self._dataset)

    def active_worker_ids(self, dp_rank, dp_world_size, num_workers):
        rows = max(0, (len(self._dataset) - dp_rank + dp_world_size - 1) // dp_world_size)
        return list(range(min(num_workers, rows)))

    def _epoch_indices(self, epoch: int) -> Iterator[int]:
        group_order = list(range(len(self._dataset.row_groups)))
        if self._shuffle:
            group_order.sort(key=lambda group_index: _stable_seed("row-group", self._seed, epoch, group_index))
        for group_index in group_order:
            group = self._dataset.row_groups[group_index]
            offsets = list(range(group.num_rows))
            if self._shuffle:
                random.Random(_stable_seed("row", self._seed, epoch, group.file_index, group.row_group_index)).shuffle(
                    offsets
                )
            for offset in offsets:
                yield group.start + offset

    def stream(self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int) -> Iterator[dict[str, Any]]:
        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        # Split by DP rank first, then split that rank's rows among its local
        # workers. A flat ``rank * num_workers + worker`` modulo assignment makes
        # later ranks empty whenever a small catalog has fewer rows than the
        # total number of workers (for example 8 rows on 8 ranks x 4 workers).
        rank_row_count = (len(self._dataset) - dp_rank + dp_world_size - 1) // dp_world_size
        if rank_row_count <= 0 or worker_id >= rank_row_count:
            return
        prefix = f"COSMOS_DL_STATE_{self._name}_" if self._name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_EPOCH", 0))
        resume_pos = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_INDEX", -1))
        epoch = resume_epoch
        while True:
            local_position = 0
            rank_position = 0
            for global_position, index in enumerate(self._epoch_indices(epoch)):
                if global_position % dp_world_size != dp_rank:
                    continue
                if rank_position % num_workers != worker_id:
                    rank_position += 1
                    continue
                rank_position += 1
                if epoch == resume_epoch and local_position <= resume_pos:
                    local_position += 1
                    continue
                observe_candidate(self._dataset, index, epoch, local_position)
                item = self._dataset[index]
                output = [
                    {
                        "_dp_epoch": epoch,
                        "_dp_stream_pos": local_position,
                        "_dp_dataset_fingerprint": self.dataset_fingerprint,
                        "_catalog_stream_id": stream_id,
                        "_catalog_total_streams": total_streams,
                        **item,
                    }
                ]
                del item
                yield output.pop()
                local_position += 1
            epoch += 1


def _episode_candidates(blocks, seed, stream_id, total_streams, resume_epoch, resume_pos):
    """Pure window enumeration shared by execution and disposable lookahead."""
    epoch = resume_epoch
    while True:
        order = sorted(range(len(blocks)), key=lambda block: _stable_seed("action-block", seed, epoch, block))
        position = 0
        for offset, block in enumerate(order):
            if offset % total_streams != stream_id:
                continue
            start, length = blocks[block]
            for index in range(start, start + length):
                if epoch != resume_epoch or position > resume_pos:
                    yield epoch, position, index
                position += 1
        epoch += 1


class _ActionCandidateAudit:
    """Worker-local execution counts, not training-consumption counts.

    Counts include speculative loader work and are reset on stream recreation.
    An instance id lets downstream analysis distinguish resumes and sum only the
    final cumulative report for each instance. No sampling state is touched.
    """

    def __init__(self, dataset: str, dp_rank: int, worker_id: int, report_every: int = 128):
        self.dataset = dataset
        self.dp_rank = dp_rank
        self.worker_id = worker_id
        self.instance = f"{os.getpid()}-{time.monotonic_ns()}"
        self.report_every = report_every
        self.counts = {
            "candidates": 0,
            "prepared": 0,
            "no_target": 0,
            "other_drop": 0,
            "media_error": 0,
            "excluded": 0,
        }
        self.last_reported = 0

    def record(self, outcome: str) -> None:
        self.counts["candidates"] += 1
        self.counts[outcome] += 1
        if self.counts["candidates"] % self.report_every == 0:
            self.report()

    def report(self) -> None:
        if self.counts["candidates"] == self.last_reported:
            return
        payload = {
            "dataset": self.dataset,
            "dp_rank": self.dp_rank,
            "worker_id": self.worker_id,
            "instance": self.instance,
            **self.counts,
        }
        log.info(
            f"Action128 candidate audit: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", rank0_only=False
        )
        self.last_reported = self.counts["candidates"]


class EpisodeBlockDistributor(DataDistributor):
    """Shuffle Action episode blocks while reading windows sequentially."""

    requires_persistent_workers = True

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        seed: int = 0,
        name: str = "",
        dataset_fingerprint: str | None = None,
        fingerprint_suffix: str | None = None,
        media_config: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(dataset, "get_shuffle_blocks"):
            raise TypeError("EpisodeBlockDistributor requires dataset.get_shuffle_blocks()")
        self._dataset = dataset
        self._seed = int(seed)
        self._name = name
        self._fingerprint_suffix = fingerprint_suffix
        self.dataset_fingerprint = self._with_processing_fingerprint(dataset_fingerprint)
        self._media_config = dict(media_config or {})
        # Optional immutable original shard addresses, installed before workers
        # start when a checkpoint is redistributed to a different DP size.
        self._virtual_shards = None

    def _with_processing_fingerprint(self, fingerprint: str | None) -> str | None:
        if fingerprint is None or self._fingerprint_suffix is None:
            return fingerprint
        return hashlib.sha256(f"{fingerprint}:{self._fingerprint_suffix}".encode()).hexdigest()

    def dataset_fingerprint_for_spec(self, spec) -> str | None:
        """Retain sample processing when validating an older Action128 registry."""
        if not hasattr(self._dataset, "fingerprint_for_spec"):
            return self.dataset_fingerprint
        return self._with_processing_fingerprint(self._dataset.fingerprint_for_spec(spec))

    def active_worker_ids(self, dp_rank, dp_world_size, num_workers):
        from itertools import islice

        # UnifiedAction128 returns an iterator, not a sized sequence. We only
        # need to know whether each local shard has at least one block; avoid
        # materializing/counting a potentially huge episode table here.
        blocks = getattr(self._dataset, "num_shuffle_blocks", None)
        if blocks is None:
            blocks = sum(1 for _ in islice(self._dataset.get_shuffle_blocks(), (dp_rank + 1) * num_workers))
        return [worker for worker in range(num_workers) if dp_rank * num_workers + worker < blocks]

    def prepare_media_lookahead(self, dp_rank, dp_world_size, worker_id, num_workers, *, shared_budget=None):
        enabled = getattr(self._dataset, "media_lookahead_enabled", None)
        if enabled is None or not enabled():
            return None
        stream_id, total_streams = dp_rank * num_workers + worker_id, dp_world_size * num_workers
        if self._virtual_shards is not None:
            stream_id, total_streams = self._virtual_shards[worker_id]
        key = (stream_id, total_streams)
        if not hasattr(self, "_media_lookaheads"):
            self._media_lookaheads = {}
        cached = self._media_lookaheads.get(key)
        if cached is not None and not cached.closed:
            return cached
        blocks = tuple(self._dataset.get_shuffle_blocks())
        if stream_id >= len(blocks):
            return None
        prefix = f"COSMOS_DL_STATE_{self._name}_" if self._name else "COSMOS_DL_STATE_"
        # Peek rather than pop: the ordinary sampler still owns resume state.
        epoch = int(os.environ.get(f"{prefix}WORKER_{worker_id}_EPOCH", 0))
        position = int(os.environ.get(f"{prefix}WORKER_{worker_id}_INDEX", -1))
        candidates = _episode_candidates(blocks, self._seed, stream_id, total_streams, epoch, position)
        cache = self._dataset.make_media_lookahead(candidates, shared_budget=shared_budget)
        self._media_lookaheads[key] = cache
        return cache

    def stream(self, dp_rank: int, dp_world_size: int, worker_id: int, num_workers: int) -> Iterator[dict[str, Any]]:
        from cosmos_framework.data.generator.action.media_failure_policy import ActionMediaFailurePolicy
        from cosmos_framework.data.generator.media_decode import MediaDecodeError

        policy = ActionMediaFailurePolicy(self._media_config, self._name, dp_rank, worker_id)
        if policy.exclusions and not hasattr(self._dataset, "media_sample_identity"):
            raise ValueError("Action window exclusions require a dataset with media_sample_identity()")
        blocks = tuple(self._dataset.get_shuffle_blocks())  # type: ignore[attr-defined]
        stream_id = dp_rank * num_workers + worker_id
        total_streams = dp_world_size * num_workers
        if self._virtual_shards is not None:
            stream_id, total_streams = self._virtual_shards[worker_id]
        if stream_id >= len(blocks):
            return
        media_prefetch = self.prepare_media_lookahead(dp_rank, dp_world_size, worker_id, num_workers)
        prefix = f"COSMOS_DL_STATE_{self._name}_" if self._name else "COSMOS_DL_STATE_"
        resume_epoch = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_EPOCH", 0))
        resume_pos = int(os.environ.pop(f"{prefix}WORKER_{worker_id}_INDEX", -1))
        audit = _ActionCandidateAudit(self._name, dp_rank, worker_id)
        try:
            for epoch, local_position, index in _episode_candidates(
                blocks, self._seed, stream_id, total_streams, resume_epoch, resume_pos
            ):
                if media_prefetch is not None:
                    media_prefetch.advance(epoch, local_position)
                get_item = getattr(self._dataset, "get_item", None)
                context = {"sample_index": index, "epoch": epoch, "stream_position": local_position}
                if policy.exclusions and hasattr(self._dataset, "media_sample_identity"):
                    context.update(self._dataset.media_sample_identity(index))
                if policy.excludes(context):
                    audit.record("excluded")
                    policy.drop(
                        {**context, "audit_instance": audit.instance, "candidate_counts": dict(audit.counts)},
                        "excluded",
                        None,
                    )
                    continue
                observe_candidate(self._dataset, index, epoch, local_position)
                try:
                    if media_prefetch is not None:
                        item = get_item(index, epoch=epoch, media_prefetch=media_prefetch)
                    else:
                        item = get_item(index, epoch=epoch) if get_item is not None else self._dataset[index]
                except MediaDecodeError as error:
                    if not policy.enabled:
                        raise
                    if hasattr(self._dataset, "media_sample_identity"):
                        context.update(self._dataset.media_sample_identity(index))
                    audit.record("media_error")
                    policy.drop(
                        {
                            **context,
                            **(error.context or {}),
                            "audit_instance": audit.instance,
                            "candidate_counts": dict(audit.counts),
                        },
                        "decode_error",
                        error,
                    )
                    # Advance the original candidate cursor, not the successful
                    # sample count. Resume must not replay the skipped window.
                    continue
                policy.success()
                if isinstance(item, dict) and item.get(DROP_SAMPLE_KEY):
                    outcome = (
                        "no_target"
                        if item.get(DROP_SAMPLE_REASON_KEY) == "Action128 window has no eligible target"
                        else "other_drop"
                    )
                    audit.record(outcome)
                    item[DROP_SAMPLE_CONTEXT_KEY] = {
                        **item.get(DROP_SAMPLE_CONTEXT_KEY, {}),
                        **context,
                        "dataset": self._name,
                        "dp_rank": dp_rank,
                        "worker_id": worker_id,
                        "audit_instance": audit.instance,
                        "candidate_counts": dict(audit.counts),
                    }
                else:
                    audit.record("prepared")
                output = [{"_dp_epoch": epoch, "_dp_stream_pos": local_position, **item}]
                del item
                yield output.pop()
        finally:
            audit.report()
            if media_prefetch is not None:
                media_prefetch.close()


__all__ = [
    "CATALOG_COLUMNS",
    "CATALOG_TASK_TYPES",
    "CatalogRowGroup",
    "CatalogSchemaError",
    "EpisodeBlockDistributor",
    "ParquetBlockDistributor",
    "ParquetCatalogDataset",
]
