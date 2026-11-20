# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, job-independent cache of projected episode metadata and window indexes.

Source signatures deliberately match the dataset fingerprint's size/mtime policy.
A hit still lists/stats source files, but never opens their Parquet payloads.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_framework.data.generator.catalog import CatalogSchemaError

EPISODE_COLUMNS = (
    "episode_index",
    "action_domain",
    "action_space",
    "embodiment_name",
    "data_role",
    "training_eligible",
    "split",
    "duration_seconds",
    "fps",
    "length",
    "data_path",
    "dataset_from_index",
    "dataset_to_index",
    "caption_path",
    "caption_index",
    "video_views",
    "view_mapping",
)
_VERSION = 1
_BATCH_ROWS = 65536
_META_KEY = b"action128_episode_cache"
_LOG = logging.getLogger(__name__)


@dataclass
class EpisodeMetadata:
    episodes: pa.Table
    metadata: list
    counts: np.ndarray
    ends: np.ndarray
    summary: dict
    cache_hit: bool = False
    cache_path: Path | None = None


def source_signature(root):
    paths = sorted((Path(root) / "meta/episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise CatalogSchemaError(f"no episode metadata: {root}")
    metadata = []
    for path in paths:
        stat = path.stat()
        metadata.append((str(path), stat.st_size, stat.st_mtime_ns))
    return metadata


def _build(metadata, stride):
    # Project before decoding: unused stats arrays can cost tens of GiB per rank.
    # Split is nullable provenance; it is never a train/test admission gate.
    tables = []
    for path, _, _ in metadata:
        path = Path(path)
        columns = list(EPISODE_COLUMNS)
        names = pq.read_schema(path).names
        if "split" not in names:
            columns.remove("split")
        if "source_times_path" in names:
            columns.append("source_times_path")
        table = pq.read_table(path, columns=columns)
        for optional in ("split", "source_times_path"):
            if optional not in table.column_names:
                table = table.append_column(optional, pa.nulls(len(table), type=pa.string()))
        tables.append(table.select([*EPISODE_COLUMNS, "source_times_path"]))
    episodes = pa.concat_tables(tables, promote_options="default")
    episodes = episodes.filter(pc.and_(pc.equal(episodes["data_role"], "training"), episodes["training_eligible"]))
    durations = episodes["duration_seconds"].to_numpy()
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise CatalogSchemaError("training episode duration must be finite and positive")
    counts = np.ceil(durations / stride - 1e-9).astype(np.int64)
    view_type = episodes.schema.field("video_views").type.value_type
    summary = {
        "action_spaces": pc.unique(episodes["action_space"]).to_pylist(),
        "domains": pc.unique(episodes["action_domain"]).to_pylist(),
        "domain_embodiments": sorted(
            set(zip(episodes["action_domain"].to_pylist(), episodes["embodiment_name"].to_pylist(), strict=True))
        ),
        "native_timeline": bool(pc.any(pc.is_valid(episodes["source_times_path"])).as_py()),
        "storages": pc.unique(pc.struct_field(pc.list_flatten(episodes["video_views"]), "storage")).to_pylist()
        if "storage" in view_type.names
        else [],
    }
    return EpisodeMetadata(episodes, metadata, counts, np.cumsum(counts), summary)


def _read(path, key, metadata):
    # Arrow buffers retain the mapping after the Python reader leaves this scope.
    with pa.memory_map(str(path), "r") as mapping:
        reader = pa.ipc.open_file(mapping)
        info = json.loads(reader.schema.metadata[_META_KEY])
        if info["key"] != key or info["version"] != _VERSION:
            raise ValueError("episode cache key/version mismatch")
        table = reader.read_all()
    counts = table["__window_counts"].to_numpy()
    ends = table["__window_ends"].to_numpy()
    episodes = table.select([*EPISODE_COLUMNS, "source_times_path"]).replace_schema_metadata(None)
    return EpisodeMetadata(episodes, metadata, counts, ends, info["summary"], True, path)


def _write(path, key, result):
    table = result.episodes.append_column("__window_counts", pa.array(result.counts))
    table = table.append_column("__window_ends", pa.array(result.ends))
    table = table.replace_schema_metadata(
        {
            _META_KEY: json.dumps(
                {
                    "key": key,
                    "version": _VERSION,
                    "summary": result.summary,
                }
            ).encode()
        }
    )
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                # Consolidate bounded slices. Combining an entire large source
                # can overflow Arrow's 32-bit nested list/string offsets.
                for start in range(0, len(table), _BATCH_ROWS):
                    writer.write_table(table.slice(start, _BATCH_ROWS).combine_chunks())
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_episode_metadata(root, *, stride=2.0, cache_dir=None):
    """Use cache_dir='' to disable; None resolves ACTION128_EPISODE_CACHE_DIR.

    Admission filtering and row order are identical with/without a cache. Global
    registry, normalization and media-publication validation stay in the caller.
    Builders serialize with a shared-filesystem flock and publish atomically.
    """
    if not np.isfinite(stride) or stride <= 0:
        raise CatalogSchemaError("window stride must be finite and positive")
    root = Path(root)
    metadata = source_signature(root)
    cache_dir = os.environ.get("ACTION128_EPISODE_CACHE_DIR", "") if cache_dir is None else cache_dir
    if not cache_dir:
        return _build(metadata, stride)
    key = hashlib.sha256(
        json.dumps(
            {
                "version": _VERSION,
                "root": str(root),
                "metadata": metadata,
                "stride": stride,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path = Path(cache_dir) / f"{key}.arrow"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Fast path does not contend with other readers.
        if path.exists():
            try:
                return _read(path, key, metadata)
            except (OSError, ValueError, KeyError, pa.ArrowException):
                _LOG.warning("Invalid episode cache; rebuilding: %s", path, exc_info=True)
        with path.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                try:
                    return _read(path, key, metadata)
                except (OSError, ValueError, KeyError, pa.ArrowException):
                    pass
            result = _build(metadata, stride)
            if source_signature(root) != metadata:
                raise CatalogSchemaError(f"episode metadata changed while building cache: {root}; retry initialization")
            _write(path, key, result)
            result = _read(path, key, metadata)
            result.cache_hit = False
            return result
    except OSError:
        _LOG.warning("Episode cache unavailable; reading source metadata: %s", path, exc_info=True)
        return _build(metadata, stride)
