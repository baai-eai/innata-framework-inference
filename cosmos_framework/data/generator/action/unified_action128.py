# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the published Action128 contract without dataset-specific action transforms.

Source data are read-only. Integer domain IDs in source metadata are local to a
dataset; only the stable domain strings participate in the global registry.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from cosmos_framework.data.generator.action.episode_metadata_cache import EPISODE_COLUMNS, load_episode_metadata
from cosmos_framework.data.generator.action.projection_routing import (
    EEF_ABSOLUTE_STATE_HEAD_ID,
    NUM_PROJECTION_SLOTS,
    PROJECTION_HEAD_IDS,
    PROJECTION_ROUTING_VERSION,
    PUBLISHED_PROJECTION_FAMILY_IDS,
    projection_family_id,
    projection_head_id,
)
from cosmos_framework.data.generator.catalog import CatalogSchemaError
from cosmos_framework.data.generator.drop_sample_contract import (
    DROP_SAMPLE_CONTEXT_KEY,
    DROP_SAMPLE_KEY,
    DROP_SAMPLE_REASON_KEY,
)
from cosmos_framework.data.generator.media_decode import MediaDecodeError, SupervisedVideoDecoder
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils import log


def stable_hash(value) -> str:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return hashlib.blake2b(json.dumps(value, sort_keys=True).encode(), digest_size=16).hexdigest()


_NORMALIZATION_REFERENCE = {
    "path": "meta/action_normalization.json",
    "schema_version": "action-normalization-v1",
    "policy_version": "domain-slot-q01-q99-v1",
    "storage": "raw_action_plus_dataloader_transform",
    "authoritative_key": "action_domain",
}
_A1_REFERENCE = {"path": "meta/a1_structural_gate.json", "status": "passed"}


def _require_published_action128_contract(root_path: Path) -> tuple[dict, dict]:
    """Load the mandatory publisher-owned transform and structural gate."""
    root = str(root_path)
    try:
        info = json.loads((root_path / "meta/info.json").read_text())
        manifest = json.loads((root_path / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CatalogSchemaError(f"missing or invalid Action128 publication metadata: {root}") from error
    reference = info.get("action_normalization")
    compact = info.get("schema_version") == "action-state-r2-parquet-delivery-20261028-v1"
    if not isinstance(reference, (dict, str)):
        raise CatalogSchemaError(f"invalid or missing Action128 normalization reference in meta/info.json: {root}")
    if isinstance(reference, str):
        reference = {"path": reference}
    if reference.get("path") != _NORMALIZATION_REFERENCE["path"]:
        raise CatalogSchemaError(f"invalid Action128 normalization path: {root}")
    manifest_reference = manifest.get("action_normalization")
    if isinstance(manifest_reference, str):
        manifest_reference = {"path": manifest_reference}
    if isinstance(manifest_reference, dict) and manifest_reference.get("path") != reference["path"]:
        raise CatalogSchemaError(f"invalid or missing Action128 normalization reference in manifest.json: {root}")
    if not compact and (
        info.get("a1_structural_gate") != _A1_REFERENCE or manifest.get("a1_structural_gate") != _A1_REFERENCE
    ):
        raise CatalogSchemaError(f"invalid or missing Action128 A1 reference: {root}")
    try:
        document = json.loads((root_path / _NORMALIZATION_REFERENCE["path"]).read_text())
        gate = json.loads((root_path / _A1_REFERENCE["path"]).read_text()) if not compact else {}
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CatalogSchemaError(f"missing or invalid mandatory Action128 sidecar: {root}") from error
    required_document = {
        "schema_version": "action-normalization-v1",
        "action_storage": "raw_physical_or_source_units",
    }
    if any(document.get(key) != value for key, value in required_document.items()):
        raise CatalogSchemaError(f"unsupported Action128 normalization contract: {root}")
    if document.get("policy_version") not in {
        "domain-slot-q01-q99-v1",
        "domain-slot-shared-state-action-q01-q99-v1",
        "domain-slot-separate-state-action-q01-q99-v1",
        "eef-local-delta-state-action-separate-q01-q99-20261029-v1",
        "robot-limit-affine-no-clip-v1",
    }:
        raise CatalogSchemaError(f"unsupported Action128 normalization policy: {root}")
    required_fit_scope = {
        "data_role": "training",
        "training_eligible": True,
        "row_mask": "action_valid",
        "slot_mask": "action_dim_mask",
        "control_mask_used_for_fit": False,
    }
    required_application = {
        "performed_by": "training dataloader",
        "parquet_values_pre_scaled": False,
        "do_not_apply_additional_mean_std": True,
        "authoritative_key": "action_domain",
    }
    # The parallel publisher omits the redundant control-mask flag when fitting
    # with action_valid/action_dim_mask, and identifies the normalized field.
    # Accept that representation alongside the original publication format.
    fit_scope = dict(document.get("fit_scope") or {})
    fit_scope.setdefault("control_mask_used_for_fit", False)
    application = dict(document.get("application") or {})
    field = application.pop("field", "action")
    if compact:
        fit_scope = required_fit_scope
        application = required_application
        field = "action"
    revised_eef = document.get("eef_state_action_statistics_fitted_separately") is True
    if revised_eef and document.get("normalization_target") == "action":
        fit_scope = required_fit_scope
    if document.get("policy_version") == "robot-limit-affine-no-clip-v1":
        from cosmos_framework.data.generator.action.physical_normalization import validate_physical_document

        validate_physical_document(document, field="action")
        fit_scope = required_fit_scope
    if fit_scope != required_fit_scope or application != required_application or field != "action":
        raise CatalogSchemaError(f"unsupported Action128 fit/application contract: {root}")
    if not compact and (
        gate.get("schema_version") != "a1-structural-action128-v1"
        or gate.get("status") != "passed"
        or gate.get("dataset_name") != document.get("dataset_name")
        or gate.get("failures")
    ):
        raise CatalogSchemaError(f"Action128 A1 structural gate has not passed: {root}")
    return document, gate


def build_action128_spec(roots: list[str]) -> dict:
    domains = {}
    heads = {}
    transforms = {}
    sources = {}
    for root in roots:
        root_path = Path(root)
        document, _ = _require_published_action128_contract(root_path)
        physical = document.get("policy_version") == "robot-limit-affine-no-clip-v1"
        compact = (
            json.loads((root_path / "meta/info.json").read_text()).get("schema_version")
            == "action-state-r2-parquet-delivery-20261028-v1"
        )
        domain_path = root_path / "meta/domains.json"
        local_domains = (
            json.loads(domain_path.read_text())["domains"] if domain_path.exists() else document.get("domains", {})
        )
        if set(local_domains) != set(document.get("domains", {})):
            raise CatalogSchemaError(f"Action128 normalization/domain registry mismatch: {root}")
        routing_path = root_path / "meta/state_action_projection_routing.json"
        published_routing = json.loads(routing_path.read_text()).get("domains", {}) if routing_path.exists() else {}
        for domain, record in document["domains"].items():
            transform = record["transform"]
            statistics = record.get("statistics", {})
            kinds = transform.get("kind")
            if compact and kinds == "domain_slot_affine_q01_q99_to_minus1_plus1":
                kinds = ["robust_quantile" if m else "identity" for m in transform["normalization_mask"]]
            required_statistics = (
                ("lower_bound", "upper_bound")
                if physical
                else (
                    ("count", "q01", "q99")
                    if compact
                    else ("count", "min", "max", "mean", "std", "q01", "q99", "sampled")
                )
            )
            if (
                record.get("slot_count") != 128
                or not isinstance(kinds, list)
                or len(kinds) != 128
                or any(
                    kind not in {"inactive", "identity", "robot_limit_affine" if physical else "robust_quantile"}
                    for kind in kinds
                )
                or any(
                    not isinstance(statistics.get(key), list) or len(statistics[key]) != 128
                    for key in required_statistics
                )
            ):
                raise CatalogSchemaError(f"incomplete Action128 per-slot contract: {domain}")
            normalized = {
                "normalization_mask": transform["normalization_mask"],
                "scale": transform["scale"],
                "bias": transform["bias"],
                "clip_min": transform.get("clip_min"),
                "clip_max": transform.get("clip_max"),
            }
            if physical:
                normalized["clip_policy"] = "none"
            if any(len(normalized[key]) != 128 for key in ("normalization_mask", "scale", "bias")):
                raise CatalogSchemaError(f"Action128 normalization arrays must have length 128: {domain}")
            mask = np.asarray(normalized["normalization_mask"], dtype=np.bool_)
            expected_mask = np.asarray(
                [kind == ("robot_limit_affine" if physical else "robust_quantile") for kind in kinds], dtype=np.bool_
            )
            scale = np.asarray(normalized["scale"], dtype=np.float64)
            bias = np.asarray(normalized["bias"], dtype=np.float64)
            if (
                not np.array_equal(mask, expected_mask)
                or not np.isfinite(scale).all()
                or not np.isfinite(bias).all()
                or (scale[mask] <= 0).any()
            ):
                raise CatalogSchemaError(f"invalid Action128 affine transform: {domain}")
            if not physical and (normalized["clip_min"] != -1.0 or normalized["clip_max"] != 1.0):
                raise CatalogSchemaError(f"unsupported Action128 clipping range: {domain}")
            if domain in transforms and transforms[domain] != normalized:
                raise CatalogSchemaError(f"conflicting Action128 normalization for shared domain: {domain}")
            previous_route = (heads[domain], sources[domain]["state_projection_id"]) if domain in heads else None
            domains[domain] = domain.split("/", 1)[0]
            route = published_routing.get(domain)
            heads[domain] = (
                projection_family_id(route["action_projection_key"])
                if route
                else projection_head_id(record["action_space"], domains[domain])
            )
            if route:
                sources[domain] = {
                    "normalization": stable_hash(record),
                    "state_projection_id": projection_family_id(route["state_projection_key"]),
                }
            transforms[domain] = normalized
            if not route:
                sources[domain] = {
                    "normalization": stable_hash(record),
                    "state_projection_id": EEF_ABSOLUTE_STATE_HEAD_ID
                    if record["action_space"] == "eef20"
                    else heads[domain],
                }
            if previous_route is not None and previous_route != (heads[domain], sources[domain]["state_projection_id"]):
                raise CatalogSchemaError(f"conflicting Action128 projection routing for shared domain: {domain}")
    names = sorted(set(domains.values()))
    ordered = sorted(domains)
    registry = {
        "version": "action128-v3",
        "domains": ordered,
        "embodiments": names,
        "domain_embodiments": [names.index(domains[d]) for d in ordered],
        "domain_projection_ids": [heads[d] for d in ordered],
        "domain_state_projection_ids": [sources[d]["state_projection_id"] for d in ordered],
    }
    return {
        "registry": registry,
        "projection_routing": {
            "version": PROJECTION_ROUTING_VERSION,
            "num_slots": NUM_PROJECTION_SLOTS,
            "keys": {**PROJECTION_HEAD_IDS, **PUBLISHED_PROJECTION_FAMILY_IDS},
        },
        "transforms": [transforms[d] for d in ordered],
        "source_digests": [sources[d] for d in ordered],
    }


def build_registry(roots: list[str]) -> dict:
    """Compatibility helper for callers that need only the global stable IDs."""
    return build_action128_spec(roots)["registry"]


class BoundedRows:
    """PID-aware handles, bounded frame caches, and bounded JSONL indexes."""

    def __init__(self, capacity: int = 8, max_row_group_bytes: int = 256 * 1024**2):
        if (
            isinstance(max_row_group_bytes, bool)
            or not isinstance(max_row_group_bytes, int)
            or max_row_group_bytes <= 0
        ):
            raise ValueError("max_row_group_bytes must be a positive integer")
        self.capacity = capacity
        self.max_row_group_bytes = max_row_group_bytes
        self._reset()

    def _reset(self):
        self.pid = os.getpid()
        self.handles = OrderedDict()
        self.offsets = OrderedDict()
        self.group_key = None
        self.group = None
        self.episode_key = None
        self.episode = None

    def __getstate__(self):
        return {
            "capacity": self.capacity,
            "max_row_group_bytes": self.max_row_group_bytes,
        }

    def __setstate__(self, state):
        self.capacity = state["capacity"]
        self.max_row_group_bytes = state.get("max_row_group_bytes", 256 * 1024**2)
        self._reset()

    def parquet(self, path: Path):
        if self.pid != os.getpid():
            self._reset()
        key = str(path)
        if key not in self.handles:
            self.handles[key] = pq.ParquetFile(path)
        self.handles.move_to_end(key)
        while len(self.handles) > self.capacity:
            _, old = self.handles.popitem(last=False)
            old.close()
        return self.handles[key]

    def row_group(self, path: Path, index: int, columns=None):
        handle = self.parquet(path)
        size = handle.metadata.row_group(index).total_byte_size
        if size > self.max_row_group_bytes:
            raise CatalogSchemaError(
                f"oversized Action128 row group: {path}:{index}; "
                f"uncompressed_bytes={size}, limit_bytes={self.max_row_group_bytes} "
                "(dataloader_train.media.action_max_row_group_bytes)"
            )
        key = (str(path), index, tuple(columns) if columns else None)
        if key != self.group_key:
            # Drop our old Arrow buffers before decoding the next group. RHS
            # evaluation otherwise keeps both groups resident during the read.
            # Invalidate the key too so a failed read cannot return a stale hit.
            self.group = None
            self.group_key = None
            # Physical loader processes already provide parallel reads. Avoid
            # invoking Arrow's column thread pool inside a forked worker.
            # faulthandler shows native read stacks but cannot include frame
            # locals. Record the exact file if an individual read is slow.
            warning = threading.Timer(
                30.0,
                log.warning,
                args=(
                    f"Action128 Parquet read still running after 30s: pid={os.getpid()}, "
                    f"path={path}, row_group={index}, uncompressed_bytes={size}, columns={columns}",
                ),
                kwargs={"rank0_only": False},
            )
            warning.daemon = True
            warning.start()
            try:
                self.group = handle.read_row_group(index, columns=columns, use_threads=False)
            finally:
                warning.cancel()
            self.group_key = key
        return self.group

    def _episode_tables(self, path: Path, index: int, columns):
        handle = self.parquet(path)
        size = handle.metadata.row_group(index).total_byte_size
        if (
            size < 64 * 1024**2 and handle.metadata.row_group(index).num_rows < 65536
        ) or size > self.max_row_group_bytes:
            # Preserve the existing size-limit error and small-group cache.
            yield self.row_group(path, index, columns)
            return
        self.group = None
        self.group_key = None
        warning = threading.Timer(
            30.0,
            log.warning,
            args=(
                f"Action128 Parquet read still running after 30s: pid={os.getpid()}, "
                f"path={path}, row_group={index}, uncompressed_bytes={size}, columns={columns}, mode=streaming",
            ),
            kwargs={"rank0_only": False},
        )
        warning.daemon = True
        warning.start()
        try:
            for batch in handle.iter_batches(batch_size=8192, row_groups=[index], columns=columns, use_threads=False):
                yield pa.Table.from_batches([batch])
        finally:
            warning.cancel()

    def episode_rows(self, path: Path, episode: dict, *, use_state=False, video_only=False):
        # Published files may include nullable provenance/eligibility columns.
        # Read and validate only the frame fields consumed by Action128 training.
        columns = (
            "index",
            "episode_index",
            "timestamp",
            "action",
            "action_dim_mask",
            "action_control_mask",
            "action_valid",
        )
        if use_state:
            columns += ("observation.state", "observation.state_mask")
        if video_only:
            columns = ("index", "episode_index", "timestamp")
        handle = self.parquet(path)
        from cosmos_framework.data.generator.action.media_timeline import episode_views

        views = episode_views(episode) if "video_views" in episode else []
        mapping_columns = {v["pts_column"] for v in views if v.get("pts_column")}
        for key in mapping_columns:
            if key not in handle.schema_arrow.names:
                raise CatalogSchemaError(f"missing declared video PTS column {key}: {path}")
        mapping_columns.update(k for k in ("source_media_pts", "source_frame_id") if k in handle.schema_arrow.names)
        columns += tuple(sorted(mapping_columns))
        episode_key = (
            str(path),
            episode["episode_index"],
            episode["dataset_from_index"],
            episode["dataset_to_index"],
            episode["length"],
            use_state,
            tuple(columns),
            self.max_row_group_bytes,
        )
        if self.episode_key == episode_key:
            return self.episode
        self.episode_key = None
        self.episode = None
        typed_fields = [
            ("action", pa.float32()),
            ("action_dim_mask", pa.bool_()),
            ("action_control_mask", pa.bool_()),
        ]
        if use_state:
            typed_fields += [
                ("observation.state", pa.float32()),
                ("observation.state_mask", pa.bool_()),
            ]
        if video_only:
            typed_fields = []
        for key, dtype in typed_fields:
            if key not in handle.schema_arrow.names:
                raise CatalogSchemaError(f"missing Action128 field {key}: {path}")
            actual = handle.schema_arrow.field(key).type
            if not pa.types.is_fixed_size_list(actual) or actual.list_size != 128 or actual.value_type != dtype:
                raise CatalogSchemaError(f"{key} must be fixed-size {dtype}[128], got {actual}: {path}")
        for key in (
            ("index", "episode_index", "timestamp")
            if video_only
            else ("index", "episode_index", "timestamp", "action_valid")
        ):
            if key not in handle.schema_arrow.names:
                raise CatalogSchemaError(f"missing Action128 field {key}: {path}")
        if not video_only and handle.schema_arrow.field("action_valid").type != pa.bool_():
            raise CatalogSchemaError("action_valid must be boolean")
        index_col = handle.schema_arrow.get_field_index("index")
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        parts = []
        largest_group_bytes = 0
        for i in range(handle.num_row_groups):
            stats = handle.metadata.row_group(i).column(index_col).statistics
            if stats and stats.has_min_max and (stats.max < start or stats.min >= end):
                continue
            largest_group_bytes = max(largest_group_bytes, handle.metadata.row_group(i).total_byte_size)
            for table in self._episode_tables(path, i, columns):
                largest_group_bytes = max(largest_group_bytes, table.nbytes)
                keep = pc.and_(
                    pc.greater_equal(table["index"], start),
                    pc.less(table["index"], end),
                )
                part = table.filter(keep)
                if part.num_rows:
                    parts.append(part)
        if not parts:
            raise CatalogSchemaError(f"episode {episode['episode_index']} has no rows in {path}")
        result = pa.concat_tables(parts)
        null_columns = [key for key in columns if result[key].null_count]
        if null_columns:
            raise CatalogSchemaError(f"nulls in Action128 frame rows: {path}; fields={null_columns}")
        if result.num_rows != episode["length"]:
            raise CatalogSchemaError(f"episode {episode['episode_index']}: metadata/data length mismatch")
        indexes = result["index"].to_numpy()
        if not np.array_equal(indexes, np.arange(start, end)):
            raise CatalogSchemaError("episode frame indexes must be ordered, contiguous and unique")
        if (
            "episode_index" in result.column_names
            and not pc.all(pc.equal(result["episode_index"], episode["episode_index"])).as_py()
        ):
            raise CatalogSchemaError("episode frame rows cross an episode boundary")
        if result.nbytes <= self.max_row_group_bytes:
            # Windows revisit the same episode. Retain its filtered frame rows
            # rather than a potentially GiB-sized group containing many other
            # episodes, once for every active source in every training rank.
            self.group = None
            self.group_key = None
            del table
            if largest_group_bytes >= 64 * 1024**2 or any(
                handle.metadata.row_group(i).num_rows >= 65536 for i in range(handle.num_row_groups)
            ):
                # Arrow's allocator can retain GiB of freed decode buffers.
                # Return unused pages after large reads, once per episode;
                # live episode tables and other live buffers remain valid.
                pa.default_memory_pool().release_unused()
            self.episode = result
            self.episode_key = episode_key
        return result

    def caption(self, path: Path, index: int):
        if self.pid != os.getpid():
            self._reset()
        key = str(path)
        if key not in self.offsets:
            positions = []
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    if not stream.readline():
                        break
                    positions.append(offset)
            self.offsets[key] = np.asarray(positions, dtype=np.int64)
        self.offsets.move_to_end(key)
        while len(self.offsets) > self.capacity:
            self.offsets.popitem(last=False)
        with path.open("rb") as stream:
            stream.seek(int(self.offsets[key][index]))
            return json.loads(stream.readline())


def render_caption(
    caption: dict,
    *,
    start: float,
    duration: float,
    fps: float,
    action_frequency_hz: float,
    width: int,
    height: int,
    embodiment: str,
    camera_layout: list[list[str | int]] | None = None,
    seed,
):
    """Use the VLM's full description as command; keep the plan as separate steps."""
    description = caption.get("source_task_description")
    if not caption.get("task_title") or not isinstance(description, str) or not description.strip():
        raise CatalogSchemaError("Action128 caption requires task_title and nonempty source_task_description")
    if not embodiment:
        raise CatalogSchemaError("Action128 caption requires embodiment_name")
    result = {
        "task_title": caption["task_title"],
        "command": description,
        "embodiment": embodiment,
        "duration": f"{duration:g}s",
        "fps": fps,
        "action_frequency_hz": action_frequency_hz,
        "resolution": {"W": width, "H": height},
        "aspect_ratio": f"{width // math.gcd(width, height)}:{height // math.gcd(width, height)}",
    }
    if camera_layout is not None:
        result["camera_layout"] = camera_layout
    draw = int(stable_hash(seed), 16) / 2**128
    subtasks = caption.get("subtasks", [])
    if draw < 0.9:
        result["task_plan"] = [s["command"] for s in subtasks if not s.get("is_idle", False)]
    if 0.6 <= draw < 0.9:
        current = [s for s in subtasks if s["start_time"] <= start < s["end_time"] and not s.get("is_idle", False)]
        if current:
            result["current_subtask"] = current[0]["command"]
    return json.dumps(result, ensure_ascii=False)


def view_layout(episode: dict):
    from cosmos_framework.data.generator.action.media_timeline import episode_views

    expanded_views = episode_views(episode)
    views = {v["role"]: v for v in expanded_views}
    mapping = episode["view_mapping"]
    for role, view in views.items():
        if mapping.get(role) != view["source_key"]:
            raise CatalogSchemaError(f"view_mapping disagrees with video_views: {role}")
    auxiliary = [v for v in expanded_views if v["role"].startswith("auxiliary_")]
    primary = views.get("main") or (auxiliary[0] if auxiliary else None)
    selected = ([primary] if primary is not None else []) + [
        views[r] for r in ("left_wrist", "right_wrist") if r in views and views[r] is not primary
    ]
    selected += [v for v in auxiliary if v not in selected]
    selected = selected[:3]
    if not selected:
        raise CatalogSchemaError("Action128 episode has no video view")
    ratio = selected[0]["width"] / selected[0]["height"]
    family = min(range(3), key=lambda i: abs(math.log(ratio / (16 / 9, 3 / 2, 4 / 3)[i])))
    sizes = (
        ((848, 480), (640, 720), (752, 640)),
        ((720, 480), (640, 848), (640, 640)),
        ((640, 480), (640, 960), (640, 720)),
    )
    width, height = sizes[family][len(selected) - 1]
    if len(selected) == 1:
        boxes = [(0, 0, width, height)]
    elif len(selected) == 2:
        boxes = [(0, 0, width, height // 2), (0, height // 2, width, height)]
    else:
        boundary = round(height * 2 / 3)
        boxes = [(0, 0, width, boundary), (0, boundary, width // 2, height), (width // 2, boundary, width, height)]
    return selected, boxes, width, height


def _copy_resized_camera(frames: list[np.ndarray], destination: torch.Tensor) -> None:
    from cosmos_framework.data.generator.camera_resize import copy_resized_camera

    copy_resized_camera(frames, destination)


_EPISODE_COLUMNS = EPISODE_COLUMNS  # Compatibility for metadata profiling tools.


class UnifiedAction128Dataset(torch.utils.data.Dataset):
    """One reader for all R2 embodiments. No source action values are rewritten."""

    def __init__(
        self,
        root,
        *,
        action128_spec=None,
        tokenizer_config=None,
        seed=42,
        mode="wam",
        fps=None,
        video_fps_max=None,
        episode_filter=None,
        source_schema_version="unified-lerobot-action128-v1",
        tail_policy="pad_mask",
        use_state=False,
        window_duration_seconds=2.0,
        window_stride_seconds=2.0,
        action_normalization=None,
        control_only=False,
        video_decoder=None,
        max_row_group_bytes=256 * 1024**2,
        episode_cache_dir=None,
    ):
        if mode != "wam" or action_normalization is not None:
            raise CatalogSchemaError("Action128 currently requires WAM and published raw loader values")
        if fps is not None and (
            isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
        ):
            raise CatalogSchemaError("video fps override must be finite and positive")
        self.video_fps = float(fps) if fps is not None else None
        if video_fps_max is not None and (
            isinstance(video_fps_max, bool)
            or not isinstance(video_fps_max, (int, float))
            or not math.isfinite(video_fps_max)
            or video_fps_max <= 0
        ):
            raise CatalogSchemaError("video fps cap must be finite and positive")
        self.video_fps_max = float(video_fps_max) if video_fps_max is not None else None
        if tail_policy != "pad_mask" or source_schema_version not in {
            "unified-lerobot-action128-v1",
            "unified-lerobot-action128-v2",
        }:
            raise CatalogSchemaError("unsupported Action128 source version/tail policy")
        if episode_filter not in (None, {"data_role": "training", "training_eligible": True}):
            raise CatalogSchemaError("Action128 only permits the published training collection")
        if window_duration_seconds <= 0 or window_stride_seconds <= 0:
            raise CatalogSchemaError("window duration/stride must be positive")
        self.root = Path(root)
        self.use_state = use_state
        self.state_transforms = None
        if use_state:
            from cosmos_framework.data.generator.action.pretrain_state import load_state_transforms

            self.state_transforms = load_state_transforms(self.root)
        self.action128_spec = action128_spec or build_action128_spec([str(self.root)])
        self.registry = self.action128_spec["registry"]
        self.domain_ids = {d: i for i, d in enumerate(self.registry["domains"])}
        self.transforms = self.action128_spec["transforms"]
        self.tokenizer = tokenizer_config
        self.seed = seed
        self.duration = window_duration_seconds
        self.stride = window_stride_seconds
        self.control_only = control_only
        self.reader = BoundedRows(max_row_group_bytes=max_row_group_bytes)
        self.decoder = video_decoder or SupervisedVideoDecoder()
        from cosmos_framework.data.generator.action.zarr_tar_video import ZarrTarVideoDecoder

        self.zarr_tar_decoder = ZarrTarVideoDecoder()
        from cosmos_framework.data.generator.action.media_timeline import MediaTimeline

        self.media_timeline = MediaTimeline()
        metadata = load_episode_metadata(self.root, stride=self.stride, cache_dir=episode_cache_dir)
        self.episodes = metadata.episodes
        self.counts, self.ends = metadata.counts, metadata.ends
        self.episode_cache_path, self.episode_cache_hit = metadata.cache_path, metadata.cache_hit
        if metadata.cache_path is not None:
            log.info(
                f"Action128 episode cache {'hit' if metadata.cache_hit else 'built'}: "
                f"root={self.root}, episodes={len(self.episodes)}, path={metadata.cache_path}"
            )
        if not set(metadata.summary["action_spaces"]).issubset({"joint128", "eef20", "ego128"}):
            raise CatalogSchemaError("unsupported published Action128 action_space")
        for domain in metadata.summary["domains"]:
            if domain not in self.domain_ids:
                raise CatalogSchemaError(f"domain missing from global registry: {domain}")
            if self.use_state and domain not in self.state_transforms["domains"]:
                raise CatalogSchemaError(f"domain missing from State128 normalization: {domain}")
        for domain, embodiment in metadata.summary["domain_embodiments"]:
            if self.registry["embodiments"][self.registry["domain_embodiments"][self.domain_ids[domain]]] != embodiment:
                raise CatalogSchemaError(f"domain/embodiment mismatch: {domain}, {embodiment}")
        self._fingerprint_inputs = {
            "root": str(self.root),
            "metadata": metadata.metadata,
            "duration": self.duration,
            "stride": self.stride,
            "control_only": control_only,
            "video_fps": self.video_fps,
            "video_fps_max": self.video_fps_max,
            "frequency_contract": "view-video-fps-cap-v3",
        }
        if metadata.summary["native_timeline"]:
            self._fingerprint_inputs["media_timeline"] = "frozen-pts-per-view-native-clock-v1"
        if use_state:
            self._fingerprint_inputs["state"] = {
                "policy": "current-h1-optional-state-no-window-shift-v3",
                "normalization": self.state_transforms,
            }
        from cosmos_framework.data.generator.jpeg_tar import STORAGE, image_storage_fingerprint

        media_contract = image_storage_fingerprint(self.root)
        if STORAGE in metadata.summary["storages"] and media_contract is None:
            raise CatalogSchemaError("JPEG views require a validated index publication contract")
        if media_contract is not None:
            self._fingerprint_inputs["image_storage"] = media_contract
        self.fingerprint = self.fingerprint_for_spec(self.action128_spec)

    def fingerprint_for_spec(self, spec):
        """Compare unchanged source data against a checkpoint's smaller global registry."""
        source = stable_hash({**self._fingerprint_inputs, "registry": spec["registry"]})
        return stable_hash([source, stable_hash(spec)])

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def get_shuffle_blocks(self):
        starts = np.concatenate(([0], self.ends[:-1]))
        return zip(starts.tolist(), self.counts.tolist(), strict=True)

    @property
    def num_shuffle_blocks(self):
        """Episode count without materializing Python lists for shard probing."""
        return len(self.counts)

    def __getitem__(self, index):
        return self.get_item(index, epoch=0)

    def media_sample_identity(self, index):
        """Stable window identity for an external quarantine list; no media reads."""
        ep_index = bisect.bisect_right(self.ends, index)
        episode = self.episodes.slice(ep_index, 1).to_pylist()[0]
        window = index - (int(self.ends[ep_index - 1]) if ep_index else 0)
        start = window * self.stride
        return {
            "root": str(self.root),
            "episode_index": int(episode["episode_index"]),
            "window_index": int(window),
            "window_start_sec": start,
            "window_end_sec": min(start + self.duration, episode["duration_seconds"]),
        }

    def media_lookahead_enabled(self):
        sources = {name.strip() for name in os.environ.get("COSMOS_ACTION128_MEDIA_PREFETCH_SOURCES", "").split(",")}
        return self.root.name in sources and isinstance(self.decoder, SupervisedVideoDecoder)

    def make_media_lookahead(self, candidates, *, shared_budget=None):
        """Create disposable pixel speculation without consuming sampling state."""
        if not self.media_lookahead_enabled():
            return None
        from cosmos_framework.data.generator.action.media_prefetch import MediaPrefetchPlan, SelectiveMediaPrefetch
        from cosmos_framework.data.generator.action.zarr_tar_video import ZarrTarVideoDecoder

        budget = int(os.environ.get("COSMOS_ACTION128_MEDIA_PREFETCH_BYTES", str(2 * 1024**3)))
        source_budgets = json.loads(os.environ.get("COSMOS_ACTION128_MEDIA_PREFETCH_SOURCE_BYTES", "{}"))
        budget = int(source_budgets.get(self.root.name, budget))
        lookahead = int(os.environ.get("COSMOS_ACTION128_MEDIA_LOOKAHEAD", "1024"))
        minimum_work = float(os.environ.get("COSMOS_ACTION128_MEDIA_MIN_PIXEL_WORK", "100000000"))
        clone = UnifiedAction128Dataset.__new__(UnifiedAction128Dataset)
        clone.root, clone.duration = self.root, self.duration
        clone.video_fps, clone.video_fps_max = self.video_fps, self.video_fps_max
        clone.decoder = self.decoder.independent_decoder()
        clone.zarr_tar_decoder = ZarrTarVideoDecoder()
        clone.reader = BoundedRows(max_row_group_bytes=self.reader.max_row_group_bytes)
        clone._media_prefetch_stop = threading.Event()

        def plan(candidate):
            epoch, position, index = candidate
            ep_index = bisect.bisect_right(self.ends, index)
            episode = self.episodes.slice(ep_index, 1).to_pylist()[0]
            window = index - (int(self.ends[ep_index - 1]) if ep_index else 0)
            start = window * self.stride
            fps = self.sampling_video_fps(episode)
            selected, _, width, height = view_layout(episode)
            if not all(str(v["path"]).endswith((".mp4", ".mcap")) for v in selected):
                return None
            count = 4 * math.ceil((math.ceil(self.duration * fps - 1e-8)) / 4) + 1
            # Cost is a structural ranking feature, not a claimed duration.
            # MCAP currently replays a three-second GOP pre-roll per window.
            work = sum(
                float(v["width"])
                * float(v["height"])
                * (count + (3 * float(v.get("source_fps", fps)) if str(v["path"]).endswith(".mcap") else 0))
                for v in selected
            )
            reserved = 3 * count * height * width + (1 + (count - 1) // 4) * (height // 16) * (width // 16) + count * 3
            return MediaPrefetchPlan(
                (epoch, index),
                (epoch, position),
                reserved,
                work,
                (episode, start, min(start + self.duration, episode["duration_seconds"]), fps),
            )

        def materialize(task):
            episode, start, end, fps = task.payload
            rows = clone.reader.episode_rows(clone.root / episode["data_path"], episode, video_only=True)
            if clone._media_prefetch_stop.is_set():
                raise MediaDecodeError("media speculation cancelled")
            return clone._video(episode, start, end, fps, rows=rows)

        def size(value):
            return sum(t.numel() * t.element_size() for t in value[:3])

        def cancel():
            clone._media_prefetch_stop.set()
            clone.decoder.close()
            clone.zarr_tar_decoder.close()
            for decoder in getattr(clone, "_parallel_view_decoders", {}).values():
                decoder.close()

        return SelectiveMediaPrefetch(
            candidates,
            plan,
            materialize,
            budget_bytes=budget,
            lookahead=lookahead,
            minimum_work=minimum_work,
            size=size,
            cancel=cancel,
            shared_budget=shared_budget,
        )

    def get_item(self, index, *, epoch, media_prefetch=None):
        timing = os.environ.get("COSMOS_DATALOADER_TIMING") == "1"
        started = time.monotonic() if timing else 0.0
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_index = bisect.bisect_right(self.ends, index)
        episode = self.episodes.slice(ep_index, 1).to_pylist()[0]
        window = index - (int(self.ends[ep_index - 1]) if ep_index else 0)
        start = window * self.stride
        end = min(start + self.duration, episode["duration_seconds"])
        fps = float(episode["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise CatalogSchemaError("invalid episode fps")
        read_started = time.monotonic() if timing else 0.0
        table = self.reader.episode_rows(self.root / episode["data_path"], episode, use_state=self.use_state)
        numeric_read = time.monotonic() - read_started if timing else 0.0
        ts = table["timestamp"].to_numpy()
        if not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
            raise CatalogSchemaError("episode timestamps must be finite and strictly increasing")
        selected = table.filter(
            pc.and_(pc.greater_equal(table["timestamp"], start - 1e-8), pc.less(table["timestamp"], end - 1e-8))
        )
        n = selected.num_rows
        steps = max(n, math.ceil(self.duration * fps - 1e-8))
        action = torch.zeros(steps, 128, dtype=torch.float32)
        dim_mask = torch.zeros(steps, 128, dtype=torch.bool)
        control_mask = torch.zeros_like(dim_mask)
        valid = torch.zeros(steps, dtype=torch.bool)
        for field, target in (("action", action), ("action_dim_mask", dim_mask), ("action_control_mask", control_mask)):
            values = np.asarray(selected[field].to_pylist()) if n else np.empty((0, 128))
            if values.shape != (n, 128):
                raise CatalogSchemaError(f"{field} must have width 128")
            target[:n] = torch.as_tensor(values)
        valid[:n] = torch.as_tensor(selected["action_valid"].to_numpy().copy())
        if not torch.isfinite(action).all() or torch.any(control_mask & ~dim_mask):
            raise CatalogSchemaError("invalid Action128 values/control mask")
        domain = self.domain_ids[episode["action_domain"]]
        transform = self.transforms[domain]
        from cosmos_framework.data.generator.action.physical_normalization import apply_affine

        apply_affine(action, transform)
        # Affine bias must not turn absent dimensions or padded frames into nonzero inputs.
        action.masked_fill_(~dim_mask, 0)
        temporal = torch.arange(steps) < n
        loss_mask = temporal[:, None] & valid[:, None] & dim_mask
        if self.control_only:
            loss_mask &= control_mask
        if not loss_mask.any():
            valid_frames = int(valid[:n].sum())
            valid_dimensions = int((valid[:n, None] & dim_mask[:n]).sum())
            controlled_dimensions = int((valid[:n, None] & dim_mask[:n] & control_mask[:n]).sum())
            if n == 0:
                category = "no_rows"
            elif valid_frames == 0:
                category = "all_action_invalid"
            elif valid_dimensions == 0:
                category = "no_valid_action_dimensions"
            else:
                category = "no_controlled_action_dimensions"
            return {
                DROP_SAMPLE_KEY: True,
                DROP_SAMPLE_REASON_KEY: "Action128 window has no eligible target",
                DROP_SAMPLE_CONTEXT_KEY: {
                    "root": str(self.root),
                    "sample_index": int(index),
                    "episode_index": int(episode["episode_index"]),
                    "window_index": int(window),
                    "window_start_sec": start,
                    "window_end_sec": end,
                    "epoch": int(epoch),
                    "action_domain": episode["action_domain"],
                    "embodiment": episode["embodiment_name"],
                    "category": category,
                    "rows": n,
                    "valid_frames": valid_frames,
                    "valid_dimensions": valid_dimensions,
                    "controlled_dimensions": controlled_dimensions,
                    "control_only": self.control_only,
                },
            }
        # Native action timestamps are not inferred from frame indexes.
        action_ts = torch.arange(steps, dtype=torch.float64) / fps + start
        action_ts[:n] = torch.as_tensor(selected["timestamp"].to_numpy().copy())
        if n and n < steps:
            action_ts[n:] = action_ts[n - 1] + torch.arange(1, steps - n + 1, dtype=torch.float64) / fps
        sample_key = f"{self.root.name}/{episode['episode_index']}/{window}"
        # Video sampling comes from camera metadata, never the action frequency.
        # Explicit sampling settings are ceilings: do not upsample slower views.
        video_fps = self.sampling_video_fps(episode)
        video_started = time.monotonic() if timing else 0.0
        prepared_video = media_prefetch.get((epoch, index)) if media_prefetch is not None else None
        if prepared_video is None:
            prepared_video = self._video(episode, start, end, video_fps, rows=table)
        video, video_valid, cameras, decode_meta, width, height = prepared_video
        video_seconds = time.monotonic() - video_started if timing else 0.0
        caption_started = time.monotonic() if timing else 0.0
        caption = self.reader.caption(self.root / episode["caption_path"], episode["caption_index"])
        caption_read = time.monotonic() - caption_started if timing else 0.0
        caption_text = render_caption(
            caption,
            start=start,
            duration=self.duration,
            fps=video_fps,
            action_frequency_hz=fps,
            width=width,
            height=height,
            embodiment=episode["embodiment_name"],
            camera_layout=decode_meta.get("camera_layout"),
            seed=(self.seed, self.fingerprint, sample_key, epoch),
        )
        if self.tokenizer is None:
            raise CatalogSchemaError("Action128 training requires a Qwen3VL processor")
        from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_generation_caption

        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        ids, text_mask = tokenize_generation_caption(caption_text, tokenizer, is_video=True)
        state_indexes = []
        relative_times = action_ts - start
        if self.use_state:
            from cosmos_framework.data.generator.action.pretrain_state import current_state

            state, state_mask, state_times = current_state(
                table, start=start, transform=self.state_transforms["domains"][episode["action_domain"]]["transform"]
            )
            # Missing/masked current State removes only the condition token.
            # Keep the original window and its video/Action targets unchanged.
            if len(state):
                state_indexes = [0]
                action = torch.cat((state, action))
                dim_mask = torch.cat((state_mask, dim_mask))
                control_mask = torch.cat((torch.zeros_like(state_mask), control_mask))
                valid = torch.cat((torch.zeros(1, dtype=torch.bool), valid))
                temporal = torch.cat((torch.ones(1, dtype=torch.bool), temporal))
                loss_mask = torch.cat((torch.zeros_like(state_mask), loss_mask))
                relative_times = torch.cat((state_times, relative_times))
        projection_id = self.registry["domain_projection_ids"][domain]
        projection_ids = torch.tensor([projection_id])
        if state_indexes:
            projection_ids = torch.full((len(action),), projection_id, dtype=torch.long)
            projection_ids[0] = self.registry["domain_state_projection_ids"][domain]
        if timing:
            decode_meta = {
                **decode_meta,
                "timing_seconds": {
                    "numeric_read": numeric_read,
                    "video": video_seconds,
                    "caption_read": caption_read,
                    "sample": time.monotonic() - started,
                },
            }
        return {
            "__key__": sample_key,
            "__url__": str(self.root),
            "ai_caption": caption_text,
            "text_token_ids": torch.tensor(ids),
            "video": video,
            "num_frames": video.shape[1],
            "image_size": torch.tensor([height, width]),
            "fps": video_fps,
            "conditioning_fps": video_fps,
            "conditioning_fps_action": fps,
            "action": action,
            "action_dim_mask": dim_mask,
            "action_control_mask": control_mask,
            "action_valid": valid,
            "temporal_valid_mask": temporal,
            "action_loss_mask": loss_mask,
            "action_timestamps": relative_times,
            "vision_loss_mask": video_valid,
            "camera_valid": cameras,
            # Existing packed domain_id is the projection ID, not the source/statistics ID.
            "domain_id": projection_ids,
            "normalization_domain_id": domain,
            "raw_action_dim": 128,
            "embodiment_id": self.registry["domain_embodiments"][domain],
            "action128_contract_fingerprint": stable_hash(self.action128_spec),
            "decode_metadata": decode_meta,
            "sequence_plan": SequencePlan(
                has_text=True,
                supervise_text=False,
                text_loss_mask=text_mask,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=True,
                action_start_frame_offset=0,
                action_timestamps_seconds=relative_times.tolist(),
                condition_frame_indexes_action=state_indexes,
                robot_state_frame_indexes=state_indexes,
            ),
        }

    def sampling_video_fps(self, episode):
        selected, _, _, _ = view_layout(episode)
        rates = []
        for view in selected:
            rate = view.get("fps")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise CatalogSchemaError(f"invalid or missing video_views fps for camera {view['role']!r}")
            rates.append(float(rate))
        # The tiled video has a shared time grid. Use the slowest selected view
        # when cameras differ so no low-FPS camera is implicitly upsampled.
        if self.video_fps is not None:
            rates.append(self.video_fps)
        if self.video_fps_max is not None:
            rates.append(self.video_fps_max)
        return min(rates)

    def _video_request(self, episode, view, box, safe_times, rows):
        from cosmos_framework.data.generator.action.media_timeline import MediaTimeline

        if not hasattr(self, "media_timeline"):
            self.media_timeline = MediaTimeline()
        query = safe_times + float(view["from_timestamp"])
        try:
            query, source_indexes, mapping = self.media_timeline.resolve(self.root, episode, view, safe_times, rows)
        except CatalogSchemaError as error:
            if str(error) != "camera reference does not cover episode window":
                raise
            failure = MediaDecodeError(str(error))
            failure.context = {
                "camera": view["role"],
                "video_path": str(self.root / view["path"]),
                "request_start_sec": float(query[0]),
                "request_end_sec": float(query[-1]),
            }
            raise failure from error
        x0, y0, x1, y1 = box
        return view, str(self.root / view["path"]), [y1 - y0, x1 - x0], query, source_indexes, mapping

    def _decode_video_view(self, request, decoder):
        view, video_path, output_size, query, source_indexes, _ = request
        try:
            stop = getattr(self, "_media_prefetch_stop", None)
            if stop is not None and stop.is_set():
                raise MediaDecodeError("media speculation cancelled")
            if view.get("storage") == "indexed_jpeg_tar_v1":
                return decoder.decode_jpeg(self.root, view, timestamps_sec=query.tolist())
            if (
                view.get("container_format") in {"mcap", "zarr", "zarr_v3", "zarr.tar"}
                or video_path.endswith(".mcap")
                or (".zarr" in video_path and view.get("stream_key"))
            ):
                return decoder.decode(
                    video_path,
                    native_view=view,
                    timestamps_sec=query.tolist(),
                    source_frame_indexes=None if source_indexes is None else source_indexes.tolist(),
                    camera_output_size=output_size,
                )
            if video_path.endswith(".zarr.tar"):
                return self.zarr_tar_decoder.decode(video_path, timestamps_sec=query.tolist())
            return decoder.decode(video_path, timestamps_sec=query.tolist(), camera_output_size=output_size)
        except MediaDecodeError as error:
            error.context = {
                "camera": view["role"],
                "video_path": video_path,
                "request_start_sec": float(query[0]),
                "request_end_sec": float(query[-1]),
            }
            raise

    def _video(self, episode, start, end, fps, *, rows=None):
        timing = os.environ.get("COSMOS_DATALOADER_TIMING") == "1"
        selected, boxes, width, height = view_layout(episode)
        native_count = math.ceil(self.duration * fps - 1e-8) + 1
        count = 4 * math.ceil((native_count - 1) / 4) + 1
        times = start + np.arange(count) / fps
        real = times < end - 1e-8
        if not real.any():
            raise CatalogSchemaError("empty video time interval")
        safe_times = np.minimum(times, times[real][-1])
        canvas = torch.empty((3, count, height, width), dtype=torch.uint8)
        valid = torch.zeros(count, height // 16, width // 16, dtype=torch.bool)
        camera_valid = torch.zeros(count, 3, dtype=torch.bool)
        metadata = []
        sources = {name.strip() for name in os.environ.get("COSMOS_ACTION128_PARALLEL_VIEW_SOURCES", "").split(",")}
        parallel = (
            self.root.name in sources
            and 1 < len(selected) <= 3
            and isinstance(self.decoder, SupervisedVideoDecoder)
            and canvas.numel() <= 512 * 1024**2
            # The legacy Zarr TAR reader is stateful and has no isolated helper.
            and all(
                str(view["path"]).endswith((".mp4", ".mcap")) and view.get("storage") != "indexed_jpeg_tar_v1"
                for view in selected
            )
        )
        executor = None
        futures = []
        try:
            if parallel:
                # Resolve timelines/RNG-independent planning on the calling thread.
                # Each camera owns a socket/codec; at most two decode concurrently.
                requests = [
                    self._video_request(episode, view, box, safe_times, rows)
                    for view, box in zip(selected, boxes, strict=True)
                ]
                if not hasattr(self, "_parallel_view_decoders"):
                    self._parallel_view_decoders = {}
                for i in range(1, len(selected)):
                    if i not in self._parallel_view_decoders:
                        self._parallel_view_decoders[i] = self.decoder.independent_decoder()
                executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="action128-view")
                futures = [
                    executor.submit(
                        self._decode_video_view, request, self.decoder if i == 0 else self._parallel_view_decoders[i]
                    )
                    for i, request in enumerate(requests)
                ]
            for i, (view, box) in enumerate(zip(selected, boxes, strict=True)):
                request = requests[i] if parallel else self._video_request(episode, view, box, safe_times, rows)
                output_size, mapping = request[2], request[5]
                if parallel:
                    frames, decoded = futures[i].result()
                    futures[i] = None
                else:
                    frames, decoded = self._decode_video_view(request, self.decoder)
                copied_at = time.monotonic() if timing else 0.0
                x0, y0, x1, y1 = box
                if decoded.get("camera_output_size") == output_size:
                    for frame_index, frame in enumerate(frames):
                        canvas[:, frame_index, y0:y1, x0:x1].copy_(torch.from_numpy(frame).permute(2, 0, 1))
                else:
                    _copy_resized_camera(frames, canvas[:, :, y0:y1, x0:x1])
                del frames
                if timing:
                    decoded = {
                        **decoded,
                        "timing_seconds": {
                            **decoded.get("timing_seconds", {}),
                            "canvas_copy": time.monotonic() - copied_at,
                        },
                    }
                camera_valid[:, i] = torch.as_tensor(real.copy())
                metadata.append({**decoded, "time_mapping": mapping})
        except MediaDecodeError:
            # Join all socket users before closing/resetting helpers after failure.
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
            self.decoder.close()
            self.zarr_tar_decoder.close()
            for decoder in getattr(self, "_parallel_view_decoders", {}).values():
                decoder.close()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        # All spatial regions, including separator pixels, are valid targets.
        valid[:] = torch.as_tensor(real.copy())[:, None, None]
        if len(selected) > 1:
            boundary = boxes[0][3]
            canvas[:, :, boundary - 2 : boundary + 2, :] = 0
        if len(selected) == 3:
            canvas[:, :, boxes[1][1] :, width // 2 - 2 : width // 2 + 2] = 0
        camera_layout = None
        if len(selected) > 1:
            positions = ["top", "bottom"] if len(selected) == 2 else ["top", "bottom_left", "bottom_right"]
            camera_layout = [
                [view["role"], position, box[2] - box[0], box[3] - box[1]]
                for view, position, box in zip(selected, positions, boxes, strict=True)
            ]
        # Conservative causal VAE mask: initial frame, then complete groups of four.
        latent_valid = torch.cat((valid[:1], valid[1:].reshape(-1, 4, height // 16, width // 16).all(1)))
        return (
            canvas,
            latent_valid,
            camera_valid,
            {
                "views": metadata,
                "selected_roles": [view["role"] for view in selected],
                "primary_role": selected[0]["role"],
                "main_role_fallback": selected[0]["role"] != "main",
                "camera_layout": camera_layout,
            },
            width,
            height,
        )
