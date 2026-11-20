# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an explicit media-only publication change before migrating cursors."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pyarrow.compute as pc

from cosmos_framework.data.generator.action.episode_metadata_cache import EPISODE_COLUMNS


def validate_media_migration(before, after):
    """Require identical ordered training windows and action/State contracts.

    Encoded video descriptors and their timestamp sidecars may change. Action
    payload equivalence is a publisher assertion; this does not scan all media
    or action Parquet values. Both dataset fingerprints remain version-specific.
    """
    if before.root.name != after.root.name:
        raise ValueError("Media migration cannot change sample ID prefixes")
    for field in [
        "action128_spec",
        "state_transforms",
        "duration",
        "stride",
        "use_state",
        "control_only",
        "video_fps",
        "video_fps_max",
    ]:
        if getattr(before, field) != getattr(after, field):
            raise ValueError(f"Media migration changed processing contract: {field}")
    if not np.array_equal(before.counts, after.counts) or not np.array_equal(before.ends, after.ends):
        raise ValueError("Media migration changed ordered window counts")
    # Preserve admission, episode ordering, action clocks, data row ranges,
    # domains, camera role mapping and caption positions. Only media descriptors
    # and the timestamp-sidecar path are intentionally excluded.
    columns = [c for c in EPISODE_COLUMNS if c not in {"video_views", "caption_path"}]
    for column in columns:
        if not before.episodes[column].combine_chunks().equals(after.episodes[column].combine_chunks()):
            raise ValueError(f"Media migration changed ordered episode field: {column}")
    old_views = before.episodes["video_views"]
    new_views = after.episodes["video_views"]
    # Do not combine entire structs: their native_metadata_json can exceed
    # Arrow's 2-GiB string-offset limit on the real Lightwheel publication.
    if not pc.list_value_length(old_views).combine_chunks().equals(pc.list_value_length(new_views).combine_chunks()):
        raise ValueError("Media migration changed camera counts")
    for field in ["role", "source_key"]:
        old_identity = pc.struct_field(pc.list_flatten(old_views), field).combine_chunks()
        new_identity = pc.struct_field(pc.list_flatten(new_views), field).combine_chunks()
        if not old_identity.equals(new_identity):
            raise ValueError(f"Media migration changed camera identity/order: {field}")

    def captions(dataset):
        paths = dataset.episodes["caption_path"].to_pylist()
        normalized = {p: os.path.normpath(os.path.join(str(dataset.root), p)) for p in set(paths)}
        return [normalized[p] for p in paths]

    if captions(before) != captions(after):
        raise ValueError("Media migration changed caption references")
    digest = hashlib.sha256()
    for episode, count in zip(before.episodes["episode_index"].to_pylist(), before.counts, strict=True):
        digest.update(f"{episode}:{int(count)}\n".encode())
    return {
        "old_root": str(before.root),
        "new_root": str(after.root),
        "training_episodes": len(before.episodes),
        "training_windows": len(before),
        "ordered_episode_window_sha256": digest.hexdigest(),
        "preserved_episode_columns": columns,
        "action_state_contract_equal": True,
        "scope": "Ordered admitted metadata, window partition and normalization contracts checked; "
        "publisher asserts action payload equivalence. Encoded pixels/frame clocks are allowed to change.",
    }
