# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Published episode clock -> native per-camera clock/frame identity.

Only video queries are mapped. Numeric State and Action are already published
on the training time grid and must never be resampled here.
"""

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.catalog import CatalogSchemaError


def episode_views(episode):
    result = []
    for view in episode["video_views"]:
        native = json.loads(view.get("native_metadata_json") or "{}")
        native.update({k: v for k, v in view.items() if v is not None and k != "native_metadata_json"})
        result.append(native)
    return result


def nearest_indices(values, targets):
    indexes = np.searchsorted(values, targets).clip(0, len(values) - 1)
    previous = np.maximum(indexes - 1, 0)
    return np.where(abs(values[previous] - targets) < abs(values[indexes] - targets), previous, indexes)


class MediaTimeline:
    """Bounded process-local frozen NPZ arrays, invalidated by file identity."""

    def __init__(self):
        self.pid = os.getpid()
        self.arrays = OrderedDict()
        self.bytes = 0

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    def array(self, path, key):
        if self.pid != os.getpid():
            self.__init__()
        path = Path(path)
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns, key)
        if identity not in self.arrays:
            with np.load(path, allow_pickle=False) as mapping:
                values = np.asarray(mapping[key], dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (np.diff(values) <= 0).any():
                raise CatalogSchemaError(f"invalid frozen media timeline: {path}:{key}")
            half_step = float(np.median(np.diff(values))) / 2 if len(values) > 1 else 0.0
            while self.arrays and (len(self.arrays) >= 8 or self.bytes + values.nbytes > 128 * 1024**2):
                _, (old, _) = self.arrays.popitem(last=False)
                self.bytes -= old.nbytes
            if values.nbytes > 128 * 1024**2:
                raise CatalogSchemaError(f"oversized frozen media timeline: {path}:{key}")
            self.arrays[identity] = (values, half_step)
            self.bytes += values.nbytes
        self.arrays.move_to_end(identity)
        return self.arrays[identity]

    def resolve(self, root, episode, view, published_times, rows=None):
        times = np.asarray(published_times, dtype=np.float64)
        if isinstance(view.get("time_scale"), bool):
            raise CatalogSchemaError("video time_scale must be numeric, not boolean")
        scale = float(view.get("time_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0:
            raise CatalogSchemaError("video time_scale must be finite and positive")
        column = view.get("pts_column")
        row_indexes = None
        if rows is not None:
            row_indexes = nearest_indices(rows["timestamp"].to_numpy(), times)
            if not column and view["role"] == "main" and "source_media_pts" in rows.column_names:
                column = "source_media_pts"
        if column:
            if rows is None or column not in rows.column_names:
                raise CatalogSchemaError(f"missing declared video PTS column: {column}")
            source = rows[column].to_numpy()[row_indexes].astype(np.float64)
            rule = column
        elif view.get("frame_mapping"):
            if rows is None:
                raise CatalogSchemaError("frame_mapping requires episode rows")
            source = self.array(Path(root) / view["frame_mapping"], "selected_pts")[0][row_indexes]
            rule = "frame_mapping"
        else:
            # Frozen R2 mappings are defined at numeric-row positions, including
            # the final measured row before a partial-window padding region.
            mapped_times = (
                rows["timestamp"].to_numpy()[row_indexes]
                if rows is not None and ("time_scale" in view or episode.get("source_times_path"))
                else times
            )
            source = float(view["from_timestamp"]) + mapped_times * scale
            rule = "affine"
        if not np.isfinite(source).all():
            raise CatalogSchemaError("nonfinite video source timestamps")
        # Coverage endpoints may be the final frame PTS, so equality is valid.
        coverage_tolerance = 1e-7
        if column or (episode.get("source_times_path") and view.get("timeline_key")):
            # Measured PTS can straddle the nominal endpoint by a half frame.
            # Do not relax coverage for ordinary affine/legacy video references.
            if view.get("source_fps") is None or isinstance(view["source_fps"], bool):
                raise CatalogSchemaError("frozen video mapping requires declared source_fps")
            source_fps = float(view["source_fps"])
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise CatalogSchemaError("invalid native source FPS")
            coverage_tolerance += 0.5 / source_fps
        if (source < float(view["from_timestamp"]) - coverage_tolerance).any() or (
            source > float(view["to_timestamp"]) + coverage_tolerance
        ).any():
            raise CatalogSchemaError("camera reference does not cover episode window")
        query = source - float(view.get("source_clock_origin_s", 0.0))
        frame_indexes = None
        if episode.get("source_times_path") and view.get("timeline_key"):
            timeline, half_step = self.array(Path(root) / episode["source_times_path"], view["timeline_key"])
            frame_indexes = nearest_indices(timeline, query)
            if (np.abs(timeline[frame_indexes] - query) > half_step + 1e-7).any():
                raise CatalogSchemaError("no synchronized view frame near the selected source time")
            query = timeline[frame_indexes]
        elif rows is not None and view["role"] == "main" and "source_frame_id" in rows.column_names:
            frame_indexes = rows["source_frame_id"].to_numpy()[row_indexes]
        return (
            query,
            None if frame_indexes is None else frame_indexes.astype(np.int64),
            {
                "published_timestamps_sec": times.tolist(),
                "mapping_rule": rule,
                "source_timestamps_sec": source.tolist(),
                "decoder_timestamps_sec": query.tolist(),
                "source_frame_indexes": None if frame_indexes is None else frame_indexes.tolist(),
                "time_scale": scale,
                "role": view["role"],
            },
        )
