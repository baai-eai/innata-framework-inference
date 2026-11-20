# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-independent indexed JPEG TAR media for the unified pretrain reader.

The sidecar describes the JPEG index, not the exclusive container of a dataset:
MP4 and JPEG TAR views may coexist. Frame requests use nearest-frame sampling
(half-frame ties go forward); only a nearest index at the valid final cell is
clamped to the last frame. Requests outside the declared half-open media range
are rejected. No dataset name, task count, or SFT sampling policy is involved.
"""

from __future__ import annotations

import io
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from cosmos_framework.data.generator.media_decode import MediaDecodeError

STORAGE = "indexed_jpeg_tar_v1"
INDEX_PATH = "meta/jpeg_offsets.npy"


def image_storage_fingerprint(root):
    """Validate published index declarations without rehashing GBs in every rank.

    The publisher verifies the full SHA and writes training_validation.json. Runtime
    fingerprints bind that declared SHA, the complete contract and index file stats.
    """
    root = Path(root)
    path = root / "meta/image_storage.json"
    info = json.loads((root / "meta/info.json").read_text())
    if not path.is_file():
        if "indexed_jpeg_tar" in str(info.get("image_storage", "")):
            raise ValueError("JPEG media require meta/image_storage.json")
        return None
    contract = json.loads(path.read_text())
    if "indexed_jpeg_tar" not in str(contract.get("schema_version", "")) and "indexed_jpeg_tar" not in str(
        info.get("image_storage", "")
    ):
        # Other generic media backends own their contracts. A JPEG view without
        # a JPEG declaration is independently rejected by the unified reader.
        return None
    validation_path = root / "training_validation.json"
    if not validation_path.is_file():
        raise ValueError("JPEG media require training_validation.json")
    validation = json.loads(validation_path.read_text())
    sha = contract.get("index_sha256")
    if (
        contract.get("schema_version") != STORAGE
        or contract.get("index_path") != INDEX_PATH
        or contract.get("index_dtype") != "uint64"
        or not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
        or validation.get("status") != "passed"
        or validation.get("image_index_sha256") != sha
    ):
        raise ValueError("Invalid JPEG index publication contract or validation SHA")
    index = root / INDEX_PATH
    header = np.load(index, mmap_mode="r", allow_pickle=False)
    try:
        if (
            header.dtype != np.uint64
            or header.ndim != 3
            or header.shape[1] < 1
            or header.shape[2] != 2
            or list(header.shape) != contract.get("index_shape")
        ):
            raise ValueError("JPEG index header disagrees with uint64[N,cameras,2] contract")
    finally:
        header._mmap.close()
    stat = index.stat()
    return {
        "contract": contract,
        "index_sha256": sha,
        "index_stat": [stat.st_size, stat.st_mtime_ns],
        "sampling_contract": "nearest_frame_half_up_valid_end_clamp_v1",
    }


class IndexedJpegTarReader:
    """Bounded, process-local file handles and an mmap of payload offsets."""

    def __init__(self, root, cache_size=8):
        if cache_size < 1:
            raise ValueError("JPEG handle cache size must be positive")
        self.root = Path(root)
        self.cache_size = cache_size
        self._pid = os.getpid()
        self._fds = OrderedDict()
        self._index = None

    def __getstate__(self):
        return {**self.__dict__, "_fds": OrderedDict(), "_index": None, "_pid": None}

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        if self._index is not None:
            self._index._mmap.close()
            self._index = None

    def _local(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if self._index is None:
            self._index = np.load(self.root / INDEX_PATH, mmap_mode="r", allow_pickle=False)
            if (
                self._index.dtype != np.uint64
                or self._index.ndim != 3
                or self._index.shape[1] < 1
                or self._index.shape[2] != 2
            ):
                self.close()
                raise ValueError("JPEG offsets require uint64[N,cameras,2]")

    def _fd(self, path):
        fd = self._fds.pop(path, None)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
        self._fds[path] = fd
        while len(self._fds) > self.cache_size:
            _, old = self._fds.popitem(last=False)
            os.close(old)
        return fd

    def decode(
        self,
        view,
        *,
        timestamps_sec,
        max_sampled_frames=256,
        max_frame_pixels=16_777_216,
        max_window_bytes=1_073_741_824,
    ):
        required = (
            "storage",
            "index_path",
            "fps",
            "width",
            "height",
            "from_timestamp",
            "frame_count",
            "jpeg_frame_start",
            "jpeg_camera_index",
            "path",
        )
        if any(view.get(key) is None for key in required):
            raise ValueError("JPEG view is missing required storage/frame metadata")
        if view["storage"] != STORAGE or view["index_path"] != INDEX_PATH:
            raise ValueError("Unsupported JPEG view/index storage")
        ts = np.asarray(timestamps_sec, dtype=np.float64)
        fps, origin = float(view["fps"]), float(view["from_timestamp"])
        integers = [view[k] for k in ("width", "height", "frame_count", "jpeg_frame_start", "jpeg_camera_index")]
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in integers):
            raise ValueError("JPEG sizes and index positions must be integers")
        width, height, count, start, camera = map(int, integers)
        if ts.ndim != 1 or not 0 < len(ts) <= max_sampled_frames or not np.isfinite(ts).all():
            raise ValueError("Invalid or oversized JPEG timestamp request")
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(origin) or min(width, height, count) < 1:
            raise ValueError("Invalid JPEG FPS/dimensions/time origin")
        if width * height > max_frame_pixels or len(ts) * width * height * 3 > max_window_bytes:
            raise ValueError("JPEG request exceeds pixel/window byte limit")
        stop = min(origin + count / fps, float(view.get("to_timestamp", origin + count / fps)))
        if not math.isfinite(stop) or stop <= origin or ts.min() < origin - 1e-8 or ts.max() >= stop:
            raise ValueError("JPEG timestamp is outside the half-open media range")
        positions = (ts - origin) * fps
        rounded = np.floor(np.maximum(positions, 0) + 0.5).astype(np.int64)
        # Requests inside the final frame's time cell may round to count; this is
        # endpoint padding, not permission to read past the episode or TAR index.
        frames = np.minimum(rounded, count - 1)
        self._local()
        if start < 0 or start + count > len(self._index) or not 0 <= camera < self._index.shape[1]:
            raise ValueError("JPEG frame/camera is outside the published index")
        unique, inverse = np.unique(frames, return_inverse=True)
        try:
            fd = self._fd(str(self.root / view["path"]))
            file_size = os.fstat(fd).st_size
            images = []
            for frame in unique:
                offset, size = map(int, self._index[start + int(frame), camera])
                if size < 1 or size > max_window_bytes or offset + size > file_size:
                    raise ValueError("Invalid JPEG payload offset/length")
                payload = os.pread(fd, size, offset)
                if len(payload) != size:
                    raise OSError("Short JPEG payload read")
                with Image.open(io.BytesIO(payload)) as image:
                    if image.format != "JPEG" or image.size != (width, height):
                        raise MediaDecodeError("JPEG payload format/dimensions disagree with metadata")
                    images.append(np.array(image.convert("RGB"), dtype=np.uint8))
        except (OSError, UnidentifiedImageError) as error:
            raise MediaDecodeError(f"JPEG TAR read/decode failed: {view['path']}: {error}") from error
        actual = frames / fps + origin
        return tuple(images[i] for i in inverse), {
            "video_backend": STORAGE,
            "requested_timestamps_sec": ts.tolist(),
            "timestamps_sec": actual.tolist(),
            "frame_indexes": frames.tolist(),
            "jpeg_camera_index": camera,
            "camera_name": view.get("camera_name"),
            "decoded_frames": len(unique),
            "sampled_frames": len(frames),
            "source_fps": fps,
            "width": width,
            "height": height,
            "sampling": "nearest_frame_half_up_valid_end_clamp_v1",
            "end_clamped_frames": int(np.count_nonzero(rounded >= count)),
            "max_timestamp_error_seconds": float(np.max(np.abs(actual - ts))),
            "jpeg_decode_passes_per_unique_frame": 1,
            "jpeg_reencode_in_training": False,
        }
