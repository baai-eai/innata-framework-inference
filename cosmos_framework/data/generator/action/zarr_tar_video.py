# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read indexed PsiBot Zarr v3 RGB chunks directly from the source TAR."""

import json
import os
from pathlib import Path

import numpy as np
import zstandard

from cosmos_framework.data.generator.media_decode import MediaDecodeError


class ZarrTarVideoDecoder:
    def __init__(self):
        self.last_path = None
        self.last_chunk = None
        self.last_pixels = None
        self.index = None
        self.fd = None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None
        self.last_path = self.last_chunk = self.last_pixels = self.index = None

    def __del__(self):
        if getattr(self, "fd", None) is not None:
            self.close()

    def decode(self, video_path: str, *, timestamps_sec):
        path = Path(video_path)
        if self.last_path != path:
            self.close()
            sidecar = Path(str(path) + ".head_rgb_index.json")
            try:
                index = json.loads(sidecar.read_text())
                resolved = path.resolve()
                st = resolved.stat()
                if index["schema_version"] != "psibot-zarr-tar-head-rgb-index-v1" or st.st_size != index["tar_size"]:
                    raise ValueError("stale video index")
                self.fd = os.open(resolved, os.O_RDONLY)
                self.index = index
                self.last_path = path
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise MediaDecodeError(f"Cannot open indexed PsiBot TAR: {path}: {error}") from error
        shape = self.index["shape"]
        chunk_frames = self.index["chunk_frames"]
        result = []
        for time_sec in timestamps_sec:
            frame = min(max(int(round(time_sec * self.index["fps"])), 0), shape[0] - 1)
            chunk, within = divmod(frame, chunk_frames)
            if self.last_chunk != chunk:
                try:
                    offset, size = self.index["chunks"][str(chunk)]
                    compressed = os.pread(self.fd, size, offset)
                    data = zstandard.ZstdDecompressor().decompress(compressed)
                    count = min(chunk_frames, shape[0] - chunk * chunk_frames)
                    # Zarr edge chunks retain the full storage shape; only their
                    # in-bounds frames belong to the logical video array.
                    pixels = np.frombuffer(data, dtype=np.uint8).reshape(chunk_frames, shape[1], shape[2], shape[3])
                    self.last_pixels = pixels[:count]
                    self.last_chunk = chunk
                except (OSError, ValueError, KeyError, zstandard.ZstdError) as error:
                    raise MediaDecodeError(f"Cannot read PsiBot TAR frame {frame}: {error}") from error
            result.append(self.last_pixels[within].copy())
        return tuple(result), {
            "path": str(path),
            "storage": "zarr_tar_head_rgb_v1",
            "fps": self.index["fps"],
            "frames": len(result),
        }
