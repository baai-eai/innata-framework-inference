# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded timestamp video decoding behind a killable helper process."""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import select
import signal
import socket
import struct
import subprocess
import sys
import time
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 2 * 1024 * 1024 * 1024
_ACTIVE_DECODERS: weakref.WeakSet = weakref.WeakSet()
_HIGH_FD_LOGGED_PID: int | None = None


class MediaDecodeError(RuntimeError):
    """Ordinary media failure that may use deterministic sample replacement."""

    context: dict[str, Any] | None = None


class MediaDecodeTimeout(MediaDecodeError):
    """The isolated decode helper exceeded its deadline and was terminated."""


class _ContainerLRU:
    def __init__(self, capacity: int, decode_threads: int = 1) -> None:
        self.capacity = int(capacity)
        if isinstance(decode_threads, bool) or not isinstance(decode_threads, int) or not 1 <= decode_threads <= 4:
            raise ValueError("video_decode_threads must be an integer from 1 to 4")
        self.decode_threads = decode_threads
        self._containers: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str):
        import av

        container = self._containers.pop(path, None)
        if container is None:
            container = av.open(path, mode="r")
            # Bound FFmpeg explicitly: automatic sizing creates a CPU-sized
            # pool for every cached stream. Opt-in frame threading helps long
            # GOP files; slice threading cannot parallelize single-slice video.
            for stream in container.streams.video:
                stream.codec_context.thread_count = self.decode_threads
                if self.decode_threads > 1:
                    stream.codec_context.thread_type = "FRAME"
        self._containers[path] = container
        while len(self._containers) > self.capacity:
            _, evicted = self._containers.popitem(last=False)
            evicted.close()
        return container

    def close(self) -> None:
        for container in self._containers.values():
            container.close()
        self._containers.clear()


def _frame_record(frame: Any, *, origin_pts: int, time_base: float) -> tuple[float, int, np.ndarray] | None:
    if frame.pts is None:
        return None
    pts = int(frame.pts)
    timestamp = (pts - origin_pts) * time_base
    if not math.isfinite(timestamp):
        return None
    return timestamp, pts, frame.to_ndarray(format="rgb24")


def _decode_sequential_probe(
    container: Any, stream: Any, *, origin_pts: int, time_base: float, limit: int
) -> list[tuple[float, int, np.ndarray]]:
    container.seek(origin_pts, stream=stream, any_frame=False, backward=True)
    records: list[tuple[float, int, np.ndarray]] = []
    for index, frame in enumerate(container.decode(stream)):
        if index >= limit:
            break
        record = _frame_record(frame, origin_pts=origin_pts, time_base=time_base)
        if record is not None:
            records.append(record)
    return records


def _decode_sequential_indices(
    container: Any,
    stream: Any,
    *,
    indices: Sequence[int],
    origin_pts: int,
    time_base: float,
    limit: int,
) -> list[tuple[float, int, np.ndarray]]:
    """Decode once while materializing only requested frame indexes."""

    container.seek(origin_pts, stream=stream, any_frame=False, backward=True)
    targets = [int(index) for index in indices]
    records: list[tuple[float, int, np.ndarray]] = []
    target_position = 0
    for index, frame in enumerate(container.decode(stream)):
        if index >= limit or target_position >= len(targets):
            break
        if index < targets[target_position]:
            continue
        record = _frame_record(frame, origin_pts=origin_pts, time_base=time_base)
        if record is None:
            continue
        while target_position < len(targets) and targets[target_position] <= index:
            records.append(record)
            target_position += 1
    if records and len(records) < len(targets):
        records.extend([records[-1]] * (len(targets) - len(records)))
    return records


def _decode_seek_groups(
    container: Any,
    stream: Any,
    requested: list[float],
    *,
    origin_pts: int,
    time_base: float,
    limit: int,
    max_window_bytes: int,
) -> list[tuple[float, int, np.ndarray, int]] | None:
    """Share decode prefixes only when seek resolves to the same file packet.

    Packet position (not approximate time/GOP duration) identifies the prefix.
    Each target still observes its own original frame budget and first-after
    stopping rule. No decoded GOP cache is retained across calls.
    """
    if not hasattr(container, "demux"):
        return None

    def seek(target: float) -> None:
        offset = origin_pts + max(int(math.floor(target / time_base)), 0)
        try:
            container.seek(offset, stream=stream, any_frame=False, backward=True)
        except Exception:
            container.seek(origin_pts, stream=stream, any_frame=False, backward=True)

    groups: dict[tuple, list[int]] = {}
    for index, target in enumerate(requested):
        seek(target)
        packet = next(container.demux(stream), None)
        # Missing packet positions cannot prove that two seek prefixes match.
        if packet is None or packet.pos is None or packet.pos < 0 or not packet.size:
            return None
        key = (packet.pos, packet.pts, packet.dts, packet.size)
        groups.setdefault(key, []).append(index)
    if len(groups) == len(requested):
        return None

    results: dict[int, tuple[float, int, np.ndarray, int]] = {}
    total_bytes = 0

    def append(index: int, record: tuple, count: int) -> None:
        nonlocal total_bytes
        timestamp, pts, frame = record
        image = frame.to_ndarray(format="rgb24")
        total_bytes += int(image.nbytes)
        if total_bytes > max_window_bytes:
            raise MediaDecodeError(f"decoded window exceeds max_window_bytes={max_window_bytes}")
        results[index] = (timestamp, pts, image.copy(), count)

    for indexes in groups.values():
        seek(requested[indexes[0]])
        pending = 0
        before = None
        decoded_count = 0
        budget_exhausted = False
        for frame in container.decode(stream):
            if decoded_count >= limit:
                budget_exhausted = True
                break
            decoded_count += 1
            if frame.pts is None:
                continue
            pts = int(frame.pts)
            timestamp = (pts - origin_pts) * time_base
            if not math.isfinite(timestamp):
                continue
            record = (timestamp, pts, frame)
            while pending < len(indexes) and timestamp > requested[indexes[pending]]:
                index = indexes[pending]
                candidates = [record] if before is None else [before, record]
                selected = min(candidates, key=lambda item: abs(item[0] - requested[index]))
                append(index, selected, decoded_count)
                pending += 1
            if pending == len(indexes):
                break
            before = record
        for index in indexes[pending:]:
            if before is None:
                raise MediaDecodeError(f"could not decode a frame near timestamp={requested[index]:.6f}")
            if budget_exhausted and before[0] + time_base / 2 < requested[index]:
                raise MediaDecodeError(f"decode frame budget={limit} exhausted before timestamp={requested[index]:.6f}")
            append(index, before, decoded_count)
    return [results[index] for index in range(len(requested))]


def decode_video_at_timestamps(
    video_path: str,
    *,
    timestamps_sec: Sequence[float] | None = None,
    uniform_sample_count: int | None = None,
    trim_head_frames: int | None = None,
    valid_frames: int | None = None,
    target_fps: float | None = None,
    max_sampled_frames: int = 256,
    max_decode_frames_per_seek: int = 2_048,
    max_frame_pixels: int = 16_777_216,
    max_window_bytes: int = 1_073_741_824,
    output_canvas_width: int | None = None,
    output_canvas_height: int | None = None,
    container_cache: _ContainerLRU | None = None,
    reuse_seek_groups: bool = False,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Decode nearest frames using bounded backward keyframe seeks.

    Declared frame count and duration are hints only.  Actual PTS timestamps are
    returned for every served frame so callers can audit seek drift. Exhausting
    the frame budget before reaching the target neighborhood is a media error.
    """

    path = str(Path(video_path).expanduser().resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    clip_contract = (trim_head_frames, valid_frames, target_fps)
    has_clip_contract = any(value is not None for value in clip_contract)
    if has_clip_contract and not all(value is not None for value in clip_contract):
        raise ValueError("trim_head_frames, valid_frames, and target_fps must be provided together")
    mode_count = int(timestamps_sec is not None) + int(uniform_sample_count is not None) + int(has_clip_contract)
    if mode_count != 1:
        raise ValueError("provide exactly one of timestamps_sec, uniform_sample_count, or the clip trim contract")
    if has_clip_contract:
        if isinstance(trim_head_frames, bool) or not isinstance(trim_head_frames, int) or trim_head_frames < 0:
            raise ValueError("trim_head_frames must be a non-negative integer")
        if isinstance(valid_frames, bool) or not isinstance(valid_frames, int) or valid_frames < 1:
            raise ValueError("valid_frames must be a positive integer")
        if isinstance(target_fps, bool) or not isinstance(target_fps, (int, float)) or target_fps <= 0:
            raise ValueError("target_fps must be a positive number")
        requested_count = valid_frames
        requested = [index / float(target_fps) for index in range(valid_frames)]
    elif uniform_sample_count is not None:
        if isinstance(uniform_sample_count, bool) or uniform_sample_count < 1:
            raise ValueError("uniform_sample_count must be a positive integer")
        requested_count = int(uniform_sample_count)
        requested: list[float] = []
    else:
        requested = [float(value) for value in timestamps_sec or ()]
        if not requested or any(not math.isfinite(value) or value < 0 for value in requested):
            raise ValueError("timestamps must be finite non-negative values")
        if any(right < left for left, right in zip(requested, requested[1:])):
            raise ValueError("timestamps must be sorted")
        requested_count = len(requested)
    if requested_count > max_sampled_frames:
        raise ValueError(f"requested {requested_count} frames exceeds max_sampled_frames={max_sampled_frames}")
    if min(max_decode_frames_per_seek, max_frame_pixels, max_window_bytes) < 1:
        raise ValueError("decode safety limits must be positive")
    output_canvas = (output_canvas_width, output_canvas_height)
    if any(value is not None for value in output_canvas):
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in output_canvas):
            raise ValueError("output_canvas_width and output_canvas_height must be positive integers provided together")

    own_container = container_cache is None
    if own_container:
        import av

        container = av.open(path, mode="r")
    else:
        container = container_cache.get(path)
    try:
        if not container.streams.video:
            raise MediaDecodeError(f"video has no video stream: {path}")
        stream = container.streams.video[0]
        if stream.time_base is None:
            raise MediaDecodeError(f"video has no time base: {path}")
        time_base = float(stream.time_base)
        origin_pts = int(stream.start_time or 0)
        width, height = int(stream.width), int(stream.height)
        if width <= 0 or height <= 0:
            raise MediaDecodeError(f"video reports invalid dimensions {width}x{height}: {path}")
        if width * height > max_frame_pixels:
            # Dimensions come from this media file, not from the caller's
            # request. Reject an unexpectedly large source as a droppable media
            # failure so one outlier cannot terminate distributed training.
            raise MediaDecodeError(
                f"video dimensions {width}x{height} exceed max_frame_pixels={max_frame_pixels}: {path}"
            )
        fps_value = stream.average_rate or getattr(stream, "base_rate", None)
        fps = float(fps_value) if fps_value else None
        declared_frames = int(stream.frames or 0)
        duration = None
        if stream.duration is not None:
            candidate = float(stream.duration) * time_base
            duration = candidate if math.isfinite(candidate) and candidate > 0 else None
        elif container.duration is not None:
            import av

            candidate = float(container.duration) / float(av.time_base)
            duration = candidate if math.isfinite(candidate) and candidate > 0 else None

        decoded_width, decoded_height = width, height
        if output_canvas_width is not None and output_canvas_height is not None:
            scale = min(output_canvas_width / width, output_canvas_height / height, 1.0)
            decoded_width = max(1, int(width * scale + 0.5))
            decoded_height = max(1, int(height * scale + 0.5))

        if has_clip_contract:
            assert trim_head_frames is not None and valid_frames is not None and target_fps is not None
            container.seek(origin_pts, stream=stream, any_frame=False, backward=True)
            targets = [index / float(target_fps) for index in range(valid_frames)]
            frames: list[np.ndarray] = []
            actual: list[float] = []
            pts_values: list[int] = []
            decoded_count = 0
            first_timestamp: float | None = None
            previous: tuple[float, int, Any] | None = None
            target_position = 0
            total_bytes = 0

            def append_frame(record: tuple[float, int, Any]) -> None:
                nonlocal total_bytes
                timestamp, pts, frame = record
                if decoded_width != width or decoded_height != height:
                    frame = frame.reformat(width=decoded_width, height=decoded_height, format="rgb24")
                image = frame.to_ndarray(format="rgb24")
                total_bytes += int(image.nbytes)
                if total_bytes > max_window_bytes:
                    raise MediaDecodeError(f"decoded window exceeds max_window_bytes={max_window_bytes}: {path}")
                frames.append(image.copy())
                assert first_timestamp is not None
                actual.append(timestamp - first_timestamp)
                pts_values.append(pts)

            for decode_index, frame in enumerate(container.decode(stream)):
                if decode_index >= max_decode_frames_per_seek:
                    break
                decoded_count += 1
                if decode_index < trim_head_frames:
                    continue
                if frame.pts is None:
                    raise MediaDecodeError(
                        f"decoded frame {decode_index} after trim has no PTS; cannot resample at {target_fps:g} fps: {path}"
                    )
                timestamp = (int(frame.pts) - origin_pts) * time_base
                if not math.isfinite(timestamp):
                    raise MediaDecodeError(f"decoded frame {decode_index} has an invalid timestamp: {path}")
                current = (timestamp, int(frame.pts), frame)
                if first_timestamp is None:
                    first_timestamp = timestamp
                    previous = current
                    append_frame(current)
                    target_position = 1
                    if target_position >= valid_frames:
                        break
                    continue
                assert previous is not None
                relative_timestamp = timestamp - first_timestamp
                while target_position < valid_frames and targets[target_position] <= relative_timestamp:
                    absolute_target = first_timestamp + targets[target_position]
                    selected = min((previous, current), key=lambda record: abs(record[0] - absolute_target))
                    append_frame(selected)
                    target_position += 1
                previous = current
                if target_position >= valid_frames:
                    break
            if len(frames) != valid_frames:
                raise MediaDecodeError(
                    f"insufficient frames after trim/resample: expected={valid_frames} served={len(frames)} "
                    f"trim_head_frames={trim_head_frames} target_fps={target_fps:g} decoded={decoded_count}: {path}"
                )
            metadata = {
                "video_backend": "trimmed_fps_resample_pyav",
                "trim_head_frames": trim_head_frames,
                "valid_frames": valid_frames,
                "target_fps": float(target_fps),
                "requested_timestamps_sec": targets,
                "timestamps_sec": actual,
                "timestamp_error_seconds": [abs(a - b) for a, b in zip(actual, targets, strict=True)],
                "decoded_frame_pts": pts_values,
                "decoded_frames": decoded_count,
                "width": width,
                "height": height,
                "decoded_width": decoded_width,
                "decoded_height": decoded_height,
                "source_fps": fps,
                "declared_frame_count": declared_frames or None,
                "declared_duration": duration,
            }
            return tuple(frames), metadata

        probe_records: list[tuple[float, int, np.ndarray]] = []
        if uniform_sample_count is not None:
            if declared_frames > 0 and fps is not None and fps > 0:
                last = max(declared_frames - 1, 0)
                indices = (
                    [0]
                    if uniform_sample_count == 1
                    else [math.floor(i * last / (uniform_sample_count - 1)) for i in range(uniform_sample_count)]
                )
                requested = [index / fps for index in indices]
                if declared_frames <= max_decode_frames_per_seek:
                    # Uniform training clips are normally short. Decode them once
                    # and select all requested frames from that pass instead of
                    # performing one backward keyframe seek per target frame.
                    probe_records = _decode_sequential_indices(
                        container,
                        stream,
                        indices=indices,
                        origin_pts=origin_pts,
                        time_base=time_base,
                        limit=max_decode_frames_per_seek,
                    )
                    if not probe_records:
                        raise MediaDecodeError(f"video contains no decodable timestamped frames: {path}")
            elif duration is not None:
                final = max(duration - (1.0 / fps if fps and fps > 0 else 0.0), 0.0)
                requested = (
                    [0.0]
                    if uniform_sample_count == 1
                    else [i * final / (uniform_sample_count - 1) for i in range(uniform_sample_count)]
                )
            else:
                probe_records = _decode_sequential_probe(
                    container,
                    stream,
                    origin_pts=origin_pts,
                    time_base=time_base,
                    limit=max_decode_frames_per_seek,
                )
                if not probe_records:
                    raise MediaDecodeError(f"video contains no decodable timestamped frames: {path}")
                final = probe_records[-1][0]
                requested = (
                    [0.0]
                    if uniform_sample_count == 1
                    else [i * final / (uniform_sample_count - 1) for i in range(uniform_sample_count)]
                )

        grouped = None
        if timestamps_sec is not None and reuse_seek_groups:
            grouped = _decode_seek_groups(
                container,
                stream,
                requested,
                origin_pts=origin_pts,
                time_base=time_base,
                limit=max_decode_frames_per_seek,
                max_window_bytes=max_window_bytes,
            )
        frames: list[np.ndarray] = []
        actual: list[float] = []
        pts_values: list[int] = []
        decoded_counts: list[int] = []
        total_bytes = 0
        for index, target in enumerate(requested):
            if grouped is not None:
                timestamp, pts, image, decoded_count = grouped[index]
                selected = (timestamp, pts, image)
            elif probe_records:
                selected = min(probe_records, key=lambda record: abs(record[0] - target))
                decoded_count = len(probe_records)
            else:
                seek_pts = origin_pts + max(int(math.floor(target / time_base)), 0)
                try:
                    container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
                except Exception:
                    container.seek(origin_pts, stream=stream, any_frame=False, backward=True)
                before = None
                after = None
                decoded_count = 0
                budget_exhausted = False
                for frame in container.decode(stream):
                    if decoded_count >= max_decode_frames_per_seek:
                        budget_exhausted = True
                        break
                    decoded_count += 1
                    # Keep only the two candidate AVFrames. Converting every
                    # frame between the keyframe and target to RGB repeats a
                    # large amount of unused work for timestamp-based windows.
                    if frame.pts is None:
                        continue
                    pts = int(frame.pts)
                    timestamp = (pts - origin_pts) * time_base
                    if not math.isfinite(timestamp):
                        continue
                    record = (timestamp, pts, frame)
                    if record[0] <= target:
                        before = record
                    else:
                        after = record
                        break
                if budget_exhausted and before is not None and before[0] + time_base / 2 < target:
                    raise MediaDecodeError(
                        f"decode frame budget={max_decode_frames_per_seek} exhausted before timestamp={target:.6f}: {path}"
                    )
                candidates = [record for record in (before, after) if record is not None]
                if not candidates:
                    raise MediaDecodeError(f"could not decode a frame near timestamp={target:.6f}: {path}")
                selected = min(candidates, key=lambda record: abs(record[0] - target))
                timestamp, pts, selected_frame = selected
                selected = (timestamp, pts, selected_frame.to_ndarray(format="rgb24"))
            timestamp, pts, image = selected
            total_bytes += int(image.nbytes)
            if total_bytes > max_window_bytes:
                raise MediaDecodeError(f"decoded window exceeds max_window_bytes={max_window_bytes}: {path}")
            frames.append(image if grouped is not None else image.copy())
            actual.append(timestamp)
            pts_values.append(pts)
            decoded_counts.append(decoded_count)
        metadata = {
            "video_backend": "timestamp_seek_pyav",
            "requested_timestamps_sec": requested,
            "timestamps_sec": actual,
            "timestamp_error_seconds": [abs(a - b) for a, b in zip(actual, requested, strict=True)],
            "decoded_frame_pts": pts_values,
            "decoded_frames_per_seek": decoded_counts,
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frame_count": declared_frames or None,
            "declared_duration": duration,
        }
        return tuple(frames), metadata
    finally:
        if own_container:
            container.close()


def _send(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"decode IPC payload is too large: {len(payload)}")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv(sock: socket.socket, *, timeout: float | None = None) -> Any:
    deadline = None if timeout is None else time.monotonic() + timeout
    poller = None
    if deadline is not None:
        # Data workers may hold enough shared tensor FDs that a newly created
        # socket exceeds select()'s FD_SETSIZE. poll() has no such FD limit
        # and, unlike epoll(), does not allocate another file descriptor.
        poller = select.poll()
        poller.register(sock, select.POLLIN)

    def exact(size: int) -> bytes:
        chunks: list[bytes] = []
        while size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                assert poller is not None
                # Round up positive sub-millisecond waits. Keep the original
                # deadline across all header and payload fragments.
                if not poller.poll(math.ceil(remaining * 1000)):
                    raise TimeoutError
                # HUP/ERR may accompany buffered data; recv handles both the
                # remaining bytes and eventual EOF/OSError.
            chunk = sock.recv(size)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    (size,) = _HEADER.unpack(exact(_HEADER.size))
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"unsafe decode IPC message size: {size}")
    return pickle.loads(exact(size))  # noqa: S301 - trusted task-owned helper


def _log_high_fd(sock: socket.socket, helper_pid: int) -> None:
    """Sample FD pressure once per worker, only when a new socket has a high FD."""
    global _HIGH_FD_LOGGED_PID
    fd = sock.fileno()
    pid = os.getpid()
    if fd < 1024 or _HIGH_FD_LOGGED_PID == pid:
        return
    _HIGH_FD_LOGGED_PID = pid
    # Enumeration may observe transient FDs. Never let optional diagnostics
    # interfere with decoding, and do not expose file paths or scan per frame.
    count = maximum = None
    try:
        fds = [int(name) for name in os.listdir("/proc/self/fd") if name.isdecimal()]
        count, maximum = len(fds), max(fds, default=None)
    except OSError:
        pass
    logging.getLogger(__name__).warning(
        "Video decoder high-FD diagnostic: worker_pid=%s helper_pid=%s socket_fd=%s "
        "observed_open_fd_count=%s observed_max_fd=%s",
        pid,
        helper_pid,
        fd,
        count,
        maximum,
    )


class SupervisedVideoDecoder:
    """Lazy per-worker helper which can be killed and rebuilt after timeout."""

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        video_handle_cache_size: int = 8,
        *,
        max_sampled_frames: int = 256,
        max_frame_pixels: int = 16_777_216,
        max_window_bytes: int = 1_073_741_824,
        max_decode_frames_per_seek: int = 2_048,
        reuse_seek_groups: bool = False,
        video_decode_threads: int = 1,
    ) -> None:
        if timeout_seconds <= 0 or video_handle_cache_size < 1:
            raise ValueError("decoder timeout and cache size must be positive")
        if min(max_sampled_frames, max_frame_pixels, max_window_bytes, max_decode_frames_per_seek) < 1:
            raise ValueError("decoder safety limits must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.video_handle_cache_size = int(video_handle_cache_size)
        if (
            isinstance(video_decode_threads, bool)
            or not isinstance(video_decode_threads, int)
            or not 1 <= video_decode_threads <= 4
        ):
            raise ValueError("video_decode_threads must be an integer from 1 to 4")
        self.video_decode_threads = video_decode_threads
        self.decode_options = {
            "max_sampled_frames": int(max_sampled_frames),
            "max_frame_pixels": int(max_frame_pixels),
            "max_window_bytes": int(max_window_bytes),
            "max_decode_frames_per_seek": int(max_decode_frames_per_seek),
            "reuse_seek_groups": bool(reuse_seek_groups),
        }
        self._pid = os.getpid()
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        _ACTIVE_DECODERS.add(self)

    def independent_decoder(self) -> SupervisedVideoDecoder:
        """Create a lazy helper with identical limits and no shared codec/socket state."""
        return SupervisedVideoDecoder(
            timeout_seconds=self.timeout_seconds,
            video_handle_cache_size=self.video_handle_cache_size,
            video_decode_threads=self.video_decode_threads,
            **self.decode_options,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pid"] = None
        state["_process"] = None
        state["_socket"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        _ACTIVE_DECODERS.add(self)

    def _ensure_process_local(self) -> None:
        if self._pid == os.getpid():
            return
        self._pid = os.getpid()
        self._process = None
        self._socket = None

    def _start(self) -> None:
        parent, child = socket.socketpair()
        command = (
            sys.executable,
            "-m",
            "cosmos_framework.data.generator.media_decode",
            "--helper",
            "--socket-fd",
            str(child.fileno()),
            "--cache-size",
            str(self.video_handle_cache_size),
            "--decode-threads",
            str(self.video_decode_threads),
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(child.fileno(),),
            close_fds=True,
            start_new_session=True,
        )
        child.close()
        self._socket = parent
        _log_high_fd(parent, self._process.pid)

    def close(self) -> None:
        sock, process = self._socket, self._process
        self._socket = None
        self._process = None
        if sock is not None:
            sock.close()
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass

    def decode(self, video_path: str, **kwargs: Any) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
        timing = os.environ.get("COSMOS_DATALOADER_TIMING") == "1"
        started = time.monotonic() if timing else 0.0
        self._ensure_process_local()
        if self._process is None:
            self._start()
        ready = time.monotonic() if timing else 0.0
        assert self._process is not None and self._socket is not None
        try:
            decode_kwargs = {**self.decode_options, **kwargs}
            _send(self._socket, {"path": video_path, "kwargs": decode_kwargs})
            response = _recv(self._socket, timeout=self.timeout_seconds)
        except TimeoutError as error:
            pid = self._process.pid
            self.close()
            raise MediaDecodeTimeout(
                f"video decode timed out after {self.timeout_seconds}s; helper_pid={pid}; path={video_path}"
            ) from error
        except (EOFError, BrokenPipeError, OSError) as error:
            self.close()
            raise MediaDecodeError(f"video decode helper failed for {video_path}: {error}") from error
        if response.get("ok"):
            if timing:
                frames, metadata = response["value"]
                phases = dict(metadata.get("timing_seconds", {}))
                request_seconds = time.monotonic() - ready
                phases.update(
                    helper_spawn=ready - started,
                    request=request_seconds,
                    request_overhead=max(
                        0.0, request_seconds - phases.get("decode_with_io", 0.0) - phases.get("resize", 0.0)
                    ),
                )
                return frames, {**metadata, "timing_seconds": phases}
            return response["value"]
        if response.get("type") in {"TypeError", "ValueError"}:
            raise ValueError(f"invalid video decode request for {video_path}: {response.get('message')}")
        raise MediaDecodeError(
            f"video decode failed for {video_path}: {response.get('type')}: {response.get('message')}"
        )

    def decode_jpeg(self, root, view, *, timestamps_sec):
        """Use the same timeout/process boundary for indexed image media."""
        return self.decode(
            str(Path(root) / view["path"]),
            jpeg_root=str(root),
            jpeg_view=dict(view),
            timestamps_sec=timestamps_sec,
        )


def _helper(socket_fd: int, cache_size: int, decode_threads: int = 1) -> None:
    sock = socket.socket(fileno=socket_fd)
    cache = _ContainerLRU(cache_size, decode_threads)
    jpeg_reader = None
    native_reader = None
    timing = os.environ.get("COSMOS_DATALOADER_TIMING") == "1"
    try:
        while True:
            try:
                request = _recv(sock)
            except EOFError:
                return
            try:
                started = time.monotonic() if timing else 0.0
                options = request["kwargs"]
                output_size = options.pop("camera_output_size", None)
                if output_size is not None:
                    if (
                        not isinstance(output_size, (list, tuple))
                        or len(output_size) != 2
                        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in output_size)
                    ):
                        raise ValueError("camera_output_size must be positive integer [height, width]")
                    if output_size[0] * output_size[1] > options.get("max_frame_pixels", 16_777_216):
                        raise ValueError("camera_output_size exceeds max_frame_pixels")
                if "native_view" in options:
                    from cosmos_framework.data.generator.action.native_video import NativeVideoReader

                    if native_reader is None:
                        native_reader = NativeVideoReader(cache_size)
                    value = native_reader.decode(request["path"], **options)
                elif "jpeg_view" in options:
                    from cosmos_framework.data.generator.jpeg_tar import IndexedJpegTarReader

                    root = Path(options["jpeg_root"])
                    if jpeg_reader is None or jpeg_reader.root != root:
                        if jpeg_reader is not None:
                            jpeg_reader.close()
                        jpeg_reader = IndexedJpegTarReader(root, cache_size)
                    value = jpeg_reader.decode(
                        options["jpeg_view"],
                        timestamps_sec=options["timestamps_sec"],
                        **{
                            key: options[key]
                            for key in ("max_sampled_frames", "max_frame_pixels", "max_window_bytes")
                            if key in options
                        },
                    )
                else:
                    value = decode_video_at_timestamps(request["path"], container_cache=cache, **options)
                decoded_at = time.monotonic() if timing else 0.0
                if output_size is not None:
                    frames, metadata = value
                    height, width = output_size
                    if len(frames) * height * width * 3 > options.get("max_window_bytes", 1_073_741_824):
                        raise ValueError("resized camera exceeds max_window_bytes")
                    if all(frame.shape == (height, width, 3) for frame in frames):
                        # Identity bilinear resize of uint8 pixels is exact.
                        value = (frames, {**metadata, "camera_output_size": [height, width]})
                    else:
                        import torch

                        from cosmos_framework.data.generator.camera_resize import copy_resized_camera

                        torch.set_num_threads(1)
                        # THWC storage makes each returned frame contiguous for IPC.
                        pixels = torch.empty((len(frames), height, width, 3), dtype=torch.uint8)
                        copy_resized_camera(frames, pixels.permute(3, 0, 1, 2))
                        value = (tuple(pixels.numpy()), {**metadata, "camera_output_size": [height, width]})
                        del pixels
                    del frames
                if timing:
                    frames, metadata = value
                    # Native/FFmpeg reads occur inside decode. These are wall
                    # times, not isolated codec CPU or storage service times.
                    value = (
                        frames,
                        {
                            **metadata,
                            "timing_seconds": {
                                "decode_with_io": decoded_at - started,
                                "resize": time.monotonic() - decoded_at,
                            },
                        },
                    )
                    del frames, metadata
                _send(sock, {"ok": True, "value": value})
                # IPC has copied the NumPy payload. Idle helpers need not keep
                # the previous camera window alive until the next decode ends.
                del value
            except Exception as error:  # noqa: BLE001 - serialize helper boundary errors
                _send(
                    sock,
                    {"ok": False, "type": type(error).__name__, "message": str(error)},
                )
    finally:
        if native_reader is not None:
            native_reader.close()
        if jpeg_reader is not None:
            jpeg_reader.close()
        cache.close()
        sock.close()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", action="store_true")
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--decode-threads", type=int, default=1)
    args = parser.parse_args()
    if not args.helper:
        raise SystemExit("media_decode is an internal helper module")
    _helper(args.socket_fd, args.cache_size, args.decode_threads)


if __name__ == "__main__":
    _main()
