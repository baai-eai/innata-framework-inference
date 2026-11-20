# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded MCAP/H.264 and Zarr-v3 JPEG media, run inside the decode helper."""

import io
import json
import os
import struct
import tarfile
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.media_decode import MediaDecodeError


class NativeVideoReader:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.handles = OrderedDict()
        self.metadata = OrderedDict()
        self.shards = OrderedDict()
        self.mcap_bounds = OrderedDict()

    def close(self):
        for stream, _ in self.handles.values():
            stream.close()
        self.handles.clear()
        self.metadata.clear()
        self.shards.clear()
        self.mcap_bounds.clear()

    def handle(self, path, kind, topic=""):
        stat = os.stat(path)
        key = (str(path), stat.st_size, stat.st_mtime_ns, kind, topic)
        if key not in self.handles:
            stream = open(path, "rb")
            try:
                if kind == "mcap":
                    from mcap.reader import make_reader
                    from mcap_protobuf.decoder import DecoderFactory

                    reader = make_reader(stream, decoder_factories=[DecoderFactory()])
                    summary = reader.get_summary()
                    if summary is None or not summary.chunk_indexes:
                        raise MediaDecodeError("MCAP requires indexed chunks for bounded random reads")
                else:
                    with tarfile.open(fileobj=stream, mode="r:") as archive:
                        reader = {}
                        for member in archive:
                            if len(reader) >= 1_000_000:
                                raise MediaDecodeError("oversized Zarr TAR member index")
                            if member.isfile():
                                reader[member.name.lstrip("./")] = (member.offset_data, member.size)
                self.handles[key] = (stream, reader)
            except Exception:
                stream.close()
                raise
        self.handles.move_to_end(key)
        while len(self.handles) > self.capacity:
            _, (old, _) = self.handles.popitem(last=False)
            old.close()
        return self.handles[key]

    def read(self, path, prefix, member, offset=0, size=None):
        key = "/".join(x for x in (prefix.strip("/"), member) if x)
        if str(path).endswith(".tar"):
            stream, index = self.handle(path, "tar")
            if key not in index and not prefix:
                roots = [n[:-10] for n in index if n.endswith("/zarr.json") and n.count("/") == 1]
                if len(roots) == 1:
                    key = roots[0] + "/" + key
            base, total = index[key]
            fd = stream.fileno()
            own = False
        else:
            fd = os.open(Path(path) / key, os.O_RDONLY)
            base, total, own = 0, os.fstat(fd).st_size, True
        try:
            offset = total + offset if offset < 0 else offset
            size = total - offset if size is None else size
            if offset < 0 or size < 0 or offset + size > total or size > 128 * 1024**2:
                raise MediaDecodeError("Zarr read exceeds member or safety limit")
            data = os.pread(fd, size, base + offset)
            if len(data) != size:
                raise MediaDecodeError("truncated Zarr member")
            return data
        finally:
            if own:
                os.close(fd)

    def zarr_frame(self, path, view, index, max_frame_pixels):
        import zstandard
        from PIL import Image

        archive = view.get("archive") or {}
        prefix = archive.get("member_prefix", view.get("member_prefix", ""))
        stream = view["stream_key"]
        meta_key = (path, prefix, stream)
        # Metadata is small; bound the cache and key it by the descriptor file identity.
        if not str(path).endswith(".tar"):
            stat = (Path(path) / prefix / stream / "zarr.json").stat()
        else:
            stat = Path(path).stat()
        meta_key += (stat.st_size, stat.st_mtime_ns)
        if meta_key not in self.metadata:
            self.metadata[meta_key] = json.loads(self.read(path, prefix, stream + "/zarr.json"))
        self.metadata.move_to_end(meta_key)
        while len(self.metadata) > self.capacity:
            self.metadata.popitem(last=False)
        meta = self.metadata[meta_key]
        if meta.get("zarr_format") != 3 or len(meta["shape"]) != 1:
            raise ValueError("native JPEG reader requires a one-dimensional Zarr-v3 array")
        if index < 0 or index >= meta["shape"][0]:
            raise MediaDecodeError("Zarr frame outside source array")
        outer = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        codecs = meta["codecs"]
        names = [c["name"] for c in codecs]
        if names == ["vlen-bytes", "zstd"] and outer == 1:
            compressed = self.read(path, prefix, f"{stream}/c/{index}")
        elif names == ["sharding_indexed"]:
            config = codecs[0]["configuration"]
            if config["chunk_shape"] != [1] or [c["name"] for c in config["codecs"]] != ["vlen-bytes", "zstd"]:
                raise ValueError("unsupported Zarr JPEG shard codecs")
            if config.get("index_location", "end") != "end" or [c["name"] for c in config["index_codecs"]] != [
                "bytes",
                "crc32c",
            ]:
                raise ValueError("unsupported Zarr shard index")
            shard = f"{stream}/c/{index // outer}"
            identity_path = Path(path) if str(path).endswith(".tar") else Path(path) / prefix / shard
            identity = identity_path.stat()
            key = (path, prefix, shard, identity.st_size, identity.st_mtime_ns)
            if key not in self.shards:
                import google_crc32c

                raw = self.read(path, prefix, shard, -(outer * 16 + 4), outer * 16 + 4)
                if google_crc32c.value(raw[:-4]) != struct.unpack("<I", raw[-4:])[0]:
                    raise MediaDecodeError("Zarr shard index checksum mismatch")
                self.shards[key] = np.frombuffer(raw[:-4], dtype="<u8").reshape(outer, 2)
            self.shards.move_to_end(key)
            while len(self.shards) > self.capacity:
                self.shards.popitem(last=False)
            offset, length = map(int, self.shards[key][index % outer])
            if offset == 2**64 - 1:
                raise MediaDecodeError("unwritten Zarr image")
            compressed = self.read(path, prefix, shard, offset, length)
        else:
            raise ValueError(f"unsupported Zarr JPEG codec chain: {names}")
        content_size = zstandard.frame_content_size(compressed)
        if content_size > 128 * 1024**2:
            raise MediaDecodeError("Zarr JPEG decompressed payload exceeds safety limit")
        payload = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=128 * 1024**2)
        count, size = struct.unpack_from("<II", payload)
        if count != 1 or size != len(payload) - 8:
            raise MediaDecodeError("invalid vlen JPEG payload")
        with Image.open(io.BytesIO(payload[8:])) as image:
            if image.width * image.height > max_frame_pixels:
                raise MediaDecodeError("Zarr image exceeds max_frame_pixels")
            return np.asarray(image.convert("RGB"))

    def mcap_frames(self, path, view, times, limits):
        import av

        if not isinstance(view["time_origin_ns"], int) or isinstance(view["time_origin_ns"], bool):
            raise ValueError("MCAP time_origin_ns must be an integer, not a rounded epoch float")
        origin = view["time_origin_ns"]
        targets = [origin + round(float(t) * 1_000_000_000) for t in times]
        _, reader = self.handle(path, "mcap", view["mcap_topic"])
        fps = float(view["source_fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("MCAP source_fps must be finite and positive")
        # A catalog view can claim a longer interval than its actual topic.
        # Retrying GOP pre-roll cannot produce frames beyond the last message,
        # and repeated retries can exhaust the whole batch's consumer deadline.
        # Read actual indexed message bounds without H.264/protobuf decoding;
        # key by file identity so replacing a source cannot leave stale bounds.
        stat = os.stat(path)
        bounds_key = (str(path), stat.st_size, stat.st_mtime_ns, view["mcap_topic"])
        if bounds_key not in self.mcap_bounds:
            first = next(reader.iter_messages(topics=[view["mcap_topic"]]), None)
            last = next(reader.iter_messages(topics=[view["mcap_topic"]], reverse=True), None)
            if first is None or last is None:
                raise MediaDecodeError(f"MCAP topic has no messages: {view['mcap_topic']!r}")
            self.mcap_bounds[bounds_key] = (first[2].log_time, last[2].log_time)
        self.mcap_bounds.move_to_end(bounds_key)
        while len(self.mcap_bounds) > self.capacity:
            self.mcap_bounds.popitem(last=False)
        first_time, last_time = self.mcap_bounds[bounds_key]
        tolerance = 2e9 / fps
        if first_time - targets[0] > tolerance or targets[-1] - last_time > tolerance:
            raise MediaDecodeError(
                f"requested timestamps outside MCAP topic timeline: topic={view['mcap_topic']!r}, "
                f"requested_ns=[{targets[0]}, {targets[-1]}], "
                f"available_ns=[{first_time}, {last_time}], tolerance_ns={tolerance}"
            )
        for preroll in (3, 12, 60):
            codec = av.CodecContext.create("h264", "r")
            codec.thread_count = 1
            previous, results, position = None, [], 0
            for count, (_, _, message, decoded) in enumerate(
                reader.iter_decoded_messages(
                    topics=[view["mcap_topic"]],
                    start_time=max(0, targets[0] - preroll * 1_000_000_000),
                    end_time=targets[-1] + 3_000_000_000,
                )
            ):
                if count >= limits["max_decode_frames_per_seek"]:
                    break
                packet = av.Packet(bytes(decoded.data))
                packet.pts = message.log_time
                packet.time_base = Fraction(1, 1_000_000_000)
                try:
                    frames = codec.decode(packet)
                except av.error.InvalidDataError as error:
                    if previous is not None:
                        raise MediaDecodeError("corrupt MCAP H.264 packet after decoding began") from error
                    continue  # pre-roll can start before an IDR
                for frame in frames:
                    if frame.width * frame.height > limits["max_frame_pixels"]:
                        raise MediaDecodeError("MCAP frame exceeds max_frame_pixels")
                    actual = (
                        round(frame.pts * frame.time_base * 1_000_000_000)
                        if frame.pts is not None
                        else message.log_time
                    )
                    current = (actual, frame)
                    while position < len(targets) and actual >= targets[position]:
                        chosen = min(
                            [current] + ([previous] if previous else []), key=lambda r: abs(r[0] - targets[position])
                        )
                        if abs(chosen[0] - targets[position]) > 2e9 / fps:
                            break
                        pixels = chosen[1].to_ndarray(format="rgb24")
                        if sum(r[1].nbytes for r in results) + pixels.nbytes > limits["max_window_bytes"]:
                            raise MediaDecodeError("MCAP window exceeds max_window_bytes")
                        results.append((chosen[0], pixels))
                        position += 1
                    previous = current
                    if position == len(targets):
                        return (
                            tuple(r[1] for r in results),
                            [(r[0] - origin) / 1e9 for r in results],
                            [r[0] for r in results],
                        )
            else:
                # A published tail timestamp can round a few nanoseconds past
                # the last message. At natural exhaustion, use that real frame
                # under the same nearest-frame tolerance as a bracketed query.
                # The for/else deliberately excludes packet-budget exhaustion.
                while previous is not None and position < len(targets):
                    if abs(previous[0] - targets[position]) > 2e9 / fps:
                        break
                    pixels = previous[1].to_ndarray(format="rgb24")
                    if sum(r[1].nbytes for r in results) + pixels.nbytes > limits["max_window_bytes"]:
                        raise MediaDecodeError("MCAP window exceeds max_window_bytes")
                    results.append((previous[0], pixels))
                    position += 1
                if position == len(targets):
                    return (
                        tuple(r[1] for r in results),
                        [(r[0] - origin) / 1e9 for r in results],
                        [r[0] for r in results],
                    )
        raise MediaDecodeError("MCAP frames unavailable within bounded GOP pre-roll")

    def decode(self, path, *, native_view, timestamps_sec, source_frame_indexes=None, **limits):
        times = np.asarray(timestamps_sec, dtype=np.float64)
        if (
            not len(times)
            or len(times) > limits["max_sampled_frames"]
            or not np.isfinite(times).all()
            or (times < 0).any()
            or (np.diff(times) < 0).any()
        ):
            raise ValueError("invalid native video query timestamps/count")
        view = native_view
        fmt = view.get("container_format") or view.get("format") or ""
        if fmt == "mcap" or path.endswith(".mcap"):
            frames, actual, pts = self.mcap_frames(path, view, times, limits)
        else:
            if source_frame_indexes is None or len(source_frame_indexes) != len(times):
                raise ValueError("Zarr requires explicit native frame indexes")
            frames, total = [], 0
            for index in source_frame_indexes:
                pixels = self.zarr_frame(path, view, int(index), limits["max_frame_pixels"])
                total += pixels.nbytes
                if total > limits["max_window_bytes"]:
                    raise MediaDecodeError("Zarr window exceeds max_window_bytes")
                frames.append(pixels)
            frames, actual, pts = tuple(frames), times.tolist(), None
        return frames, {
            "path": path,
            "storage": fmt,
            "topic": view.get("mcap_topic"),
            "stream_key": view.get("stream_key"),
            "requested_timestamps_sec": times.tolist(),
            "actual_timestamps_sec": actual,
            "pts": pts,
            "pts_time_base_seconds": 1e-9 if pts is not None else None,
            "source_frame_indexes": source_frame_indexes,
            "timestamp_error_seconds": np.abs(times - actual).tolist(),
        }
