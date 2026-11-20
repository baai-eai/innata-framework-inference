# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Contracts and bounded materialization for pre-encoded VAE training latents."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_LINEAGE_FIELDS = ("dataset", "vae_name", "vae_tag", "vae_weights_sha256")


class VaeLatentLoadError(RuntimeError):
    """A pre-encoded latent could not be loaded without violating its contract."""


@dataclass(frozen=True)
class VaeLatentRef:
    """Lightweight catalog reference passed through admission before tensor I/O."""

    path: str
    tensor_key: str
    latent_shape: tuple[int, int, int, int]
    target_fps: float
    catalog_id: str
    dataset: str

    @property
    def expected_nbytes(self) -> int:
        # The offline contract fixes storage dtype to bf16 (two bytes per element).
        return math.prod(self.latent_shape) * 2


def parse_vae_latent_ref(
    value: Any,
    *,
    row_id: str,
    dataset: str,
    expected_channels: int,
    max_tensor_bytes: int,
    fps: Any = None,
) -> VaeLatentRef:
    """Validate one ``gen_video.vae_latent`` object without opening its shard."""

    # catalog.py imports the dataflow package (and therefore its collators),
    # while collators import this module for the lightweight reference type.
    # Delay the exception import until row processing to keep module loading
    # acyclic and retain the public CatalogSchemaError exception identity.
    from cosmos_framework.data.generator.catalog import CatalogSchemaError

    if not isinstance(value, dict):
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent must be an object")
    missing = [field for field in ("path", "tensor_key", "latent_shape") if field not in value]
    if missing:
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent is missing required fields {missing}")

    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent.path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent.path must be absolute, got {raw_path!r}")
    if path.suffix != ".safetensors":
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent.path must end in .safetensors")
    path = path.resolve()

    tensor_key = value["tensor_key"]
    if not isinstance(tensor_key, str) or not tensor_key:
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent.tensor_key must be a non-empty string")

    raw_shape = value["latent_shape"]
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 4
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in raw_shape)
    ):
        raise CatalogSchemaError(
            f"catalog row {row_id!r}: vae_latent.latent_shape must be four positive integers [C,T,H,W]"
        )
    latent_shape = tuple(int(dim) for dim in raw_shape)
    if latent_shape[0] != expected_channels:
        raise CatalogSchemaError(
            f"catalog row {row_id!r}: vae_latent C={latent_shape[0]} does not match model state_ch={expected_channels}"
        )

    # Standard catalogs carry fps beside vae_latent; retain old demo catalogs.
    target_fps = fps if fps is not None else value.get("target_fps")
    if (
        isinstance(target_fps, bool)
        or not isinstance(target_fps, (int, float))
        or not math.isfinite(float(target_fps))
        or float(target_fps) <= 0
    ):
        raise CatalogSchemaError(f"catalog row {row_id!r}: vae_latent.target_fps must be a finite positive number")

    ref = VaeLatentRef(
        path=str(path),
        tensor_key=tensor_key,
        latent_shape=latent_shape,
        target_fps=float(target_fps),
        catalog_id=row_id,
        dataset=dataset,
    )
    if ref.expected_nbytes > max_tensor_bytes:
        raise CatalogSchemaError(
            f"catalog row {row_id!r}: declared vae_latent tensor uses {ref.expected_nbytes} bytes, "
            f"limit is {max_tensor_bytes}"
        )
    return ref


@dataclass
class _CachedSafetensorsHandle:
    context: Any
    reader: Any
    keys: frozenset[str]

    def close(self) -> None:
        self.context.__exit__(None, None, None)


class SafetensorsLatentReader:
    """Read one tensor at a time through a process-local bounded handle cache."""

    def __init__(self, *, handle_cache_size: int = 8, max_tensor_bytes: int = 1_073_741_824) -> None:
        if handle_cache_size < 1:
            raise ValueError("handle_cache_size must be positive")
        if max_tensor_bytes < 1:
            raise ValueError("max_tensor_bytes must be positive")
        self.handle_cache_size = int(handle_cache_size)
        self.max_tensor_bytes = int(max_tensor_bytes)
        self._pid: int | None = os.getpid()
        self._handles: OrderedDict[str, _CachedSafetensorsHandle] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_pid"] = None
        state["_handles"] = OrderedDict()
        return state

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        # A forked worker inherits the parent's mmap descriptors. Close that
        # inherited cache before establishing an independent worker-local LRU.
        inherited_handles = self._handles
        self._pid = pid
        self._handles = OrderedDict()
        while inherited_handles:
            _, handle = inherited_handles.popitem(last=False)
            handle.close()

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _handle(self, ref: VaeLatentRef) -> tuple[_CachedSafetensorsHandle, bool]:
        self._ensure_process_local()
        cached = self._handles.pop(ref.path, None)
        cache_hit = cached is not None
        if cached is None:
            path = Path(ref.path)
            if not path.is_file():
                raise VaeLatentLoadError(
                    f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: "
                    f"VAE latent shard does not exist: {ref.path}"
                )
            context = None
            try:
                from safetensors import safe_open

                context = safe_open(ref.path, framework="pt", device="cpu")
                reader = context.__enter__()
                cached = _CachedSafetensorsHandle(context=context, reader=reader, keys=frozenset(reader.keys()))
            except Exception as error:
                if context is not None:
                    context.__exit__(type(error), error, error.__traceback__)
                raise VaeLatentLoadError(
                    f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: "
                    f"failed to open VAE latent shard {ref.path}: {error}"
                ) from error
        self._handles[ref.path] = cached
        while len(self._handles) > self.handle_cache_size:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return cached, cache_hit

    def load(self, ref: VaeLatentRef) -> tuple[torch.Tensor, dict[str, int | float | bool]]:
        """Load and own one CPU bf16 tensor, returning bounded I/O metrics."""

        if ref.expected_nbytes > self.max_tensor_bytes:
            raise VaeLatentLoadError(
                f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: declared VAE latent uses "
                f"{ref.expected_nbytes} bytes, limit is {self.max_tensor_bytes}"
            )
        started = time.perf_counter_ns()
        handle, cache_hit = self._handle(ref)
        if ref.tensor_key not in handle.keys:
            raise VaeLatentLoadError(
                f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: tensor key {ref.tensor_key!r} "
                f"is missing from {ref.path}"
            )
        try:
            tensor = handle.reader.get_tensor(ref.tensor_key)
        except Exception as error:
            raise VaeLatentLoadError(
                f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: failed to read tensor "
                f"{ref.tensor_key!r} from {ref.path}: {error}"
            ) from error
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != ref.latent_shape:
            raise VaeLatentLoadError(
                f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: tensor {ref.tensor_key!r} "
                f"shape mismatch, expected {ref.latent_shape}, got {actual_shape}"
            )
        if tensor.dtype != torch.bfloat16:
            raise VaeLatentLoadError(
                f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: tensor {ref.tensor_key!r} "
                f"dtype mismatch, expected torch.bfloat16, got {tensor.dtype}"
            )
        actual_nbytes = tensor.numel() * tensor.element_size()
        if actual_nbytes > self.max_tensor_bytes:
            raise VaeLatentLoadError(
                f"catalog row {ref.catalog_id!r} dataset={ref.dataset!r}: tensor {ref.tensor_key!r} uses "
                f"{actual_nbytes} bytes, limit is {self.max_tensor_bytes}"
            )
        # Own the storage independently of the mmap handle before the sample leaves this worker.
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return tensor, {
            "io_ms": elapsed_ms,
            "nbytes": actual_nbytes,
            "cache_hit": cache_hit,
        }


def validate_offline_vae_manifests(
    dataset_specs: list[dict[str, Any]],
    *,
    expected_vae_name: str,
    expected_vae_weights_sha256: str | None,
) -> list[dict[str, str]]:
    """Fail before worker construction unless every offline dataset has compatible lineage."""

    expected_sha = (expected_vae_weights_sha256 or "").strip().lower()
    if expected_sha and _SHA256_PATTERN.fullmatch(expected_sha) is None:
        raise ValueError("expected_vae_weights_sha256 must be a 64-character lowercase hexadecimal SHA256")

    lineages: list[dict[str, str]] = []
    for index, spec in enumerate(dataset_specs):
        manifest_value = spec.get("vae_manifest_path")
        if not isinstance(manifest_value, str) or not manifest_value.strip():
            raise ValueError(f"offline VAE dataset_specs[{index}] requires vae_manifest_path")
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file():
            raise ValueError(f"offline VAE manifest does not exist: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"failed to read offline VAE manifest {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise ValueError(f"offline VAE manifest must contain a JSON object: {manifest_path}")
        missing = [
            field
            for field in _REQUIRED_LINEAGE_FIELDS
            if not isinstance(manifest.get(field), str) or not manifest[field]
        ]
        if missing:
            raise ValueError(f"offline VAE manifest {manifest_path} is missing lineage fields {missing}")
        sha = str(manifest["vae_weights_sha256"]).lower()
        if _SHA256_PATTERN.fullmatch(sha) is None:
            raise ValueError(f"offline VAE manifest {manifest_path} has an invalid vae_weights_sha256")
        if manifest["vae_name"] != expected_vae_name:
            raise ValueError(
                f"offline VAE manifest {manifest_path} has vae_name={manifest['vae_name']!r}, "
                f"expected {expected_vae_name!r}"
            )
        if expected_sha and sha != expected_sha:
            raise ValueError(
                f"offline VAE manifest {manifest_path} has vae_weights_sha256={sha}, expected {expected_sha}"
            )
        lineages.append(
            {
                "dataset": str(manifest["dataset"]),
                "vae_name": str(manifest["vae_name"]),
                "vae_tag": str(manifest["vae_tag"]),
                "vae_weights_sha256": sha,
                "manifest_path": str(manifest_path),
            }
        )

    compatible_fields = ("vae_name", "vae_tag", "vae_weights_sha256")
    if lineages:
        baseline = lineages[0]
        for lineage in lineages[1:]:
            mismatched = [field for field in compatible_fields if lineage[field] != baseline[field]]
            if mismatched:
                raise ValueError(
                    "offline VAE dataset manifests are incompatible for fields "
                    f"{mismatched}: {baseline['manifest_path']} vs {lineage['manifest_path']}"
                )
    return lineages


__all__ = [
    "SafetensorsLatentReader",
    "VaeLatentLoadError",
    "VaeLatentRef",
    "parse_vae_latent_ref",
    "validate_offline_vae_manifests",
]
