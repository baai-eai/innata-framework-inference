# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Configure PyTorch's native compiler cache directory.

Cosmos does not serialize, copy, or publish compiler artifacts. Once enabled,
PyTorch reads and writes its native cache tree directly through
``TORCHINDUCTOR_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from cosmos_framework.utils import log

DEFAULT_COMPILE_CACHE_PATH = ".compile_cache"
REPO_ROOT = Path(__file__).resolve().parents[2]


class CompileCacheError(RuntimeError):
    """Raised when the configured PyTorch compiler cache cannot be enabled."""


def _get_path(obj: Any, *path: str, default: Any = None) -> Any:
    current = obj
    for key in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            try:
                current = getattr(current, key)
            except (AttributeError, KeyError):
                return default
    return current


def _expanded_path(raw_value: str, *, relative_to: Path) -> Path:
    if not raw_value.strip():
        raise CompileCacheError("trainer.compile_cache.path must not be empty")
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_value)))
    if not expanded.is_absolute():
        expanded = relative_to / expanded
    resolved = expanded.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise CompileCacheError(f"refusing broad trainer.compile_cache.path: {resolved}")
    return resolved


def resolve_compile_cache_path(raw_value: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a configured cache path, anchoring relative paths at the repo root."""

    return _expanded_path(raw_value, relative_to=repo_root.resolve(strict=False))


def configure_compile_cache(
    config: Any,
    *,
    deterministic: bool = False,
    repo_root: Path = REPO_ROOT,
) -> Path | None:
    """Set ``TORCHINDUCTOR_CACHE_DIR`` before model construction.

    Disabled configuration is deliberately a no-op, including when the caller
    has already configured PyTorch through the environment.
    """

    cache_config = _get_path(config, "trainer", "compile_cache", default=None)
    if cache_config is None or not bool(_get_path(cache_config, "enabled", default=False)):
        return None

    if deterministic:
        raise CompileCacheError("compile cache cannot be enabled together with --deterministic")
    if not bool(_get_path(config, "model", "config", "compile", "enabled", default=False)):
        raise CompileCacheError("compile cache requires model.config.compile.enabled=true")

    raw_path = str(_get_path(cache_config, "path", default=DEFAULT_COMPILE_CACHE_PATH))
    cache_path = resolve_compile_cache_path(raw_path, repo_root=repo_root)
    if os.environ.get("COSMOS_COMPILE_CACHE_PER_RANK") == "1":
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is None or not local_rank.isdecimal():
            raise CompileCacheError("COSMOS_COMPILE_CACHE_PER_RANK requires a numeric LOCAL_RANK")
        cache_path = cache_path / f"rank_{local_rank}"

    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CompileCacheError(f"cannot create trainer.compile_cache.path {cache_path}: {exc}") from exc
    if not cache_path.is_dir():
        raise CompileCacheError(f"trainer.compile_cache.path is not a directory: {cache_path}")

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    log.info(f"PyTorch compile cache directory: {cache_path}", rank0_only=False)
    return cache_path


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory.

    This remains public for tokenizer ``.pt2`` package integrity checks and is
    unrelated to compiler-cache eligibility.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_COMPILE_CACHE_PATH",
    "REPO_ROOT",
    "CompileCacheError",
    "configure_compile_cache",
    "resolve_compile_cache_path",
    "sha256_file",
]
