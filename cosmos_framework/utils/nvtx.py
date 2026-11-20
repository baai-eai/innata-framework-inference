# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-overhead, runtime-configurable NVTX annotations.

The training NVTX callback configures this module once per process. Model code can
then describe meaningful phase/module boundaries without carrying profiler config
through every call. When detailed NVTX is disabled, the context managers are no-ops.
"""

from __future__ import annotations

import contextlib
from enum import IntEnum
from typing import Iterator

import torch


class NVTXGranularity(IntEnum):
    """Cumulative detail levels for Cosmos-authored NVTX ranges."""

    COARSE = 0
    PHASE = 1
    MODULE = 2
    OPERATOR = 3

    @classmethod
    def parse(cls, value: str | NVTXGranularity) -> NVTXGranularity:
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except (AttributeError, KeyError) as error:
            choices = ", ".join(level.name.lower() for level in cls)
            raise ValueError(f"Unknown NVTX granularity {value!r}; expected one of: {choices}") from error


_granularity = NVTXGranularity.COARSE


def configure_nvtx(granularity: str | NVTXGranularity) -> NVTXGranularity:
    """Set this process's effective detailed-NVTX granularity."""

    global _granularity
    _granularity = NVTXGranularity.parse(granularity)
    return _granularity


def disable_nvtx() -> None:
    """Disable model-level ranges for this process."""

    configure_nvtx(NVTXGranularity.COARSE)


def nvtx_enabled(required: str | NVTXGranularity = NVTXGranularity.PHASE) -> bool:
    """Whether the configured granularity includes ``required``."""

    return _granularity >= NVTXGranularity.parse(required)


@contextlib.contextmanager
def nvtx_range(
    name: str,
    required: str | NVTXGranularity = NVTXGranularity.PHASE,
) -> Iterator[None]:
    """Emit a ``cosmos.*`` NVTX range when the requested level is active."""

    if not nvtx_enabled(required):
        yield
        return
    torch.cuda.nvtx.range_push(f"cosmos.{name}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_mark(name: str, required: str | NVTXGranularity = NVTXGranularity.PHASE) -> None:
    """Emit a point annotation when the requested level is active."""

    if nvtx_enabled(required):
        torch.cuda.nvtx.mark(f"cosmos.{name}")
