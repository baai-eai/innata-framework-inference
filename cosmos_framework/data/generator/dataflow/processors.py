# Copied from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/data/generator/dataflow/processors.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Built-in RawItemProcessor implementations."""

from __future__ import annotations

from typing import Any

from cosmos_framework.data.generator.dataflow.base import RawItemProcessor


class IdentityProcessor(RawItemProcessor):
    """No-op processor: the dataset already yields training-ready samples."""

    def process(self, item: Any) -> Any:
        return item
