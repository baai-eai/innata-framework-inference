# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal callable base used by retained Action and inference transforms."""

from typing import Any


class Augmentor:
    def __init__(self, input_keys: list, output_keys: list | None = None, args: dict | None = None) -> None:
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("Augmentor not implemented")
