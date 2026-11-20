# Copied from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/model/tokenizer/utils/__init__.py
# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# -----------------------------------------------------------------------------

"""Tokenizer utility helpers."""

# Keep this initializer dependency-light. Launcher scripts import
# ``projects.cosmos3.tokenizer.utils.paths`` before training dependencies such
# as Torch are installed, so utility symbols should be imported from their
# concrete submodules instead of re-exported here.
