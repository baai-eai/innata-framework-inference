# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU RNG snapshots for logical data streams, independent of physical workers."""

from __future__ import annotations

import pickle
import random

import numpy as np
import torch
from torch.utils.data._utils.worker import _generate_state


def capture_rng() -> bytes:
    # Use numpy rather than tensor pickling: no shared-memory lifetime/FD and
    # bytewise-comparable snapshots, also safe to carry as CPU batch metadata.
    return pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()), protocol=5)


def restore_rng(state: bytes) -> None:
    python_state, numpy_state, torch_state = pickle.loads(state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch.from_numpy(torch_state.copy()))


def seed_worker_rng(base_seed: int, worker_id: int) -> None:
    """Match PyTorch DataLoader's Python, numpy, and torch CPU worker seeds."""
    random.seed(base_seed + worker_id)
    np.random.seed(_generate_state(base_seed, worker_id))
    # Inline execution must not reseed the training process's CUDA RNG.
    torch.random.default_generator.manual_seed(base_seed + worker_id)
