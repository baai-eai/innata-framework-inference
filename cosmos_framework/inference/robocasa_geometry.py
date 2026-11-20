# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa Control12 v3 offline/online geometry; no training dependencies.

Matches the publisher's robocasa-control12-state128-two-domains-base9-v3
adapter. Quaternions are xyzw; EEF orientation is the legacy hand body frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = "robocasa-control12-state128-two-domains-base9-v3"
ADAPTER_SHA256 = "7cfaca4b3ac14a8447a33958d033fb1b598afc2cb223b436e4b47efb89f09845"
EMBODIMENT = "panda_omron_mobile__parallel_gripper"
ACTION_KEY = f"native_control12/{EMBODIMENT}"
STATE_KEY = f"state_eef20_base9/{EMBODIMENT}"


def state128(native_state, native_mask=None):
    x = np.asarray(native_state, dtype=np.float64)
    if x.shape[-1:] != (16,):
        raise ValueError("expected source State16")
    shape = x.shape[:-1]
    x = x.reshape(-1, 16)
    m = np.ones_like(x, bool) if native_mask is None else np.asarray(native_mask, bool).reshape(x.shape)
    m = m & np.isfinite(x)
    out = np.zeros((len(x), 128), np.float32)
    valid = np.zeros_like(out, bool)
    rb = np.broadcast_to(np.eye(3), (len(x), 3, 3)).copy()
    qb = m[:, 3:7].all(1) & (abs(np.linalg.norm(x[:, 3:7], axis=1) - 1) <= 1e-3)
    rb[qb] = Rotation.from_quat(x[qb, 3:7]).as_matrix()
    bp = m[:, :3].all(1)
    out[bp, 96:99] = x[bp, :3]
    valid[bp, 96:99] = True
    out[qb, 99:105] = np.concatenate([rb[qb, :, 0], rb[qb, :, 1]], axis=1)
    valid[qb, 99:105] = True
    eq = m[:, 10:14].all(1) & (abs(np.linalg.norm(x[:, 10:14], axis=1) - 1) <= 1e-3)
    pose = bp & qb & m[:, 7:10].all(1) & eq
    rr = Rotation.from_quat(x[pose, 10:14]).as_matrix()
    rsource = rb[pose] @ rr
    p = x[pose, :3] + np.einsum("nij,nj->ni", rb[pose], x[pose, 7:10])
    p += np.einsum("nij,j->ni", rsource, np.array([0.0, 0.0, -0.0036]))
    r = rsource @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    out[pose, :9] = np.concatenate([p, r[:, :, 0], r[:, :, 1]], axis=1)
    valid[pose, :9] = True
    gap = x[:, 14] - x[:, 15] - 0.001
    gm = m[:, 14:16].all(1) & (gap >= -0.001) & (gap <= 0.080 + 1e-12)
    out[gm, 9] = np.clip(gap[gm], 0.0, 0.079) / 0.079
    valid[gm, 9] = True
    return out.reshape(*shape, 128), valid.reshape(*shape, 128)


def source_state_from_observation(obs, prefix="robot0_"):
    keys = ["base_pos", "base_quat", "base_to_eef_pos", "base_to_eef_quat", "gripper_qpos"]
    widths = [3, 4, 3, 4, 2]
    parts = [np.asarray(obs[prefix + k], dtype=np.float64) for k in keys]
    if any(p.shape != (n,) for p, n in zip(parts, widths, strict=True)):
        raise ValueError("observation fields do not match PandaOmron source layout")
    return np.concatenate(parts)
