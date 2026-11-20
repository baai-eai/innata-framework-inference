# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate publisher-owned robot-limit transforms and apply their clip policy."""

import numpy as np
import torch

from cosmos_framework.data.generator.catalog import CatalogSchemaError

PHYSICAL_POLICY = "robot-limit-affine-no-clip-v1"


def validate_physical_document(document, *, field):
    """Physical bounds are a robot specification, never fitted sample quantiles."""
    scope = {
        "data_role": "robot_specification",
        "training_eligible": False,
        "row_mask": "not_applicable",
        "slot_mask": "action_dim_mask" if field == "action" else "observation.state_mask",
        "control_mask_used_for_fit": False,
    }
    fit_scope = dict(document.get("fit_scope") or {})
    fit_scope.setdefault("control_mask_used_for_fit", False)
    application = {
        "performed_by": "training dataloader",
        "parquet_values_pre_scaled": False,
        "do_not_apply_additional_mean_std": True,
        "authoritative_key": "action_domain",
        "field": field,
    }
    if (
        document.get("schema_version") != "action-normalization-v1"
        or document.get("policy_version") != PHYSICAL_POLICY
        or document.get("action_storage") != "raw_physical_or_source_units"
        or fit_scope != scope
        or document.get("application") != application
        or not document.get("domains")
    ):
        raise CatalogSchemaError(f"invalid physical-limit {field} publication contract")
    for domain, record in document["domains"].items():
        transform = record.get("transform", {})
        statistics = record.get("statistics", {})
        arrays = [transform.get(name) for name in ("kind", "normalization_mask", "scale", "bias")]
        bounds = [statistics.get(name) for name in ("lower_bound", "upper_bound")]
        if record.get("slot_count") != 128 or any(not isinstance(a, list) or len(a) != 128 for a in [*arrays, *bounds]):
            raise CatalogSchemaError(f"invalid physical-limit transform width: {domain}")
        kinds, mask, scale, bias = arrays
        mask = np.asarray(mask, dtype=np.bool_)
        scale, bias = np.asarray(scale, dtype=np.float64), np.asarray(bias, dtype=np.float64)
        lower, upper = (np.asarray(array, dtype=np.float64) for array in bounds)
        if (
            any(kind not in {"robot_limit_affine", "identity", "inactive"} for kind in kinds)
            or not np.array_equal(mask, [kind == "robot_limit_affine" for kind in kinds])
            or not all(np.isfinite(array).all() for array in (scale, bias, lower, upper))
            or (upper[mask] <= lower[mask]).any()
            or not np.allclose(scale[mask], 2 / (upper[mask] - lower[mask]), rtol=1e-12, atol=1e-12)
            or not np.allclose(bias[mask], -1 - lower[mask] * scale[mask], rtol=1e-12, atol=1e-12)
            or not np.array_equal(scale[~mask], np.ones((~mask).sum()))
            or not np.array_equal(bias[~mask], np.zeros((~mask).sum()))
            or statistics.get("source") != "robot_urdf_limits"
            or statistics.get("active_slot_count") != int(mask.sum())
            or transform.get("clip_policy") != "none"
            or transform.get("clip_min") is not None
            or transform.get("clip_max") is not None
            or transform.get("nominal_min") != -1.0
            or transform.get("nominal_max") != 1.0
        ):
            raise CatalogSchemaError(f"invalid physical-limit affine/no-clip transform: {domain}")


def apply_affine(values, transform):
    """Transform in place; legacy publications retain their original clipping."""
    mask = torch.tensor(transform["normalization_mask"], dtype=torch.bool, device=values.device)
    scale = torch.tensor(transform["scale"], dtype=torch.float32, device=values.device)
    bias = torch.tensor(transform["bias"], dtype=torch.float32, device=values.device)
    normalized = values[:, mask] * scale[mask] + bias[mask]
    if transform.get("clip_policy") != "none":
        normalized = normalized.clamp(float(transform.get("clip_min", -1)), float(transform.get("clip_max", 1)))
    values[:, mask] = normalized
    return values
