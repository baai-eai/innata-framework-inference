# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One current State128 observation as optional clean pretraining context."""

import json

import numpy as np
import torch

from cosmos_framework.data.generator.catalog import CatalogSchemaError


def load_state_transforms(root):
    """Read fixed, independently fitted State transforms; never fit at runtime."""
    info = json.loads((root / "meta/info.json").read_text())
    reference = info.get("state_normalization", {})
    expected = {
        "path": "meta/state_normalization.json",
        "field": "observation.state",
        "mask": "observation.state_mask",
        "authoritative_key": "action_domain",
    }
    if (
        reference.get("path") != expected["path"]
        or reference.get("authoritative_key", "action_domain") != "action_domain"
    ):
        raise CatalogSchemaError(f"missing published State128 normalization reference: {root}")
    storage = info.get("state_storage")
    # Subset publications declare the same inline columns on the normalization
    # reference instead of repeating them in state_storage. Never infer names.
    if storage is None:
        declared = reference.get("field") == expected["field"] and reference.get("mask") == expected["mask"]
    else:
        declared = storage.get("value") == expected["field"] and storage.get("loader_mask") == expected["mask"]
    if not declared:
        raise CatalogSchemaError(f"missing inline State128 storage contract: {root}")
    document = json.loads((root / expected["path"]).read_text())
    application = document.get("application", {})
    if (
        document.get("schema_version") != "action-normalization-v1"
        or document.get("policy_version")
        not in {
            "domain-slot-q01-q99-v1",
            "domain-slot-shared-state-action-q01-q99-v1",
            "domain-slot-separate-state-action-q01-q99-v1",
            "eef-local-delta-state-action-separate-q01-q99-20261029-v1",
            "robot-limit-affine-no-clip-v1",
        }
        or application.get("parquet_values_pre_scaled", False) is not False
        or application.get("authoritative_key", "action_domain") != "action_domain"
        or application.get("field", "observation.state") != "observation.state"
        or not document.get("domains")
    ):
        raise CatalogSchemaError(f"unsupported State128 normalization contract: {root}")
    if document.get("policy_version") == "robot-limit-affine-no-clip-v1":
        from cosmos_framework.data.generator.action.physical_normalization import validate_physical_document

        validate_physical_document(document, field="observation.state")
        return document
    for domain, record in document["domains"].items():
        t = record["transform"]
        kinds = t["kind"]
        if kinds == "domain_slot_affine_q01_q99_to_minus1_plus1":
            kinds = ["robust_quantile" if m else "identity" for m in t["normalization_mask"]]
        if (
            record["slot_count"] != 128
            or any(len(t[k]) != 128 for k in ("scale", "bias", "normalization_mask"))
            or len(kinds) != 128
        ):
            raise CatalogSchemaError(f"invalid State128 transform width: {domain}")
        mask = np.asarray(t["normalization_mask"], dtype=bool)
        scale, bias = np.asarray(t["scale"]), np.asarray(t["bias"])
        if (
            not np.isfinite(scale).all()
            or not np.isfinite(bias).all()
            or (scale[mask] <= 0).any()
            or t["clip_min"] != -1.0
            or t["clip_max"] != 1.0
            or any(k not in {"robust_quantile", "identity", "inactive"} for k in kinds)
            or not np.array_equal(mask, [k == "robust_quantile" for k in kinds])
        ):
            raise CatalogSchemaError(f"invalid State128 transform: {domain}")
    return document


def current_state(table, *, start, transform):
    """Select the latest observation at/before t0; a masked row yields no token.

    Never substitute an older valid state for a masked current observation.
    Float32 boundary tolerance is only for timestamp representation error.
    """
    timestamps = table["timestamp"].to_numpy()
    tolerance = max(1e-8, abs(start) * np.finfo(np.float32).eps)
    eligible = np.flatnonzero(timestamps <= start + tolerance)
    if not len(eligible):
        return torch.empty(0, 128), torch.empty(0, 128, dtype=torch.bool), torch.empty(0, dtype=torch.float64)
    index = int(eligible[-1])
    values = torch.tensor(table["observation.state"][index].as_py(), dtype=torch.float32).reshape(1, 128)
    mask = torch.tensor(table["observation.state_mask"][index].as_py(), dtype=torch.bool).reshape(1, 128)
    if not torch.isfinite(values[mask]).all():
        raise CatalogSchemaError("nonfinite valid State128 observation")
    if not mask.any():
        return values[:0], mask[:0], torch.empty(0, dtype=torch.float64)
    from cosmos_framework.data.generator.action.physical_normalization import apply_affine

    apply_affine(values, transform)
    values.masked_fill_(~mask, 0)
    times = torch.tensor([min(0.0, float(timestamps[index]) - start)], dtype=torch.float64)
    return values, mask, times
