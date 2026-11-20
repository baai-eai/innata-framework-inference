# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Redistribute immutable episode shards, keeping every committed shard cursor.

This changes shard ownership, never the original modulo partition. It is not
an elastic model/optimizer restore or a promise of identical training batches.
"""

from __future__ import annotations

import copy


def validate_virtual_layout(shards, seeds, sources):
    if not isinstance(shards, list) or not isinstance(seeds, list) or len(shards) != sources or len(seeds) != sources:
        raise ValueError("Invalid virtual-shard source count")
    for addresses, rng_seeds in zip(shards, seeds, strict=True):
        if not addresses or len(addresses) != len(rng_seeds):
            raise ValueError("Every source needs virtual shards and matching seeds")
        seen = set()
        for address, seed in zip(addresses, rng_seeds, strict=True):
            if len(address) != 2 or len(seed) != 2:
                raise ValueError("Invalid virtual shard or RNG seed")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in [*address, *seed]):
                raise ValueError("Virtual shard and RNG coordinates must be integers")
            index, total = address
            if not 0 <= index < total or min(seed) < 0 or tuple(address) in seen:
                raise ValueError("Invalid or duplicate virtual shard")
            seen.add(tuple(address))


def redistribute_episode_states(states, target_dp_size):
    """Return rank states for CP=TP=PP=1 episode loaders, without reading media.

    Input must contain every old global rank, in rank order. A current bound
    loader verifies source fingerprints and distributor type when loading.
    """
    if isinstance(target_dp_size, bool) or not isinstance(target_dp_size, int) or target_dp_size < 1 or not states:
        raise ValueError("Need all old rank states and a positive target DP size")
    first = states[0]
    names = list(first["__fingerprints__"])
    if not names:
        raise ValueError("Source fingerprints are required")
    sources, old_size = len(names), len(states)
    entries = [[] for _ in names]
    for rank, state in enumerate(states):
        for key in ["global_id", "__schedule__", "__fingerprints__"]:
            if state[key] != first[key]:
                raise ValueError(f"Old rank {rank} has inconsistent {key}")
        if list(state["__fingerprints__"]) != names:
            raise ValueError("Source order changed between ranks")
        ex = state["__execution__"]
        topology = ex["topology"]
        if ex["version"] != 1 or topology["dp"] != [[rank, old_size]] * sources:
            raise ValueError("Migration requires one global rank per DP coordinate and all rank files")
        if any(len(topology[k]) != sources for k in ["logical_workers", "active_workers"]):
            raise ValueError("Invalid source topology")
        if ex["base_seeds"] is None or len(ex["base_seeds"]) != sources:
            raise ValueError("Checkpoint must contain initialized source base seeds")
        if "virtual_shards" in ex:
            validate_virtual_layout(ex["virtual_shards"], ex["virtual_rng_seeds"], sources)
        for source, name in enumerate(names):
            count = max(1, topology["logical_workers"][source])
            active = topology["active_workers"][source]
            if not active or len(set(active)) != len(active) or any(w < 0 or w >= count for w in active):
                raise ValueError("Invalid active worker IDs")
            cursors = state.get(name, {})
            if set(cursors) - set(active):
                raise ValueError("Cursor references an inactive worker")
            for worker in active:
                if "virtual_shards" in ex:
                    address = ex["virtual_shards"][source][worker]
                    seed = ex["virtual_rng_seeds"][source][worker]
                else:
                    address = [rank * count + worker, old_size * count]
                    seed = [ex["base_seeds"][source], worker]
                cursor = cursors.get(worker, {"epoch": 0, "index": -1})
                if (
                    set(cursor) != {"epoch", "index"}
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in cursor.values())
                    or cursor["epoch"] < 0
                    or cursor["index"] < -1
                ):
                    raise ValueError("Invalid committed shard cursor")
                entries[source].append((address, seed, cursor, ex["rng"].get((source, worker))))
    for group in entries:
        addresses = [tuple(e[0]) for e in group]
        if len(set(addresses)) != len(addresses) or len({a[1] for a in addresses}) != 1:
            raise ValueError("Original virtual shards overlap or use inconsistent partition sizes")
        if len(group) < target_dp_size:
            raise ValueError("Target DP size exceeds active original shards for a source")

    result = []
    for rank in range(target_dp_size):
        state = {key: copy.deepcopy(first[key]) for key in ["global_id", "__schedule__", "__fingerprints__"]}
        shards, seeds, rng, workers = [], [], {}, []
        for source, name in enumerate(names):
            owned = entries[source][rank::target_dp_size]
            shards.append([copy.deepcopy(e[0]) for e in owned])
            seeds.append([copy.deepcopy(e[1]) for e in owned])
            workers.append(len(owned))
            state[name] = {worker: copy.deepcopy(e[2]) for worker, e in enumerate(owned)}
            rng.update({(source, worker): e[3] for worker, e in enumerate(owned) if e[3] is not None})
        state["__execution__"] = {
            "version": 1,
            "topology": {
                "logical_workers": workers,
                "active_workers": [list(range(n)) for n in workers],
                "dp": [[rank, target_dp_size]] * sources,
            },
            "next_workers": [0] * sources,
            "base_seeds": list(first["__execution__"]["base_seeds"]),
            "rng": rng,
            "virtual_shards": shards,
            "virtual_rng_seeds": seeds,
        }
        result.append(state)
    return result
