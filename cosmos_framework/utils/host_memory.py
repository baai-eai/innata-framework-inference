# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host memory attribution without adding shared RSS pages as physical usage."""

from pathlib import Path

import psutil

_GIB = 1024**3


def cgroup_memory(proc_root: Path = Path("/proc")) -> dict[str, float]:
    """Read this process's visible cgroup v2 counters; -1 means unavailable.

    File includes shmem: these counters must not be added together. The host's
    available memory and this cgroup's usage describe different scopes.
    """
    metrics = {f"cpu_cgroup_{key}_gb": -1.0 for key in ("current", "limit", "anon", "file", "shmem")}
    try:
        membership = next(
            line[3:] for line in (proc_root / "self/cgroup").read_text().splitlines() if line.startswith("0::")
        )
        for line in (proc_root / "self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            try:
                relative = Path(membership).relative_to(fields[3])
            except ValueError:
                continue
            root = Path(fields[4]) / relative
            metrics["cpu_cgroup_current_gb"] = int((root / "memory.current").read_text()) / _GIB
            try:
                limit = (root / "memory.max").read_text().strip()
                if limit != "max":
                    metrics["cpu_cgroup_limit_gb"] = int(limit) / _GIB
            except (OSError, ValueError):
                pass
            values = dict(row.split() for row in (root / "memory.stat").read_text().splitlines())
            for key in ("anon", "file", "shmem"):
                if key in values:
                    metrics[f"cpu_cgroup_{key}_gb"] = int(values[key]) / _GIB
            break
    except (OSError, ValueError, StopIteration):
        pass
    return metrics


def process_tree_memory(process: psutil.Process) -> dict[str, float]:
    """Keep historical RSS sums, and add PSS/USS for shared-page attribution.

    Read each live descendant independently so decoder exit does not erase
    the whole measurement. PSS/USS are -1 if any live process denies access.
    PSS apportions pages shared with processes outside this tree as well.
    """
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        children = []
        complete = False
    else:
        complete = True
    rss = [0, 0]
    pss = uss = 0
    count = 0
    for index, item in enumerate([process, *children]):
        try:
            value = item.memory_info().rss
        except psutil.NoSuchProcess:
            continue
        except psutil.Error:
            complete = False
            continue
        rss[int(index > 0)] += value
        count += 1
        try:
            detail = item.memory_full_info()
            pss += detail.pss
            uss += detail.uss
        except psutil.NoSuchProcess:
            complete = False
        except (psutil.Error, AttributeError):
            complete = False
    host = psutil.virtual_memory()
    return {
        "cpu_mem_gb": sum(rss) / _GIB,  # Compatibility: shared pages are counted repeatedly.
        "cpu_trainer_rss_gb": rss[0] / _GIB,
        "cpu_children_rss_gb": rss[1] / _GIB,
        "cpu_tree_pss_gb": pss / _GIB if complete else -1.0,
        "cpu_tree_uss_gb": uss / _GIB if complete else -1.0,
        "cpu_tree_process_count": count,
        "cpu_host_available_gb": host.available / _GIB,
        "cpu_host_used_gb": (host.total - host.available) / _GIB,
        **cgroup_memory(),
    }
