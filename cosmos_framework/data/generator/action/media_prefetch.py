# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep metadata lookahead with a separate byte budget for selected media.

The cache is disposable. It never owns a sampler cursor, packing state or RNG.
Failed speculative work is retried by the ordinary foreground path when due.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class MediaPrefetchPlan:
    key: tuple[int, int]  # epoch, dataset sample index
    position: tuple[int, int]  # epoch, logical stream position
    reserved_bytes: int
    work: float
    payload: Any = None


class SharedMediaBudget:
    """One output-tensor reservation budget shared by every worker of a rank."""

    def __init__(self, context, limit_bytes):
        if limit_bytes < 1:
            raise ValueError("media byte limit must be positive")
        self.limit_bytes = limit_bytes
        self._values = context.Array("q", [0, 0, 0, 0, 0, 0], lock=True)

    def count(self, name):
        index = {"hits": 2, "misses": 3, "completed": 4, "planned": 5}[name]
        with self._values.get_lock():
            self._values[index] += 1

    def reserve(self, count):
        with self._values.get_lock():
            if self._values[0] + count > self.limit_bytes:
                return False
            self._values[0] += count
            self._values[1] = max(self._values[1], self._values[0])
            return True

    def release(self, count):
        with self._values.get_lock():
            assert 0 <= count <= self._values[0]
            self._values[0] -= count

    @property
    def stats(self):
        with self._values.get_lock():
            return {
                "reserved_bytes": self._values[0],
                "peak_reserved_bytes": self._values[1],
                "limit_bytes": self.limit_bytes,
                "hits": self._values[2],
                "misses": self._values[3],
                "completed": self._values[4],
                "planned": self._values[5],
            }


class SelectiveMediaPrefetch:
    """One producer; arbitrary deep cheap plans, bounded decoded tensor bytes.

    Byte reservations cover running and completed outputs, not decoder working
    memory. Each materializer must provide a conservative output-size estimate.
    Plans stay in original stream order; only plans selected by ``work`` execute.
    """

    def __init__(
        self,
        candidates: Iterable[tuple[int, int, int]],
        plan: Callable,
        materialize: Callable,
        *,
        budget_bytes: int,
        lookahead: int,
        minimum_work: float,
        size: Callable,
        cancel: Callable | None = None,
        shared_budget: SharedMediaBudget | None = None,
    ):
        if budget_bytes < 1 or lookahead < 1 or minimum_work < 0:
            raise ValueError("positive media budget/lookahead and nonnegative minimum work required")
        self.budget_bytes = budget_bytes
        self.lookahead = lookahead
        self.minimum_work = minimum_work
        self._candidates = iter(candidates)
        self._plan = plan
        self._materialize = materialize
        self._size = size
        self._cancel = cancel
        self._shared_budget = shared_budget
        self._condition = threading.Condition()
        self._plans = deque()
        self._entries = {}
        self._skipped = set()
        self._reserved = 0
        self._frontier = (-1, -1)
        self._exhausted = False
        self._closed = False
        self._stats = dict(
            planned=0,
            selected=0,
            completed=0,
            failed=0,
            oversized=0,
            hits=0,
            misses=0,
            wait_seconds=0.0,
            peak_reserved_bytes=0,
        )
        self._thread = threading.Thread(target=self._run, name="action128-media-lookahead", daemon=True)
        self._thread.start()

    @property
    def closed(self):
        return self._closed

    @property
    def stats(self):
        with self._condition:
            return {
                **self._stats,
                "reserved_bytes": self._reserved,
                "plan_count": len(self._plans),
                "cached_or_running": len(self._entries),
                "budget_bytes": self.budget_bytes,
            }

    def _release(self, key):
        entry = self._entries.pop(key)
        self._reserved -= entry["plan"].reserved_bytes
        if self._shared_budget is not None:
            self._shared_budget.release(entry["plan"].reserved_bytes)
        return entry

    def advance(self, epoch, position):
        with self._condition:
            frontier = (int(epoch), int(position))
            if frontier < self._frontier:
                raise ValueError("media lookahead frontier must advance monotonically")
            self._frontier = frontier
            while self._plans and self._plans[0].position < frontier:
                stale = self._plans.popleft()
                self._skipped.discard(stale.key)
                entry = self._entries.get(stale.key)
                if entry is not None and entry["done"]:
                    self._release(stale.key)
            self._condition.notify_all()

    def get(self, key):
        started = time.monotonic()
        with self._condition:
            while key in self._entries and not self._entries[key]["done"] and not self._closed:
                self._condition.wait(timeout=0.1)
            self._stats["wait_seconds"] += time.monotonic() - started
            if self._closed or key not in self._entries:
                self._stats["misses"] += 1
                if self._shared_budget is not None:
                    self._shared_budget.count("misses")
                return None
            entry = self._release(key)
            self._skipped.add(key)
            value = entry["value"]
            self._stats["hits" if value is not None else "misses"] += 1
            if self._shared_budget is not None:
                self._shared_budget.count("hits" if value is not None else "misses")
            self._condition.notify_all()
            return value

    def _run(self):
        try:
            while True:
                # Planning does not hold the consumer lock during metadata I/O.
                with self._condition:
                    if self._closed:
                        return
                    refill = len(self._plans) < self.lookahead and not self._exhausted
                if refill:
                    try:
                        candidate = next(self._candidates)
                    except StopIteration:
                        with self._condition:
                            self._exhausted = True
                        continue
                    try:
                        plan = self._plan(candidate)
                    except Exception:
                        # Prediction failures must not fail training early.
                        plan = None
                    if plan is None:
                        epoch, position, index = candidate
                        # Even unclassified candidates occupy the bounded plan
                        # horizon; never scan an infinite light-only stream.
                        plan = MediaPrefetchPlan((epoch, index), (epoch, position), 0, 0)
                    if plan is not None:
                        with self._condition:
                            if plan.position >= self._frontier:
                                self._plans.append(plan)
                                self._stats["planned"] += 1
                                if self._shared_budget is not None:
                                    self._shared_budget.count("planned")
                    continue
                with self._condition:
                    selected = next(
                        (
                            p
                            for p in self._plans
                            if p.position > self._frontier
                            and p.key not in self._entries
                            and p.key not in self._skipped
                            and p.work >= self.minimum_work
                            and 0 < p.reserved_bytes <= self.budget_bytes - self._reserved
                        ),
                        None,
                    )
                    if selected is None:
                        self._condition.wait(timeout=0.1)
                        continue
                    if self._shared_budget is not None and not self._shared_budget.reserve(selected.reserved_bytes):
                        self._condition.wait(timeout=0.1)
                        continue
                    self._reserved += selected.reserved_bytes
                    self._stats["selected"] += 1
                    self._stats["peak_reserved_bytes"] = max(self._reserved, self._stats["peak_reserved_bytes"])
                    self._entries[selected.key] = {"plan": selected, "done": False, "value": None}
                value = None
                try:
                    value = self._materialize(selected)
                    oversized = self._size(value) > selected.reserved_bytes
                    with self._condition:
                        self._stats["oversized" if oversized else "completed"] += 1
                        if not oversized and self._shared_budget is not None:
                            self._shared_budget.count("completed")
                    if oversized:
                        value = None
                except Exception:
                    with self._condition:
                        self._stats["failed"] += 1
                with self._condition:
                    entry = self._entries[selected.key]
                    if self._closed or selected.position < self._frontier:
                        self._release(selected.key)
                    else:
                        entry.update(done=True, value=value)
                    self._condition.notify_all()
                # Do not retain the last delivered result while waiting for a
                # later job or another byte reservation.
                del entry, value
        finally:
            if self._cancel is not None:
                self._cancel()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        if self._cancel is not None:
            self._cancel()
        self._thread.join(timeout=5)
        with self._condition:
            # A running reservation remains until its producer exits.
            for key in list(self._entries):
                if self._entries[key]["done"]:
                    self._release(key)
            self._plans.clear()
            self._skipped.clear()
