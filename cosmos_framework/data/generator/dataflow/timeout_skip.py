# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Worker-local microbatch deadlines and an audit trail of discarded candidates.

SIGALRM interrupts Python and interruptible IO, including supervised media waits.
Native calls that do not return to Python remain covered by worker_timeout's
fail-fast watchdog. No deadline is installed in the trainer process.
"""

from __future__ import annotations

import json
import os
import signal
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from cosmos_framework.utils import log

_observer = ContextVar("microbatch_candidate_observer", default=None)


def observe_candidate(dataset, index, epoch, position):
    """Called before media IO so even the interrupted sample has an identity."""
    observer = _observer.get()
    if observer is not None:
        observer.candidates.append((dataset, int(index), int(epoch), int(position)))


class MicrobatchDeadlineExceeded(BaseException):
    """Bypass sample-level Exception retry/drop handlers."""


class MicrobatchTimeoutPolicy:
    def __init__(self, seconds, max_skips, directory, run_id, physical_worker):
        self.seconds = seconds
        self.max_skips = max_skips
        self.directory = Path(directory)
        self.run_id = run_id
        self.physical_worker = physical_worker
        self.pending = {}
        self.candidates = []
        self.log_path = None

    @contextmanager
    def deadline(self, key):
        self.candidates = self.pending.setdefault(key, [])
        token = _observer.set(self)
        previous = signal.getsignal(signal.SIGALRM)

        def expired(*_):
            raise MicrobatchDeadlineExceeded()

        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            _observer.reset(token)
            raise RuntimeError("Microbatch deadline requires an unused ITIMER_REAL timer")
        signal.signal(signal.SIGALRM, expired)
        started = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            _observer.reset(token)
            self.elapsed = time.monotonic() - started

    def accepted(self, key, batch):
        # Retain lookahead candidates until they are actually delivered. They
        # too are discarded if the generator is abandoned on a later timeout.
        cursor = (int(batch["sample_epoch"].max()), int(batch["sample_index"].max()))
        self.pending[key] = [row for row in self.pending[key] if (row[2], row[3]) > cursor]

    def discard(self, key, *, global_id, dataset_name, dp_rank, attempt):
        rows = self.pending.pop(key)
        if not rows:
            raise RuntimeError("Microbatch timed out before reaching a resumable sample cursor")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.log_path is None:
            rank = os.environ.get("RANK", str(dp_rank))
            self.log_path = self.directory / (
                f"{self.run_id}-rank{rank}-worker{self.physical_worker}-pid{os.getpid()}.jsonl"
            )
            # Never truncate records, even if PID/rank are reused after resume.
            self.log_path.touch(exist_ok=False)
        skip_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a") as output:
            for dataset, index, epoch, position in rows:
                identity = dataset.media_sample_identity(index) if hasattr(dataset, "media_sample_identity") else {}
                record = {
                    **identity,
                    "reason": "microbatch_timeout",
                    "timestamp": timestamp,
                    "run_id": self.run_id,
                    "skip_id": skip_id,
                    "global_microbatch_id": global_id,
                    "dataset": dataset_name,
                    "rank": int(os.environ.get("RANK", dp_rank)),
                    "dp_rank": dp_rank,
                    "logical_worker": key[1],
                    "physical_worker": self.physical_worker,
                    "sample_index": index,
                    "epoch": epoch,
                    "stream_position": position,
                    "timeout_sec": self.seconds,
                    "elapsed_sec": self.elapsed,
                    "attempt": attempt,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
        log.warning(
            f"Microbatch {global_id} timed out after {self.elapsed:.2f}s; discarded {len(rows)} candidates "
            f"from {dataset_name!r}, retry={attempt}/{self.max_skips or 'unlimited'}; records={self.log_path}",
            rank0_only=False,
        )
        return max((row[2], row[3]) for row in rows)
