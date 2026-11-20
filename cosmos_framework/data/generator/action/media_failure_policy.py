# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY Action media quarantine, NOT a substitute for upstream data cleaning.

Skipping changes the effective sample distribution. Keep exclusions immutable
within a run (including resume), and retain logs for the eventual cleaning job.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.media_decode import MediaDecodeError

_LOG = logging.getLogger(__name__)


class ActionMediaFailurePolicy:
    """Worker-local accounting; no collectives or shared mutable blacklist."""

    def __init__(self, config: dict[str, Any], dataset: str, dp_rank: int, worker_id: int):
        self.enabled = bool(config.get("action_skip_media_errors", False))
        self.limit = int(config.get("action_max_consecutive_skips", 100))
        if self.limit < 1:
            raise ValueError("action_max_consecutive_skips must be positive")
        self.dataset = dataset
        self.dp_rank = dp_rank
        self.worker_id = worker_id
        self.consecutive = self.skipped = self.accepted = 0
        self.exclusions: set[tuple[str, int, int]] = set()
        self.log_path: Path | None = None
        exclusion_path = config.get("action_exclusion_jsonl")
        if exclusion_path and not self.enabled:
            raise ValueError("action_exclusion_jsonl requires action_skip_media_errors=true")
        if not self.enabled:
            return
        log_dir = config.get("action_failure_log_dir")
        output_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT") or os.environ.get("OUTPUT_ROOT") or "outputs"
        directory = Path(log_dir) if log_dir else Path(output_root) / "media_failures"
        directory.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(dataset.encode()).hexdigest()[:12]
        rank = os.environ.get("RANK", str(dp_rank))
        self.log_path = directory / f"action-{tag}-rank{rank}-worker{worker_id}-pid{os.getpid()}.jsonl"
        if exclusion_path:
            with Path(exclusion_path).open() as source:
                for line in source:
                    if line.strip():
                        self.exclusions.add(self._key(json.loads(line)))
        _LOG.warning(
            "TEMPORARY Action media skipping enabled: dataset=%s, limit=%d, evidence=%s; "
            "this is NOT data repair and changes the sample distribution",
            dataset,
            self.limit,
            self.log_path,
        )

    @staticmethod
    def _key(context: dict[str, Any]) -> tuple[str, int, int]:
        return (str(Path(context["root"]).resolve()), int(context["episode_index"]), int(context["window_index"]))

    def excludes(self, context: dict[str, Any]) -> bool:
        return bool(self.exclusions) and self._key(context) in self.exclusions

    def success(self) -> None:
        self.accepted += 1
        self.consecutive = 0

    def drop(self, context: dict[str, Any], reason: str, error: MediaDecodeError | None) -> None:
        self.skipped += 1
        self.consecutive += 1
        record = {
            **context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset,
            "dp_rank": self.dp_rank,
            "worker_id": self.worker_id,
            "reason": reason,
            "error": str(error) if error is not None else None,
            "accepted": self.accepted,
            "skipped": self.skipped,
            "skip_fraction": self.skipped / (self.skipped + self.accepted),
            "consecutive_skips": self.consecutive,
        }
        assert self.log_path is not None
        with self.log_path.open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        _LOG.warning("Action window dropped: %s", json.dumps(record, ensure_ascii=False))
        if self.consecutive >= self.limit:
            raise RuntimeError(
                f"Action source {self.dataset!r} reached {self.limit} consecutive skipped windows; "
                f"refusing to loop indefinitely. See {self.log_path}"
            ) from error
