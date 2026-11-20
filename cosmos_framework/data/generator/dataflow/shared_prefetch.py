# Copyright (c) 2026, BAAI. All rights reserved.
# Developed for Innata on the NVIDIA Cosmos Framework; original notices follow.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""One bounded prefetch window for the globally scheduled mixture of batches.

Logical shards own processing/packing state. Physical CPU processes execute
requests for those shards without any independent per-source prefetch.
"""

from __future__ import annotations

import faulthandler
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.multiprocessing as mp
from torch.utils.data._utils import worker as worker_module
from torch.utils.data._utils.pin_memory import pin_memory

from cosmos_framework.data.generator.dataflow.loader import JointCosmosDataLoader, _DataflowIterableDataset
from cosmos_framework.data.generator.dataflow.rng import capture_rng, restore_rng, seed_worker_rng
from cosmos_framework.data.generator.dataflow.timeout_skip import MicrobatchDeadlineExceeded, MicrobatchTimeoutPolicy
from cosmos_framework.utils import log

LOADER_META_KEY = "_cosmos_loader"
EXECUTION_STATE_KEY = "__execution__"
_TIMING_KEY = "_cosmos_prefetch_timing"


@dataclass
class CosmosDataStream:
    """Synchronous source specification, deliberately not a torch DataLoader.

    num_workers preserves the legacy logical topology (0 means one shard).
    Only SharedPrefetchDataLoader creates physical workers.
    """

    dataset: _DataflowIterableDataset
    num_workers: int = 0


class _StreamRunner:
    def __init__(self, streams, base_seeds, resume_rng, timeout_policy=None, names=None):
        self.streams = streams
        self.base_seeds = base_seeds
        self.rng = dict(resume_rng)
        self.iterators = {}
        self.timeout_policy = timeout_policy
        self.names = names
        self.media_lookaheads = []

    def prepare_media_lookaheads(self, assigned_streams, media_budget=None):
        for source, worker in assigned_streams:
            stream = self.streams[source]
            ds = stream.dataset
            prepare = getattr(ds._distributor, "prepare_media_lookahead", None)
            if prepare is not None:
                cache = prepare(
                    ds._dp_rank, ds._dp_world_size, worker, max(1, stream.num_workers), shared_budget=media_budget
                )
                if cache is not None:
                    self.media_lookaheads.append(cache)

    def next_batch(self, source: int, worker: int, global_id=None):
        if self.timeout_policy is None:
            return self._with_worker_info(source, worker)
        key = (source, worker)
        policy = self.timeout_policy
        attempt = 0
        while True:
            attempt += 1
            try:
                with policy.deadline(key):
                    batch = self._with_worker_info(source, worker)
                policy.accepted(key, batch)
                return batch
            except MicrobatchDeadlineExceeded:
                ds = self.streams[source].dataset
                epoch, position = policy.discard(
                    key,
                    global_id=global_id,
                    dataset_name=self.names[source],
                    dp_rank=ds._dp_rank,
                    attempt=attempt,
                )
                iterator = self.iterators.pop(key, None)
                if iterator is not None:
                    iterator.close()
                self.rng[key] = capture_rng()
                _close_decoders()
                if policy.max_skips and attempt > policy.max_skips:
                    raise RuntimeError(
                        f"Microbatch {global_id} exceeded {policy.max_skips} timeout replacements; "
                        f"see {policy.log_path}"
                    ) from None
                # Rebuild only this logical stream past every discarded sample,
                # including the interrupted candidate and buffered lookahead.
                name = ds._distributor._name
                prefix = f"COSMOS_DL_STATE_{name}_" if name else "COSMOS_DL_STATE_"
                os.environ[f"{prefix}WORKER_{worker}_EPOCH"] = str(epoch)
                os.environ[f"{prefix}WORKER_{worker}_INDEX"] = str(position)

    def _with_worker_info(self, source: int, worker: int):
        previous_info = worker_module._worker_info
        stream = self.streams[source]
        base_seed, seed_worker = self._seed_coordinates(source, worker)
        worker_module._worker_info = worker_module.WorkerInfo(
            id=worker,
            num_workers=max(1, stream.num_workers),
            seed=base_seed + seed_worker,
            dataset=stream.dataset,
        )
        try:
            return self._next_batch(source, worker)
        finally:
            worker_module._worker_info = previous_info

    def _next_batch(self, source: int, worker: int):
        key = (source, worker)
        if key not in self.iterators:
            seed_worker_rng(*self._seed_coordinates(source, worker))
            if key in self.rng:
                restore_rng(self.rng[key])
            stream = self.streams[source]
            self.iterators[key] = stream.dataset.iter_stream(
                worker, max(1, stream.num_workers), capture_resume_rng=True
            )
        else:
            restore_rng(self.rng[key])
        try:
            batch = next(self.iterators[key])
        except StopIteration as error:
            raise RuntimeError(
                f"Data stream source={source} logical_worker={worker} exhausted; "
                "check empty shards or finite sources (catalog streams must be infinite)"
            ) from error
        self.rng[key] = capture_rng()
        return batch

    def _seed_coordinates(self, source, worker):
        seeds = getattr(self.streams[source].dataset, "_virtual_rng_seeds", None)
        return seeds[worker] if seeds is not None else (self.base_seeds[source], worker)

    def close(self):
        for iterator in self.iterators.values():
            iterator.close()
        self.iterators.clear()
        for cache in self.media_lookaheads:
            cache.close()
        self.media_lookaheads.clear()


def _close_decoders():
    # Helpers are subprocess.Popen children (not multiprocessing children).
    # Explicitly close them on orderly shutdown; a closed socket also makes a
    # supervised helper exit when its owning worker is killed.
    from cosmos_framework.data.generator.media_decode import _ACTIVE_DECODERS

    for obj in list(_ACTIVE_DECODERS):
        if obj._pid == os.getpid():
            obj.close()


class _ResultQueue:
    """Serialize results in the worker, within the existing global credits.

    multiprocessing.Queue serializes on a feeder thread after put() returns.
    A reducer failure there drops the result without reaching the worker's
    exception handler. SimpleQueue completes serialization and the pipe write
    before the worker can mark this task submitted or start another read.
    """

    def __init__(self, context):
        self._queue = context.SimpleQueue()

    def put(self, result):
        # A diagnostic SIGUSR1 can interrupt posix_fallocate in PyTorch's
        # tensor-sharing reducer. Affected builds misread errno instead of the
        # return value, turning EINTR into "shared memory ... Success (0)".
        # Defer only our diagnostic signal until serialization/transfer ends;
        # reads and decoding remain interruptible for stack collection.
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        try:
            self._queue.put(result)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def get(self, timeout):
        # There is one consumer: either the main thread or the pin thread.
        # SimpleQueue.get has no timeout; poll its pipe before receiving.
        if not self._queue._reader.poll(timeout):
            raise queue.Empty
        return self._queue.get()

    def cancel_join_thread(self):
        # SimpleQueue has no feeder thread to join.
        pass

    def close(self):
        self._queue.close()


def _worker_main(
    requests,
    results,
    streams,
    base_seeds,
    resume_rng,
    physical,
    worker_state,
    timeout_config,
    names,
    collect_timing=False,
    assigned_streams=(),
    media_budget=None,
):
    # Physical workers already parallelize tokenization. A Rayon pool per
    # worker creates thousands of idle threads and is unsafe across forks.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Forked CPU workers must not run the trainer's inherited SIGTERM
    # checkpoint handler when the executor terminates an unfinished task.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    worker_state[2 * physical] = -1
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    policy = MicrobatchTimeoutPolicy(*timeout_config, physical) if timeout_config is not None else None
    runner = _StreamRunner(streams, base_seeds, resume_rng, policy, names)
    try:
        runner.prepare_media_lookaheads(assigned_streams, media_budget)
        while True:
            task = requests.get()
            if task is None:
                return
            global_id, source, worker, next_worker = task
            worker_state[2 * physical] = global_id
            try:
                started = time.monotonic_ns() if collect_timing else None
                batch = runner.next_batch(source, worker, global_id)
                if collect_timing:
                    batch[_TIMING_KEY] = {
                        "worker_started_ns": started,
                        "worker_processed_ns": time.monotonic_ns(),
                    }
                results.put((global_id, batch, None))
                del batch
            except BaseException:
                results.put((global_id, None, traceback.format_exc()))
                # Remain alive so torch's FD-based tensor sharing stays valid
                # until the parent drains results or terminates this worker.
            finally:
                worker_state[2 * physical + 1] = global_id
                worker_state[2 * physical] = -1
    finally:
        runner.close()
        _close_decoders()


def _receive_results(results, ready, done, device, stats):
    """Drain worker IPC ahead; optionally pin within the existing task credits.

    Keep this target independent of the loader so the thread cannot prevent
    loader destruction. Like torch's pin thread, retain no batch between calls.
    """

    def put(result):
        while not done.is_set():
            try:
                ready.put(result, timeout=0.1)
                return
            except queue.Full:
                continue

    if device is not None:
        try:
            torch.set_num_threads(1)
            torch.cuda.set_device(device)
        except BaseException:
            put((None, None, "Pin memory thread initialization failed:\n" + traceback.format_exc()))
            return

    def step():
        try:
            global_id, batch, error = results.get(timeout=0.1)
        except queue.Empty:
            return
        if error is None and _TIMING_KEY in batch:
            # get() has completed unpickling/tensor FD reconstruction. This is
            # transfer latency including blocked sends and receiver scheduling,
            # rather than a pure pipe bandwidth measurement.
            batch[_TIMING_KEY]["received_ns"] = time.monotonic_ns()
        if done.is_set():
            return
        if error is None and device is not None:
            try:
                # Checkpoint RNG must remain ordinary CPU memory. It is not
                # model input and must not take part in asynchronous H2D.
                resume_rng = batch.pop("_cosmos_resume_rng", None)
                started = time.perf_counter()
                batch = pin_memory(batch)
                stats["seconds"] += time.perf_counter() - started
                stats["batches"] += 1
                if resume_rng is not None:
                    batch["_cosmos_resume_rng"] = resume_rng
            except BaseException:
                batch, error = None, "Pin memory failed:\n" + traceback.format_exc()
        put((global_id, batch, error))

    while not done.is_set():
        try:
            step()
        except Exception:
            put((None, None, "Worker result transfer failed:\n" + traceback.format_exc()))
            return


class SharedPrefetchDataLoader(JointCosmosDataLoader):
    """Ordered joint loader with a global, worker-independent prefetch budget.

    ``prefetch_factor=2`` admits at most two not-yet-delivered batch tasks
    across the entire mixture, including running and completed tasks. Physical
    workers do not multiply this limit. With num_workers=0 execution is inline.

    ``COSMOS_DATALOADER_TIMING=1`` adds worker queue, preparation, transfer and
    ready-queue wall times to ``_cosmos_loader['timing']`` for multiprocessing
    batches. This diagnostic metadata is not part of committed loader state.
    """

    def __init__(
        self,
        dataloaders: dict,
        seed: int = 42,
        selection_mode: str = "exact_periodic",
        *,
        num_workers: int = 0,
        prefetch_factor: int = 1,
        pin_memory: bool = False,
        pin_memory_in_background: bool = True,
        receive_in_background: bool = True,
        worker_assignment: list[int] | None = None,
        worker_timeout: float = 300.0,
        microbatch_timeout_sec: float = 0.0,
        microbatch_timeout_max_skips: int = 2,
        microbatch_skip_log_dir: str | None = None,
    ):
        super().__init__(dataloaders, seed, selection_mode)
        if num_workers < 0 or isinstance(prefetch_factor, bool) or prefetch_factor < 1 or worker_timeout <= 0:
            raise ValueError("num_workers >= 0, global prefetch_factor >= 1, and worker_timeout > 0 required")
        if microbatch_timeout_sec < 0 or microbatch_timeout_max_skips < 0:
            raise ValueError("microbatch_timeout_sec >= 0 and microbatch_timeout_max_skips >= 0 required")
        self._timeout_config = None
        if microbatch_timeout_sec:
            from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor, ParquetBlockDistributor
            from cosmos_framework.data.generator.dataflow.batchers import (
                SequentialPackingBatcher,
                SequentialTokenBudgetBatcher,
                SimpleBatcher,
            )
            from cosmos_framework.data.generator.dataflow.distributors import MapDistributor

            # Unlimited replacement is safe during prefetch: the consumer's
            # watchdog still bounds actual trainer waiting, independently of
            # how many attempts completed before this task reached the head.
            if not num_workers or microbatch_timeout_sec * (microbatch_timeout_max_skips + 1) >= worker_timeout:
                raise ValueError(
                    "Microbatch skipping requires workers and timeout_sec * (max_skips + 1) < worker_timeout"
                )
            for stream in self._loaders:
                ds = stream.dataset
                if (
                    not ds._enable_map_style_resume
                    or not isinstance(
                        ds._distributor, (MapDistributor, EpisodeBlockDistributor, ParquetBlockDistributor)
                    )
                    or not isinstance(
                        ds._batcher, (SimpleBatcher, SequentialPackingBatcher, SequentialTokenBudgetBatcher)
                    )
                ):
                    raise ValueError(
                        "Microbatch skipping requires resumable map/episode/parquet streams and sequential batching"
                    )
            root = os.environ.get("OUTPUT_ROOT") or os.environ.get("IMAGINAIRE_OUTPUT_ROOT") or "outputs"
            directory = microbatch_skip_log_dir or os.path.join(root, "logs", "skipped_samples")
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:12]
            self._timeout_config = (microbatch_timeout_sec, microbatch_timeout_max_skips, directory, run_id)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_in_background = pin_memory_in_background
        self.receive_in_background = receive_in_background
        self._collect_timing = os.environ.get("COSMOS_DATALOADER_TIMING") == "1"
        self._enqueued_ns = {}
        self._pin_in_background = False
        self.worker_timeout = worker_timeout
        self._active = []
        self._affinity = {}
        slot = 0
        for source, stream in enumerate(self._loaders):
            if not isinstance(stream, CosmosDataStream):
                raise TypeError("shared prefetch requires synchronous CosmosDataStream specifications")
            ds = stream.dataset
            active = ds._distributor.active_worker_ids(ds._dp_rank, ds._dp_world_size, max(1, stream.num_workers))
            if not active:
                raise ValueError(
                    f"Dataset {self._names[source]!r} has no active logical shards on DP rank {ds._dp_rank}"
                )
            self._active.append(active)
            for worker in active:
                self._affinity[source, worker] = slot % max(1, num_workers)
                slot += 1
        self._worker_assignment = list(worker_assignment) if worker_assignment is not None else None
        self._apply_worker_assignment()
        self._next_worker = [0] * len(self._names)
        self._committed_worker = list(self._next_worker)
        self._base_seeds = None
        self._resume_rng = {}
        self._virtual_shards = None
        self._virtual_rng_seeds = None
        self._committed_rng = {}
        self._committed_global_id = 0
        self._next_schedule_id = 0
        self._tasks = deque()
        self._ready = {}
        self._requests = []
        self._processes = []
        self._worker_state = None
        self._results = None
        self._received_results = None
        self._result_thread = None
        self._results_done = threading.Event()
        self._pin_stats = {"batches": 0, "seconds": 0.0}
        self._receive_seconds = 0.0
        self._inline_pin_seconds = 0.0
        self._runner = None
        self._started = False
        self._closed = False
        self._max_observed_inflight = 0
        self._media_budget = None
        log.info(
            f"SharedPrefetchDataLoader: global prefetch={prefetch_factor}, physical workers={num_workers}, "
            f"logical shards={[max(1, stream.num_workers) for stream in self._loaders]}"
        )

    @property
    def committed_global_id(self):
        return self._committed_global_id

    @property
    def prefetch_stats(self):
        stats = {
            "inflight": len(self._tasks),
            "ready": len(self._ready) + (self._received_results.qsize() if self._received_results is not None else 0),
            "limit": self.prefetch_factor if self.num_workers else 0,
            "max_observed_inflight": self._max_observed_inflight,
            "delivered_uncommitted": self._global_id - self._committed_global_id,
            "background_receive": self._result_thread is not None,
            "background_pin_memory": self._pin_in_background,
            "pin_memory_batches": self._pin_stats["batches"],
            "pin_memory_seconds": self._pin_stats["seconds"] + self._inline_pin_seconds,
            "receive_seconds": self._receive_seconds,
        }
        if self._media_budget is not None:
            stats["media_prefetch"] = self._media_budget.stats
        return stats

    def _topology(self):
        return {
            "logical_workers": [stream.num_workers for stream in self._loaders],
            "active_workers": self._active,
            "dp": [[stream.dataset._dp_rank, stream.dataset._dp_world_size] for stream in self._loaders],
        }

    def execution_state_dict(self):
        state = {
            "version": 1,
            "topology": self._topology(),
            "next_workers": list(self._committed_worker),
            "base_seeds": self._base_seeds,
            "rng": dict(self._committed_rng),
        }
        if self._virtual_shards is not None:
            state.update(virtual_shards=self._virtual_shards, virtual_rng_seeds=self._virtual_rng_seeds)
        return state

    def _apply_worker_assignment(self):
        assignment = self._worker_assignment
        if assignment is None:
            return
        if len(assignment) != len(self._affinity) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= self.num_workers
            for value in assignment
        ):
            raise ValueError("worker_assignment needs one valid physical worker id per active logical shard")
        self._affinity = dict(zip(self._affinity, assignment, strict=True))

    def _restore_virtual_shards(self, state):
        from cosmos_framework.data.generator.catalog import EpisodeBlockDistributor
        from cosmos_framework.data.generator.dataflow.reshard import validate_virtual_layout

        shards, seeds = state["virtual_shards"], state["virtual_rng_seeds"]
        validate_virtual_layout(shards, seeds, len(self._loaders))
        if any(not isinstance(s.dataset._distributor, EpisodeBlockDistributor) for s in self._loaders):
            raise ValueError("Virtual-shard resume supports EpisodeBlockDistributor only")
        if self._worker_assignment is not None and len(self._worker_assignment) != sum(map(len, shards)):
            raise ValueError(
                "worker_assignment length does not match active logical shards after virtual-shard restore"
            )
        self._affinity = {}
        self._active = []
        slot = 0
        for source, stream in enumerate(self._loaders):
            stream.num_workers = len(shards[source])
            stream.dataset._distributor._virtual_shards = shards[source]
            stream.dataset._virtual_rng_seeds = seeds[source]
            self._active.append(list(range(stream.num_workers)))
            for worker in self._active[-1]:
                self._affinity[source, worker] = slot % max(1, self.num_workers)
                slot += 1
        # Physical placement is a runtime tuning choice, not checkpoint state.
        # Rebuilding logical shards must not silently reset an explicit mapping.
        self._apply_worker_assignment()
        self._virtual_shards, self._virtual_rng_seeds = shards, seeds

    def restore_execution_state(self, state, global_id):
        if self._started:
            raise RuntimeError("Restore dataloader state before starting shared workers")
        if state is not None:
            if "virtual_shards" in state:
                self._restore_virtual_shards(state)
            if state.get("version") != 1 or state.get("topology") != self._topology():
                raise ValueError("Shared dataloader logical topology changed since checkpoint")
            self._next_worker = list(state["next_workers"])
            self._base_seeds = state["base_seeds"]
            self._resume_rng = dict(state["rng"])
            self._committed_rng = dict(self._resume_rng)
        else:
            # Older checkpoints save cursor but not the next source-local worker.
            # Infinite catalog streams have a deterministic active-shard rotation.
            counts = self._selection_counts(global_id)
            self._next_worker = [count % len(active) for count, active in zip(counts, self._active, strict=True)]
            if global_id:
                log.warning(
                    "Migrating legacy loader checkpoint: preserving source/shard cursors and pack order; "
                    "legacy checkpoints do not contain worker augmentation RNG."
                )
        self._committed_worker = list(self._next_worker)
        self.set_start_iteration(global_id)

    def _selection_counts(self, global_id):
        counts = [0] * len(self._names)
        start = 0
        if self._exact_period is not None:
            cycles, remainder = divmod(global_id, len(self._exact_period))
            counts = [cycles * self._exact_period.count(source) for source in range(len(self._names))]
            start = global_id - remainder
        for index in range(start, global_id):
            counts[self.dataset_index(index)] += 1
        return counts

    def set_start_iteration(self, iteration):
        if iteration < 0:
            raise ValueError("global microbatch id must be nonnegative")
        if self._started:
            raise RuntimeError("Cannot seek an active shared loader; close and rebuild it")
        super().set_start_iteration(iteration)
        self._committed_global_id = iteration
        self._next_schedule_id = iteration

    def commit_batch(self, batch):
        meta = batch.get(LOADER_META_KEY)
        if meta is None or meta["global_id"] != self._committed_global_id:
            raise ValueError("Shared dataloader batch commits must follow global microbatch order")
        if meta["global_id"] >= self._global_id:
            raise ValueError("Cannot commit a batch that was not delivered by this loader")
        source = meta["source"]
        self._committed_worker[source] = meta["next_worker"]
        if meta["resume_rng"] is not None:
            self._committed_rng[source, meta["worker"]] = meta["resume_rng"]
        self._committed_global_id += 1

    def _start(self):
        if self._closed:
            raise RuntimeError("Shared dataloader has been closed; build a new loader")
        # Match the legacy per-source base-seed draws on a fresh run; preserve
        # restored training RNG when rebuilding after a checkpoint.
        with torch.random.fork_rng(devices=[]) if self._preserve_resume_rng else nullcontext():
            seeds = [int(torch.empty((), dtype=torch.int64).random_().item()) for _ in self._loaders]
        if self._base_seeds is None:
            self._base_seeds = seeds
        self._preserve_resume_rng = False
        if self.num_workers:
            # Linux fork matches the existing PyTorch loader. Workers are CPU
            # only and never invoke CUDA. Components may contain local size_fn
            # closures, so silently switching to spawn is not safe.
            context = mp.get_context("fork")
            if os.environ.get("COSMOS_ACTION128_MEDIA_PREFETCH_SOURCES"):
                from cosmos_framework.data.generator.action.media_prefetch import SharedMediaBudget

                limit = int(os.environ.get("COSMOS_ACTION128_MEDIA_PREFETCH_TOTAL_BYTES", str(2 * 1024**3)))
                self._media_budget = SharedMediaBudget(context, limit)
            self._results = _ResultQueue(context)
            # Diagnostic counters only: -2 means the signal handler is not yet
            # installed, -1 means idle. No IPC queue credits are used by status.
            self._worker_state = context.Array("q", [-2, -1] * self.num_workers, lock=False)
            try:
                for physical in range(self.num_workers):
                    requests = context.Queue(maxsize=self.prefetch_factor)
                    process = context.Process(
                        target=_worker_main,
                        args=(
                            requests,
                            self._results,
                            self._loaders,
                            self._base_seeds,
                            self._resume_rng,
                            physical,
                            self._worker_state,
                            self._timeout_config,
                            self._names,
                            self._collect_timing,
                            [key for key, assigned in self._affinity.items() if assigned == physical],
                            self._media_budget,
                        ),
                        daemon=True,
                        name=f"cosmos-data-{physical}",
                    )
                    process.start()
                    self._requests.append(requests)
                    self._processes.append(process)
                # Fork all CPU workers before creating a CUDA-aware thread.
                # Credits remain in _tasks until delivery, including batches
                # in this queue, so this stage does not add another K window.
                self._pin_in_background = (
                    self.pin_memory and self.pin_memory_in_background and torch.cuda.is_available()
                )
                if self.receive_in_background or self._pin_in_background:
                    self._received_results = queue.Queue(maxsize=self.prefetch_factor)
                    self._result_thread = threading.Thread(
                        target=_receive_results,
                        args=(
                            self._results,
                            self._received_results,
                            self._results_done,
                            torch.cuda.current_device() if self._pin_in_background else None,
                            self._pin_stats,
                        ),
                        name="cosmos-data-pin" if self._pin_in_background else "cosmos-data-receive",
                        daemon=True,
                    )
                    self._result_thread.start()
            except BaseException:
                self.close()
                raise
        else:
            self._runner = _StreamRunner(self._loaders, self._base_seeds, self._resume_rng)
        self._started = True

    def _plan_task(self):
        global_id = self._next_schedule_id
        source = self.dataset_index(global_id)
        rotation = self._next_worker[source]
        worker = self._active[source][rotation]
        self._next_worker[source] = (rotation + 1) % len(self._active[source])
        self._next_schedule_id += 1
        return global_id, source, worker, self._next_worker[source]

    def _fill(self):
        while len(self._tasks) < self.prefetch_factor:
            task = self._plan_task()
            physical = self._affinity[task[1], task[2]]
            if self._collect_timing:
                self._enqueued_ns[task[0]] = time.monotonic_ns()
            self._requests[physical].put(task)
            self._tasks.append(task)
        self._max_observed_inflight = max(self._max_observed_inflight, len(self._tasks))

    def _waiting_context(self, global_id):
        task = next(task for task in self._tasks if task[0] == global_id)
        physical = self._affinity[task[1], task[2]]
        active, submitted = self._worker_state[2 * physical : 2 * physical + 2]
        return (
            f"global microbatch {global_id}, dataset={self._names[task[1]]!r}, logical_worker={task[2]}, "
            f"physical_worker={physical}, pid={self._processes[physical].pid}, "
            f"active_microbatch={active}, last_submitted_microbatch={submitted}, "
            f"ready_microbatches={sorted(self._ready)}"
        )

    def _dump_waiting_worker(self, global_id):
        task = next(task for task in self._tasks if task[0] == global_id)
        physical = self._affinity[task[1], task[2]]
        process = self._processes[physical]
        log.warning(f"Shared dataloader still waiting for {self._waiting_context(global_id)}", rank0_only=False)
        if self._worker_state[2 * physical] != -2 and process.is_alive():
            try:
                os.kill(process.pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass
        if self._result_thread is not None:
            frame = sys._current_frames().get(self._result_thread.ident)
            if frame is not None:
                log.warning(
                    "Shared dataloader result thread stack:\n" + "".join(traceback.format_stack(frame)),
                    rank0_only=False,
                )

    def _receive(self, global_id):
        started = time.monotonic()
        deadline = started + self.worker_timeout
        diagnostic_interval = min(60.0, self.worker_timeout / 2)
        next_diagnostic = started + diagnostic_interval
        results = self._received_results if self._received_results is not None else self._results
        while global_id not in self._ready:
            dead = [p for p in self._processes if not p.is_alive()]
            if dead:
                raise RuntimeError(f"Shared dataloader worker exited: {[(p.pid, p.exitcode) for p in dead]}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Shared dataloader timed out after {self.worker_timeout}s waiting for "
                    f"{self._waiting_context(global_id)}; worker Python stacks were requested during the wait"
                )
            if time.monotonic() >= next_diagnostic:
                self._dump_waiting_worker(global_id)
                next_diagnostic = time.monotonic() + diagnostic_interval
            try:
                result_id, batch, error = results.get(timeout=min(0.2, remaining, diagnostic_interval))
            except queue.Empty:
                if self._result_thread is not None and not self._result_thread.is_alive():
                    raise RuntimeError("Shared dataloader result receiving thread exited")
                continue
            if error is not None:
                if result_id is None:
                    raise RuntimeError(error)
                task = next(task for task in self._tasks if task[0] == result_id)
                raise RuntimeError(
                    f"Shared dataloader failed at microbatch={result_id}, dataset={self._names[task[1]]!r}, "
                    f"logical_worker={task[2]}:\n{error}"
                )
            if _TIMING_KEY in batch:
                batch[_TIMING_KEY].setdefault("received_ns", time.monotonic_ns())
            self._ready[result_id] = batch
        return self._ready.pop(global_id)

    def _batch_timing(self, batch, task):
        # "lead" is the task's total residence before its ordered selection,
        # not time saved by prefetching. Transfer includes serialization, blocked
        # sends and receiver scheduling. Ready time includes background pinning;
        # optional main-thread pinning happens after these diagnostic timestamps.
        timing = batch.pop(_TIMING_KEY, None)
        if timing is None:
            return None
        timing["enqueued_ns"] = self._enqueued_ns.pop(task[0])
        timing["delivered_ns"] = time.monotonic_ns()
        for name, before, after in [
            ("queue", "enqueued_ns", "worker_started_ns"),
            ("prepare", "worker_started_ns", "worker_processed_ns"),
            ("transfer", "worker_processed_ns", "received_ns"),
            ("ready", "received_ns", "delivered_ns"),
            ("lead", "enqueued_ns", "delivered_ns"),
        ]:
            timing[f"{name}_seconds"] = (timing[after] - timing[before]) / 1e9
        timing["physical_worker"] = self._affinity[task[1], task[2]]
        return timing

    def __iter__(self):
        # This facade is its own iterator; repeated iter(loader) cannot create
        # another producer or reset a live logical stream.
        if not self._started:
            self._start()
        return self

    def __next__(self):
        if not self._started:
            self._start()
        if self._closed:
            raise RuntimeError("Shared dataloader has been closed")
        try:
            if self.num_workers:
                self._fill()
                task = self._tasks[0]
                received = time.perf_counter()
                batch = self._receive(task[0])
                receive_seconds = time.perf_counter() - received
                timing = self._batch_timing(batch, task)
                self._receive_seconds += receive_seconds
                if receive_seconds >= 10.0:
                    # Several shorter waits can slow a step without reaching
                    # the 60s in-progress stack-dump interval. Include IPC and
                    # optional background receiving/pinning, not just preparation.
                    log.warning(
                        f"Shared dataloader batch received after {receive_seconds:.2f}s: "
                        f"{self._waiting_context(task[0])}" + (f", timing={timing}" if timing is not None else ""),
                        rank0_only=False,
                    )
                self._tasks.popleft()
            else:
                timing = None
                task = self._plan_task()
                training_rng = capture_rng()
                try:
                    batch = self._runner.next_batch(task[1], task[2])
                finally:
                    restore_rng(training_rng)
            global_id, source, worker, next_worker = task
            resume_rng = batch.pop("_cosmos_resume_rng", None)
            # Pin only model-facing data, not CPU checkpoint metadata.
            if self.pin_memory and not self._pin_in_background and torch.cuda.is_available():
                pin_started = time.perf_counter()
                batch = pin_memory(batch)
                self._inline_pin_seconds += time.perf_counter() - pin_started
            batch["dataset_name"] = self._names[source]
            batch[LOADER_META_KEY] = {
                "global_id": global_id,
                "source": source,
                "worker": worker,
                "next_worker": next_worker,
                "resume_rng": resume_rng,
            }
            if timing is not None:
                batch[LOADER_META_KEY]["timing"] = timing
            self._global_id = global_id + 1
            # Replace the delivered slot now, so preparation overlaps training.
            if self.num_workers:
                self._fill()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Give normal receives/pinning time to stop before their producers.
        # A partial pipe message or stalled FD transfer cannot observe done;
        # do not let that block the worker watchdog's cleanup indefinitely.
        self._results_done.set()
        if self._result_thread is not None and self._result_thread.ident is not None:
            self._result_thread.join(timeout=1.0)
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        # Do not join queue feeder threads: on early break results may be full.
        # Give idle workers a chance to close helpers, then terminate stragglers.
        for requests in self._requests:
            try:
                requests.put_nowait(None)
            except queue.Full:
                pass
        deadline = time.monotonic() + 1.0
        for process in self._processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        for channel in [*self._requests, self._results]:
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        if self._result_thread is not None and self._result_thread.ident is not None:
            self._result_thread.join(timeout=2.0)
            if self._result_thread.is_alive():
                raise RuntimeError("Shared dataloader result thread did not stop after worker shutdown")
        if self._received_results is not None:
            while True:
                try:
                    self._received_results.get_nowait()
                except queue.Empty:
                    break
        self._ready.clear()
        self._tasks.clear()
        self._enqueued_ns.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()
