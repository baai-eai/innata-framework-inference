# Copyright (c) 2026, BAAI. All rights reserved.
# Adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/utils/wandb_util.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import attrs
import torch
import wandb
import wandb.util
from omegaconf import DictConfig

from cosmos_framework.utils import distributed, log, object_store
from cosmos_framework.utils.device import accelerator_device
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.launch_defaults import get_wandb_entity, get_wandb_project
from cosmos_framework.utils.lazy_config.lazy import LazyConfig

if TYPE_CHECKING:
    from cosmos_framework.model._base import ImaginaireModel
    from cosmos_framework.utils.config import CheckpointConfig, Config, JobConfig

JOB_INFO = {}


def set_wandb_job_info(job_info: dict) -> None:
    """Set the job info for the W&B logger.

    Args:
        job_info (dict): The job info.
    """
    JOB_INFO.update(job_info)


def init_wandb(config: Config, model: ImaginaireModel) -> None:
    """Initialize W&B on rank 0 and synchronize the result across all ranks.

    All ranks wait here until rank 0 has either initialized W&B or failed. This
    prevents a rank-0-only logging failure from leaving the remaining ranks
    blocked indefinitely in their next NCCL collective.
    """
    error: Exception | None = None
    if distributed.is_rank0():
        try:
            _init_wandb_rank0(config, model)
        except Exception as exc:
            error = exc
            try:
                wandb.teardown()
            except Exception:
                pass
            log.error(f"W&B initialization failed; stopping every training rank: {exc}")

    failed = _broadcast_wandb_init_status(error is not None)
    if failed:
        if error is not None:
            raise error
        raise RuntimeError("W&B initialization failed on rank 0; terminating this training rank")


def _init_wandb_rank0(config: Config, model: ImaginaireModel) -> None:
    """Initialize Weights & Biases (wandb) logger.

    Args:
        config (Config): The config object for the Imaginaire codebase.
        model (ImaginaireModel): The PyTorch model.
    """
    if isinstance(config.job, DictConfig):
        from cosmos_framework.utils.config import JobConfig

        config_job = JobConfig(**config.job)
    else:
        config_job = config.job
    require_online = os.environ.get("COSMOS_REQUIRE_WANDB_ONLINE", "0") == "1"
    if require_online and config_job.wandb_mode != "online":
        raise RuntimeError("COSMOS_REQUIRE_WANDB_ONLINE=1 requires job.wandb_mode=online")
    config_checkpoint = config.checkpoint
    wandb_project = get_wandb_project(config_job.project)
    # Try to fetch the W&B job ID for resuming training.
    wandb_id = _read_wandb_id(config_job, config_checkpoint)
    if wandb_id is None:
        # Generate a new W&B job ID.
        wandb_id = wandb.util.generate_id()
        _write_wandb_id(config_job, config_checkpoint, wandb_id=wandb_id)
        log.info(f"Generating new wandb ID: {wandb_id}")
    else:
        log.info(f"Resuming with existing wandb ID: {wandb_id}")
    # refactor config so that wandb better understands it
    local_safe_yaml_fp = LazyConfig.save_yaml(config, os.path.join(config_job.path_local, "config.yaml"))
    if os.path.exists(local_safe_yaml_fp):
        config_resolved = easy_io.load(local_safe_yaml_fp)
    else:
        config_resolved = attrs.asdict(config)
    # Initialize the wandb library. If we attempt to resume an existing run
    # but the current user does not have permission to update that run
    # (common when re-using an ID created by someone else), fall back to
    # creating a fresh run ID and re-initializing.
    try:
        wandb.init(
            force=True,
            entity=get_wandb_entity(),
            id=wandb_id,
            project=wandb_project,
            group=config_job.group,
            name=config_job.name,
            config=config_resolved,
            dir=config_job.path_local,
            resume="allow",
            mode=config_job.wandb_mode,
        )
    except Exception as e:
        if require_online:
            # The outer wrapper broadcasts this failure to every training rank.
            raise RuntimeError("Required online W&B initialization failed; refusing fallback or a new run ID") from e
        # Detect common permission / upload errors from wandb and recover
        msg = str(e)
        if (
            "member role does not have Update Run permission" in msg
            or "Error uploading run" in msg
            or "returned error 403" in msg
        ):
            log.warning("W&B run exists but current user lacks update permission; starting a new run instead.")
            # Generate and persist a new wandb id, then create a fresh run.
            wandb_id = wandb.util.generate_id()
            _write_wandb_id(config_job, config_checkpoint, wandb_id=wandb_id)
            wandb.init(
                force=True,
                entity=get_wandb_entity(),
                id=wandb_id,
                project=wandb_project,
                group=config_job.group,
                name=config_job.name,
                config=config_resolved,
                dir=config_job.path_local,
                mode=config_job.wandb_mode,
            )
        elif "returned error 401" in msg or "user is not logged in" in msg:
            log.warning(f"W&B authentication failed (401); falling back to offline mode. Error: {msg}")
            wandb.init(
                force=True,
                entity=get_wandb_entity(),
                id=wandb_id,
                project=wandb_project,
                group=config_job.group,
                name=config_job.name,
                config=config_resolved,
                dir=config_job.path_local,
                mode="offline",
            )
        else:
            raise

    if require_online and (wandb.run is None or wandb.run.settings.mode != "online"):
        raise RuntimeError("W&B did not create the required online run")
    if wandb.run:
        wandb.run.config.update({f"JOB_INFO/{k}": v for k, v in JOB_INFO.items()}, allow_val_change=True)


def _broadcast_wandb_init_status(rank0_failed: bool) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return rank0_failed
    backend = torch.distributed.get_backend()
    device = accelerator_device(torch.accelerator.current_device_index()) if backend == "nccl" else torch.device("cpu")
    status = torch.tensor([int(rank0_failed)], dtype=torch.int32, device=device)
    torch.distributed.broadcast(status, src=0)
    return bool(status.item())


def _read_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig) -> str | None:
    """Read the W&B job ID. If it doesn't exist, return None.

    Reads from the same location as ``_write_wandb_id`` (save object store / local),
    so resume works when load and save buckets differ.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.

    Returns:
        wandb_id (str | None): W&B job ID.
    """
    wandb_id = None
    if config_checkpoint.save_to_object_store.enabled:
        object_store_loader = object_store.ObjectStore(config_checkpoint.save_to_object_store)
        wandb_id_path = f"{config_job.path}/wandb_id.txt"
        if object_store_loader.object_exists(key=wandb_id_path):
            wandb_id = object_store_loader.load_object(key=wandb_id_path, type="text").strip()
    else:
        wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
        if os.path.isfile(wandb_id_path):
            wandb_id = open(wandb_id_path).read().strip()
    return wandb_id


def _write_wandb_id(config_job: JobConfig, config_checkpoint: CheckpointConfig, wandb_id: str) -> None:
    """Write the generated W&B job ID.

    Args:
        config_wandb (JobConfig): The config object for the W&B logger.
        config_checkpoint (CheckpointConfig): The config object for the checkpointer.
        wandb_id (str): The W&B job ID.
    """
    content = f"{wandb_id}\n"
    if config_checkpoint.save_to_object_store.enabled:
        object_store_saver = object_store.ObjectStore(config_checkpoint.save_to_object_store)
        wandb_id_path = f"{config_job.path}/wandb_id.txt"
        object_store_saver.save_object(content, key=wandb_id_path, type="text")
    else:
        wandb_id_path = f"{config_job.path_local}/wandb_id.txt"
        with open(wandb_id_path, "w") as file:
            file.write(content)
