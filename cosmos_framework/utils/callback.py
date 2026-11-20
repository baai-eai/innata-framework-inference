# Copyright (c) 2026, BAAI. All rights reserved.
# Adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/utils/callback.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import sys
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

import omegaconf
import torch
import torch.distributed as dist
import torch.utils.data
import tqdm
import wandb

from cosmos_framework.utils import distributed, log, misc, wandb_util
from cosmos_framework.utils.device import accelerator_device
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor
from cosmos_framework.utils.nvtx import (
    NVTXGranularity,
    configure_nvtx,
    disable_nvtx,
    nvtx_mark,
)

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None


if TYPE_CHECKING:
    from cosmos_framework.model._base import ImaginaireModel
    from cosmos_framework.trainer import ImaginaireTrainer
    from cosmos_framework.utils.config import Config


class CallBackGroup:
    """A class for hosting a collection of callback objects.

    It is used to execute callback functions of multiple callback objects with the same method name.
    When callbackgroup.func(args) is executed, internally it loops through the objects in self._callbacks and runs
    self._callbacks[0].func(args), self._callbacks[1].func(args), etc. The method name and arguments should match.

    Attributes:
        _callbacks (list[Callback]): List of callback objects.
    """

    def __init__(self, config: Config, trainer: ImaginaireTrainer) -> None:
        """Initializes the list of callback objects.

        Args:
            config (Config): The config object for the Imaginaire codebase.
            trainer (ImaginaireTrainer): The main trainer.
        """
        self._callbacks = []
        callback_configs = config.trainer.callbacks
        if callback_configs:
            if isinstance(callback_configs, list) or isinstance(callback_configs, omegaconf.listconfig.ListConfig):
                warnings.warn(
                    "The 'config.trainer.callbacks' parameter should be a dict instead of a list. "
                    "Please update your code",
                    DeprecationWarning,
                    stacklevel=2,
                )
                callback_configs = {f"callback_{i}": v for i, v in enumerate(callback_configs)}
            for callback_name, current_callback_cfg in callback_configs.items():
                if "_target_" not in current_callback_cfg:
                    log.critical(
                        f"Callback {callback_name} is missing the '_target_' field. \n SKip {current_callback_cfg}"
                    )
                    continue
                log.critical(f"Instantiating callback {callback_name}: {current_callback_cfg}")
                _callback = instantiate(current_callback_cfg)
                if not isinstance(_callback, Callback):
                    missing_hooks = _missing_callback_hooks(_callback)
                    if missing_hooks:
                        raise TypeError(
                            f"{current_callback_cfg} is not a valid callback; "
                            f"missing required callback hooks: {', '.join(missing_hooks)}"
                        )
                try:
                    _callback.config = config
                    _callback.trainer = trainer
                except (AttributeError, TypeError) as error:
                    raise TypeError(
                        f"{current_callback_cfg} is not a valid callback; "
                        "cannot assign required callback metadata 'config' and 'trainer'"
                    ) from error
                self._callbacks.append(_callback)

    def __getattr__(self, method_name: str) -> Callable:
        """Loops through the callback objects to call the corresponding callback function.

        Args:
            method_name (str): Callback method name.
        """

        def multi_callback_wrapper(*args, **kwargs) -> None:
            for callback in self._callbacks:
                assert hasattr(callback, method_name)
                method = getattr(callback, method_name)
                assert callable(method)
                _ = method(*args, **kwargs)

        return multi_callback_wrapper


class Callback:
    """The base class for all callbacks.

    All callbacks should inherit from this class and adhere to the established method names and signatures.
    """

    def __init__(self, config: Optional["Config"] = None, trainer: Optional["ImaginaireTrainer"] = None):
        """Initializes a Callback object.

        Args:
            config (Optional[Config]): The configuration object for the Imaginaire codebase, if available.
            trainer (Optional[ImaginaireTrainer]): The main trainer handling the training loop, if available.

        Notes:
            The config and trainer parameters are optional to maintain backward compatibility.
            In future releases, these parameters will be removed. Upon using these parameters, a deprecation
            warning will be issued.

        """
        if config is not None or trainer is not None:
            warnings.warn(
                "The 'config' and 'trainer' parameters are deprecated and will be removed in a future release. "
                "Please update your code to create Callback instances without these parameters.",
                DeprecationWarning,
                stacklevel=2,
            )
        del config, trainer

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        """
        Called before the training step, for each batch. This is paired with on_training_step_end() but note that
        when using gradient accumulation, while on_training_step_end() is only called when the optimizer is updated,
        this function is called for every batch.
        Use on_training_step_batch_start and on_training_step_batch_end if you need callbacks that are called
        for every batch, albeit with the same iteration number.
        FIXME - should this either be deprecated, or called only when a new training step is started after having updated
        the optimizer?
        """
        pass

    def on_training_step_batch_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """
        Called before the training step, for each batch, similarly to on_training_step_start(). This function is paired with
        on_training_step_batch_end(), and both functions are called for every batch even when using gradient accumulation.
        Note that the iteration is only updated when the optimizer is updated, and therefore it may be the same for multiple invocations.
        """
        pass

    def on_before_forward(self, iteration: int = 0) -> None:
        pass

    def on_after_forward(self, iteration: int = 0) -> None:
        pass

    def on_before_backward(self, model: ImaginaireModel, loss: torch.Tensor, iteration: int = 0) -> None:
        pass

    def on_after_backward(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_before_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_after_dataloading(self, iteration: int = 0) -> None:
        pass

    def on_optimizer_init_start(self) -> None:
        pass

    def on_optimizer_init_end(self) -> None:
        pass

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_before_zero_grad(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        pass

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Called at the end of a training step for every batch even when using gradient accumulation.
        This is paired with on_training_step_batch_start(). Note that the iteration is only updated when the optimizer is updated,
        and therefore it may be the same for multiple batches.
        """
        pass

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """
        Called at the end of a training step, but note that when using gradient accumulation, this is only called
        when the optimizer is updated, and the iteration incremented, whereas on_training_step_start is called every time.
        Use on_training_step_batch_start and on_training_step_batch_end if you need callbacks that are called
        for every batch.
        """
        pass

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        pass

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        pass

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_load_checkpoint_start(self, model: ImaginaireModel) -> None:
        pass

    def on_load_checkpoint_end(
        self, model: ImaginaireModel, iteration: int = 0, checkpoint_path: Optional[str] = None
    ) -> None:
        pass

    def on_load_checkpoint(self, model: ImaginaireModel, state_dict: dict[Any]) -> None:
        """
        Called when checkpoint loading is about to start, but after on_save_checkpoint_start().
        FIXME - why do we need this callback, can't we just use on_save_checkpoint_start()?
        """
        pass

    def on_save_checkpoint_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """
        Called when checkpoint saving is about to start.
        """
        pass

    def on_save_checkpoint_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """
        Called when the synchronous part of checkpointing is finished, this function can be used
        along with on_save_checkpoint_start() to measure the exposed (synchronous) checkpoint time.
        Note that for asynchronous checkpoint, the checkpoint may still be ongoing, so this function
        does not mean the checkpoint is finished for the asynchronous case, use on_save_checkpoint_success()
        for that.
        """
        pass

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        """
        Called when checkpoint saving is fully finished, and succeeded. Not called if checkpoint failed.
        For synchronous checkpoint, it is called at the same time as on_save_checkpoint_end(), but for asynchronous
        checkpoint, it is called after the asynchronous part has also finished. For checkpointers with out-of-process
        checkpointing, this function is called as soon as the notification is received from the checkpointer process,
        which may not be immediately after the checkpoint has completed but later on. Therefore, if you need to measure
        the full checkpoint duration for the asynchronous part, use the elapsed_time parameter, do not measure it directly
        as this would be a significant overestimate.
        """
        pass

    def on_save_checkpoint(self, model: ImaginaireModel, state_dict: dict[Any]) -> None:
        pass

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        pass

    def on_app_end(self) -> None:
        pass


def _missing_callback_hooks(callback: object) -> list[str]:
    """Return required callback hooks that are absent or non-callable."""
    return sorted(
        name
        for name, member in vars(Callback).items()
        if name.startswith("on_") and callable(member) and not callable(getattr(callback, name, None))
    )


class EMAModelCallback(Callback):
    """The callback class for tracking EMA model weights."""

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Set up the EMA model weight tracker.
        if model.config.ema.enabled:
            assert hasattr(model, "ema"), "EMA should be initialized from ImaginaireModel"
            # EMA model must be kept in FP32 precision.
            model.ema = model.ema.to(dtype=torch.float32)
        else:
            assert not hasattr(model, "ema"), "There should be no EMA initialized."

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Update the EMA model with the new regular weights.
        if model.config.ema.enabled:
            model.ema.update_average(model, iteration)


class ProgressBarCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.train_pbar.update()

    @distributed.rank0_only
    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        if self.config.trainer.max_val_iter is not None:
            num_iter = self.config.trainer.max_val_iter
        else:
            num_iter = len(dataloader_val)
        assert num_iter is not None and num_iter > 0, f"Invalid number of validation iterations: {num_iter}"
        self.val_pbar = tqdm.trange(num_iter, desc="Validating", position=1, leave=False)

    @distributed.rank0_only
    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        self.val_pbar.update()

    @distributed.rank0_only
    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.val_pbar.close()

    @distributed.rank0_only
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        self.trainer.checkpointer.finalize()
        self.train_pbar.close()


class IterationLoggerCallback(Callback):
    """The callback class for visualizing the training/validation progress bar in the console."""

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # self.train_pbar = tqdm.trange(self.config.trainer.max_iter, initial=iteration, desc="Training")
        self.start_iteration_time = time.time()
        self.elapsed_iteration_time = 0

    @distributed.rank0_only
    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        self.start_iteration_time = time.time()

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # FIXME - this is not correct when using gradient accumulation since self.start_iteration_time is updated every batch
        # but this is only called when the optimizer is updated, so it's only the time for the last batch.
        self.elapsed_iteration_time += time.time() - self.start_iteration_time

        if iteration % self.config.trainer.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / self.config.trainer.logging_iter
            log.info(f"Iteration: {iteration}, average iter time: {avg_time:2f}, total loss {loss.item():4f}")

            self.elapsed_iteration_time = 0


class WandBCallback(Callback):
    """The callback class for logging to Weights and Biases (W&B).

    By default, WandBCallback logs the following training stats to W&B every config.trainer.logging_iter:
    - iteration: The current iteration number (useful for visualizing the training progress over time).
    - train/loss: The computed overall loss in the training batch.
    - optim/lr: The current learning rate.
    - timer/*: The averaged timing results of each code block recorded by trainer.training_timer.
    For validation, WandBCallback logs:
    - val/loss: The computed overall loss in the validation dataset.
    """

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb_util.init_wandb(self.config, model=model)

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            wandb.log({"optim/lr": scheduler.get_last_lr()[0]}, step=iteration)
            wandb.log({"optim/grad_scale": grad_scaler.get_scale()}, step=iteration)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Log the timing results (over a number of iterations) and the training loss.
        if iteration % self.config.trainer.logging_iter == 0:
            timer_results = self.trainer.training_timer.compute_average_results()

            # reduce loss
            sample_size = torch.tensor(misc.get_data_batch_size(data_batch), device=accelerator_device())
            loss_sum = loss * sample_size
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)
            avg_loss = loss_sum.item() / sample_size.item()

            if distributed.is_rank0():
                wandb.log({f"timer/{key}": value for key, value in timer_results.items()}, step=iteration)
                wandb.log({"train/loss": avg_loss}, step=iteration)
                wandb.log({"iteration": iteration}, step=iteration)
            self.trainer.training_timer.reset()

    def on_validation_start(
        self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device=accelerator_device()),
            sample_size=torch.tensor(0, device=accelerator_device()),
        )

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = misc.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss:4f}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_app_end(self) -> None:
        # Validation callbacks may still log in on_train_end. Finish only once
        # the entire callback group and checkpoint finalization have completed.
        wandb.finish()


class LowPrecisionCallback(Callback):
    """The callback class handling low precision training"""

    def __init__(self, config: Config, trainer: ImaginaireTrainer, update_iter: int):
        self.update_iter = update_iter

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if model.precision == torch.float32:
            log.critical("Using fp32. We should disable master weights update.")
            self.update_iter = sys.maxsize
        else:
            assert model.precision in [
                torch.bfloat16,
                torch.float16,
                torch.half,
            ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_validation_step_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_before_zero_grad(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if iteration % self.update_iter == 0:
            if getattr(optimizer, "master_weights", False):
                params, master_params = [], []
                for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
                    for p, p_master in zip(group["params"], group_master["params"]):
                        params.append(get_local_tensor_if_DTensor(p).data)
                        master_params.append(get_local_tensor_if_DTensor(p_master).data)
                torch._foreach_copy_(params, master_params)


class NVTXCallback(Callback):
    """Create hierarchical NVTX ranges around the training pipeline.

    ``granularity`` is cumulative:

    - ``coarse``: legacy dataloading/forward/backward/optimizer ranges.
    - ``phase``: optimizer-step/micro-step identity plus model pipeline phases.
    - ``module``: per-VAE-item and per-decoder-layer ranges.
    - ``operator``: every autograd operation, with forward/backward sequence IDs.

    ``detail_target_ranks`` limits phase-and-deeper annotations while retaining
    the legacy coarse ranges on every rank. This is useful for keeping an
    operator-level multi-GPU Nsight Systems report manageable.

    ``operator_start_iteration`` delays ``emit_nvtx`` until the requested
    optimizer iteration. When omitted during an nsys run, it is inferred from
    the nsys active window so model compilation and warmup are not perturbed.

    Optimizer ranges span two callback hooks, so they use NVTX start/end handles
    instead of the thread-local push/pop stack. This keeps trainer-owned nested
    ranges balanced while the wider optimizer range remains open.
    """

    def __init__(
        self,
        synchronize: bool = False,
        granularity: str = "coarse",
        detail_target_ranks: Optional[list[int]] = None,
        record_shapes: bool = False,
        operator_start_iteration: Optional[int] = None,
        config: Optional["Config"] = None,
        trainer: Optional["ImaginaireTrainer"] = None,
    ):
        super().__init__(config, trainer)
        self.synchronize = synchronize
        self.granularity = NVTXGranularity.parse(granularity)
        self.detail_target_ranks = detail_target_ranks
        self.record_shapes = record_shapes
        self.operator_start_iteration = operator_start_iteration
        self._effective_granularity = NVTXGranularity.COARSE
        self._operator_profiler: Any | None = None
        self._effective_operator_start_iteration = 0
        self._grad_accum_iter = 1
        self._micro_step = 0
        self._rank = 0
        self._micro_range_open = False
        self._optimizer_range_ids: list[int] = []

    def _context_name(self, iteration: int) -> str:
        return (
            f"train.rank_{self._rank:03d}.step_{iteration + 1:06d}."
            f"micro_{self._micro_step + 1:02d}_of_{self._grad_accum_iter:02d}"
        )

    @property
    def _detail_enabled(self) -> bool:
        return self._effective_granularity >= NVTXGranularity.PHASE

    @staticmethod
    def _shape_summary(value: Any, *, limit: int = 64) -> list[tuple[int, ...]]:
        shapes: list[tuple[int, ...]] = []

        def collect(item: Any) -> None:
            if len(shapes) >= limit:
                return
            if isinstance(item, torch.Tensor):
                shapes.append(tuple(item.shape))
            elif isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)

        collect(value)
        return shapes

    @staticmethod
    def _compact(value: Any, *, limit: int = 768) -> str:
        text = repr(value).replace(" ", "")
        return text if len(text) <= limit else f"{text[: limit - 3]}..."

    @staticmethod
    def _sequence_plan_summary(value: Any, *, limit: int = 64) -> list[str]:
        modes: list[str] = []

        def collect(item: Any) -> None:
            if len(modes) >= limit:
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    collect(child)
                return
            mode = (
                item.get("vision_conditioning")
                if isinstance(item, dict)
                else getattr(item, "vision_conditioning", None)
            )
            if mode is not None:
                modes.append(str(mode))

        collect(value)
        return modes

    def _mark_input_metadata(self, data: dict[str, torch.Tensor], iteration: int) -> None:
        fields = []
        for key, label in (("video", "video"), ("images", "image"), ("action", "action"), ("sound", "sound")):
            if key in data:
                fields.append(f"{label}_shapes={self._compact(self._shape_summary(data[key]))}")
        if "text_token_ids" in data:
            fields.append(f"text_shapes={self._compact(self._shape_summary(data['text_token_ids']))}")
        if "sequence_plan" in data:
            fields.append(f"vision_modes={self._compact(self._sequence_plan_summary(data['sequence_plan']))}")
        nvtx_mark(f"meta.input.{self._context_name(iteration)};{'|'.join(fields) or 'no_modalities'}")

    def _mark_output_metadata(self, output: dict[str, Any], iteration: int) -> None:
        fields = [f"kind={'image' if output.get('is_image_batch') else 'video'}"]
        for key, label in (
            ("batch_size", "batch"),
            ("und_token_length", "real_und"),
            ("gen_token_length", "real_gen"),
            ("physical_und_token_length", "physical_und"),
            ("physical_gen_token_length", "physical_gen"),
            ("vae_pixel_shapes", "vae_shapes"),
            ("split_lens", "split_lens"),
        ):
            if key in output:
                fields.append(f"{label}={self._compact(output[key])}")
        nvtx_mark(f"meta.output.{self._context_name(iteration)};{'|'.join(fields)}")

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        self._rank = distributed.get_rank()
        is_detail_rank = self.detail_target_ranks is None or self._rank in self.detail_target_ranks
        self._effective_granularity = self.granularity if is_detail_rank else NVTXGranularity.COARSE
        configure_nvtx(self._effective_granularity)
        self._grad_accum_iter = int(self.config.trainer.grad_accum_iter)
        if self._effective_granularity >= NVTXGranularity.OPERATOR:
            if self.config.trainer.profiling.enable_profiling:
                raise ValueError("NVTX granularity='operator' cannot run concurrently with the Torch profiler")
            if self.operator_start_iteration is not None:
                self._effective_operator_start_iteration = self.operator_start_iteration
            elif self.config.trainer.profiling.enable_nsys:
                self._effective_operator_start_iteration = (
                    self.config.trainer.profiling.profile_freq - self.config.trainer.profiling.profile_active
                )
            else:
                self._effective_operator_start_iteration = iteration
            self._start_operator_profiler_if_needed(iteration)

    def _start_operator_profiler_if_needed(self, iteration: int) -> None:
        if (
            self._effective_granularity >= NVTXGranularity.OPERATOR
            and self._operator_profiler is None
            and iteration >= self._effective_operator_start_iteration
        ):
            self._operator_profiler = torch.autograd.profiler.emit_nvtx(record_shapes=self.record_shapes)
            self._operator_profiler.__enter__()

    def _cleanup(self) -> None:
        try:
            self._close_optimizer_ranges()
        finally:
            try:
                if self._operator_profiler is not None:
                    self._operator_profiler.__exit__(None, None, None)
                    self._operator_profiler = None
            finally:
                disable_nvtx()

    def _close_optimizer_ranges(self) -> None:
        """Close cross-hook optimizer ranges without touching the NVTX push/pop stack."""

        range_ids, self._optimizer_range_ids = self._optimizer_range_ids, []
        for range_id in reversed(range_ids):
            torch.cuda.nvtx.range_end(range_id)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        self._cleanup()

    def on_app_end(self) -> None:
        self._cleanup()

    def on_training_step_batch_start(
        self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        del model
        if self._detail_enabled:
            self._start_operator_profiler_if_needed(iteration)
            if self._micro_range_open:
                raise RuntimeError("NVTX micro-step range was not closed before the next batch")
            torch.cuda.nvtx.range_push(f"cosmos.{self._context_name(iteration)}")
            self._micro_range_open = True
            self._mark_input_metadata(data, iteration)

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch, loss
        if self._detail_enabled:
            self._mark_output_metadata(output_batch, iteration)
            if not self._micro_range_open:
                raise RuntimeError("NVTX micro-step range ended without a matching start")
            torch.cuda.nvtx.range_pop()
            self._micro_range_open = False
        self._micro_step = (self._micro_step + 1) % self._grad_accum_iter

    def on_before_forward(self, iteration: int = 0) -> None:
        if self.synchronize:
            torch.accelerator.synchronize()
        torch.cuda.nvtx.range_push("forward")
        if self._detail_enabled:
            torch.cuda.nvtx.range_push(f"cosmos.{self._context_name(iteration)}.forward")

    def on_after_forward(self, iteration: int = 0) -> None:
        del iteration
        if self.synchronize:
            torch.accelerator.synchronize()
        if self._detail_enabled:
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    def on_before_backward(self, model: ImaginaireModel, loss: torch.Tensor, iteration: int = 0) -> None:
        del model, loss
        if self.synchronize:
            torch.accelerator.synchronize()
        torch.cuda.nvtx.range_push("backward")
        if self._detail_enabled:
            torch.cuda.nvtx.range_push(f"cosmos.{self._context_name(iteration)}.backward")

    def on_after_backward(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        if self.synchronize:
            torch.accelerator.synchronize()
        if self._detail_enabled:
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model, optimizer, scheduler, grad_scaler
        if self.synchronize:
            torch.accelerator.synchronize()
        if self._optimizer_range_ids:
            raise RuntimeError("NVTX optimizer range was not closed before the next optimizer step")

        range_ids = [torch.cuda.nvtx.range_start("optimizer_step")]
        try:
            if self._detail_enabled:
                range_ids.append(torch.cuda.nvtx.range_start(f"cosmos.{self._context_name(iteration)}.optimizer"))
        except Exception:
            for range_id in reversed(range_ids):
                torch.cuda.nvtx.range_end(range_id)
            raise
        self._optimizer_range_ids = range_ids

    def on_before_zero_grad(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        del model, optimizer, scheduler, iteration
        if self.synchronize:
            torch.accelerator.synchronize()
        if not self._optimizer_range_ids:
            raise RuntimeError("NVTX optimizer range ended without a matching start")
        self._close_optimizer_ranges()

    def on_before_dataloading(self, iteration: int = 0) -> None:
        torch.cuda.nvtx.range_push("dataloading")
        if self._detail_enabled:
            torch.cuda.nvtx.range_push(f"cosmos.{self._context_name(iteration)}.dataloading")

    def on_after_dataloading(self, iteration: int = 0) -> None:
        del iteration
        if self._detail_enabled:
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()


# End of callback definitions.
