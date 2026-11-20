# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""torchrun-backed Action128 server; only rank zero owns the HTTP listener."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

from cosmos_framework.inference.action_serving import (
    PreparedAction128Request,
    create_app,
    prepare_wam_request,
    resolve_checkpoint,
)
from cosmos_framework.utils.device import accelerator_device


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--checkpoint-path",
        "--checkpoint-root",
        dest="checkpoint_root",
        type=Path,
        required=True,
        help="Training OUTPUT_ROOT, run directory, checkpoints directory, or completed iter/model",
    )
    result.add_argument("--output-dir", type=Path, default=Path("outputs/action_serving"))
    result.add_argument("--model-name", default="action128")
    result.add_argument("--host", default="0.0.0.0")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--vae-path", type=Path)
    result.add_argument("--tokenizer-path", type=Path)
    result.add_argument("--attention-backend", choices=("sdpa", "flash3_video", "flash3"), default="flash3")
    result.add_argument("--fuse-qkv", action="store_true", help="Experimental post-load immutable QKV buffers")
    result.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the three-tower decoder layers (enabled by default)",
    )
    result.add_argument("--cuda-graphs", action="store_true", help="Enable compile CUDA graphs (requires --compile)")
    result.add_argument(
        "--compile-recompile-limit",
        type=int,
        default=256,
        help="Decoder compile cache entries for varying request lengths (applied after model initialization)",
    )
    result.add_argument(
        "--reuse-small-action-und", action="store_true", help="Experimental request-local small tower UND reuse"
    )
    result.add_argument(
        "--reuse-small-action-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse request-local layout/mask/RoPE metadata only (enabled by default; single GPU)",
    )
    result.add_argument(
        "--warmup-input", type=Path, help="Single-GPU startup warmup: Cosmos request JSON object/list, twice each"
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Resolve checkpoint/config and print plan without loading GPU weights"
    )
    result.add_argument(
        "--weights",
        choices=("regular", "ema"),
        default="regular",
        help="Explicit DCP weight namespace; EMA never falls back to regular",
    )
    return result


def deployment_plan(args: argparse.Namespace) -> dict[str, Any]:
    model, config = resolve_checkpoint(args.checkpoint_root)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "checkpoint_path": str(model),
        "config_file": str(config),
        "world_size": world_size,
        "model_name": args.model_name,
        "sft_profile": getattr(args, "sft_profile", None),
        "output_dir": str(args.output_dir.resolve()),
        "dp_shard_size": world_size,
        "cp_size": 1,
        "cfgp_size": 1,
        "attention_backend": args.attention_backend,
        "compile": args.compile,
        "fuse_qkv": args.fuse_qkv,
        "cuda_graphs": args.cuda_graphs,
        "reuse_small_action_und": args.reuse_small_action_und,
        "reuse_small_action_metadata": args.reuse_small_action_metadata,
        "warmup": {"requested": args.warmup_input is not None, "completed": False},
        "ema_weights": args.weights == "ema",
        "weight_namespace": "net_ema" if args.weights == "ema" else "net",
        "checkpoint_selection": "startup_snapshot",
        "fixed_training_buckets": False,
        "host": args.host,
        "port": args.port,
    }


def model_overrides(args: argparse.Namespace) -> list[str]:
    """Retain learned architecture/contract while disabling training padding buckets."""
    overrides = [
        "model.config.ema.enabled=false",
        "model.config.action_tower_init=checkpoint",
        "model.config.fixed_sequence_length.enabled=false",
        "model.config.action_tower_und_short_tokens=0",
        f"model.config.action_tower_attention_backend={args.attention_backend}",
    ]
    if args.vae_path:
        overrides.append(f"model.config.tokenizer.vae_path={json.dumps(str(args.vae_path.resolve()))}")
    if args.tokenizer_path:
        overrides.append(
            f"model.config.vlm_config.tokenizer.pretrained_model_name={json.dumps(str(args.tokenizer_path.resolve()))}"
        )
    return overrides


def initialize_single_rank_distributed() -> None:
    """Respect torchrun's agent store even when WORLD_SIZE is one."""
    from torch import distributed as dist

    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", init_method="env://")
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


class DistributedActionService:
    def __init__(self, pipe: Any, group: Any, info: dict[str, Any]):
        from omegaconf import OmegaConf
        from torch import distributed as dist

        self.model = pipe.model
        self.model.eval()
        if info.get("fuse_qkv", False):
            from cosmos_framework.inference.action_optimization import fuse_small_action_qkv

            info["fused_qkv_bytes"] = fuse_small_action_qkv(self.model)
        self.group = group
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.video_key = self.model.input_video_key
        saved_spec = self.model.config.action128_spec
        self.spec = OmegaConf.to_container(saved_spec, resolve=True) if OmegaConf.is_config(saved_spec) else saved_spec
        if not isinstance(self.spec, dict) or not self.model.config.action_gen:
            raise ValueError("This service requires an Action128 action-generating checkpoint")
        normalizer = getattr(self.model.net, "action_normalization", None)
        if normalizer is None:
            raise ValueError("Checkpoint has no Action128 normalization contract")
        normalizer.validate_contract()
        # DCP copies buffers directly and does not run load_state_dict hooks.
        # Recover the exact FP32 transforms from the verified saved contract.
        normalizer._reset_transforms()
        self.info = info
        if info.get("sft_profile"):
            from cosmos_framework.inference.sft_action_serving import sft_action_schema, validate_profile

            validate_profile(self.spec, info["sft_profile"])
            info["action_schema"] = sft_action_schema(info["sft_profile"])

    def prepare(self, request, output_dir):
        if self.info.get("sft_profile"):
            from cosmos_framework.inference.sft_action_serving import prepare_sft_wam_request

            return prepare_sft_wam_request(
                request, self.spec, self.model.config, output_dir, self.video_key, self.info["sft_profile"]
            )
        return prepare_wam_request(request, self.spec, self.model.config, output_dir, self.video_key)

    def execute(self, request: PreparedAction128Request, batch: dict) -> Any:
        import torch

        from cosmos_framework.utils import misc

        # FastAPI's synchronous route executes on a thread. CUDA device is thread-local.
        torch.accelerator.set_device_index(int(os.environ.get("LOCAL_RANK", "0")))
        start = time.monotonic()
        batch = misc.to(batch, device=accelerator_device())
        with torch.inference_mode():
            sampling = {}
            if request.sample_args:
                sampling = {
                    key: request.sample_args[key] for key in ("guidance_interval", "normalize_cfg", "sigma_max")
                }
                sampling["has_negative_prompt"] = "neg_ai_caption" in batch
            if self.info.get("reuse_small_action_und", False):
                sampling["reuse_small_action_und"] = True
            if self.info.get("reuse_small_action_metadata", False):
                sampling["reuse_small_action_metadata"] = True
            samples = self.model.generate_samples_from_batch(
                batch,
                seed=[request.seed],
                num_steps=request.num_steps,
                guidance=request.guidance,
                shift=request.shift,
                **sampling,
            )
            if self.rank != 0:
                return None
            # The processing record deliberately has no legacy normalizer. The
            # generator therefore returns model-space Action128 values here.
            normalized = samples["action"][0].float()
            domain = self.spec["registry"]["domains"].index(request.action_domain)
            native = self.spec["registry"]["domain_projection_ids"][domain] == 32
            history = request.state_history_length or (1 if native else 0)
            normalized = normalized[history:]
            if native:
                normalized = normalized[:, :12]  # State is conditioning, never an env command.
                physical = normalized.clone()  # Native controller units are identity.
            else:
                physical = self.model.net.action_normalization.denormalize(normalized, domain)
            mask = torch.tensor(
                request.action_dim_mask[:12] if native else request.action_dim_mask,
                device=physical.device,
                dtype=torch.bool,
            )
            physical = physical.masked_fill(~mask, 0)
            normalized = normalized.masked_fill(~mask, 0)
            if not torch.isfinite(physical).all() or not torch.isfinite(normalized).all():
                raise RuntimeError("Model returned non-finite actions")
            output_mask = list(request.action_dim_mask[:12] if native else request.action_dim_mask)
            output_schema = None
            sft_profile = self.info.get("sft_profile", "robocasa" if native else "robodojo")
            if request.state_history_length:
                from cosmos_framework.inference.sft_action_serving import sft_action_schema

                output_schema = sft_action_schema(sft_profile)
                slots = output_schema["model_slots"]
                physical, normalized = physical[:, slots], normalized[:, slots]
                output_mask = [request.action_dim_mask[slot] for slot in slots]
            libero = sft_profile in {"libero", "libero_plus"}
            if libero:
                from cosmos_framework.inference.libero_sft import control10_to_action7

                physical = torch.from_numpy(control10_to_action7(physical.cpu().numpy())).to(physical.device)
                output_mask = [True] * 7
        content = {
            "action": physical.cpu().tolist(),
            "normalized_action": normalized.cpu().tolist(),
            "action_units": "raw_physical_or_source_units",
            "action_domain": request.action_domain,
            "action_dim_mask": output_mask if output_schema else request.action_dim_mask,
            "action_fps": request.action_fps,
            "action_timestamps_seconds": [
                value - (history - 1) / request.action_fps if history else value
                for value in batch["sequence_plan"][0].action_timestamps_seconds[history:]
            ],
            "state_history_length": history,
            "seed": request.seed,
            "checkpoint_path": self.info["checkpoint_path"],
            "conditioning_caption_sha256": hashlib.sha256(batch["ai_caption"][0].encode()).hexdigest(),
            "elapsed_seconds": time.monotonic() - start,
        }
        if output_schema:
            content["action_schema"] = output_schema
            content["action_units"] = "native_controller_input" if native else "joint_absolute_rad_gripper_openness_0_1"
        if sft_profile == "robodojo":
            from cosmos_framework.inference.sft_action_serving import ROBODOJO_CAPTION_VERSION

            content["caption_serialization_version"] = ROBODOJO_CAPTION_VERSION
        if libero:
            # Rotvec7 and normalized rot6d10 are different representations. Do not
            # mislabel diagnostics as seven normalized controller coordinates.
            content["normalized_control10"] = content.pop("normalized_action")
            content["action_units"] = "native_OSC_POSE_translation_rotvec_signed_gripper"
        if native:
            content["action_units"] = "native_controller_input"
            content["control_range_check"] = {
                "expected": [-1, 1],
                "outside_count": int((physical.abs() > 1).sum().item()),
                "max_abs": float(physical.abs().max().item()),
                "clipped": False,
            }
        if not request.sample_args:
            return content  # Internal contract tests; HTTP always uses Cosmos SampleOutputs.
        from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs
        from cosmos_framework.tools.visualize.video import save_img_or_video

        output_dir = Path(request.sample_args["output_dir"])
        video_file = output_dir / "vision.mp4"
        files = []
        if request.return_video:
            with torch.inference_mode():
                video = ((self.model.decode(samples["vision"][0]) + 1) / 2).clamp(0, 1).squeeze(0)
            save_img_or_video(
                video,
                str(video_file.with_suffix("")),
                fps=request.sample_args["fps"],
                quality=request.sample_args["video_save_quality"],
            )
            files.append(video_file.relative_to(Path(self.info["output_dir"])))
        content["elapsed_seconds"] = time.monotonic() - start
        action = content.pop("action")
        result = SampleOutputs(
            args=request.sample_args,
            outputs=[
                SampleOutput(
                    content={"action": action, "action128": content},
                    files=files,
                )
            ],
        )
        (output_dir / "sample_args.json").write_text(json.dumps(request.sample_args))
        (output_dir / "sample_outputs.json").write_text(result.model_dump_json())
        return result

    def predict(self, request: PreparedAction128Request, batch: dict) -> Any:
        from torch import distributed as dist

        try:
            if dist.get_world_size() > 1:
                dist.broadcast_object_list([("predict", request.model_dump(), batch)], src=0, group=self.group)
            result = self.execute(request, batch)
            # Workers have finished before admitting another request.
            dist.barrier(group=self.group)
            return result
        except Exception:
            # A failed collective may poison the group. torchrun must terminate
            # all ranks instead of accepting another request on a partial group.
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)

    def worker_loop(self) -> None:
        from torch import distributed as dist

        while True:
            message = [None]
            dist.broadcast_object_list(message, src=0, group=self.group)
            command, payload, batch = message[0]
            if command == "stop":
                return
            try:
                self.execute(PreparedAction128Request.model_validate(payload), batch)
                dist.barrier(group=self.group)
            except Exception:
                traceback.print_exc()
                sys.stderr.flush()
                os._exit(1)


def warmup_service(service: Any, input_path: Path, output_dir: Path) -> None:
    """Warm explicit request buckets before opening HTTP; no shape admission policy."""
    from cosmos_framework.inference.args import OmniSampleOverrides

    input_path = input_path.resolve()
    raw = input_path.read_bytes()
    payloads = json.loads(raw)
    if isinstance(payloads, dict):
        payloads = [payloads]
    if not isinstance(payloads, list) or not payloads or not all(isinstance(item, dict) for item in payloads):
        raise ValueError("--warmup-input requires a nonempty Cosmos request object/list")
    with contextlib.chdir(input_path.parent):
        requests = [OmniSampleOverrides.model_validate(item) for item in payloads]
    status = {"requested": True, "completed": False, "input_sha256": hashlib.sha256(raw).hexdigest(), "requests": []}
    service.info["warmup"] = status
    for request in requests:
        timings = []
        for _ in range(2):
            start = time.monotonic()
            prepared, batch = service.prepare(request, output_dir)
            service.predict(prepared, batch)
            timings.append(time.monotonic() - start)
        status["requests"].append(
            {
                "name": request.name,
                "seed": prepared.seed,
                "image_size": prepared.sample_args.get("image_size"),
                "num_frames": prepared.sample_args.get("num_frames"),
                "action_chunk_size": prepared.sample_args.get("action_chunk_size"),
                "guidance": prepared.guidance,
                "return_video": prepared.return_video,
                "conditioning_caption_sha256": hashlib.sha256(batch["ai_caption"][0].encode()).hexdigest(),
                "repeat_seconds": timings,
            }
        )
    status["completed"] = True


def main(sft_profile=None) -> None:
    args = parser().parse_args()
    args.sft_profile = sft_profile
    if args.warmup_input is not None and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("--warmup-input currently requires one GPU per server replica")
    if args.cuda_graphs and not args.compile:
        raise ValueError("--cuda-graphs requires --compile")
    if (args.reuse_small_action_und or args.reuse_small_action_metadata) and int(
        os.environ.get("WORLD_SIZE", "1")
    ) != 1:
        raise ValueError(
            "Small action UND/metadata reuse currently requires one GPU per server replica; "
            "for multi-rank serving, pass --no-reuse-small-action-metadata and leave --reuse-small-action-und off"
        )
    if args.dry_run:
        print(json.dumps(deployment_plan(args), indent=2))
        return

    # Establish inference flags before importing model/runtime modules.
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import torch
    import uvicorn
    from torch import distributed as dist

    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference

    if not torch.accelerator.is_available():
        raise RuntimeError("Action128 serving requires CUDA GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.accelerator.set_device_index(local_rank)
    initialize_single_rank_distributed()
    # Idle serving waits on a CPU group, avoiding NCCL timeout while no request arrives.
    group = dist.new_group(backend="gloo", timeout=timedelta(days=7))
    snapshot = [deployment_plan(args) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(snapshot, src=0, group=group)
    plan = snapshot[0]
    overrides = model_overrides(args)
    setup = OmniSetupOverrides(
        checkpoint_path=plan["checkpoint_path"],
        config_file=plan["config_file"],
        output_dir=args.output_dir.resolve(),
        experiment_overrides=overrides,
        use_ema_weights=plan["ema_weights"],
        use_torch_compile=args.compile,
        compiled_region="language",
        compile_dynamic=False,
        use_cuda_graphs=args.cuda_graphs,
        guardrails=False,
        dp_shard_size=dist.get_world_size(),
        dp_replicate_size=1,
        cp_size=1,
        cfgp_size=1,
    ).build_setup(world_size=dist.get_world_size())
    service = DistributedActionService(OmniInference.create(setup), group, plan)
    if args.compile:
        if args.compile_recompile_limit < 1:
            raise ValueError("--compile-recompile-limit must be positive")
        from cosmos_framework.utils import misc

        misc.set_torch_compile_options(recompile_limit=args.compile_recompile_limit, use_duck_shape=False)
        print(f"Serving effective recompile_limit={torch._dynamo.config.recompile_limit}", flush=True)
    dist.barrier(group=group)
    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.warmup_input is not None:
            warmup_service(service, args.warmup_input, args.output_dir.resolve())
        (args.output_dir / "serving_snapshot.json").write_text(json.dumps(plan, indent=2) + "\n")
        try:
            uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)
        finally:
            if dist.get_world_size() > 1:
                dist.broadcast_object_list([("stop", None, None)], src=0, group=group)
    else:
        service.worker_loop()
    dist.destroy_process_group()
