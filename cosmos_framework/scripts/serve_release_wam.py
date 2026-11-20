# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Serve an Innata RoboDojo or RoboCasa HF EMA checkpoint for closed-loop evaluation."""

import argparse
import json
import os
from pathlib import Path

from cosmos_framework.inference.release_models import WAM_CHECKPOINT_NAMES, read_hf_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    release_names = " or ".join(WAM_CHECKPOINT_NAMES.values())
    parser.add_argument("--checkpoint-path", type=Path, required=True, help=f"Local HF directory for {release_names}")
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint_path.resolve(strict=True)
    vae = args.vae_path.resolve(strict=True)
    profile = read_hf_config(checkpoint).get("innata_wam_profile")
    if profile not in WAM_CHECKPOINT_NAMES or not (checkpoint / "model.safetensors.index.json").is_file():
        parser.error(f"--checkpoint-path must be an exported {release_names} HF EMA checkpoint")
    if not vae.is_file() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("use a local Wan2.2 VAE and one process per GPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from cosmos_framework.inference.common.init import init_script

    init_script(env={"COSMOS_TRAINING": "0", "COSMOS_INTERNAL": "0"})

    import torch
    import uvicorn

    from cosmos_framework.inference.action_server import DistributedActionService, initialize_single_rank_distributed
    from cosmos_framework.inference.action_serving import create_app
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.robocasa_serving import RoboCasaService
    from cosmos_framework.utils import misc

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    overrides = [
        "model.config.ema.enabled=false",
        "model.config.fixed_sequence_length.enabled=false",
        "model.config.action_tower_und_short_tokens=0",
        "model.config.action_tower_init=checkpoint",
        "model.config.action_tower_attention_backend=flash3",
        "model.config.compile.mix_order_reduction_split_size=null",
        "model.config.tokenizer.bucket_name=''",
        f"model.config.tokenizer.vae_path={json.dumps(str(vae))}",
    ]
    if profile == "robodojo":
        overrides.append("model.config.rectified_flow_training_config.train_time_weight=uniform")
    output = args.output_dir.resolve()
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        output_dir=output,
        experiment_overrides=overrides,
        use_torch_compile=True,
        compiled_region="language",
        compile_dynamic=False,
        use_cuda_graphs=False,
        guardrails=False,
        dp_shard_size=1,
        dp_replicate_size=1,
        cp_size=1,
        cfgp_size=1,
    ).build_setup()
    info = {
        "checkpoint_path": str(checkpoint),
        "output_dir": str(output),
        "weight_variant": "ema",
        "ema_weights": True,
        "compile": True,
        "reuse_small_action_metadata": True,
    }
    if profile == "robodojo":
        info["sft_profile"] = profile
    initialize_single_rank_distributed()
    try:
        pipe = setup.get_inference_cls().create(setup)
        service_class = RoboCasaService if profile == "robocasa" else DistributedActionService
        service = service_class(pipe, None, info)
        misc.set_torch_compile_options(recompile_limit=256, use_duck_shape=False)
        uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
