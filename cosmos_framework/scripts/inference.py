# Mainly adapted from https://github.com/NVIDIA/cosmos-framework/blob/c14617c2bc93dacbf69674fb964eec93182933d9/cosmos_framework/scripts/inference.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.inference.common.init import init_script, is_rank0

init_script(env={"COSMOS_TRAINING": "0", "COSMOS_INTERNAL": "0"})

import json
from pathlib import Path
from typing import Annotated

import pydantic
import tyro

from cosmos_framework.inference.args import OmniSetupOverrides, is_reasoner_only
from cosmos_framework.inference.common.args import SampleOutputs, SetupOverrides, tyro_cli
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.release_models import (
    ACWM_CHECKPOINT_NAME,
    UNDERSTANDING_CHECKPOINT_NAME,
    WAM_CHECKPOINT_NAMES,
)
from cosmos_framework.utils import log


class InferenceArgs(pydantic.BaseModel):
    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s) or a unified visual-generation Parquet catalog.

    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.

    Accepts glob patterns (e.g. `inputs/*.json`) and Parquet catalog directories.
    """

    vae_path: Path | None = None
    """Optional local video VAE file for native Cosmos checkpoints."""

    setup: SetupOverrides = OmniSetupOverrides.model_construct()
    """Setup arguments."""


def inference(args: InferenceArgs):
    from cosmos_framework.inference.catalog import load_inference_inputs
    from cosmos_framework.inference.common.inference import sync_distributed_errors
    from cosmos_framework.inference.release_models import generate_understanding, generate_wam, read_hf_config

    release_config = read_hf_config(args.setup.checkpoint_path)
    if release_config.get("model_type") == "qwen3_vl":
        if args.setup.output_dir is None:
            raise ValueError("'output_dir' is required")
        samples = load_inference_inputs(args.input_files)
        generate_understanding(args.setup.checkpoint_path, samples, Path(args.setup.output_dir))
        return
    profile = release_config.get("innata_wam_profile")

    with sync_distributed_errors():
        if args.setup.output_dir is None:
            raise ValueError("'output_dir' is required")
        setup_args = args.setup.build_setup()
        if args.vae_path is not None:
            vae = args.vae_path.expanduser().resolve(strict=True)
            if not vae.is_file():
                raise ValueError("--vae-path must point to a VAE checkpoint file")
            setup_args.experiment_overrides.extend(
                [
                    "model.config.tokenizer.bucket_name=''",
                    f"model.config.tokenizer.vae_path={json.dumps(str(vae))}",
                ]
            )
        init_output_dir(setup_args.output_dir)
        log.debug(f"{args.__class__.__name__}({args})")
        sample_overrides_list = load_inference_inputs(args.input_files, global_overrides=setup_args.sample_overrides)
        log.info(f"Loaded {len(sample_overrides_list)} samples")
        if is_reasoner_only(sample_overrides_list):
            setup_args.experiment_overrides.append("model.config.load_vision_tokenizer=false")
            log.info("Reasoner-only inputs detected; generation vision tokenizer will not be loaded")
        for sample_overrides in sample_overrides_list:
            assert sample_overrides.name
            sample_overrides.output_dir = setup_args.output_dir / sample_overrides.name
            sample_overrides.download(sample_overrides.output_dir / "inputs")

    use_release_wam = profile and any(s.sample_meta.model_mode.value == "wam" for s in sample_overrides_list)
    if use_release_wam:
        if any(s.sample_meta.model_mode.value != "wam" for s in sample_overrides_list):
            raise ValueError("Put release WAM and video requests in separate invocations")
        setup_args.experiment_overrides.extend(
            [
                "model.config.fixed_sequence_length.enabled=false",
                "model.config.action_tower_und_short_tokens=0",
                "model.config.action_tower_init=checkpoint",
            ]
        )
    if profile == "robodojo":
        # The archived training loss schedule is unused for sampling.
        setup_args.experiment_overrides.append("model.config.rectified_flow_training_config.train_time_weight=uniform")
    pipe = setup_args.get_inference_cls().create(setup_args)
    if use_release_wam:
        generate_wam(profile, pipe, sample_overrides_list, setup_args.output_dir, str(args.setup.checkpoint_path))
        return
    sample_args_list = []
    for overrides in sample_overrides_list:
        try:
            sample_args_list.append(overrides.build_sample(model_config=pipe.model_config))
        except ValueError as e:
            if not setup_args.skip_invalid_samples:
                raise
            msg = f"Skipping sample '{overrides.name}': {e}"
            log.warning(msg)
            overrides.output_dir.mkdir(parents=True, exist_ok=True)
            skip_output = SampleOutputs(args=overrides.model_dump(mode="json"), status="skip", message=msg)
            (overrides.output_dir / "sample_outputs.json").write_text(skip_output.model_dump_json())
    pipe.generate(sample_args_list)

    if setup_args.benchmark and is_rank0():
        benchmark_file = setup_args.output_dir / "benchmark.json"
        benchmark_file.write_text(json.dumps(pipe.get_timer_results(), indent=2, sort_keys=True))
        log.success(f"Saved benchmark to '{benchmark_file}'")


def main():
    release_names = ", ".join((UNDERSTANDING_CHECKPOINT_NAME, ACWM_CHECKPOINT_NAME, *WAM_CHECKPOINT_NAMES.values()))
    args = tyro_cli(
        InferenceArgs,
        description=f"Run model inference. Innata-0 Hugging Face checkpoint directories include {release_names}.",
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    inference(args)


if __name__ == "__main__":
    main()
