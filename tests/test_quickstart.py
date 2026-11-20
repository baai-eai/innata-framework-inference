# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import tyro
from PIL import Image

from cosmos_framework.inference.action import get_action_sample_data
from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.catalog import load_inference_inputs
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.scripts.inference import InferenceArgs
from cosmos_framework.utils.checkpoint_db import CheckpointConfig


def test_quickstart_example_parses():
    example = Path(__file__).resolve().parents[1] / "examples" / "text-to-video.json"
    (sample,) = load_inference_inputs([example])
    assert sample.name == "clouds"
    assert sample.model_mode == "text2video"


def test_wam_example_builds_action_batch(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "wam.json"
    request = tmp_path / "wam.json"
    copyfile(example, request)
    Image.new("RGB", (256, 256)).save(tmp_path / "observation.png")

    (sample,) = load_inference_inputs([request])
    assert sample.model_mode == ModelMode.WAM
    assert Path(sample.vision_path) == tmp_path / "observation.png"
    sample.output_dir = tmp_path / "output"
    sample_args = sample.build_sample(
        model_config=SimpleNamespace(
            resolution="256",
            vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            action_gen=True,
            sound_gen=False,
        )
    )

    batch = get_action_sample_data(
        model_config=SimpleNamespace(input_video_key="video"),
        batch_size=1,
        prompt=sample_args.prompt,
        vision_path=sample_args.vision_path,
        model_mode=sample_args.model_mode,
        action_path=None,
        domain_name=sample_args.domain_name,
        view_point=sample_args.view_point,
        resolution=str(sample_args.image_size),
        action_chunk_size=sample_args.action_chunk_size,
        max_action_dim=64,
        fps=sample_args.fps,
        device="cpu",
    )
    assert batch["mode"] == ["wam"]
    assert batch["action"][0][0].shape == (16, 64)
    assert batch["raw_action_dim"][0].item() == 10
    assert batch["sequence_plan"][0].condition_frame_indexes_vision == [0]
    assert batch["sequence_plan"][0].condition_frame_indexes_action == []


def test_cli_accepts_explicit_dcp_model_config(tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    model = tmp_path / "checkpoint" / "model"
    model.mkdir(parents=True)
    (model / ".metadata").touch()
    (model / "__0_0.distcp").touch()
    config = tmp_path / "model.yaml"
    config.write_text("model: {}\n")

    args = tyro.cli(
        InferenceArgs,
        args=[
            "-i",
            str(request),
            "-o",
            str(tmp_path / "output"),
            "--checkpoint-path",
            str(model.parent),
            "--config-file",
            str(config),
        ],
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    checkpoint = args.setup.build_checkpoint(checkpoints={})
    assert checkpoint.checkpoint_type == CheckpointType.DCP
    assert Path(checkpoint.config_file) == config


def test_wan_registry_does_not_require_object_storage():
    register_checkpoints()
    checkpoint = CheckpointConfig.from_uri("pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth")
    assert checkpoint.s3 is None
    assert checkpoint.hf.repository == "Wan-AI/Wan2.2-TI2V-5B"
