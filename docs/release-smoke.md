<!-- SPDX-FileCopyrightText: Copyright (c) 2026, BAAI -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Release checkpoint smoke tests

Each command runs a single request through the repository's offline inference
entry point, then checks the output. The [fixtures](../examples/release-smoke)
include synthetic observations, request JSON, robot states and ACWM actions.
Use matching local Hugging Face weights. ACWM and both WAM models also need a
local `Wan2.2_VAE.pth`. The smoke tests run with guardrails disabled for
these bundled inputs.

## Install

From this repository's root on Linux with an NVIDIA CUDA 13 environment,
Python 3.13 and uv 0.11.3 or newer, run the following command. Installation
requires access to the wheel indexes configured by the repository.

```bash
uv sync --no-dev --group cu130
```

## Run a checkpoint

Set `INNATA_CHECKPOINT` to the local weights for the model in the heading.
For the last three commands, also set `WAN_VAE` to the local VAE file.
Choose an idle GPU and run the matching command.

### Innata-0-Understanding

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python examples/release-smoke/run.py understanding \
  --checkpoint "$INNATA_CHECKPOINT"
```

The runner checks for a nonempty text response to a single-image question.

### Innata-0-ACWM

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python examples/release-smoke/run.py acwm \
  --checkpoint "$INNATA_CHECKPOINT" --vae "$WAN_VAE"
```

The request supplies one 736×544 image and 16 normalized 20-channel actions.
It uses UniPC 30 steps, CFG 1, shift 5, seed 0 and `torch.compile`. The
runner checks `vision.mp4` for 17 frames at 16 fps.

### Innata-0-WAM-RoboDojo

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python examples/release-smoke/run.py robodojo \
  --checkpoint "$INNATA_CHECKPOINT" --vae "$WAN_VAE"
```

The request supplies five three-camera observations and five Joint14 states.
The runner checks that the native and normalized outputs each contain 16
finite Joint14 actions at the request's 25 Hz.

### Innata-0-WAM-RoboCasa365

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python examples/release-smoke/run.py robocasa \
  --checkpoint "$INNATA_CHECKPOINT" --vae "$WAN_VAE"
```

The request supplies a three-camera observation, a State16 vector and a
task instruction. The runner checks that the native and normalized outputs
each contain 16 finite Control12 actions at the request's 20 Hz.

Both WAM commands compile the static language region. Their requests include
the camera layout, state history and sampling parameters.

## Read the result

Each run gets a fresh subdirectory under `outputs/release-smoke/`, or
under the directory given by `--output`. The runner prints its location
and returns a nonzero exit code on inference or validation failure.
Inspect `sample_outputs.json` in the case subdirectory for the response.
The smoke tests check the inference interface and output format on a
fixed synthetic request. For benchmark inputs and scoring, see
[release evaluation](release-evaluation.md).

## Use your own request

To try your own media, copy a bundled JSON request and its input files
into one directory, then update the JSON. Media paths resolve relative to
the JSON file. Run the offline CLI with your request and a fresh output
directory.

```bash
uv run --no-sync innata-infer -i "$REQUEST" -o "$OUTPUT" \
  --checkpoint-path "$INNATA_CHECKPOINT"
```

For ACWM, add `--vae-path "$WAN_VAE" --use-torch-compile`. For either
release WAM, add `--vae-path "$WAN_VAE" --use-torch-compile
--compiled-region language --no-compile-dynamic`. Use the same camera
layout and state format as the bundled example for that model.
