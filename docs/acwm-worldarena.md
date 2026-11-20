# Innata-0-ACWM on WorldArena 2.0 Track 1

`Innata-0-ACWM` scored **67.63** on the 1,000 WorldArena
RoboTwin episodes. The score is the equally weighted mean of 15 metrics on a
0–100 scale. Dynamic degree, optical flow score and motion smoothness are
capped at the corresponding ground-truth normalized score for each episode
before averaging. The exact aggregate of the archived evaluation is
`67.63327617069078`.

## Downloads

- `Innata-0-ACWM` Hugging Face checkpoint with its tokenizer, seven
  safetensors shards and `innata_acwm_profile: "a115"` in `config.json`.
- `Wan2.2_VAE.pth` from the [Wan2.2 release](https://github.com/Wan-Video/Wan2.2).
- The matching WorldArena input bundle with `SHA256SUMS`, 1,000
  prepared images in `first_frames/`, 1,000 normalized absolute
  end-effector action arrays in `actions/`, `prompts-1000.jsonl`,
  `eval-prompts-1000.jsonl` and the three archived reference scores in
  `gt-perdim/`. Use the prepared images for generation. The original
  320×240 first frames were upscaled with Real-ESRGAN x4plus and resized
  to 736×544 with area interpolation.
- The [WorldArena RoboTwin dataset](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0),
  including the original test videos and first frames for scoring.

## Generate videos

Follow the repository installation instructions. Set these arguments to
your local directories and run from the repository checkout.

```bash
repo="$PWD"
checkpoint=/path/to/Innata-0-ACWM
vae=/path/to/Wan2.2_VAE.pth
inputs=/path/to/acwm-a115-worldarena-inputs
run="$PWD/outputs/acwm-worldarena"
(cd "$inputs" && sha256sum -c SHA256SUMS)

CUDA_VISIBLE_DEVICES=0 python -m cosmos_framework.scripts.acwm_worldarena \
  --checkpoint-path "$checkpoint" --vae-path "$vae" --inputs "$inputs" \
  --output-dir "$run/videos" --episode episode576 --episode episode1
```

`episode576` produces 73 frames in one chunk. `episode1` produces 673
frames in three chunks. Remove both `--episode` arguments to generate all
1,000 episodes. To divide the work, launch one process per GPU with the same
`--num-shards` and different zero-based `--shard-index` values. All processes
can write to the same video directory. A rerun skips completed episodes
with matching video metadata. Use a fresh output directory when changing
the checkpoint or input bundle.

The rollout uses 20-channel normalized actions at 16 Hz, the whole-episode
QT R6 caption, UniPC at 30 steps with shift 5, guidance 1, seed 0 and
`torch.compile`. Each episode uses consecutive chunks of up to 300 actions,
with the total rounded down to a multiple of four. The previous chunk's
decoded last frame conditions the next. The output contains the initial
frame and each predicted frame once, encoded at 736×544 and 30 fps.

## Evaluate the complete set

Use the [pinned upstream WorldArena evaluator](https://github.com/WorldArena2/WorldArena-2.0/tree/222f7b9bd225639e88e37f605a1067e9d2621801/video_quality_ood).
Its README specifies the separate video, VLM and JEPA dependencies and
scorer weights. Use Python 3.10 for each environment. The archived run used
PyTorch 2.10.0+cu128 for video metrics, 2.9.1+cu128 for VLM metrics and
2.4.1+cu121 for JEPA. The VLM judge uses Qwen3-VL-8B-Instruct, greedy
decoding and 16 sampled frames. The VLM judge reads the original dataset
instructions from `eval-prompts-1000.jsonl`.

```bash
git clone https://github.com/WorldArena2/WorldArena-2.0.git "$run/evaluator"
git -C "$run/evaluator" checkout 222f7b9bd225639e88e37f605a1067e9d2621801
evaluator="$run/evaluator/video_quality_ood"
cp "$evaluator/config/config.yaml" "$run/scorers.yaml"
```

Fill the `ckpt` entries of `scorers.yaml` with the scorer checkpoints
documented by WorldArena. Replace the upstream path placeholders with
valid YAML strings for your installation. In the pinned checkout,
remove the unused `CACHE_DIR` import from
`WorldArena/subject_consistency.py`. Remove the top-level
`from .action_following import compute_action_following` line in
`WorldArena/__init__.py` if your checkout lacks `action_following.py`.
Table 6 uses the 15 Track 1 metrics. Use the installed JEPA dependency and
weight layout from its setup guide within the pinned checkout. Paths in the
upstream guide beginning with `video_quality/` refer to `video_quality_ood/`
at this revision.

```bash
dataset=/path/to/WorldArena_Robotwin2.0/test_dataset/test_dataset
gt_videos="$dataset/video"
work="$run/scoring"
python -m cosmos_framework.scripts.acwm_worldarena_eval prepare \
  --inputs "$inputs" --videos "$run/videos" --gt-videos "$gt_videos" \
  --gt-first-frames "$dataset/first_frame/fixed_scene_task" --work-dir "$work"

export PYTHONPATH="$repo"
conda run --no-capture-output -n WorldArena \
  python -m cosmos_framework.scripts.acwm_worldarena_eval video \
  --work-dir "$work" --evaluator "$evaluator" --config "$run/scorers.yaml"
conda run --no-capture-output -n WorldArena_VLM \
  python -m cosmos_framework.scripts.acwm_worldarena_eval vlm \
  --work-dir "$work" --evaluator "$evaluator" --config "$run/scorers.yaml" \
  --videos "$run/videos"
conda run --no-capture-output -n WorldArena_JEPA \
  python -m cosmos_framework.scripts.acwm_worldarena_eval jepa \
  --work-dir "$work" --evaluator "$evaluator" --videos "$run/videos" \
  --gt-videos "$gt_videos"

python -m cosmos_framework.scripts.acwm_worldarena_score \
  --video-results "$work/perdim" --gt-results "$inputs/gt-perdim" \
  --vlm-results "$work/vlm" --jepa-result "$work/jepa/results.json" \
  --jepa-pairs "$work/intersection_names.json" --output "$work/score.json"
```

Use the bundle's ground-truth scores to match the reference cap. The video
phase also recomputes the three ground-truth motion metrics in
`$work/gt-perdim`, which can be used to check the cap under your scorer
environment. The final command checks that each metric covers the full
episode set and reports the 15 scores and their mean.

For a smoke evaluation, prepare one generated episode with `--episode
episode576` in a separate scoring work directory. Then run the video phase
with `--metric image_quality`. The final aggregation still requires the
full episode set.

The 67.63 reference comes from the archived run. The rollout command has
been exercised on the two episodes above and the aggregation has been
checked against the full archived results. A fresh full scoring run depends
on obtaining the matching scorer weights and environment.
