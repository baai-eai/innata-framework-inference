# Reproduce the Innata-0 evaluation

Use the matching Hugging Face checkpoint for each benchmark. Keep the
checkpoint, evaluator sources, assets and case manifest fixed across
runs. Write every benchmark to a new output directory.

| Benchmark | Checkpoint | Evaluation |
| --- | --- | --- |
| Understanding, Table 5 | `Innata-0-Understanding`, 1 tower | 18 reported metrics, including MMSI-Video Uniform-50 |
| WorldArena 2.0 Track 1, Table 6 | `Innata-0-ACWM`, 2 towers | [1,000-episode generation and scoring](acwm-worldarena.md) |
| RoboDojo-Sim, Table 9 | `Innata-0-WAM-RoboDojo`, 3 towers | VLM + WAM, frozen seed-0 layouts and Plan v9 |
| RoboCasa365, Table 8 | `Innata-0-WAM-RoboCasa365`, 3 towers | Target50 closed-loop, command-only |

[Innata-0 technical report](https://arxiv.org/pdf/2611.30254)

## Understanding

Set `UNDERSTANDING_HF` to the local `Innata-0-Understanding` directory.
The evaluation adapter uses the frozen `wm_evaluation` source and its
`models/cosmos_qwen3_vl` adapter at
`7ee247c0d2a1be39631b66d23c26e05fd81ef940`. Populate `models/` in the
evaluator checkout and prepare the benchmark datasets according to its
`DATA_PREPARATION.md`. Use the directory names listed in the bundled
[profiles](../cosmos_framework/inference/defaults/release_eval/understanding.json).
For Table 5, the validated evaluator base is
`d6c2369a832bb161831d32493133a55b405a3e76`.
The extended suite also needs the historical RefSpatial adapter.

Use vLLM 0.19.1+cu130 with PyTorch 2.10.0+cu130, Transformers 4.57.6 and
FlashAttention 3. The vLLM server executable can live in its own
environment. The evaluator reads datasets using its Python environment.
For each `--gpu-id`, the launcher starts one TP=1 service. Use CUDA
device indices visible to the evaluator. The default suite selects the
runners behind the paper's 18 metrics. `--suite extended` runs the
extended handoff suite of 26 metrics.

```bash
python -m cosmos_framework.scripts.eval_understanding standard \
  --evaluator "$WM_EVAL" --vllm "$VLLM_BIN" \
  --checkpoint "$UNDERSTANDING_HF" --data-root "$UNDERSTANDING_DATA" \
  --gpu-id "$GPU" --output-dir "$RUN" --run-name Innata-0-Understanding

python -m cosmos_framework.scripts.eval_understanding video \
  --evaluator "$WM_EVAL" --vllm "$VLLM_BIN" \
  --checkpoint "$UNDERSTANDING_HF" --data-root "$UNDERSTANDING_DATA" \
  --gpu-id "$GPU" --output-dir "$RUN" --run-name Innata-0-Understanding

python -m cosmos_framework.scripts.score_understanding \
  --standard "$RUN/Innata-0-Understanding-standard" \
  --video "$RUN/Innata-0-Understanding-video" --output "$RUN/understanding-scores.json"
```

Run the full BLINK benchmark first by adding `--runner blink` to the
standard command and using a separate run name. It produces BlinkDepth
and BlinkSpatial. Use `--limit` to qualify a handful of cases before
long runs. The reference score checker reports coverage and per-metric
differences. Use `--extended` with corresponding extended results.

## Start a WAM evaluation service

Run one model service per GPU. The offline HF exports contain the EMA
weights. The HTTP service reuses the repository's WAM input contracts
and compiled language path. Use a local copy of `Wan2.2_VAE.pth`.

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m cosmos_framework.scripts.serve_release_wam \
  --checkpoint-path "$WAM_HF" --vae-path "$WAN_VAE" \
  --output-dir "$RUN/service" --port "$WAM_PORT"
```

Run the gateway or the rollout client on the same host, with access to
the model service's input paths. The service binds to loopback by default.
The `/info` endpoint reports the loaded checkpoint and the SFT profile.

## RoboDojo-Sim

Set `WAM_HF` to the local `Innata-0-WAM-RoboDojo` directory.
Use the [RoboDojo simulator](https://github.com/RoboDojo-Benchmark/RoboDojo),
the [public simulator assets](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/1127bda94f172d757386945af3eb3a6754d1a4e6)
and the matching frozen Innata Plan v9 harness. Keep the complete case
list, policy case list, Qwen verifier weights, prompts and per-task
configuration from the release evaluation bundle. The simulator and
Qwen verifier remain separately installed. The layout manifest uses the
official seed 0. The observed layout IDs, corresponding hashes and
benchmark horizon are part of the case identity.

```bash
python -m cosmos_framework.scripts.eval_robodojo layouts \
  --cases "$DOJO_CASES" --assets "$DOJO_ASSETS" \
  --output "$RUN/local-cases.json"

python -m cosmos_framework.scripts.eval_robodojo gateway \
  --harness "$PLAN_V9_HARNESS" --model-url "$WAM_URL" \
  --checkpoint "$WAM_HF" --request-dir "$RUN/gateway" \
  --run-id "$RUN_ID" --port "$GATEWAY_PORT"
```

In the frozen harness, set the WAM endpoint to the gateway URL. Use the
same run ID and checkpoint path as the gateway, the pinned local case
manifest and the frozen Plan v9 verifier. Run the native simulator
against the `DojoPlanHarness` policy with ARX-X5 joint control and the
original layout IDs. The WAM conditions on five frames, predicts 16
actions and executes eight at 25 Hz. It uses 30 UniPC steps, shift 5,
guidance 1 and seed `42 + chunk_index`. Keep the frozen JPEG95 and
episode-first observation replacement rules. The verifier checks
progress every 48 control steps against the original, fixed plans.

Save each episode as `<case_id>/OFFICIAL_RESULT.json` and record native
invalid layouts and simulation instability in the unscored receipt.

```bash
python -m cosmos_framework.scripts.eval_robodojo score \
  --cases "$DOJO_CASES" --results "$RUN/episodes" \
  --unscored "$RUN/unscored.json" --official-scorer "$DOJO_SCORER" \
  --output "$RUN/robodojo-scores.json"
```

The historical comparison uses 2,100 planned cases and 2,084 native
scored episodes. The five-family mean is 23.04 score and 15.98% success.
The scorer also reports the unweighted episode success rate and the
number of cases missing an outcome.

## RoboCasa365

Set `WAM_HF` to the local `Innata-0-WAM-RoboCasa365` directory.
Use the [RoboCasa365 simulator](https://robocasa.ai/) with the frozen
Target50 scenes, original horizons, detailed commands and environment
SDK. Validate the initial simulator state and the three initial RGB
views against the frozen snapshots. The environment service checks
success each control step using the native controller.

The command-only recipe uses H1 State16, JPEG95 camera encoding,
Control12 at 20 Hz, 16 predicted and executed actions, 30 denoising
steps, CFG 2 and seed `42 + chunk_index`. The 29k release handoff
has a mixed-shift profile, with shift 5 for Atomic-Seen and shift 8 for
both Composite groups. `uniform-shift5` evaluates every group at
shift 5.

```bash
python -m cosmos_framework.scripts.eval_robocasa \
  --cases "$CASA_CASES" --checkpoint "$WAM_HF" \
  --env-url "$CASA_ENV_URL" --token-file "$CASA_ENV_TOKEN_FILE" \
  --sdk-zip "$CASA_ENV_SDK" --model-url "$WAM_URL" \
  --protocol mixed-shift --output-dir "$RUN/episodes"

python -m cosmos_framework.scripts.score_robocasa \
  --cases "$CASA_CASES" --results "$RUN/episodes" \
  --checkpoint "$WAM_HF" --protocol mixed-shift \
  --output "$RUN/robocasa-scores.json"
```

For a simulator qualification run, add a known `--case` and
`--max-control-steps 32`. Run with a separate output directory. For the
full 2,500 cases, use one `--shard-index` per independent client with the
same `--num-shards`. Failed or partial episodes must be completed before
the complete-set scorer accepts the result.
