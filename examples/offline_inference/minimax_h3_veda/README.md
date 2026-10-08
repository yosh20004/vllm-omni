# MiniMax-H3 Veda experiment

Opt-in experiment for resident, unquantized, single-GPU Ref2VA inference.
Importing these modules does not install model patches. This is independent
of Omni's built-in Turbo/LightX2V LoRA support.

## Dependencies

The original B300 experiment used these exact revisions:

| Component | Revision |
| --- | --- |
| vLLM-Omni base | `65699fe94c355e235f1acabddd866bce0bba1c02` |
| Miowtion | `20da18c38e6ab29a726d0ce83342d2c83df9654b` |
| FlashAttention CuTe | `d15f1531a460ba456f41b01a774f33ab2db8febf` |
| MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Larry Turbo LoRA | `43a74557ac3f6539db8e0f2a959d03feb7a81480` |
| Veda predictor | `1598b407905b7c0af85aa0c13acdfa6f01ff2982` |

Runtime: Torch 2.13.0+cu130, vLLM 0.31.0, transformers 5.14.1,
diffusers 0.40.0, Triton 3.7.1, apache-tvm-ffi 0.1.12, FA4 4.0.0b32.
Install Miowtion and FA4 at the revisions above; neither is vendored here.
FA4 on B300 uses the upstream SM100 kernel, with Miowtion's block lists.

Download the assets separately, preserving their pinned revisions:

- `larryvrh/MiniMax-H3-Turbo-Lora`:
  `minimax_h3_turbo_v4_step600_ema.safetensors` → `assets/larry_turbo.safetensors`.
- `Veda-Sparse/Minimax-H3-R2VA-Veda-Preview`:
  `minimax_h3_r2va_veda_preview_fp8.safetensors` → `assets/predictor.safetensors`.

## Bridge

`h3_miowtion_bridge.install()` must run in the model worker before weight
loading. It merges 259 Larry LoRA targets using FP32 `W + B @ A`, rounds to
BF16, installs the published video/audio Turbo8 schedules (shifts 12/3),
and optionally routes trunk attention through the pretrained Veda selector.
Token refiners retain Omni's original dense attention.

Set `H3_LARRY_LORA` and `H3_VEDA_PREDICTOR` to the downloaded files.
Set `H3_ENCODER_GRAPH_DIR` to an existing output directory for merge evidence.
In that same worker, set `h3_miowtion_bridge.MODE` to `"dense"` or `"veda"`.
The default is `"dense"`; even this mode uses Larry weights after `install()`.
No serving CLI or automatic installation is added by this experiment.

Only batch size 1, 56 heads, head dimension 128, and 8 Ref2VA denoise steps
are supported. The published per-head plan and 32/32 tile budgets are kept.
The measured 576×1024 geometry transfers a plan calibrated at 768p; quality
equivalence is not established.

Before allocating a GPU, run the checkpoint/layout/schedule checks:

```bash
python examples/offline_inference/minimax_h3_veda/h3_miowtion_preflight.py \
  --assets assets --output cpu_preflight.json
```

The same script with `--gpu --output gpu_preflight.json` checks B300
gather/pooling, partial tiles and several sparse masks without H3 weights.

## Existing full-model evidence

On B300, 576×1024, 243 frames, Turbo8, the same merged weights were used for
dense warmup, dense timing, Veda warmup, Veda timing and Veda trace, in that
order. Each video has eight steps. One bare timing per path:

| Path | Denoise | HTTP request | Video VAE |
| --- | ---: | ---: | ---: |
| Dense | 26.3391 s | 29.9592 s | 2.78085 s |
| Veda | 18.8902 s | 22.4611 s | 2.78191 s |

This run also enabled regional compilation and the fused VAE tile graph
experiment. It captured only a Veda trace, not a matching Turbo8 dense trace.
These results do not establish video quality parity or statistical variance.
Weights, videos and raw traces are deliberately excluded from Git.

## Dense-prefix routing

Set `H3_VEDA_PREFIX_BACKEND=fa4` to enable `SplitStudent` when the bridge is
in `veda` mode. `trtllm` is an alternative requiring FlashInfer's
`trtllm_ragged_attention_deepseek`; `none` (default) preserves the original
Miowtion path. The student cache distinguishes backend and device.

The supported layout has exactly one final target video span and a nonempty
global prefix. Other layouts raise an error. Routing splits **query rows**:

- Prefix queries use dense attention over every valid key, once across all heads.
- Video queries use the original selector and sparse attention, retaining all
  global keys in the same softmax as the selected video keys.
- Only video results are scattered; the dense prefix is copied into its
  original positions. Tail padding remains zero. No LSE merge is needed.

The scatter-only layout view sets `n_global_tiles=0`. It must never be passed
to the selector or attention: doing so would remove required global keys.
Layers explicitly configured as dense retain their full dense path.
The adapter is inference-only; it does not build a predictor autograd graph.

Run CPU routing tests with the pinned Miowtion package installed:

```bash
python -m pytest -o addopts='' -q \
  examples/offline_inference/minimax_h3_veda/test_prefix_split.py
```

These tests replace GPU kernel boundaries with mathematical CPU references;
they cover routing semantics, masks, partial tiles, head chunking, padding,
backend selection and invalid layouts. They do not validate GPU code generation.
The prefix split has prior B300 synthetic validation, but the newly exposed
bridge option has not been measured in a full-model generation.

## Reproduce the prefix benchmark

On a B300 with the pinned runtime, run:

```bash
python examples/offline_inference/minimax_h3_veda/h3_prefix_bench.py \
  --predictor assets/predictor.safetensors --output outputs/prefix-run
python examples/offline_inference/minimax_h3_veda/h3_prefix_analyze.py \
  outputs/prefix-run
```

The output directory must not exist. The runner loads only the predictor,
uses synthetic BF16 Q/K/V of shape `[44096, 56, 128]`, and checks small
partial-tile cases before warming all 50 actual layer plans. It compares the
original sparse path, dense-FA4 prefix, and dense-TRTLLM prefix. Each path
runs three rotating-order bare measurements, followed by a diagnostic trace
of layers 0, 1 and 17. It does not load the H3 model or save Q/K/V.
The original path explicitly disables prefix routing, even if the enabling
environment variable is set. The benchmark requires both dense backends.

Outputs include `preflight.json`, `benchmark.json`, `trace.json.gz` and,
after analysis, `trace_analysis.json`. The analysis uses CPU-launch
correlation and layer annotations: FA4 dense and sparse kernels share names,
so names alone cannot distinguish them. GPU busy time is interval coverage,
not hardware utilization.

### Existing B300 measurements (2026-10-08)

These measurements predate code extraction into this branch. Each method
includes selector, gather, block-list preparation, attention, scatter and
prefix copy. Median wall time for 50 layers, three measurements:

| Method | Time | Speedup |
| --- | ---: | ---: |
| Original Miowtion | 994.015 ms | 1.000× |
| Dense FA4 prefix + sparse video | 799.769 ms | 1.243× |
| Dense TRTLLM prefix + sparse video | 817.504 ms | 1.216× |

All 50 layers retained bitwise-equal video-query outputs. Prefix maximum
absolute error was 0.00024414, relative MSE approximately 9.94e-6; full-output
relative MSE was below 9.2e-8. Dense prefix reduction order differs from
tile-ordered sparse attention. Small cases also passed against an FP32
reference at Q/K scales 1 and 2. These results use synthetic activations
reused across layers, not the actual denoising trajectory. The speedups
are attention microbenchmark results, not end-to-end video speedups.

### Sparse kernel configuration findings

Separate synthetic kernel trials after removing prefix queries did not
improve on the original M=128, N=128, Q-stage=1, 1CTA configuration:

| Candidate | Outcome versus baseline |
| --- | --- |
| N=64 | 1.13–1.16× slower |
| Q-stage=2, union two query masks | 5.38–5.61× slower |
| 2CTA, M=128, union two query masks | 5.82–6.07× slower |
| 2CTA and Q-stage=2, union four masks | 9.16–9.56× slower |
| M=64, Q-stage=2 or 2CTA | TMEM layout/alignment compilation failures |

The union candidates preserved each original 128-row mask with a fine mask,
but increased work. These are negative results for that adaptation, not a
general limit on sparse 2CTA. Two CTAs with 64 rows each could share one
128-row selection; the tested upstream TMEM path did not compile for it.
No experimental FA4 patches or configuration changes are enabled here.

This branch also retains the earlier
[fused VAE tile CUDA Graph experiment](../minimax_h3_fused_vae_graph/README.md).
