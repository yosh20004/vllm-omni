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
