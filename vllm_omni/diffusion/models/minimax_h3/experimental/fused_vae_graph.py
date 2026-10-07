# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Experimental CUDA graphs around official fused H3 video VAE tiles only.

Temporal/spatial tiling, blending, padding and uint8 writeback remain outside
capture. Only shape-derived, immutable attention masks are cached. Graph
outputs are cloned because the checkpoint mutates/blends decoded tiles.
"""

import json
import os
import sys
import time
from types import MethodType

import torch

from . import graph_runner as common

METRICS = {}
BYPASS = False


def save():
    common.ROOT.mkdir(parents=True, exist_ok=True)
    (common.ROOT / "vae_graph_metrics.json").write_text(json.dumps(METRICS, indent=2))


def install(video_cls):
    """Patch the video adapter explicitly; the caller controls the opt-in gate."""
    Video = video_cls
    for name in (
        "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE",
        "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE",
    ):
        if os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "off"):
            raise RuntimeError(f"Experimental fused VAE graph requires {name}=0")

    if getattr(Video, "_h3_video_fused_graph_enabled", False):
        return
    original_video = Video.decode_latent

    def prepare_video(self):
        if self.parallel_size != 1 or any(
            getattr(self, name, None) is not None for name in ("_stager", "_encoder_stager", "_decoder_stager")
        ):
            raise RuntimeError("Experimental VAE graph requires resident single-GPU weights")
        model = self.model
        if not getattr(model.decoder, "_omni_h3_vae_optimizations_installed", False):
            raise RuntimeError("Install official fused VAE ops before capture")
        if any(b.ff._compile_forward_enabled for b in model.decoder.transformer_blocks):
            raise RuntimeError("Use official fused FF, not native compiled FF, for this experiment")
        namespace = model.decoder.__class__.__module__.rsplit(".", 1)[0]
        flash = sys.modules[namespace + ".flash"]
        original_mask = flash._mask_mod_to_dense
        cache = {}

        def cached_mask(mask_mod, batch, heads, q_len, kv_len, device, aux_tensors=None):
            if BYPASS or aux_tensors is not None:
                return original_mask(mask_mod, batch, heads, q_len, kv_len, device, aux_tensors)
            mask_key = getattr(mask_mod, "block_sparse_cache_key", None)
            if mask_key is None:
                raise RuntimeError("Only shape-derived H3 attention masks can be captured")
            key = (mask_key, batch, heads, q_len, kv_len, str(device))
            # A graph can reuse immutable masks, but cannot build CPU-synchronized
            # attention metadata during capture. Side-stream warmup fills this cache.
            if key not in cache:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Attention metadata must be prepared before graph capture")
                cache[key] = original_mask(mask_mod, batch, heads, q_len, kv_len, device, aux_tensors)
            return cache[key]

        flash._mask_mod_to_dense = cached_mask
        original_decode = model.decode
        runner = common.Runner(
            "video_vae_tile",
            original_decode,
            max_entries=4,
            warmups=2,
            eager_repeats=1,
            replay_repeats=2,
            profile_components=False,
        )

        def decode(owner, z):
            if BYPASS:
                return original_decode(z)
            if z.device.type != "cuda":
                raise RuntimeError("Experimental VAE graph requires CUDA tile inputs")
            return runner.run((z,))

        model.decode = MethodType(decode, model)
        object.__setattr__(self, "_h3_vae_runner", runner)

    def measured(self, latent, original, prepare, name):
        global BYPASS
        if not hasattr(self, "_h3_vae_runner"):
            prepare(self)
        start = time.perf_counter()
        result = original(self, latent)
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        if name not in METRICS:
            try:
                BYPASS = True
                eager, eager_time = common.timed(lambda: original(self, latent), 1)
                BYPASS = False
                graphed, graph_time = common.timed(lambda: original(self, latent), 1)
                METRICS[name] = {
                    "first_call_with_capture_validation_seconds": elapsed,
                    "warm_original_eager": eager_time,
                    "warm_graph": graph_time,
                    "output_check": common.check(graphed, eager),
                    "captured_shapes": len(self._h3_vae_runner.entries),
                    "scope": "official fused post_quant_conv + ViT per tile",
                }
                del eager, graphed
                save()
                print(
                    "H3 full VAE graph comparison:",
                    name,
                    json.dumps(METRICS[name]),
                    flush=True,
                )
            finally:
                BYPASS = False
        METRICS[name]["calls"] = METRICS[name].get("calls", 0) + 1
        METRICS[name]["replay_count"] = self._h3_vae_runner.replays
        save()
        return result

    @torch.inference_mode()
    def video(self, latent):
        return measured(self, latent, original_video, prepare_video, "video_vae")

    Video.decode_latent = video
    Video._h3_video_fused_graph_enabled = True
    print(
        "H3 official fused video VAE tile CUDA graphs enabled; encoder/audio unchanged",
        flush=True,
    )
