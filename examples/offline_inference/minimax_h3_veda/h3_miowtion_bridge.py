"""Experiment-only Larry Turbo + Miowtion adapter for pinned vLLM-Omni H3.

Miowtion is used as a pinned external library; its selector and sparse kernel
are unmodified. The experiment supports one resident, unquantized Ref2VA DiT
on one CUDA device. Import has no model patching side effects.
"""

import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

MODE = "dense"
BUNDLE = None
STUDENTS = {}
COUNTS = {"dense_trunk": 0, "sparse_trunk": 0, "dense_refiner": 0}
EVIDENCE = {"layouts": [], "lora": None}


@dataclass(frozen=True)
class LayoutView:
    """The structural subset of PackedLayout consumed by ClipTiling."""

    seq_len: int
    used: int
    spans: tuple

    @property
    def target(self):
        return self.spans[-1]


def adapt_layout(video_layout, seq_len, used):
    from miowtion.h3.layout import VideoSpan

    if not 0 < used <= seq_len:
        raise ValueError((used, seq_len))
    raw = video_layout.video_spans
    if not raw:
        raise ValueError("Ref2VA must publish all physical visual spans")
    spans = tuple(VideoSpan(s.start, tuple(s.latent_grid), s.role) for s in raw)
    if video_layout.used_len != used:
        raise ValueError("Attention length and video layout disagree")
    if sum(s.role == "target" for s in spans) != 1:
        raise ValueError("Expected exactly one target video")
    previous_stop = 0
    for span in spans:
        if span.start < previous_stop or any(d <= 0 for d in span.grid):
            raise ValueError("Invalid/overlapping visual spans")
        previous_stop = span.stop
    if spans[-1].role != "target" or spans[-1].stop != used:
        raise ValueError("Target must be the final valid token span")
    return LayoutView(seq_len, used, spans)


def turbo_sigmas(*, num_steps=8, shift_scale=12.0, base_schedule=None):
    from miowtion.h3.schedule import ShiftScales, turbo_schedule

    if num_steps != 8 or shift_scale not in (12.0, 3.0):
        raise ValueError("This experiment pins Turbo 8, video shift 12/audio 3")
    if base_schedule is not None and tuple(base_schedule) != tuple(1.0 - i / 8 for i in range(9)):
        raise ValueError("Unexpected sigma positions")
    schedule = turbo_schedule(8, ShiftScales(12.0, 3.0))
    return list(schedule.video if shift_scale == 12.0 else schedule.audio)


def expected_lora_targets():
    projection = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    return {
        *(f"blocks.{i}.{suffix}" for i in range(50) for suffix in (*projection, "adaln_proj.linear")),
        *(f"token_refiner.blocks.{i}.{suffix}" for i in range(2) for suffix in projection),
        "final_layer.adaln_proj.linear",
    }


def validate_lora(checkpoint, parameters):
    keys = set(checkpoint.keys())
    expected = expected_lora_targets()
    expected_keys = {f"{name}.lora_{side}.weight" for name in expected for side in "AB"}
    if keys != expected_keys:
        raise ValueError(f"Unexpected LoRA keys: {keys ^ expected_keys}")
    metadata = checkpoint.metadata() or {}
    if metadata.get("application") != "W_eff = W + lora_B @ lora_A":
        raise ValueError(f"Unknown Larry LoRA scaling: {metadata}")
    for name in sorted(expected):
        a = checkpoint.get_slice(name + ".lora_A.weight").get_shape()
        b = checkpoint.get_slice(name + ".lora_B.weight").get_shape()
        p = parameters[name + ".weight"]
        rank = 16 if "adaln_proj" in name else 64
        if a[0] != b[1] or a[0] != rank or tuple(p.shape) != (b[0], a[1]):
            raise ValueError(f"LoRA shape mismatch {name}: {a}, {b}, {p.shape}")
        if p.dtype != torch.bfloat16:
            raise ValueError(f"LoRA target is not unquantized BF16: {name}")
    return sorted(expected)


@torch.no_grad()
def merge_tensor(weight, a, b):
    # Same operation order as Miowtion train.lora.merged_weight. At this
    # boundary both implementations have QKV=[all Q; all K; all V] and
    # native MLP fc1=[gate; up], so no second permutation is appropriate.
    delta = b.to(weight.device, torch.float32) @ a.to(weight.device, torch.float32)
    weight.copy_((weight.float() + delta).to(weight.dtype))


@torch.no_grad()
def merge_lora(model, path):
    started = time.monotonic()
    parameters = dict(model.named_parameters())
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        targets = validate_lora(checkpoint, parameters)
        for index, name in enumerate(targets):
            weight = parameters[name + ".weight"]
            if weight.device.type != "cuda":
                raise ValueError("This experiment requires resident CUDA weights")
            merge_tensor(
                weight, checkpoint.get_tensor(name + ".lora_A.weight"), checkpoint.get_tensor(name + ".lora_B.weight")
            )
            if (index + 1) % 50 == 0:
                print(f"Larry Turbo merge: {index + 1}/{len(targets)}", flush=True)
    torch.cuda.synchronize()  # noqa: TID251 - CUDA-only experiment
    if getattr(model, "adaln_cache", None) is not None:
        model.adaln_cache.clear()
    return {
        "merged_targets": len(targets),
        "scale": 1.0,
        "method": "FP32 W + B@A, rounded back to BF16 after base load",
        "seconds": time.monotonic() - started,
    }


def prepare_bundle():
    global BUNDLE
    if BUNDLE is None:
        from miowtion.veda import bundle

        # Predictor initialization must not perturb generation RNG state.
        with torch.random.fork_rng(devices=[]):
            BUNDLE = bundle.load(os.environ["H3_VEDA_PREDICTOR"], "cuda")
        assert (len(BUNDLE.predictor.layers), BUNDLE.metadata["num_heads"], BUNDLE.metadata["head_dim"]) == (
            50,
            "56",
            "128",
        )
        assert BUNDLE.tile_conditions
        assert BUNDLE.target_budget.tiles == BUNDLE.ref_budget.tiles == 32
        print("Loaded Ref2VA Veda selector: 50 layers, 56 heads, 32/32 tiles", flush=True)


def student_for(video_layout, seq_len, used, device, *, prefix_backend=None):
    from miowtion.h3.geometry import Geometry
    from miowtion.veda.attention import ClipTiling, SparseStudent, VedaConfig

    view = adapt_layout(video_layout, seq_len, used)
    if prefix_backend is None:
        prefix_backend = os.environ.get("H3_VEDA_PREFIX_BACKEND", "none")
    if prefix_backend not in ("none", "fa4", "trtllm"):
        raise ValueError("H3_VEDA_PREFIX_BACKEND must be none, fa4 or trtllm")
    key = (seq_len, used, tuple((s.start, s.grid, s.role) for s in view.spans), str(device), prefix_backend)
    if key not in STUDENTS:
        prepare_bundle()
        t, h, w = view.target.grid
        divisor = math.gcd(w, h)
        geometry = Geometry(
            f"{w // divisor}:{h // divisor}", w * 32, h * 32, 17 * ((t - 2) // 5) + 5, t, h * 2, w * 2, 0
        )
        plan = BUNDLE.plans.select(geometry)
        # Keep the published per-head plan; its grid is calibration provenance.
        # ClipTiling constructs permutations using the actual requested grid.
        config = VedaConfig(
            target_budget=BUNDLE.target_budget,
            ref_budget=BUNDLE.ref_budget,
            tile_conditions=True,
            collect_bytes=2048 * 2**20,
        )
        clip = ClipTiling(view, config, device)
        for layer in range(50):
            for group in plan.head_groups(layer, device):
                clip.get(group.shape)
        student = SparseStudent(clip, plan, BUNDLE.predictor)
        if prefix_backend != "none":
            from h3_prefix_split import SplitStudent

            student = SplitStudent(student, prefix_backend)
        STUDENTS[key] = student
        evidence = {
            "seq_len": seq_len,
            "used": used,
            "spans": [{"start": s.start, "grid": s.grid, "role": s.role} for s in view.spans],
            "selected_plan": plan.geometry,
            "calibration_grid": plan.grid,
            "actual_grid": view.target.grid,
            "collect_bytes": config.collect_bytes,
            "tile_shapes": [str(s) for s in plan.shapes],
            "prefix_backend": prefix_backend,
        }
        EVIDENCE["layouts"].append(evidence)
        print("Veda layout:", json.dumps(evidence), flush=True)
    return STUDENTS[key]


def snapshot():
    return {
        "mode": MODE,
        "calls": dict(COUNTS),
        **EVIDENCE,
        "predictor_loaded": BUNDLE is not None,
        "video_sigmas": turbo_sigmas(),
        "audio_sigmas": turbo_sigmas(shift_scale=3.0),
    }


def install():
    from vllm.distributed import get_tensor_model_parallel_world_size

    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as dit
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipe

    if getattr(dit.MiniMaxH3DiTModel, "_h3_larry_patch", False):
        return
    original_post = dit.MiniMaxH3DiTModel.post_load_weights
    original_attention = dit.MiniMaxH3Attention._run_packed_attention
    assert "video_layout" in inspect.signature(original_attention).parameters

    def post_load(model):
        original_post(model)
        if getattr(model, "_h3_larry_merged", False):
            return
        assert get_tensor_model_parallel_world_size() == 1
        assert not getattr(model, "_diffusers_weights", False)
        assert model.arch.num_layers == 50
        EVIDENCE["lora"] = merge_lora(model, os.environ["H3_LARRY_LORA"])
        model._h3_larry_merged = True
        for index, block in enumerate(model.blocks):
            block.attn._h3_veda_layer = index
        save_path = Path(os.environ["H3_ENCODER_GRAPH_DIR"]) / "lora_merge.json"
        save_path.write_text(json.dumps(EVIDENCE["lora"], indent=2))

    @torch.compiler.disable
    def attention(
        self,
        q,
        k,
        v,
        *,
        cu_seqlens,
        max_seqlen,
        packed_total,
        num_requests=1,
        video_layout=None,
        vsa_prefix_segments=(),
        gate_compress=None,
    ):
        layer = getattr(self, "_h3_veda_layer", None)
        if layer is None or MODE == "dense":
            COUNTS["dense_refiner" if layer is None else "dense_trunk"] += 1
            return original_attention(
                self,
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                packed_total=packed_total,
                num_requests=num_requests,
                video_layout=video_layout,
                vsa_prefix_segments=vsa_prefix_segments,
                gate_compress=gate_compress,
            )
        assert MODE == "veda" and num_requests == 1 and gate_compress is None
        assert q.shape == k.shape == v.shape == (packed_total, 56, 128)
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert video_layout is not None
        student = student_for(video_layout, packed_total, max_seqlen, q.device)
        COUNTS["sparse_trunk"] += 1
        # Broad annotation includes selector, permutations, list construction
        # and sparse kernel, so trace can account for their combined cost.
        with torch.profiler.record_function("H3Veda.attention"):
            return student(q, k, v, layer)

    def resolve_sigmas(self, task, sampling):
        if task != "ref2va" or int(sampling.num_inference_steps) != 8:
            raise ValueError("Larry/Veda experiment requires Ref2VA, 8 steps")
        return tuple(1.0 - i / 8 for i in range(9)), 8

    dit.MiniMaxH3DiTModel.post_load_weights = post_load
    dit.MiniMaxH3DiTModel._h3_larry_patch = True
    dit.MiniMaxH3Attention._run_packed_attention = attention
    pipe.MiniMaxH3Pipeline._resolve_sigma_positions = resolve_sigmas
    pipe.minimax_h3_time_shift_sigmas = turbo_sigmas
    print("Installed experiment Larry Turbo merge and Miowtion attention bridge", flush=True)
