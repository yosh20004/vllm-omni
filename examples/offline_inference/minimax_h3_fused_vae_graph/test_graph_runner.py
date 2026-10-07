# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-CUDA leaf/runner tests without weights or a vLLM installation.

The import-only Triton adapter does not mock kernel execution. SM86 bypasses
upstream hardware dispatch and tests two leaves, not full checkpoint decode.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]
GRAPH_PACKAGE = SOURCE / "vllm_omni/diffusion/models/minimax_h3/experimental"
OPS = SOURCE / "vllm_omni/diffusion/models/minimax_h3/ops/vae"


def load_official_leaf(name):
    """Supply the missing dependency's Triton imports, not its operators."""
    adapter = types.ModuleType("vllm.triton_utils")
    adapter.HAS_TRITON = True
    adapter.triton = triton
    adapter.tl = tl
    package = types.ModuleType("vllm")
    package.__path__ = []
    spec = importlib.util.spec_from_file_location("local_h3_" + name, OPS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, {"vllm": package, "vllm.triton_utils": adapter}):
        spec.loader.exec_module(module)
    return module


def error_stats(actual, expected):
    return {
        "exact": torch.equal(actual, expected),
        "max_abs": (actual.float() - expected.float()).abs().max().item(),
    }


def qk_reference(value, cos, sin):
    normalized = F.rms_norm(value.float(), (64,), None, 1e-5).to(value.dtype)
    rotary = normalized[..., :48]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, normalized[..., 48:]), dim=-1)


def make_inputs(batch=1, sequence=195):
    torch.manual_seed(1700 + batch + sequence)
    qkv = torch.randn(batch, sequence, 32, 192, device="cuda", dtype=torch.float16)
    angles = torch.randn(batch, sequence, 1, 48, device="cuda")
    cos, sin = angles.cos().half(), angles.sin().half()
    residual = torch.randn(batch, sequence, 2048, device="cuda")
    scale = torch.randn(2048, device="cuda") * 0.1
    return qkv, cos, sin, residual, scale


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    assert torch.cuda.is_available(), "A real CUDA device is required; do not count skips as validation"
    artifacts = tmp_path_factory.mktemp("h3_fused_graph")
    spec = importlib.util.spec_from_file_location("h3_learning_graph_runner", GRAPH_PACKAGE / "graph_runner.py")
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    runner_module.ROOT = artifacts
    qk = load_official_leaf("qk_norm_rope")
    residual = load_official_leaf("scaled_residual")
    source_files = [OPS / "qk_norm_rope.py", OPS / "scaled_residual.py", GRAPH_PACKAGE / "graph_runner.py"]
    evidence = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_commit": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "hardware_dispatch_bypassed": True,
        "official_kernel_source_modified": False,
        "scope": "two official fused leaves + synthetic SDPA chain + extracted graph runner",
        "not_validated": ["vLLM fused SwiGLU", "checkpoint masks/tile decode", "full model", "B300 performance"],
        "checks": {},
    }
    yield types.SimpleNamespace(
        qk=qk, residual=residual, runner_module=runner_module, artifacts=artifacts, evidence=evidence
    )
    evidence["peak_cuda_allocated_bytes"] = torch.accelerator.max_memory_allocated()
    evidence["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (artifacts / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


@pytest.mark.parametrize("sequence", [1, 195, 1797])
@torch.inference_mode()
def test_official_qk_kernel_matches_reference(experiment, sequence):
    qkv, cos, sin, _, _ = make_inputs(sequence=sequence)
    q, k, _ = qkv.chunk(3, dim=-1)
    outputs = experiment.qk.try_qk_norm_rope_exact(q, k, (cos, sin), 1e-5)
    assert outputs is not None, "Fused operator unexpectedly fell back"
    checks = []
    for actual, value in zip(outputs, (q, k), strict=True):
        expected = qk_reference(value, cos, sin)
        checks.append(error_stats(actual, expected))
        # SM86 is not an upstream bit-exactness-validated target; record exactness
        # separately instead of assuming it from the SM90/SM103 contract.
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=2e-3)
    experiment.evidence["checks"][f"qk_reference_sequence_{sequence}"] = checks


@torch.inference_mode()
def test_official_scaled_residual_is_exact(experiment):
    _, _, _, residual, scale = make_inputs()
    branch = torch.randn_like(residual, dtype=torch.float16)
    actual = experiment.residual.try_scaled_residual_exact(residual, branch, scale)
    assert actual is not None, "Fused operator unexpectedly fell back"
    expected = residual + branch * scale
    assert torch.equal(actual, expected)
    experiment.evidence["checks"]["residual_reference"] = error_stats(actual, expected)


def make_runner(experiment, name):
    calls = {"body": 0}

    def body(qkv, cos, sin, residual, scale):
        calls["body"] += 1
        q, k, v = qkv.chunk(3, dim=-1)
        fused_qk = experiment.qk.try_qk_norm_rope_exact(q, k, (cos, sin), 1e-5)
        assert fused_qk is not None
        q, k = fused_qk
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        branch = attended.transpose(1, 2).contiguous().view(*residual.shape)
        output = experiment.residual.try_scaled_residual_exact(residual, branch, scale)
        assert output is not None
        return output

    runner = experiment.runner_module.Runner(
        name, body, max_entries=4, warmups=2, eager_repeats=1, replay_repeats=2, profile_components=False
    )
    return runner, body, calls


@pytest.mark.parametrize("batch", [1, 2])
@torch.inference_mode()
def test_extracted_runner_replays_fused_chain_without_python_dispatch(experiment, batch):
    args = make_inputs(batch=batch)
    runner, body, calls = make_runner(experiment, f"video_fused_local_b{batch}")
    first = runner.run(args)
    assert len(runner.entries) == 1
    graph, _, static_output = next(iter(runner.entries.values()))
    assert isinstance(graph, torch.cuda.CUDAGraph)
    assert first.data_ptr() != static_output.data_ptr(), "Replay must return an owned output"
    expected = body(*args)
    assert torch.equal(first, expected)
    saved = first.clone()
    before = calls["body"]
    again = runner.run(args)
    assert calls["body"] == before, "Cached replay re-entered Python/kernel dispatch"
    assert torch.equal(again, expected)

    # Change every runtime tensor, including rotary metadata, with the same shape.
    qkv, cos, sin, residual, scale = args
    changed = (qkv + 0.25, cos.roll(1, dims=-1), sin.flip(-1), -residual, scale * -0.5)
    changed_expected = body(*changed)
    before = calls["body"]
    changed_actual = runner.run(changed)
    assert calls["body"] == before
    assert len(runner.entries) == 1, "Same-shaped new inputs should reuse the captured graph"
    assert torch.equal(changed_actual, changed_expected)
    assert not torch.equal(changed_actual, first), "Replay ignored changed inputs"
    assert torch.equal(first, saved), "A subsequent replay overwrote an earlier result"

    # Tile blending may mutate a returned tensor. It must not mutate graph storage.
    changed_actual.add_(10)
    restored = runner.run(changed)
    assert torch.equal(restored, changed_expected)
    experiment.evidence["checks"][f"runner_batch_{batch}"] = {
        "graph_vs_fused_eager_exact": True,
        "all_inputs_updated": True,
        "earlier_output_preserved": True,
        "output_mutation_does_not_corrupt_graph": True,
        "python_body_calls_during_cached_replay": 0,
        "replays": runner.replays,
        "extracted_runner_metrics": experiment.runner_module.METRICS[f"video_fused_local_b{batch}_shape0"],
    }


@torch.inference_mode()
def test_trace_contains_graph_launch_and_both_official_fused_kernels(experiment):
    args = make_inputs()
    runner, _, calls = make_runner(experiment, "video_fused_local_trace")
    runner.run(args)  # Warmup, compilation, capture and validation stay outside trace.
    torch.accelerator.synchronize()
    before = calls["body"]
    trace_path = experiment.artifacts / "fused_graph_replay_trace.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        with torch.profiler.record_function("local.fused.graph_replay"):
            runner.run(args)
            torch.accelerator.synchronize()
    prof.export_chrome_trace(str(trace_path))
    assert calls["body"] == before
    events = json.loads(trace_path.read_text())["traceEvents"]
    graph_launches = [e for e in events if "cudaGraphLaunch" in e.get("name", "")]
    kernel_names = [e["name"] for e in events if e.get("cat") == "kernel"]
    qk_kernels = [name for name in kernel_names if "qk_rms_norm_rope_exact_kernel" in name]
    residual_kernels = [name for name in kernel_names if "scaled_residual_exact_kernel" in name]
    assert graph_launches, "No actual CUDA Graph launch was observed"
    assert qk_kernels, "Graph replay trace does not contain the official fused Q/K kernel"
    assert residual_kernels, "Graph replay trace does not contain the official fused residual kernel"
    experiment.evidence["checks"]["trace"] = {
        "path": str(trace_path),
        "cuda_graph_launches": len(graph_launches),
        "qk_kernel_launches": len(qk_kernels),
        "scaled_residual_kernel_launches": len(residual_kernels),
        "kernel_names": sorted(set(kernel_names)),
        "python_body_calls_during_trace": 0,
    }


@pytest.fixture
def tile_adapter_module(monkeypatch, tmp_path):
    # Import this experimental package without importing vLLM's heavy parent package.
    name = "h3_learning_adapter"
    spec = importlib.util.spec_from_file_location(
        name, GRAPH_PACKAGE / "__init__.py", submodule_search_locations=[str(GRAPH_PACKAGE)]
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    module = __import__(name + ".fused_vae_graph", fromlist=["install"])
    module.common.ROOT = tmp_path
    for var in (
        "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE",
        "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE",
    ):
        monkeypatch.setenv(var, "0")
    module.METRICS.clear()
    module.common.METRICS.clear()
    return module


@pytest.mark.parametrize(
    "var",
    [
        "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE",
        "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE",
    ],
)
def test_tile_adapter_rejects_native_compile(tile_adapter_module, monkeypatch, var):
    class Video:
        pass

    monkeypatch.setenv(var, "1")
    with pytest.raises(RuntimeError, match="requires " + var + "=0"):
        tile_adapter_module.install(Video)


@torch.inference_mode()
def test_tile_adapter_prepares_mask_and_keeps_outer_decode_eager(tile_adapter_module, monkeypatch):
    assert torch.cuda.is_available(), "Real CUDA is required"
    module = tile_adapter_module
    counters = {"body": 0, "mask": 0, "outer": 0}
    flash = types.ModuleType("h3_learning_checkpoint.flash")

    def dense_mask(mask_mod, batch, heads, q_len, kv_len, device, aux_tensors=None):
        assert not torch.cuda.is_current_stream_capturing()
        counters["mask"] += 1
        mask = torch.ones(q_len, kv_len, device=device)
        # Simulate the checkpoint metadata's CPU synchronization: forbidden in capture.
        assert mask.sum().item() == q_len * kv_len
        return mask

    flash._mask_mod_to_dense = dense_mask
    monkeypatch.setitem(sys.modules, flash.__name__, flash)
    decoder_cls = type("Decoder", (), {"__module__": "h3_learning_checkpoint.base"})
    decoder = decoder_cls()
    decoder._omni_h3_vae_optimizations_installed = True
    decoder.transformer_blocks = [types.SimpleNamespace(ff=types.SimpleNamespace(_compile_forward_enabled=False))]
    mask_mod = types.SimpleNamespace(block_sparse_cache_key=(16, 16))

    def decode(z):
        counters["body"] += 1
        mask = flash._mask_mod_to_dense(mask_mod, 1, 1, 16, 16, z.device)
        return z * mask + z.square()

    class Video:
        parallel_size = 1

        def __init__(self):
            self.model = types.SimpleNamespace(decoder=decoder, decode=decode)

        def decode_latent(self, z):
            counters["outer"] += 1
            # Outer tiling/blending is still Python and may mutate its owned tile.
            tile = self.model.decode(z)
            tile.add_(1)
            return tile

    module.install(Video)
    owner = Video()
    z = torch.randn(16, 16, device="cuda")
    first = owner.decode_latent(z)
    assert torch.equal(first, z + z.square() + 1)
    saved = first.clone()
    before = counters.copy()
    changed = z + 0.25
    second = owner.decode_latent(changed)
    assert torch.equal(second, changed + changed.square() + 1)
    assert torch.equal(first, saved)
    assert counters["body"] == before["body"], "Tile Python body ran instead of graph replay"
    assert counters["mask"] == before["mask"], "Shape mask rebuilt after capture"
    assert counters["outer"] == before["outer"] + 1
    assert len(owner._h3_vae_runner.entries) == 1

    # Residency guard must fail before installing a new graph/mask wrapper.
    owner2 = Video()
    owner2.parallel_size = 2
    with pytest.raises(RuntimeError, match="resident single-GPU"):
        owner2.decode_latent(z)
