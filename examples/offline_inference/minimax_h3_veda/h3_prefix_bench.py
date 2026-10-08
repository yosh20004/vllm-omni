"""CPU-launched B300 synthetic attention benchmark; no H3 model weights."""

import gzip
import json
import shutil
import statistics
import time
from pathlib import Path

import torch
from h3_prefix_split import DensePrefix, SplitStudent


def compare(got, want):
    a, b = got.float(), want.float()
    diff = a - b
    return dict(
        max_abs=diff.abs().max().item(),
        relative_mse=(diff.square().mean() / b.square().mean().clamp_min(1e-20)).item(),
        bitwise_equal=torch.equal(got, want),
    )


@torch.inference_mode()
def preflight():
    from dataclasses import replace

    from miowtion.kernels import fa4, reference
    from miowtion.kernels import tile_gather_triton as tg
    from miowtion.veda import mask, tiling

    reports = []
    torch.manual_seed(1987)
    layout = tiling.build_tile_layout([tiling.TiledSpan(137, (5, 8, 8), tiling.TileShape(2, 8, 8))], 457, 512, "cuda")
    heads = torch.arange(8, device="cuda")
    for scale in (1.0, 2.0):
        q, k, v = [torch.randn(512, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        q, k = q * scale, k * scale
        qt, kt, vt = [tg.gather_tiles(x, layout, heads) for x in (q, k, v)]
        scores = torch.randn(8, layout.n_video_tiles, layout.n_tiles, device="cuda")
        selected = mask.select_video_blocks(scores, layout, mask.column_blocks(layout, mask.Budget(tiles=1)))
        full_mask = mask.dense_block_mask(selected, layout)
        video_mask = mask.dense_block_mask(selected, layout, global_rows=False)
        assert torch.equal(full_mask[:, : layout.n_video_tiles], video_mask)
        ref = reference.block_sparse_attention(qt, kt, vt, full_mask, layout.valid_count)
        want = torch.zeros(513, 8, 128, device="cuda", dtype=torch.bfloat16)
        tg.scatter_tiles_(want, ref, layout, heads)
        sparse = fa4.block_sparse_attention(qt[: layout.n_video_tiles * 128], kt, vt, video_mask, layout)
        for backend in ("fa4", "trtllm"):
            got = torch.zeros_like(want)
            tg.scatter_tiles_(got, sparse, replace(layout, n_global_tiles=0), heads)
            got[:137].copy_(DensePrefix(137, 457, backend)(q, k, v))
            metrics = compare(got[:457], want[:457])
            torch.testing.assert_close(got[:457].float(), want[:457].float(), atol=0.025, rtol=0.025)
            assert metrics["relative_mse"] < 1e-4, metrics
            assert not got[457:].count_nonzero()
            reports.append(dict(backend=backend, qk_scale=scale, **metrics))
    print("GPU partial-tile / sparse-mask / dense-prefix checks:", reports, flush=True)
    return reports


@torch.inference_mode()
def run(folder, predictor_path):
    import os
    from types import SimpleNamespace

    import h3_miowtion_bridge as bridge

    assert torch.cuda.get_device_capability() == (10, 3)
    torch.set_num_threads(8)
    report = dict(
        scope="Synthetic Q/K/V; pretrained selector and actual 50-layer plan; no model/video inference",
        gpu=torch.cuda.get_device_name(),
        preflight=preflight(),
    )
    (folder / "preflight.json").write_text(json.dumps(report, indent=2))
    os.environ["H3_VEDA_PREDICTOR"] = str(predictor_path)
    raw = SimpleNamespace(
        used_len=44035, video_spans=(SimpleNamespace(start=2563, latent_grid=(72, 32, 18), role="target"),)
    )
    original = bridge.student_for(raw, 44096, 44035, torch.device("cuda"), prefix_backend="none")
    methods = {
        "original": original,
        "split_fa4": SplitStudent(original, "fa4"),
        "split_trtllm": SplitStudent(original, "trtllm"),
    }
    torch.manual_seed(3100)
    qkv = [torch.randn(44096, 56, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    report["qkv_shape"] = list(qkv[0].shape)
    report["checks"] = []
    # Compile all 50 actual plan configurations, and validate each split output.
    for layer in range(50):
        want = original(*qkv, layer)
        for name in ("split_fa4", "split_trtllm"):
            got = methods[name](*qkv, layer)
            metrics = compare(got, want)
            prefix = compare(got[:2563], want[:2563])
            video_equal = torch.equal(got[2563:], want[2563:])
            assert video_equal, (layer, name, "Video-query outputs changed")
            assert metrics["max_abs"] < 0.025 and metrics["relative_mse"] < 1e-4, (layer, name, metrics)
            report["checks"].append(
                dict(layer=layer, method=name, video_bitwise_equal=video_equal, prefix=prefix, **metrics)
            )
        if layer % 10 == 0:
            print(f"Warmup + correctness: {layer + 1}/50 layers", flush=True)
    del got, want
    torch.cuda.synchronize()  # noqa: TID251 - CUDA-only benchmark
    report["measurements"] = []
    # Rotating order controls drift; include selector, gather, scatter and copy.
    names = list(methods)
    for repeat in range(3):
        for name in names[repeat:] + names[:repeat]:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()  # noqa: TID251 - CUDA-only benchmark
            start = time.perf_counter()
            a.record()
            for layer in range(50):
                output = methods[name](*qkv, layer)
                del output
            b.record()
            b.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000
            item = dict(repeat=repeat, method=name, wall_ms=wall_ms, cuda_ms=a.elapsed_time(b))
            report["measurements"].append(item)
            print("Measured:", item, flush=True)
    medians = {
        name: statistics.median(x["wall_ms"] for x in report["measurements"] if x["method"] == name) for name in names
    }
    report["median_50_layer_ms"] = medians
    report["speedup"] = {name: medians["original"] / ms for name, ms in medians.items()}
    report["best"] = min(medians, key=medians.get)
    (folder / "benchmark.json").write_text(json.dumps(report, indent=2))
    print("Median 50 layers:", medians, "speedups:", report["speedup"], flush=True)
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for name, method in methods.items():
            with record_function("PrefixBench." + name):
                for layer in (0, 1, 17):
                    with record_function(f"PrefixBench.{name}.layer{layer}"):
                        output = method(*qkv, layer)
                        del output
                torch.cuda.synchronize()  # noqa: TID251 - CUDA-only benchmark
    raw_trace = folder / "trace.json"
    prof.export_chrome_trace(str(raw_trace))
    with raw_trace.open("rb") as src, gzip.open(folder / "trace.json.gz", "wb", compresslevel=1) as dst:
        shutil.copyfileobj(src, dst)
    raw_trace.unlink()
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.predictor.is_file():
        parser.error("--predictor must point to the pinned checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.output, args.predictor)
