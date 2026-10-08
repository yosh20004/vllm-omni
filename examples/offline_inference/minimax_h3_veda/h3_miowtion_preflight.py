"""Preflight the experiment before any expensive model loading."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h3_miowtion_bridge as bridge
import torch
from safetensors import safe_open


def cpu_check(assets):
    from miowtion.h3.config import H3Config
    from miowtion.h3.model import H3DiT
    from miowtion.h3.schedule import ShiftScales, turbo_schedule
    from miowtion.train.lora import merged_weight
    from miowtion.veda import bundle, tiling

    torch.set_num_threads(1)
    with torch.device("meta"):
        model = H3DiT(H3Config())
    with safe_open(str(assets / "larry_turbo.safetensors"), framework="pt") as f:
        targets = bridge.validate_lora(f, dict(model.named_parameters()))
    assert len(targets) == 259
    # Verify the merging operation against the library's reference, including
    # the BF16 rounding step; layout transforms are deliberately absent here.
    gen = torch.Generator().manual_seed(73)
    for rows, cols, rank in ((384, 64, 16), (192, 128, 64)):
        w = torch.randn(rows, cols, generator=gen).bfloat16()
        a = torch.randn(rank, cols, generator=gen).bfloat16() * 0.02
        b = torch.randn(rows, rank, generator=gen).bfloat16() * 0.02
        want = merged_weight(w, a, b)
        got = w.clone()
        bridge.merge_tensor(got, a, b)
        assert torch.equal(got, want)
    schedule = turbo_schedule(8, ShiftScales(12, 3))
    assert tuple(bridge.turbo_sigmas()) == schedule.video
    assert tuple(bridge.turbo_sigmas(shift_scale=3)) == schedule.audio
    raw = SimpleNamespace(
        used_len=500,
        video_spans=(
            SimpleNamespace(start=50, latent_grid=(1, 4, 4), role="reference"),
            SimpleNamespace(start=180, latent_grid=(5, 8, 8), role="target"),
        ),
    )
    view = bridge.adapt_layout(raw, 512, 500)
    assert (view.target.start, view.target.stop) == (180, 500)
    for bad_used in (499, 513):
        try:
            bridge.adapt_layout(raw, 512, bad_used)
        except ValueError:
            pass
        else:
            raise AssertionError("Bad valid length accepted")
    loaded = bundle.load(str(assets / "predictor.safetensors"))
    plan = loaded.plans.plans["9x16_t72"]
    assert len(plan.head_shape) == 50 and all(len(row) == 56 for row in plan.head_shape)
    # Check that a geometry with an image reference, interleaved dense rows,
    # non-divisible video tiles and tail padding round-trips exactly.
    spans = [tiling.TiledSpan(s.start, s.grid, tiling.least_padding_shape(s.grid)) for s in view.spans]
    lay = tiling.build_tile_layout(spans, view.used, view.seq_len, "cpu")
    original = torch.arange(512 * 3 * 4).reshape(512, 3, 4)
    tiles = tiling.gather_tiles(original, lay, None)
    out = torch.zeros(513, 3, 4, dtype=original.dtype)
    tiling.scatter_tiles_(out, tiles, lay, None)
    assert torch.equal(original[:500], out[:500])
    assert not out[500:512].count_nonzero()
    return {
        "status": "passed",
        "lora_targets": len(targets),
        "merge_matches_miowtion_bitwise": True,
        "dual_schedule_matches_miowtion_exactly": True,
        "layout_roundtrip_exact": True,
        "selector_plan": plan.geometry,
        "selector_calibration_grid": plan.grid,
        "experiment_target_grid": [72, 32, 18],
        "geometry_note": "Published head tilings transferred from 768p to 576p; quality not established",
        "asset_sha256": {
            p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in assets.glob("*.safetensors")
        },
    }


@torch.inference_mode()
def gpu_check():
    from miowtion.kernels import fa4, reference, tile_gather_triton
    from miowtion.veda import predictor, tiling

    assert torch.cuda.get_device_capability() == (10, 3)
    assert fa4.available(torch.device("cuda")), "Miowtion FA4 unavailable"
    span = tiling.TiledSpan(137, (5, 8, 8), tiling.TileShape(2, 8, 8))
    used = span.start + span.num_rows
    lay = tiling.build_tile_layout([span], used, 512, "cuda")
    torch.manual_seed(734)
    tensors = [torch.randn(512, 3, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    heads = torch.arange(3, device="cuda")
    qt, features = tile_gather_triton.gather_and_pool(tensors[0], lay, heads)
    expected_q = tiling.gather_tiles(tensors[0], lay, heads)
    assert torch.equal(qt, expected_q)
    ref_features = predictor.pool_tiles(expected_q, lay)
    torch.testing.assert_close(features, ref_features, rtol=1e-5, atol=1e-6)
    kt, vt = [tiling.gather_tiles(t, lay, heads) for t in tensors[1:]]
    real = lay.slot_valid.bool()
    n = lay.n_tiles
    all_blocks = torch.ones(3, n, n, device="cuda", dtype=torch.bool)
    masks = [
        all_blocks,
        torch.eye(n, device="cuda", dtype=torch.bool)[None].expand(3, -1, -1).clone(),
        torch.eye(n, device="cuda", dtype=torch.bool).roll(1, -1)[None].expand(3, -1, -1).clone(),
    ]
    results, outputs = [], []
    for label, mask in zip(("all", "diagonal", "shifted"), masks):
        got = fa4.block_sparse_attention(qt, kt, vt, mask, lay)
        want = reference.block_sparse_attention(qt, kt, vt, mask, lay.valid_count)
        assert torch.isfinite(got[real]).all()
        torch.testing.assert_close(got[real].float(), want[real].float(), rtol=0.02, atol=0.02)
        results.append({"mask": label, "max_abs": float((got[real].float() - want[real].float()).abs().max())})
        outputs.append(got[real].clone())
    assert (outputs[0] - outputs[1]).float().abs().max() > 0.05
    assert (outputs[1] - outputs[2]).float().abs().max() > 0.05
    torch.cuda.synchronize()  # noqa: TID251 - CUDA-only experiment
    return {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "capability": [10, 3],
        "partial_tiles": lay.partial_tiles.numel(),
        "mask_changes_affect_output": True,
        "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = gpu_check() if args.gpu else cpu_check(args.assets)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
