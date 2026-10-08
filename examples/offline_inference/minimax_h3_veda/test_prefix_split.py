"""CPU routing checks using real Miowtion plans, masks and references."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from h3_miowtion_bridge import LayoutView
from h3_prefix_split import DensePrefix, SplitStudent
from miowtion.h3.layout import VideoSpan
from miowtion.kernels import fa4, reference, tile_gather_triton
from miowtion.veda import attention, mask, plan, predictor, tiling


@pytest.fixture
def kernels(monkeypatch):
    """Substitute math at kernel boundaries; keep routing/selection real."""
    calls = {"sparse": [], "dense": [], "scatter": []}

    def sparse(q, k, v, block_mask, layout):
        calls["sparse"].append((q.shape[0], block_mask.clone(), layout))
        return reference.block_sparse_attention(q, k, v, block_mask, layout.valid_count)

    def dense(q, k, v, softmax_scale):
        calls["dense"].append((q.shape, k.shape, v.shape))
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * softmax_scale
        return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float()).to(q.dtype)

    def gather_and_pool(x, layout, heads):
        tiles = tiling.gather_tiles(x, layout, heads)
        return tiles, predictor.pool_tiles(tiles, layout)

    def scatter(out, tiled, layout, heads):
        calls["scatter"].append(layout)
        # Match the Triton launch domain, including the scatter-only view.
        count = layout.num_slots
        valid = layout.slot_valid[:count].bool()
        rows = layout.scatter_index[:count][valid]
        out[rows[:, None], heads[None, :]] = tiled[valid]

    monkeypatch.setattr(fa4, "available", lambda *args: True)
    monkeypatch.setattr(fa4, "block_sparse_attention", sparse)
    monkeypatch.setattr(fa4, "interface", lambda: SimpleNamespace(flash_attn_func=dense))
    monkeypatch.setattr(tile_gather_triton, "gather_tiles", tiling.gather_tiles)
    monkeypatch.setattr(tile_gather_triton, "gather_and_pool", gather_and_pool)
    monkeypatch.setattr(tile_gather_triton, "scatter_tiles_", scatter)
    return calls


def make_student(*, chunked=False, dense_layers=frozenset()):
    layout = LayoutView(512, 457, (VideoSpan(137, (5, 8, 8), "target"),))
    config = attention.VedaConfig(
        target_budget=mask.Budget(tiles=1),
        collect_bytes=1 if chunked else 2**30,
        dense_layers=dense_layers,
    )
    clip = attention.ClipTiling(layout, config, torch.device("cpu"))
    tile_plan = plan.TilePlan("test", (5, 8, 8), [tiling.TileShape(2, 8, 8), tiling.TileShape(4, 4, 8)], [[0, 1, 0]])
    scorer = predictor.TileScorePredictor(1, 3, 8)
    return attention.SparseStudent(clip, tile_plan, scorer)


@pytest.mark.parametrize("chunked", [False, True])
@torch.no_grad()
def test_split_preserves_selection_global_keys_and_padding(kernels, chunked):
    torch.manual_seed(3100)
    baseline = make_student(chunked=chunked)
    split = SplitStudent(baseline, "fa4")
    q, k, v = [torch.randn(512, 3, 8) for _ in range(3)]
    # Padding keys would dominate softmax if accidentally included.
    k[457:] = 1000
    v[457:] = 1000
    want = baseline(q, k, v, 0)
    original = list(kernels["sparse"])
    kernels["sparse"].clear()
    got = split(q, k, v, 0)
    assert len(original) == len(kernels["sparse"]) == (3 if chunked else 2)
    for (_, full_mask, layout), (nq, video_mask, video_layout) in zip(original, kernels["sparse"]):
        assert video_layout is layout
        assert nq == layout.n_video_tiles * tiling.TILE_SIZE
        assert torch.equal(video_mask, full_mask[:, : layout.n_video_tiles])
        assert video_mask[:, :, layout.n_video_tiles :].all()
    assert all(view.n_global_tiles == 0 for view in kernels["scatter"])
    assert kernels["dense"] == [(torch.Size([1, 137, 3, 8]), torch.Size([1, 457, 3, 8]), torch.Size([1, 457, 3, 8]))]
    torch.testing.assert_close(got[:457], want[:457], atol=1e-6, rtol=1e-5)
    assert torch.equal(got[457:], torch.zeros_like(got[457:]))
    assert got.shape == q.shape


def test_inference_does_not_build_predictor_autograd(kernels):
    split = SplitStudent(make_student(), "fa4")
    qkv = [torch.randn(512, 3, 8, requires_grad=True) for _ in range(3)]
    assert not split(*qkv, 0).requires_grad


def test_dense_layer_is_not_split(kernels, monkeypatch):
    split = SplitStudent(make_student(dense_layers=frozenset({0})), "fa4")
    expected = torch.randn(512, 3, 8)
    monkeypatch.setattr(attention.h3_attention, "dense_attention", lambda *args: (expected, None))
    assert split(expected, expected, expected, 0) is expected
    assert not kernels["sparse"] and not kernels["dense"]


@pytest.mark.parametrize("case", ["empty_prefix", "extra_span", "trailing_global", "condition"])
def test_unsupported_layout_rejected(kernels, case):
    baseline = make_student()
    if case == "empty_prefix":
        baseline.clip.layout = LayoutView(512, 320, (VideoSpan(0, (5, 8, 8), "target"),))
    elif case == "extra_span":
        baseline.clip.layout = replace(
            baseline.clip.layout, spans=(VideoSpan(0, (1, 1, 1), "reference"), baseline.clip.layout.target)
        )
    elif case == "trailing_global":
        baseline.clip.layout = replace(baseline.clip.layout, used=458)
    else:
        baseline.clip._condition_spans = [tiling.TiledSpan(0, (1, 1, 1), tiling.TileShape(2, 8, 8))]
    with pytest.raises(ValueError, match="one final target"):
        SplitStudent(baseline, "fa4")


def test_invalid_backend_rejected(kernels):
    with pytest.raises(ValueError):
        DensePrefix(137, 457, "unknown")


def test_bridge_routes_and_caches_backends(kernels, monkeypatch):
    import h3_miowtion_bridge as bridge

    baseline = make_student()
    baseline.plan.head_shape *= 50
    bundle = SimpleNamespace(
        plans=SimpleNamespace(select=lambda geometry: baseline.plan),
        predictor=baseline.predictor,
        target_budget=mask.Budget(tiles=1),
        ref_budget=mask.Budget(tiles=1),
    )
    monkeypatch.setattr(bridge, "BUNDLE", bundle)
    monkeypatch.setattr(bridge, "STUDENTS", {})
    monkeypatch.setattr(bridge, "EVIDENCE", {"layouts": []})
    raw = SimpleNamespace(
        used_len=457,
        video_spans=(SimpleNamespace(start=137, latent_grid=(5, 8, 8), role="target"),),
    )
    monkeypatch.delenv("H3_VEDA_PREFIX_BACKEND", raising=False)
    original = bridge.student_for(raw, 512, 457, "cpu")
    assert type(original) is attention.SparseStudent
    monkeypatch.setenv("H3_VEDA_PREFIX_BACKEND", "fa4")
    split = bridge.student_for(raw, 512, 457, "cpu")
    assert isinstance(split, SplitStudent)
    assert bridge.student_for(raw, 512, 457, "cpu") is split
    assert bridge.student_for(raw, 512, 457, "cpu", prefix_backend="none") is original
    assert [entry["prefix_backend"] for entry in bridge.EVIDENCE["layouts"]] == ["none", "fa4"]
    monkeypatch.setenv("H3_VEDA_PREFIX_BACKEND", "typo")
    with pytest.raises(ValueError, match="H3_VEDA_PREFIX_BACKEND"):
        bridge.student_for(raw, 512, 457, "cpu")
