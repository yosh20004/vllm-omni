"""Experimental prefix-query split; leaves video-query softmax unchanged.

External adapter for Miowtion 20da18c, not an upstream modification.
Only supports the benchmark layout: one final video span, dense prefix.
Inputs are unbatched [sequence, heads, head_dim] BF16 CUDA tensors.
"""

from dataclasses import replace

import torch
from miowtion.kernels import fa4, tile_gather_triton
from miowtion.veda import attention, mask


class DensePrefix:
    """Compute prefix queries against every valid key in original order."""

    def __init__(self, prefix, used, backend, device="cuda"):
        if not 0 < prefix < used:
            raise ValueError("Expected a nonempty prefix followed by video tokens")
        self.prefix, self.used, self.backend = prefix, used, backend
        if backend == "trtllm":
            from flashinfer.prefill import trtllm_ragged_attention_deepseek

            self.call = trtllm_ragged_attention_deepseek
            self.workspace = torch.empty(128 * 2**20, device=device, dtype=torch.uint8)
            self.lengths = torch.tensor([used], device=device, dtype=torch.int32)
            self.cu_q = torch.tensor([0, prefix], device=device, dtype=torch.int32)
            self.cu_k = torch.tensor([0, used], device=device, dtype=torch.int32)
        elif backend != "fa4":
            raise ValueError(backend)

    def __call__(self, q, k, v):
        q, k, v = q[: self.prefix], k[: self.used], v[: self.used]
        if self.backend == "fa4":
            out = fa4.interface().flash_attn_func(q[None], k[None], v[None], softmax_scale=q.shape[-1] ** -0.5)
            return (out[0] if isinstance(out, tuple) else out)[0]
        return self.call(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self.workspace,
            seq_lens=self.lengths,
            max_q_len=self.prefix,
            max_kv_len=self.used,
            bmm1_scale=q.shape[-1] ** -0.5,
            bmm2_scale=1.0,
            o_sf_scale=-1.0,
            batch_size=1,
            window_left=-1,
            cum_seq_lens_q=self.cu_q,
            cum_seq_lens_kv=self.cu_k,
            enable_pdl=False,
            is_causal=False,
            return_lse=False,
            skip_softmax_threshold_scale_factor=None,
            skip_all_rows_active_check=True,
        )


class SplitStudent(attention.SparseStudent):
    """Route global query rows to dense and video query rows to sparse FA4.

    This inference-only adapter retains global keys in the video softmax.
    It does not split keys or merge independently normalized outputs.
    """

    def __init__(self, baseline, backend):
        layout = baseline.clip.layout
        if (
            len(layout.spans) != 1
            or layout.target.start <= 0
            or layout.target.stop != layout.used
            or layout.target.role != "target"
            or baseline.clip._condition_spans
        ):
            raise ValueError("Prefix split requires one final target span and no tiled conditions")
        super().__init__(baseline.clip, baseline.plan, baseline.predictor)
        self.prefix = layout.target.start
        self.dense = DensePrefix(self.prefix, layout.used, backend, self.clip.device)
        # Scatter reads only leading video slots. All other fields are shared;
        # this view must never be passed to attention or predictor selection.
        self.scatter_views = {s: replace(self.clip.get(s), n_global_tiles=0) for s in self.plan.shapes}

    @torch.no_grad()
    def __call__(self, q, k, v, layer_index):
        out = super().__call__(q, k, v, layer_index)
        if layer_index not in self.clip.config.dense_layers:
            out[: self.prefix].copy_(self.dense(q, k, v))
        return out

    def _attend(self, q, k, v, layer_index, tile_layout, heads, out):
        # Same predictor inputs and top-k selection as the original student.
        qt, fq = tile_gather_triton.gather_and_pool(q, tile_layout, heads)
        kt, fk = tile_gather_triton.gather_and_pool(k, tile_layout, heads)
        vt = tile_gather_triton.gather_tiles(v, tile_layout, heads)
        logits = self.predictor.layers[layer_index](fq, fk, heads)
        selected = mask.select_video_blocks(
            logits[:, : tile_layout.n_video_tiles], tile_layout, self.clip.blocks(tile_layout)
        )
        sparse_mask = mask.dense_block_mask(selected, tile_layout, global_rows=False)
        result = fa4.block_sparse_attention(qt[: tile_layout.n_video_tiles * 128], kt, vt, sparse_mask, tile_layout)
        shape = next(s for s in self.plan.shapes if self.clip.get(s) is tile_layout)
        tile_gather_triton.scatter_tiles_(out, result, self.scatter_views[shape], heads)
