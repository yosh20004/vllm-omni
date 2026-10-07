# MiniMax-H3 fused VAE CUDA Graph experiment

See the [Chinese code walkthrough](../../../docs/design/minimax_h3_fused_vae_graph.md) for capture/replay, output ownership, scope, configuration and existing full-model evidence.

This branch preserves an opt-in, resident single-GPU learning experiment. It is disabled by default. It retains first-use comparisons; it is not a production serving graph API.

Run the weight-free real-CUDA tests from the repository root:

```bash
python -m pytest -o addopts='' -q examples/offline_inference/minimax_h3_fused_vae_graph/test_graph_runner.py
```

Requires PyTorch, Triton, pytest and a real CUDA GPU. Tests run official fused leaf kernels with a Triton import adapter; on SM86 they bypass upstream hardware dispatch. They do not load checkpoint weights or validate full-model performance.
