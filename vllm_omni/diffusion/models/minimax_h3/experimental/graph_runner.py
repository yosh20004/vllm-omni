# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validated experimental copy/replay/clone runner, extracted from the H3 trial.

The tile adapter uses one CUDA tensor argument. This is not a general graph
wrapper for arbitrary Python state, CPU tensors, random/stateful functions or
concurrent requests. First-use comparisons are intentionally retained.
"""

import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

ROOT = Path(os.environ.get("H3_VAE_GRAPH_DIR", "/tmp/h3_vae_graph"))
METRICS = {}


def check(actual, expected):
    a, _ = tree_flatten(actual)
    b, _ = tree_flatten(expected)
    checks = []
    for x, y in zip(a, b, strict=True):
        if x.dtype == torch.uint8:
            # Quantization can differ by 1 at a rounding boundary; report it.
            maximum = 0.0
            for a_chunk, b_chunk in zip(x.flatten().split(1000000), y.flatten().split(1000000), strict=True):
                maximum = max(maximum, (a_chunk.float() - b_chunk.float()).abs().max().item())
            if maximum > 1:
                raise AssertionError(f"Quantized VAE mismatch: max_abs={maximum}")
            checks.append({"exact": torch.equal(x, y), "max_abs": maximum})
        else:
            torch.testing.assert_close(x, y, rtol=0.01, atol=0.01)
            checks.append({"exact": torch.equal(x, y), "max_abs": (x.float() - y.float()).abs().max().item()})
    return checks


def timed(fn, repeats):
    wall, span = [], []
    for _ in range(repeats):
        torch.accelerator.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        result = fn()
        end.record()
        end.synchronize()
        wall.append(time.perf_counter() - start)
        span.append(begin.elapsed_time(end) / 1000)
    return result, {
        "wall_seconds_median": statistics.median(wall),
        "cuda_event_span_seconds_median": statistics.median(span),
        "repeats": repeats,
    }


class Runner:
    def __init__(self, name, fn, max_entries=2, warmups=3, eager_repeats=5, replay_repeats=10, profile_components=True):
        self.name, self.fn = name, fn
        self.entries = {}
        self.max_entries, self.warmups = max_entries, warmups
        self.eager_repeats, self.replay_repeats = eager_repeats, replay_repeats
        self.replays = 0
        self.profile_components = profile_components

    def run(self, args):
        leaves, spec = tree_flatten(args)
        key = tuple(
            (tuple(x.shape), x.dtype, str(x.device), tuple(x.flatten().tolist()) if x.device.type == "cpu" else None)
            if isinstance(x, torch.Tensor)
            else x
            for x in leaves
        )
        if key not in self.entries:
            # Bound retained static buffers and graph-private allocator pools.
            if len(self.entries) >= self.max_entries:
                self.entries.clear()
            # [1] Graph kernels retain addresses: allocate persistent input storage.
            static = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            label = self.name + (f"_shape{len(self.entries)}" if self.name.startswith(("video_", "audio_")) else "")
            print("H3 graph preparing:", label, str(key)[:500], flush=True)
            # [2] Warm lazy kernel/allocator setup on a side stream before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmups):
                    self.fn(*static)
            torch.cuda.current_stream().wait_stream(stream)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            started = time.perf_counter()
            # [3] Execute Python once to record the GPU operations and dependencies.
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                output = self.fn(*static)
            torch.accelerator.synchronize()
            capture_seconds = time.perf_counter() - started
            self.entries[key] = (graph, static, output)
            eager, eager_time = timed(lambda: self.fn(*args), self.eager_repeats)
            replay, replay_time = timed(lambda: self.replay(key, args), self.replay_repeats)
            comparison = check(replay, eager)
            # New content, same shape: prove replay does not reuse stale inputs.
            changed = list(leaves)
            changed[0] = changed[0] * 0.99
            changed_args = tree_unflatten(changed, spec)
            change_check = check(self.replay(key, changed_args), self.fn(*changed_args))
            self.replay(key, args)
            METRICS[label] = {
                "capture_seconds": capture_seconds,
                "prepared_body_eager": eager_time,
                "copy_replay_clone": replay_time,
                "output_check": comparison,
                "changed_input_check": change_check,
            }
            ROOT.mkdir(parents=True, exist_ok=True)
            (ROOT / "component_graph_metrics.json").write_text(json.dumps(METRICS, indent=2))
            if self.profile_components:
                # Short component traces; no Python stacks or aggregation.
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                ) as prof:
                    with torch.profiler.record_function(label + ".warm_eager"):
                        self.fn(*args)
                        torch.accelerator.synchronize()
                    with torch.profiler.record_function(label + ".graph_replay"):
                        self.replay(key, args)
                        torch.accelerator.synchronize()
                prof.export_chrome_trace(str(ROOT / (label + ".json")))
            print("H3 graph component:", self.name, json.dumps(METRICS[label]), flush=True)
        return self.replay(key, args)

    def replay(self, key, args):
        graph, static, output = self.entries[key]
        a, _ = tree_flatten(args)
        b, _ = tree_flatten(static)
        # [4] Refresh values at the same addresses; replay does not rerun Python.
        for src, dst in zip(a, b, strict=True):
            if isinstance(src, torch.Tensor) and src.device.type == "cuda":
                dst.copy_(src)
        graph.replay()
        self.replays += 1
        # [5] The captured output is reused each replay. Return caller-owned storage.
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output)
