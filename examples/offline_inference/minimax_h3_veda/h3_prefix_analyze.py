"""Local CPU analysis of the prefix split benchmark's diagnostic trace."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def union_us(intervals):
    """Measure GPU event coverage without double-counting overlap."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(folder):
    with gzip.open(folder / "trace.json.gz", "rt") as file:
        events = json.load(file)["traceEvents"]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "LaunchKernel" in e["name"]
    }
    results = []
    for method in ("original", "split_fa4", "split_trtllm"):
        annotation = next(
            e for e in events if e.get("cat") == "user_annotation" and e["name"] == "PrefixBench." + method
        )

        def owned(e, a):
            launch = launches.get(e["args"].get("correlation"))
            return (
                launch
                and launch["tid"] == a["tid"]
                and launch["pid"] == a["pid"]
                and a["ts"] <= launch["ts"] < a["ts"] + a["dur"]
            )

        kernels = [e for e in gpu if owned(e, annotation)]
        times, counts = Counter(), Counter()
        for e in kernels:
            times[e["name"]] += e["dur"] / 1000
            counts[e["name"]] += 1
        parts = Counter()
        for layer in (0, 1, 17):
            a = next(
                e
                for e in events
                if e.get("cat") == "user_annotation" and e["name"] == f"PrefixBench.{method}.layer{layer}"
            )
            attn = sorted(
                (
                    e
                    for e in kernels
                    if owned(e, a) and ("FlashAttentionForwardSm100" in e["name"] or "fmhaSm103aKernel" in e["name"])
                ),
                key=lambda e: e["ts"],
            )
            assert len(attn) == (2 if method == "original" else 3)
            # Adapter submits its two video head groups before one dense prefix.
            for i, e in enumerate(attn):
                label = (
                    "original_all_queries"
                    if method == "original"
                    else "video_queries_sparse"
                    if i < 2
                    else "prefix_queries_dense"
                )
                parts[label] += e["dur"] / 1000
        results.append(
            dict(
                method=method,
                layers=[0, 1, 17],
                cpu_wall_ms=annotation["dur"] / 1000,
                gpu_sum_ms=sum(times.values()),
                gpu_events=len(kernels),
                gpu_busy_ms=union_us((e["ts"], e["ts"] + e["dur"]) for e in kernels) / 1000,
                attention_parts_ms=dict(parts),
                kernels=[dict(name=n, ms=v, count=counts[n]) for n, v in times.most_common()],
            )
        )
    (folder / "trace_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "kernels"} for r in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
