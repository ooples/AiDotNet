"""Compare an AiDotNet report against a PyTorch report.

The two harnesses emit the same schema with different key casing (C# records
serialize PascalCase; the Python dataclasses serialize snake_case). This script
normalizes both, then prints a per-model / per-batch-size inference table with
the latency ratio and a verdict.

Gate (from AIsEval Reporting/findings.md): a build "wins" a row when
    p95(AiDotNet) < mean(PyTorch)
i.e. our worst-of-95% steady-state latency still beats their average — a
deliberately conservative bar that is robust to rig-contention noise.

Usage:
    python compare.py ../results/aidotnet.json ../results/pytorch.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _get(d: dict, *names: str, default=None) -> object:
    """Fetch the first present key from a set of casing variants."""
    for n in names:
        if n in d:
            return d[n]
    return default


def _norm_inference(row: dict) -> dict:
    return {
        "batch_size": _get(row, "batch_size", "BatchSize"),
        "avg_ms": _get(row, "steady_state_latency_ms_avg", "SteadyStateLatencyMsAvg"),
        "p95_ms": _get(row, "steady_state_latency_ms_p95", "SteadyStateLatencyMsP95"),
        "throughput": _get(row, "throughput_samples_per_second", "ThroughputSamplesPerSecond"),
        "mem_mb": _get(row, "memory_mb_peak", "MemoryMbPeak"),
    }


def _index(report: dict) -> dict[str, dict[int, dict]]:
    """model -> batch_size -> normalized inference row."""
    out: dict[str, dict[int, dict]] = {}
    for model in _get(report, "results", "Results", default=[]):
        name = _get(model, "model", "Model")
        rows = {}
        for r in _get(model, "inference", "Inference", default=[]):
            n = _norm_inference(r)
            rows[n["batch_size"]] = n
        out[name] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AiDotNet vs PyTorch benchmark JSON reports.")
    parser.add_argument("aidotnet", type=Path, help="Path to the AiDotNet report JSON.")
    parser.add_argument("pytorch", type=Path, help="Path to the PyTorch report JSON.")
    args = parser.parse_args()

    ai = json.loads(args.aidotnet.read_text(encoding="utf-8"))
    pt = json.loads(args.pytorch.read_text(encoding="utf-8"))

    ai_idx = _index(ai)
    pt_idx = _index(pt)

    print(f"AiDotNet: {_get(ai, 'framework', 'Framework')}  runtime={_get(ai, 'dotnetRuntime', 'DotNetRuntime')}")
    print(f"PyTorch:  {pt.get('framework')} {pt.get('torch')}  threads={pt.get('torch_num_threads')}  device={pt.get('device')}")
    print()
    header = f"{'model':<13}{'bs':>5}{'AiDotNet p95':>14}{'PyTorch avg':>14}{'ratio':>9}  verdict"
    print(header)
    print("-" * len(header))

    wins = 0
    total = 0
    for model in sorted(set(ai_idx) | set(pt_idx)):
        for bs in sorted(set(ai_idx.get(model, {})) | set(pt_idx.get(model, {}))):
            a = ai_idx.get(model, {}).get(bs)
            p = pt_idx.get(model, {}).get(bs)
            if not a or not p:
                continue
            ai_p95 = a["p95_ms"]
            pt_avg = p["avg_ms"]
            if ai_p95 is None or pt_avg in (None, 0):
                continue
            ratio = ai_p95 / pt_avg
            win = ai_p95 < pt_avg
            wins += int(win)
            total += 1
            verdict = "WIN " if win else "lose"
            print(f"{model:<13}{bs:>5}{ai_p95:>14.3f}{pt_avg:>14.3f}{ratio:>8.2f}x  {verdict}")

    print("-" * len(header))
    print(f"AiDotNet wins {wins}/{total} rows (p95(AiDotNet) < mean(PyTorch)).")


if __name__ == "__main__":
    main()
