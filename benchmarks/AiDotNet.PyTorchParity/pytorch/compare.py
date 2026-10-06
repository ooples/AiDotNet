"""Compare an AiDotNet report against a PyTorch report, like-for-like.

The two harnesses emit the same schema with different key casing (C# records
serialize PascalCase; the Python dataclasses serialize snake_case). This script
normalizes both and prints one table with a TRAINING row (steady-state seconds
per step) and an INFERENCE row per batch size (steady-state latency) for every
model the two reports share.

Fairness rules:
- The same statistic on both sides: median vs median, with each side's IQR
  (p25-p75) shown. Both harnesses compute quantiles with the same linear
  interpolation, so the numbers are the same function of the samples.
- A row is compared only when both sides ran on the same device. A CPU number
  against a CUDA number is refused, not printed.
- Verdict per row:
    WIN   our median < theirs and our p75 < their p25 (IQRs do not overlap)
    win   our median < theirs, but the IQRs overlap (not decisive)
    lose  / LOSE  the mirror images
  Only decisive rows count towards the wins/losses tally.

Usage:
    python compare.py ../results/aidotnet.json ../results/pytorch.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


def _get(d: dict, *names: str, default=None) -> object:
    """Fetch the first present key from a set of casing variants."""
    for n in names:
        if n in d:
            return d[n]
    return default


@dataclass
class Stat:
    median: float
    p25: float
    p75: float


@dataclass
class ModelRows:
    device: str | None
    training: Stat | None
    inference: dict[int, Stat]


def _training_stat(row: dict) -> Stat | None:
    """Steady-state seconds per training STEP (epoch stats / steps per epoch)."""
    steps = _get(row, "steps_per_epoch", "StepsPerEpoch")
    median = _get(row, "steady_state_epoch_seconds_median", "SteadyStateEpochSecondsMedian")
    p25 = _get(row, "steady_state_epoch_seconds_p25", "SteadyStateEpochSecondsP25")
    p75 = _get(row, "steady_state_epoch_seconds_p75", "SteadyStateEpochSecondsP75")
    if not steps or median is None or p25 is None or p75 is None:
        return None
    return Stat(median / steps * 1000, p25 / steps * 1000, p75 / steps * 1000)


def _inference_stat(row: dict) -> Stat | None:
    median = _get(row, "steady_state_latency_ms_median", "SteadyStateLatencyMsMedian")
    p25 = _get(row, "steady_state_latency_ms_p25", "SteadyStateLatencyMsP25")
    p75 = _get(row, "steady_state_latency_ms_p75", "SteadyStateLatencyMsP75")
    if median is None or p25 is None or p75 is None:
        return None
    return Stat(median, p25, p75)


def _index(report: dict) -> dict[str, ModelRows]:
    out: dict[str, ModelRows] = {}
    report_device = _get(report, "device", "Device")
    for model in _get(report, "results", "Results", default=[]):
        name = _get(model, "model", "Model")
        device = _get(model, "device", "Device", default=report_device)
        training_row = _get(model, "training", "Training")
        training = _training_stat(training_row) if isinstance(training_row, dict) else None
        inference: dict[int, Stat] = {}
        for r in _get(model, "inference", "Inference", default=[]):
            stat = _inference_stat(r)
            if stat is not None:
                inference[_get(r, "batch_size", "BatchSize")] = stat
        out[name] = ModelRows(device, training, inference)
    return out


def _verdict(ours: Stat, theirs: Stat) -> tuple[str, int]:
    """(label, +1 decisive win / -1 decisive loss / 0 overlap)."""
    if ours.median < theirs.median:
        return ("WIN ", 1) if ours.p75 < theirs.p25 else ("win ", 0)
    return ("LOSE", -1) if ours.p25 > theirs.p75 else ("lose", 0)


def _fmt(s: Stat) -> str:
    return f"{s.median:9.3f} [{s.p25:.3f}-{s.p75:.3f}]"


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
    print(f"PyTorch:  {pt.get('framework')} {pt.get('torch')}  threads={pt.get('torch_num_threads')}")
    print("All times in ms: training = per step, inference = per forward. Cells are median [p25-p75].")
    print()
    header = f"{'model':<13}{'device':<7}{'row':<10}{'AiDotNet':>30}{'PyTorch':>30}{'ratio':>8}  verdict"
    print(header)
    print("-" * len(header))

    decisive_wins = decisive_losses = total = 0
    refused: list[str] = []
    for model in sorted(set(ai_idx) & set(pt_idx)):
        a, p = ai_idx[model], pt_idx[model]
        if a.device != p.device:
            refused.append(f"{model}: AiDotNet ran on {a.device}, PyTorch on {p.device}")
            continue
        pairs: list[tuple[str, Stat | None, Stat | None]] = [("train", a.training, p.training)]
        for bs in sorted(set(a.inference) & set(p.inference)):
            pairs.append((f"infer bs{bs}", a.inference[bs], p.inference[bs]))
        for label, ours, theirs in pairs:
            if ours is None or theirs is None or theirs.median == 0:
                continue
            verdict, score = _verdict(ours, theirs)
            decisive_wins += score == 1
            decisive_losses += score == -1
            total += 1
            print(f"{model:<13}{a.device or '?':<7}{label:<10}{_fmt(ours):>30}{_fmt(theirs):>30}"
                  f"{ours.median / theirs.median:>7.2f}x  {verdict}")

    print("-" * len(header))
    print(f"{total} rows: {decisive_wins} decisive wins, {decisive_losses} decisive losses, "
          f"{total - decisive_wins - decisive_losses} within noise (IQRs overlap).")
    for line in refused:
        print(f"REFUSED (device mismatch) {line}")
    only = sorted(set(ai_idx) ^ set(pt_idx))
    if only:
        print(f"Present in only one report (not compared): {', '.join(only)}")


if __name__ == "__main__":
    main()
