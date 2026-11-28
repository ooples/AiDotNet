#!/usr/bin/env python3
"""
Turnkey ABBA Multi-Seed Benchmark Orchestrator for AiDotNet vs Compiled PyTorch.

Executes interleaved ABBA passes across >= 3 seeds (default: 42, 1337, 2026) comparing:
  - Arm A: AiDotNet (ExecutionMode.AutoCompiled + MaxAutotune + DerivedWeightCaching)
  - Arm D: AiDotNet (CompilationOptions.PureDirectPtx — self-contained zero-vendor-library path)
  - Arm B: PyTorch (torch.compile(backend="inductor", mode="max-autotune") + @triton.jit + fused AdamW)

Safety gates:
  - GPU 0 is strictly forbidden; --gpu-index 0 immediately aborts.
  - --cpu-dry-run sets CUDA_VISIBLE_DEVICES="" and AIDOTNET_DISABLE_GPU="1" and runs --device cpu --smoke.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SUITE_DIR = Path(__file__).resolve().parent
CSPROJ_PATH = SUITE_DIR / "AiDotNet.CrossFamilyCompiledSuite.csproj"
PYTORCH_RUNNER = SUITE_DIR / "pytorch_compiled_runner.py"


def percentile_sorted(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = p * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def aggregate_metric(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "median": 0.0, "iqr": 0.0, "p95": 0.0, "mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if (n % 2 == 1) else 0.5 * (s[n // 2 - 1] + s[n // 2])
    q25 = percentile_sorted(s, 0.25)
    q75 = percentile_sorted(s, 0.75)
    p95 = percentile_sorted(s, 0.95)
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if n > 1 else 0.0
    return {
        "n": n,
        "median": med,
        "iqr": q75 - q25,
        "p95": p95,
        "mean": mean,
        "sd": sd,
        "min": s[0],
        "max": s[-1],
    }


def _find_built_suite_dll() -> Path:
    bin_dir = SUITE_DIR / "bin"
    dll_candidates = sorted(
        [
            p
            for p in bin_dir.rglob("AiDotNet.CrossFamilyCompiledSuite.dll")
            if "release" in [part.lower() for part in p.parts]
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not dll_candidates:
        raise FileNotFoundError("Could not locate built AiDotNet.CrossFamilyCompiledSuite.dll under bin/**/Release.")
    return dll_candidates[0]


def build_dotnet_suite(local_tensors_root: Optional[str], env: Dict[str, str]) -> Path:
    cmd = [
        "dotnet",
        "build",
        str(CSPROJ_PATH),
        "-c",
        "Release",
        "-p:GeneratePackageOnBuild=false",
        "-v",
        "minimal",
    ]
    if local_tensors_root:
        cmd.extend(["-p:UseLocalTensors=true", f"-p:LocalTensorsRoot={local_tensors_root}"])
    print(f"[ABBA] Building C# benchmark runner: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    return _find_built_suite_dll()


def run_arm_pass(
    arm_label: str,
    cmd: List[str],
    out_json: Path,
    env: Dict[str, str],
) -> List[Dict[str, Any]]:
    print(f"\n[ABBA] >>> Running {arm_label}: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    if not out_json.exists():
        raise FileNotFoundError(f"Expected benchmark output JSON was not written: {out_json}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def format_markdown_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cross-Family ABBA Benchmark Summary (AiDotNet vs Compiled PyTorch)")
    lines.append("")
    lines.append(f"- **Timestamp (UTC):** `{summary['timestamp_utc']}`")
    lines.append(f"- **Device:** `{summary['device']}` (`smoke={summary['smoke']}`)")
    lines.append(f"- **Precision Tier:** `{summary['precision']}`")
    lines.append(f"- **Seeds:** `{summary['seeds']}`")
    lines.append(f"- **ABBA Schedule per Seed:** `AiDotNet(auto) -> AiDotNet(direct-ptx) -> PyTorch(inductor) -> PyTorch(inductor) -> AiDotNet(direct-ptx) -> AiDotNet(auto)`")
    lines.append("")
    lines.append(
        "| Workload | Arm | Params | B=1 Med (ms) | Batch Infer Med (ms) | Train Step Med (ms) | Train Mean ± SD (ms) | Train [Min, Max] (ms) | Alloc/Step (KB) | Final Loss | Overlap |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for wid, arm_map in summary["workloads"].items():
        for arm_name, stats_obj in arm_map.items():
            b1 = stats_obj["b1_median_ms"]
            bi = stats_obj["batch_infer_median_ms"]
            tr = stats_obj["train_step_median_ms"]
            alloc_kb = stats_obj.get("managed_alloc_kb_median", 0.0)
            floss = stats_obj.get("final_loss_median", 0.0)
            lines.append(
                f"| `{wid}` | `{arm_name}` | {stats_obj['parameter_count']:,} | "
                f"{b1['median']:.3f} | {bi['median']:.3f} | {tr['median']:.3f} | "
                f"{tr['mean']:.3f} ± {tr['sd']:.3f} | [{tr['min']:.3f}, {tr['max']:.3f}] | "
                f"{alloc_kb:,.1f} | {floss:.4f} | "
                f"{stats_obj['max_overlap']} |"
            )

    lines.append("")
    return "\n".join(lines)


def resolve_torch_python(explicit_exe: Optional[str]) -> str:
    candidates: List[str] = []
    if explicit_exe:
        candidates.append(explicit_exe)
    env_exe = os.environ.get("AIDOTNET_TORCH_PYTHON", "")
    if env_exe:
        candidates.append(env_exe)
    candidates.append(sys.executable)
    if os.name == "nt":
        candidates.append(r"C:\torchbench\venv-win\Scripts\python.exe")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        for ver in ("Python312", "Python311", "Python313", "Python310"):
            candidates.append(str(Path(local_app_data) / "Programs" / "Python" / ver / "python.exe"))

    probe_env = dict(os.environ)
    probe_env["CUDA_VISIBLE_DEVICES"] = ""
    seen = set()
    for exe in candidates:
        if not exe or exe in seen:
            continue
        seen.add(exe)
        if os.path.sep in exe and not Path(exe).exists():
            continue
        try:
            res = subprocess.run(
                [exe, "-c", "import torch; print(torch.__version__)"],
                env=probe_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
            if res.returncode == 0:
                return exe
        except Exception:
            continue

    raise RuntimeError(
        "Could not find a Python interpreter with 'torch' installed. Pass --python-exe <path-to-python>."
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ABBA multi-seed orchestrator for AiDotNet vs Compiled PyTorch.")
    p.add_argument("--cpu-dry-run", action="store_true", help="Run smoke dry-run strictly on CPU (GPU disabled).")
    p.add_argument("--gpu-index", type=int, default=1, help="CUDA GPU index when not in --cpu-dry-run (GPU 0 forbidden).")
    p.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "bf16", "fp16"])
    p.add_argument("--wave", default="all", choices=["all", "1", "2", "wave1", "wave2"])
    p.add_argument("--filter", default=None, help="Substring filter on workload_id.")
    p.add_argument("--seeds", default="42,1337,2026", help="Comma-separated integer seeds (>=3 recommended).")
    p.add_argument("--abba-repeats", type=int, default=2, choices=[1, 2], help="2 = full ABBA mirror per seed; 1 = forward ABC/CBA.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip-before", action="store_true", help="Omit the Before (uncached-eager) baseline arm.")
    p.add_argument("--skip-direct-ptx", action="store_true", help="Omit the PureDirectPtx arm if running 2-arm ABBA only.")
    p.add_argument("--skip-build", action="store_true", help="Reuse an already built AiDotNet.CrossFamilyCompiledSuite.dll.")
    p.add_argument("--local-tensors-root", default=None, help="Optional LocalTensorsRoot path for dotnet build.")
    p.add_argument("--python-exe", default=None, help="Python interpreter with PyTorch installed (auto-detected if omitted).")
    p.add_argument("--output-dir", default=None, help="Directory for raw pass JSONs + aggregated report.")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    env = dict(os.environ)

    if args.cpu_dry_run:
        device = "cpu"
        smoke = True
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["AIDOTNET_DISABLE_GPU"] = "1"
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        if args.seeds == "42,1337,2026":
            # Keep CPU dry-run fast unless user explicitly overrides --seeds; still exercises multi-pass ABBA.
            seeds = [42]
            abba_repeats = 1
        else:
            abba_repeats = args.abba_repeats
    else:
        if args.gpu_index == 0:
            raise SystemExit("ERROR: Safety gate violation — GPU 0 is strictly off-limits. Use --gpu-index 1 or --cpu-dry-run.")
        device = "cuda"
        smoke = False
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        env.pop("AIDOTNET_DISABLE_GPU", None)
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        if len(seeds) < 3:
            print(f"[ABBA] WARNING: Gate 10 requires >= 3 seeds for publication claims (got {len(seeds)}: {seeds}).")
        abba_repeats = args.abba_repeats

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (SUITE_DIR / "results" / ("cpu_dry_run" if args.cpu_dry_run else f"gpu{args.gpu_index}_{args.precision}"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_build:
        suite_dll = _find_built_suite_dll()
    else:
        suite_dll = build_dotnet_suite(args.local_tensors_root, env)

    arms: List[Tuple[str, Any]] = []

    if not args.skip_before:
        arms.append(
            (
                "AiDotNet-Before-UncachedEager",
                lambda seed, json_path: [
                    "dotnet",
                    str(suite_dll),
                    "--device",
                    device,
                    "--mode",
                    "uncached-eager",
                    "--precision",
                    args.precision,
                    "--wave",
                    args.wave,
                    "--seed",
                    str(seed),
                    "--warmup",
                    str(2 if smoke else args.warmup),
                    "--steps",
                    str(3 if smoke else args.steps),
                    "--output",
                    str(json_path),
                ]
                + (["--smoke"] if smoke else [])
                + (["--filter", args.filter] if args.filter else []),
            )
        )

    arms.append(
        (
            "AiDotNet-AutoCompiled",
            lambda seed, json_path: [
                "dotnet",
                str(suite_dll),
                "--device",
                device,
                "--mode",
                "auto",
                "--precision",
                args.precision,
                "--wave",
                args.wave,
                "--seed",
                str(seed),
                "--warmup",
                str(2 if smoke else args.warmup),
                "--steps",
                str(3 if smoke else args.steps),
                "--output",
                str(json_path),
            ]
            + (["--smoke"] if smoke else [])
            + (["--filter", args.filter] if args.filter else []),
        )
    )

    if not args.skip_direct_ptx:
        arms.append(
            (
                "AiDotNet-PureDirectPtx",
                lambda seed, json_path: [
                    "dotnet",
                    str(suite_dll),
                    "--device",
                    device,
                    "--mode",
                    "direct-ptx",
                    "--precision",
                    args.precision,
                    "--wave",
                    args.wave,
                    "--seed",
                    str(seed),
                    "--warmup",
                    str(2 if smoke else args.warmup),
                    "--steps",
                    str(3 if smoke else args.steps),
                    "--output",
                    str(json_path),
                ]
                + (["--smoke"] if smoke else [])
                + (["--filter", args.filter] if args.filter else []),
            )
        )

    torch_python = resolve_torch_python(args.python_exe)
    print(f"[ABBA] Using PyTorch Python interpreter: {torch_python}")

    arms.append(
        (
            "PyTorch-InductorMaxAutotune",
            lambda seed, json_path: [
                torch_python,
                "-u",
                str(PYTORCH_RUNNER),
                "--device",
                device,
                "--mode",
                "inductor-max-autotune",
                "--precision",
                args.precision,
                "--wave",
                args.wave,
                "--seed",
                str(seed),
                "--warmup",
                str(2 if smoke else args.warmup),
                "--steps",
                str(3 if smoke else args.steps),
                "--output",
                str(json_path),
            ]
            + (["--smoke"] if smoke else [])
            + (["--filter", args.filter] if args.filter else []),
        )
    )

    collected: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for seed_idx, seed in enumerate(seeds):
        forward_order = list(arms) if (seed_idx % 2 == 0) else list(reversed(arms))
        schedule = forward_order + list(reversed(forward_order)) if abba_repeats == 2 else forward_order

        for pass_idx, (arm_name, cmd_builder) in enumerate(schedule):
            raw_json = out_dir / f"raw_seed{seed}_pass{pass_idx}_{arm_name}.json"
            records = run_arm_pass(f"{arm_name} (seed={seed}, pass={pass_idx})", cmd_builder(seed, raw_json), raw_json, env)
            for rec in records:
                wid = rec["workload_id"]
                collected.setdefault(wid, {}).setdefault(arm_name, []).append(rec)

    workloads_summary: Dict[str, Dict[str, Any]] = {}
    for wid, arm_records in collected.items():
        workloads_summary[wid] = {}
        for arm_name, recs in arm_records.items():
            b1_meds = [float(r["b1_inference_latency"]["median_ms"]) for r in recs]
            bi_meds = [float(r["batched_inference_latency"]["median_ms"]) for r in recs]
            tr_meds = [float(r["batched_train_step_latency"]["median_ms"]) for r in recs]
            alloc_kbs = [float(r.get("managed_alloc_bytes_per_step", 0.0)) / 1024.0 for r in recs]
            final_losses = [float(r.get("final_eval_loss", 0.0)) for r in recs]
            overlaps = [int(r.get("train_eval_window_overlap", 0)) for r in recs]
            if any(ov != 0 for ov in overlaps):
                raise RuntimeError(f"Adversarial Gate 7 violation in {wid}/{arm_name}: non-zero overlap detected.")
            workloads_summary[wid][arm_name] = {
                "runs": len(recs),
                "parameter_count": int(recs[0]["model_spec"]["parameter_count"]),
                "max_overlap": max(overlaps) if overlaps else 0,
                "b1_median_ms": aggregate_metric(b1_meds),
                "batch_infer_median_ms": aggregate_metric(bi_meds),
                "train_step_median_ms": aggregate_metric(tr_meds),
                "managed_alloc_kb_median": aggregate_metric(alloc_kbs)["median"],
                "final_loss_median": aggregate_metric(final_losses)["median"],
            }

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "smoke": smoke,
        "precision": args.precision,
        "wave": args.wave,
        "seeds": seeds,
        "abba_repeats": abba_repeats,
        "workloads": workloads_summary,
    }

    summary_json = out_dir / "abba_summary.json"
    summary_md = out_dir / "abba_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(format_markdown_report(summary), encoding="utf-8")
    print(f"\n[ABBA] Completed! Summary JSON: {summary_json}")
    print(f"[ABBA] Markdown Report:       {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
