"""AiDotNet vs PyTorch training scoreboard: every parity family on every device.

For each (family, device) cell this measures training milliseconds per step for

  * AiDotNet -- the C# harness (../Program.cs) from a prebuilt harness directory
    (--ours-bin), so the scoreboard scores exactly the DLLs a perf track ships, and
  * PyTorch  -- benchmark.py in EVERY execution mode a PyTorch user would reach for
    on that device, then scores AiDotNet against the FASTEST of them.

Protocol (each "run" is a fresh process; its value is the median steady-state epoch
time divided by steps per epoch, i.e. compare.py's training row):

  1. Mode selection: every candidate PyTorch mode runs --select-runs times (default 3),
     interleaved. A mode that fails (compiler missing, compile error) is recorded with
     its error and dropped.
  2. Scoring: AiDotNet, the --finalists best selection modes (default 2) and PyTorch
     eager (the kernel-level reference) each run --runs times (default 9), interleaved
     round by round. The finalists are re-measured from scratch so a lucky selection
     run cannot bias the bar; the finalist with the lowest scored median is the bar.
  3. The cell reports the median of the --runs values with their p25-p75, the ratio
     ours/theirs and compare.py's verdict (WIN/win/lose/LOSE: capitals only when the
     IQRs do not overlap).

Every run holds the machine-wide Global\\AiDotNetBenchLock mutex (the same lock
_bench/bench.ps1 uses) for its own duration only, so other benchmark tracks
interleave between runs instead of waiting for the whole sweep, and no two timed
runs ever overlap.

Outputs (in --output-dir): scoreboard.json (every raw run value, the selection
table, skipped modes with reasons, machine + DLL provenance) and scoreboard.md.

Usage (from this folder, with the torch venv's python):
    python scoreboard.py --ours-bin C:\\Users\\yolan\\source\\repos\\_bench\\baseline-bin
    python scoreboard.py --ours-bin <dir> --models cnn --devices cpu --runs 9
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from compare import Stat, _index, _verdict

HERE = Path(__file__).resolve().parent
FAMILIES = ["mlp", "cnn", "lstm", "transformer"]
DEVICES = ["cpu", "cuda"]
LOCK_NAME = "Global\\AiDotNetBenchLock"


@dataclass(frozen=True)
class PyTorchMode:
    """One PyTorch execution mode: its label (benchmark.py's describe_mode), its flags, and what it needs."""
    label: str
    flags: tuple[str, ...]
    devices: tuple[str, ...]
    needs: str | None = None  # "msvc" (CPU Inductor C++), "triton" (GPU Inductor)


def _modes() -> list[PyTorchMode]:
    fused = ("--optimizer-impl", "fused")
    out: list[PyTorchMode] = []
    for opt_label, opt_flags in (("", ()), ("+fused-adamw", fused)):
        out.append(PyTorchMode(f"eager{opt_label}", opt_flags, ("cpu", "cuda")))
        out.append(PyTorchMode(f"compile[inductor]{opt_label}", ("--compile",) + opt_flags, ("cpu",), "msvc"))
        out.append(PyTorchMode(f"compile[inductor]{opt_label}", ("--compile",) + opt_flags, ("cuda",), "triton"))
        out.append(PyTorchMode(f"compile[inductor/reduce-overhead]{opt_label}",
                               ("--compile", "--compile-mode", "reduce-overhead") + opt_flags, ("cuda",), "triton"))
        out.append(PyTorchMode(f"compile[cudagraphs]{opt_label}",
                               ("--compile", "--compile-backend", "cudagraphs") + opt_flags, ("cuda",)))
    return out


# Modes deliberately not run, with the reason recorded in every scoreboard.
NOT_RUN = {
    "compile[inductor/max-autotune*]": "autotuning benchmarks GEMM/conv templates for minutes per process while "
                                       "holding the shared bench lock, and for these small shapes it targets the same "
                                       "kernels reduce-overhead already graphs; not run.",
    "compile[inductor/reduce-overhead], compile[cudagraphs] on cpu": "CUDA graphs exist only on CUDA.",
    "compile[aot_eager]": "a debugging backend that runs the eager kernels through AOTAutograd; never faster than eager.",
}


# ---------------------------------------------------------------------------------------------- the lock

class BenchLock:
    """The Win32 named mutex bench.ps1 takes ([System.Threading.Mutex] 'Global\\AiDotNetBenchLock').
    WAIT_ABANDONED (a previous holder died) still grants ownership, as in bench.ps1."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateMutexW.restype = ctypes.c_void_p
        self._k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        self._k32.WaitForSingleObject.restype = ctypes.c_uint32
        self._k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._k32.ReleaseMutex.argtypes = [ctypes.c_void_p]
        self._k32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._handle = None

    def __enter__(self) -> BenchLock:
        handle = self._k32.CreateMutexW(None, False, self.name)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"CreateMutexW({self.name}) failed")
        result = self._k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result not in (0x0, 0x80):  # WAIT_OBJECT_0, WAIT_ABANDONED
            self._k32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(), f"WaitForSingleObject({self.name}) returned {result:#x}")
        self._handle = handle
        return self

    def __exit__(self, *exc: object) -> None:
        self._k32.ReleaseMutex(self._handle)
        self._k32.CloseHandle(self._handle)
        self._handle = None


# ------------------------------------------------------------------------------------ toolchain probing

def find_msvc_env(vcvars: str | None) -> tuple[dict[str, str] | None, str]:
    """CPU TorchInductor compiles C++ with cl.exe. Return the environment of vcvars64.bat
    (or None plus the reason) so the compile modes can run from a plain shell."""
    if shutil.which("cl"):
        return dict(os.environ), "cl.exe already on PATH"
    candidates: list[str] = [vcvars] if vcvars else []
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not candidates and vswhere.exists():
        found = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                               capture_output=True, text=True).stdout.strip()
        if found:
            candidates.append(str(Path(found.splitlines()[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"))
    if not candidates:
        for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            candidates += sorted(glob.glob(os.path.join(root, "Microsoft Visual Studio", "*", "*", "VC", "Auxiliary", "Build", "vcvars64.bat")), reverse=True)
    for bat in candidates:
        if not Path(bat).exists():
            continue
        proc = subprocess.run(f'cmd /d /c ""{bat}" >nul 2>&1 && set"', capture_output=True, text=True, shell=False)
        env = {}
        for line in proc.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep and key:
                env[key] = value
        if env and shutil.which("cl", path=env.get("Path") or env.get("PATH")):
            return env, f"vcvars64: {bat}"
    return None, "no MSVC cl.exe found (install VS Build Tools with the C++ workload, or pass --vcvars)"


def probe_triton(python: str) -> tuple[bool, str]:
    proc = subprocess.run([python, "-c", "import triton, sys; sys.stdout.write(triton.__version__)"],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return True, f"triton {proc.stdout.strip()}"
    return False, "triton is not importable (pip install triton-windows); GPU Inductor needs it"


def file_provenance(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    version = None
    try:
        version = subprocess.run(["pwsh", "-NoProfile", "-Command", f"(Get-Item -LiteralPath '{path}').VersionInfo.ProductVersion"],
                                 capture_output=True, text=True, timeout=60).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"path": str(path), "sha256": digest,
            "lastWriteUtc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
            "productVersion": version}


def machine_info(python: str) -> dict[str, object]:
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    torch_info = subprocess.run(
        [python, "-c", "import json, torch; print(json.dumps({'torch': torch.__version__, 'threads': torch.get_num_threads(), "
                       "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"],
        capture_output=True, text=True)
    dotnet = subprocess.run(["dotnet", "--version"], capture_output=True, text=True)
    return {
        "os": platform.platform(),
        "cpu": cpu,
        "logicalProcessors": os.cpu_count(),
        "python": platform.python_version(),
        "pytorch": json.loads(torch_info.stdout) if torch_info.returncode == 0 else torch_info.stderr[-500:],
        "dotnetSdk": dotnet.stdout.strip(),
    }


# ---------------------------------------------------------------------------------------------- runs

@dataclass
class Series:
    """The per-run ms/step values of one contender in one cell."""
    values: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def stat(self) -> Stat | None:
        if not self.values:
            return None
        s = sorted(self.values)
        return Stat(_quantile(s, 0.5), _quantile(s, 0.25), _quantile(s, 0.75))

    def to_json(self) -> dict[str, object]:
        st = self.stat()
        return {"runsMs": [round(v, 4) for v in self.values],
                "medianMs": None if st is None else round(st.median, 4),
                "p25Ms": None if st is None else round(st.p25, 4),
                "p75Ms": None if st is None else round(st.p75, 4),
                "minMs": round(min(self.values), 4) if self.values else None,
                "maxMs": round(max(self.values), 4) if self.values else None,
                "errors": self.errors}


def _quantile(sorted_values: list[float], q: float) -> float:
    # benchmark.quantile is the same function, but importing benchmark loads torch into this orchestrator.
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


class Runner:
    def __init__(self, args: argparse.Namespace, msvc_env: dict[str, str] | None) -> None:
        self.args = args
        self.lock = BenchLock(args.lock_name)
        self.workdir = Path(tempfile.mkdtemp(prefix="scoreboard-"))
        self.ours_env = dict(os.environ)
        self.torch_env = dict(msvc_env or os.environ)
        if args.threads > 0:
            self.ours_env["AIDOTNET_BLAS_THREADS"] = str(args.threads)
        self.count = 0

    def _common(self) -> list[str]:
        a = self.args
        return ["--epochs", str(a.epochs), "--train-batches", str(a.train_batches), "--batch-size", str(a.batch_size),
                "--seed", str(a.seed)]

    def _run(self, cmd: list[str], env: dict[str, str], model: str, device: str, label: str) -> tuple[float | None, str | None]:
        out = self.workdir / f"run-{self.count}.json"
        self.count += 1
        started = time.perf_counter()
        try:
            with self.lock:
                proc = subprocess.run(cmd + ["--output", str(out)], env=env, cwd=self.workdir, capture_output=True,
                                      text=True, encoding="utf-8", errors="replace", timeout=self.args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  {model:<12}{device:<5}{label:<44} TIMED OUT after {self.args.timeout}s", flush=True)
            return None, f"timed out after {self.args.timeout}s"
        wall = time.perf_counter() - started
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            err = next((line for line in reversed(tail) if line.strip()), f"exit {proc.returncode}")
            print(f"  {model:<12}{device:<5}{label:<44} FAILED ({err[:160]})", flush=True)
            return None, err[:500]
        report = json.loads(out.read_text(encoding="utf-8"))
        out.unlink()
        rows = _index(report).get(model)
        if rows is None or rows.training is None or rows.device != device:
            return None, f"report has no {model} training row on {device} (device={rows.device if rows else None})"
        print(f"  {model:<12}{device:<5}{label:<44}{rows.training.median:10.3f} ms/step  ({wall:.0f}s wall)", flush=True)
        return rows.training.median, None

    def ours(self, model: str, device: str) -> tuple[float | None, str | None]:
        cmd = ["dotnet", str(Path(self.args.ours_bin) / "AiDotNet.PyTorchParity.dll"), "--models", model, "--device", device,
               # The harness has no training-only switch; one inference iteration per batch size keeps it negligible.
               "--inference-iterations", "1", "--warmup-iterations", "1"] + self._common()
        return self._run(cmd, self.ours_env, model, device, "AiDotNet")

    def pytorch(self, model: str, device: str, mode: PyTorchMode) -> tuple[float | None, str | None]:
        cmd = [self.args.python, str(HERE / "benchmark.py"), "--models", model, "--device", device, "--skip-inference"] \
              + list(mode.flags) + self._common()
        if self.args.threads > 0:
            cmd += ["--threads", str(self.args.threads)]
        return self._run(cmd, self.torch_env, model, device, f"PyTorch {mode.label}")


def score_cell(runner: Runner, model: str, device: str, candidates: list[PyTorchMode],
               unavailable: dict[str, str], select_runs: int, runs: int, finalist_count: int) -> dict[str, object]:
    print(f"[scoreboard] {model}/{device}: selecting the best PyTorch mode from {len(candidates)} candidates", flush=True)
    selection: dict[str, Series] = {m.label: Series() for m in candidates}
    live = list(candidates)
    for _ in range(select_runs):
        for mode in list(live):
            value, err = runner.pytorch(model, device, mode)
            if value is None:
                selection[mode.label].errors.append(err or "failed")
                live.remove(mode)  # a mode that cannot run is not retried
            else:
                selection[mode.label].values.append(value)
    ranked = sorted((m for m in live if selection[m.label].values), key=lambda m: selection[m.label].stat().median)
    if not ranked:
        raise RuntimeError(f"{model}/{device}: no PyTorch mode ran")
    # Selection runs are few and process-to-process noise is large (GPU especially), so the top
    # `finalists` modes are ALL re-measured from scratch and the best FINAL median is the bar. Eager is
    # always re-measured too: it is the kernel-level reference column.
    finalists = ranked[:max(1, finalist_count)]
    eager = next(m for m in candidates if m.label == "eager")
    contenders = finalists + ([eager] if eager not in finalists else [])

    print(f"[scoreboard] {model}/{device}: scoring AiDotNet vs PyTorch {', '.join(m.label for m in contenders)}, "
          f"{runs} runs each", flush=True)
    ours = Series()
    final: dict[str, Series] = {m.label: Series() for m in contenders}
    for _ in range(runs):
        jobs = [(ours, lambda: runner.ours(model, device))]
        jobs += [(final[m.label], (lambda m=m: runner.pytorch(model, device, m))) for m in contenders]
        for series, fn in jobs:
            value, err = fn()
            if value is None:
                series.errors.append(err or "failed")
            else:
                series.values.append(value)
    scored = [m for m in finalists if final[m.label].values]
    if not scored:
        raise RuntimeError(f"{model}/{device}: no PyTorch finalist completed a scored run")
    best = min(scored, key=lambda m: final[m.label].stat().median)
    theirs, eager_series = final[best.label], final[eager.label]

    ours_stat, theirs_stat = ours.stat(), theirs.stat()
    verdict = ratio = None
    if ours_stat and theirs_stat:
        verdict = _verdict(ours_stat, theirs_stat)[0].strip()
        ratio = round(ours_stat.median / theirs_stat.median, 3)
    return {
        "model": model, "device": device,
        "aidotnet": ours.to_json(),
        "pytorchBestMode": best.label,
        "pytorchBest": theirs.to_json(),
        "pytorchEager": eager_series.to_json(),
        "ratioOursOverBest": ratio,
        "ratioOursOverEager": round(ours_stat.median / eager_series.stat().median, 3) if ours_stat and eager_series.stat() else None,
        "verdict": verdict,
        "selection": {label: s.to_json() for label, s in selection.items()},
        "final": {label: s.to_json() for label, s in final.items()},
        "modesUnavailable": unavailable,
    }


# -------------------------------------------------------------------------------------------- reports

def _cell(series: dict[str, object]) -> str:
    if series.get("medianMs") is None:
        return "n/a"
    return f"{series['medianMs']:.3f} [{series['p25Ms']:.3f}-{series['p75Ms']:.3f}]"


def write_markdown(report: dict[str, object], path: Path) -> None:
    cfg, machine = report["config"], report["machine"]
    torch_info = machine["pytorch"] if isinstance(machine["pytorch"], dict) else {}
    lines = [
        "# AiDotNet vs PyTorch training scoreboard",
        "",
        f"Generated {report['generatedUtc']} by `pytorch/scoreboard.py`. Training **ms per step** (forward + backward + "
        f"clip + AdamW), batch {cfg['batchSize']}, {cfg['trainBatches']} steps/epoch, {cfg['epochs']} epochs per process "
        f"(epoch 0 is warmup). Each cell is the **median of {cfg['runs']} process runs** with p25-p75; every run held "
        f"`{cfg['lockName']}` on its own.",
        "",
        "PyTorch is scored in its **fastest mode for that cell**: every candidate mode ran "
        f"{cfg['selectRuns']}x first, then the best {cfg['finalists']} (plus eager) were re-measured {cfg['runs']}x from "
        "scratch, interleaved with AiDotNet, and the lowest final median is the bar (see *Mode selection*).",
        "",
        "| family | device | AiDotNet ms/step | best PyTorch mode | PyTorch ms/step | ours / PyTorch | verdict | PyTorch eager ms/step | ours / eager |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cells"]:
        lines.append(f"| {c['model']} | {c['device']} | {_cell(c['aidotnet'])} | {c['pytorchBestMode']} | {_cell(c['pytorchBest'])} | "
                     f"{c['ratioOursOverBest'] if c['ratioOursOverBest'] is not None else 'n/a'}x | {c['verdict'] or 'n/a'} | "
                     f"{_cell(c['pytorchEager'])} | {c['ratioOursOverEager'] if c['ratioOursOverEager'] is not None else 'n/a'}x |")
    wins = sum(1 for c in report["cells"] if c["verdict"] == "WIN")
    losses = sum(1 for c in report["cells"] if c["verdict"] == "LOSE")
    lines += [
        "",
        f"**{wins} decisive wins, {losses} decisive losses, {len(report['cells']) - wins - losses} within noise** "
        "(verdicts as in `compare.py`: WIN/LOSE only when the run IQRs do not overlap; lowercase = medians differ but IQRs overlap).",
        "",
        "## Mode selection",
        "",
        f"Median ms/step of the {cfg['selectRuns']} selection runs; for the re-measured modes, `-> ` the median of the "
        f"{cfg['runs']} scored runs. Bold = the mode scored against.",
        "",
    ]
    labels = sorted({label for c in report["cells"] for label in c["selection"]})
    lines.append("| family | device | " + " | ".join(labels) + " |")
    lines.append("|---|---|" + "---|" * len(labels))
    for c in report["cells"]:
        row = []
        for label in labels:
            s = c["selection"].get(label)
            if s is None:
                row.append("-")
            elif s["medianMs"] is None:
                row.append("failed")
            else:
                mark = "**" if label == c["pytorchBestMode"] else ""
                fin = c["final"].get(label)
                tail = f" -> {fin['medianMs']:.3f}" if fin and fin["medianMs"] is not None else ""
                row.append(f"{mark}{s['medianMs']:.3f}{tail}{mark}")
        lines.append(f"| {c['model']} | {c['device']} | " + " | ".join(row) + " |")
    failures = [(c["model"], c["device"], label, s["errors"][0]) for c in report["cells"]
                for label, s in c["selection"].items() if s["errors"]]
    lines += ["", "## Modes skipped or failed", ""]
    for label, reason in report["modesNotRun"].items():
        lines.append(f"- `{label}`: {reason}")
    for label, reason in report["modesUnavailable"].items():
        lines.append(f"- `{label}`: unavailable on this machine: {reason}")
    for model, device, label, err in failures:
        lines.append(f"- `{label}` on {model}/{device} failed: `{err[:200].replace('`', '')}`")
    ours = report["aidotnet"]
    lines += [
        "",
        "## Provenance",
        "",
        f"- Machine: {machine['cpu']} ({machine['logicalProcessors']} logical), GPU {torch_info.get('gpu')}, {machine['os']}",
        f"- PyTorch {torch_info.get('torch')} (CUDA {torch_info.get('cuda')}, {torch_info.get('threads')} intra-op threads), "
        f"Python {machine['python']}, {report['toolchain']['triton']}, MSVC: {report['toolchain']['msvc']}",
        f"- AiDotNet harness: `{cfg['oursBin']}` (.NET SDK {machine['dotnetSdk']}); thread pin: "
        f"{cfg['threads'] if cfg['threads'] > 0 else 'none (each side at its default)'}",
    ]
    for name, prov in ours.items():
        lines.append(f"  - {name}: {prov.get('productVersion')} sha256 {str(prov.get('sha256'))[:16]}")
    lines += ["", "Raw per-run values: `scoreboard.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="AiDotNet vs best-mode PyTorch training scoreboard (median of N, locked).")
    parser.add_argument("--ours-bin", required=True, type=Path,
                        help="Directory holding a built AiDotNet.PyTorchParity.dll (e.g. _bench\\baseline-bin or a track's B dir).")
    parser.add_argument("--models", default=",".join(FAMILIES))
    parser.add_argument("--devices", default=",".join(DEVICES))
    parser.add_argument("--runs", type=int, default=9, help="scored process runs per contender per cell")
    parser.add_argument("--select-runs", type=int, default=3, help="runs per candidate PyTorch mode during mode selection")
    parser.add_argument("--finalists", type=int, default=2,
                        help="how many of the best selection modes are re-measured --runs times; the best final median wins")
    parser.add_argument("--modes", default="", help="comma-separated subset of PyTorch mode labels to consider (default: all)")
    parser.add_argument("--epochs", type=int, default=5, help="epochs per process (epoch 0 is warmup, the rest steady state)")
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0,
                        help="pin both sides' CPU threads (torch --threads + AIDOTNET_BLAS_THREADS); 0 = each side's default")
    parser.add_argument("--python", default=sys.executable, help="python with torch for the PyTorch side")
    parser.add_argument("--vcvars", default=None, help="vcvars64.bat for CPU TorchInductor (auto-detected by default)")
    parser.add_argument("--lock-name", default=LOCK_NAME)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds before a single run is abandoned")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "results")
    args = parser.parse_args()

    if not (args.ours_bin / "AiDotNet.PyTorchParity.dll").exists():
        parser.error(f"{args.ours_bin} has no AiDotNet.PyTorchParity.dll")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    wanted = {m.strip() for m in args.modes.split(",") if m.strip()}

    msvc_env, msvc_note = find_msvc_env(args.vcvars)
    triton_ok, triton_note = probe_triton(args.python)
    unavailable: dict[str, str] = {}
    candidates_by_device: dict[str, list[PyTorchMode]] = {}
    for device in devices:
        chosen = []
        for mode in _modes():
            if device not in mode.devices or (wanted and mode.label not in wanted and mode.label != "eager"):
                continue
            if mode.needs == "msvc" and msvc_env is None:
                unavailable[f"{mode.label} on {device}"] = msvc_note
                continue
            if mode.needs == "triton" and not triton_ok:
                unavailable[f"{mode.label} on {device}"] = triton_note
                continue
            chosen.append(mode)
        candidates_by_device[device] = chosen

    runner = Runner(args, msvc_env)
    started = dt.datetime.now(dt.timezone.utc)
    cells = []
    try:
        for device in devices:
            for model in models:
                cell_unavailable = {k: v for k, v in unavailable.items() if k.endswith(f" on {device}")}
                cells.append(score_cell(runner, model, device, candidates_by_device[device], cell_unavailable,
                                        args.select_runs, args.runs, args.finalists))
    finally:
        shutil.rmtree(runner.workdir, ignore_errors=True)

    report = {
        "generatedUtc": started.isoformat(timespec="seconds"),
        "finishedUtc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": {"oursBin": str(args.ours_bin), "models": models, "devices": devices, "runs": args.runs,
                   "selectRuns": args.select_runs, "finalists": args.finalists, "epochs": args.epochs, "trainBatches": args.train_batches,
                   "batchSize": args.batch_size, "seed": args.seed, "threads": args.threads, "lockName": args.lock_name,
                   "statistic": "per run: median steady-state epoch seconds / steps (epoch 0 excluded); per cell: median of runs"},
        "machine": machine_info(args.python),
        "toolchain": {"msvc": msvc_note, "triton": triton_note},
        "aidotnet": {name: file_provenance(args.ours_bin / name)
                     for name in ("AiDotNet.dll", "AiDotNet.Tensors.dll", "AiDotNet.PyTorchParity.dll")},
        "modesNotRun": NOT_RUN,
        "modesUnavailable": unavailable,
        "cells": cells,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scoreboard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output_dir / "scoreboard.md")
    print((args.output_dir / "scoreboard.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
