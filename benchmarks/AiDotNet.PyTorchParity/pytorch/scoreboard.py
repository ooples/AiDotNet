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

The lock does not stop builds and test runs, which on a shared box inflated CPU
steps up to 6x. Three defences, applied identically to both sides:

  * PRIORITY: every timed process starts at --priority (default above-normal), so
    the normal-priority builds and test hosts of other tracks yield the CPUs to it
    instead of time-slicing with it. (Not inherited by TorchInductor's compile
    workers, which run before the timed epochs anyway.)
  * QUIET GATE: before a run starts (lock held), the system-wide CPU load is
    sampled; above --max-background-pct the run waits with the lock RELEASED, for at
    most --quiet-wait seconds, then proceeds.
  * DISCARD: afterwards the load from OTHER processes during the run (system busy
    CPU time minus the CPU time of the run's whole process tree, from a Win32 job
    object) is computed; a run above --max-background-pct is discarded and redone,
    up to --max-retries times, after which the last value is kept AND flagged.
    Every kept run's background load is in the JSON and summarised in the markdown,
    and every discarded run is listed with its value and load.

Interleaving (ours and each PyTorch finalist alternate run by run) makes whatever
noise remains hit both sides alike rather than one side's whole series.

Outputs (in --output-dir): scoreboard.json (every raw run value, the selection
table, skipped modes with reasons, machine + DLL provenance) and scoreboard.md.

Usage (from this folder, with the torch venv's python):
    python scoreboard.py --ours-bin C:\\Users\\yolan\\source\\repos\\_bench\\baseline-bin
    python scoreboard.py --ours-bin <dir> --models cnn --devices cpu --runs 9
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
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

import psutil

from compare import Stat, _index, _verdict

HERE = Path(__file__).resolve().parent
FAMILIES = ["mlp", "cnn", "lstm", "transformer"]
DEVICES = ["cpu", "cuda"]
LOCK_NAME = "Global\\AiDotNetBenchLock"
# Win32 priority classes a timed run can start at (--priority); both sides always get the same one.
PRIORITY_CLASSES = {"normal": 0x00000020, "above-normal": 0x00008000, "high": 0x00000080}


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


# ------------------------------------------------------------------------------------- background load

def _system_busy_seconds() -> float:
    """Busy CPU seconds summed over all logical CPUs since boot (user + kernel, idle excluded)."""
    t = psutil.cpu_times()
    return t.user + t.system


class _JobAccounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", ctypes.c_uint32), ("TotalProcesses", ctypes.c_uint32),
                ("ActiveProcesses", ctypes.c_uint32), ("TotalTerminatedProcesses", ctypes.c_uint32)]


class ProcessTree:
    """A Win32 job object holding a run's process and every process it spawns (TorchInductor compiles in
    worker subprocesses), so the run's OWN CPU time covers the whole tree, including processes that have
    already exited, and a timed-out run can be killed as a tree."""

    def __init__(self) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.restype = ctypes.c_void_p
        self._k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self._k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._k32.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        self._k32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._k32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._job = self._k32.CreateJobObjectW(None, None)
        if not self._job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    def adopt(self, process_handle: int) -> None:
        if not self._k32.AssignProcessToJobObject(self._job, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def cpu_seconds(self) -> float:
        info = _JobAccounting()
        if not self._k32.QueryInformationJobObject(self._job, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7

    def kill(self) -> None:
        self._k32.TerminateJobObject(self._job, 1)

    def close(self) -> None:
        self._k32.CloseHandle(self._job)


def sample_background_pct(seconds: float = 1.0) -> float:
    """System-wide CPU load over `seconds`, as % of all logical CPUs (this process is idle meanwhile)."""
    return psutil.cpu_percent(interval=seconds)


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
    """The per-run ms/step values of one contender in one cell, with each kept run's background
    load and the runs discarded for a noisy machine."""
    values: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    background: list[float] = field(default_factory=list)
    discarded: list[dict[str, float]] = field(default_factory=list)

    def add(self, result: RunResult) -> None:
        self.discarded.extend(result.discarded)
        if result.value is None:
            self.errors.append(result.error or "failed")
        else:
            self.values.append(result.value)
            self.background.append(result.background_pct)

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
                "backgroundCpuPct": [round(v, 1) for v in self.background],
                "discardedNoisyRuns": self.discarded,
                "errors": self.errors}


@dataclass
class RunResult:
    value: float | None
    error: str | None = None
    background_pct: float = 0.0
    discarded: list[dict[str, float]] = field(default_factory=list)


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
        self.ours_bin = Path(args.ours_bin)
        if args.snapshot:
            # --ours-bin is often a shared directory (the bench baseline) that another track may refresh during
            # a multi-hour sweep; running a private copy keeps every AiDotNet run on one build.
            self.ours_bin = self.workdir / "ours-bin"
            shutil.copytree(args.ours_bin, self.ours_bin)
        self.ours_env = dict(os.environ)
        self.torch_env = dict(msvc_env or os.environ)
        if args.threads > 0:
            self.ours_env["AIDOTNET_BLAS_THREADS"] = str(args.threads)
        self.count = 0

    def _common(self) -> list[str]:
        a = self.args
        return ["--epochs", str(a.epochs), "--train-batches", str(a.train_batches), "--batch-size", str(a.batch_size),
                "--seed", str(a.seed)]

    def _acquire_quiet(self) -> float:
        """Take the lock on a quiet machine; returns the pre-run load. Waits with the lock RELEASED so
        other tracks are not blocked by our wait; after --quiet-wait seconds it proceeds anyway (the
        post-run background check still discards the run if the noise persists)."""
        deadline = time.monotonic() + self.args.quiet_wait
        while True:
            self.lock.__enter__()
            load = sample_background_pct()
            if load <= self.args.max_background_pct or time.monotonic() >= deadline:
                return load
            self.lock.__exit__()
            print(f"    machine busy ({load:.0f}% CPU > {self.args.max_background_pct:.0f}%), waiting", flush=True)
            time.sleep(self.args.quiet_poll)

    def _run_once(self, cmd: list[str], env: dict[str, str], out: Path) -> tuple[subprocess.CompletedProcess | None, float, float, float]:
        """One locked, quiet-gated run. Returns (proc or None on timeout, pre-run load %, background load % during
        the run, wall seconds)."""
        pre = self._acquire_quiet()
        tree = ProcessTree()
        try:
            busy0, started = _system_busy_seconds(), time.perf_counter()
            with subprocess.Popen(cmd + ["--output", str(out)], env=env, cwd=self.workdir, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                  creationflags=PRIORITY_CLASSES[self.args.priority]) as popen:
                # Adopted before the interpreter/runtime has started, so every later child is in the job too.
                tree.adopt(int(popen._handle))
                try:
                    stdout, stderr = popen.communicate(timeout=self.args.timeout)
                except subprocess.TimeoutExpired:
                    tree.kill()
                    popen.communicate()
                    return None, pre, 0.0, time.perf_counter() - started
                wall = time.perf_counter() - started
                busy = _system_busy_seconds() - busy0
                own = tree.cpu_seconds()
                tree.kill()  # reap lingering compile workers so they cannot load the next run
        finally:
            tree.close()
            self.lock.__exit__()
        background = max(0.0, busy - own) / (wall * (os.cpu_count() or 1)) * 100
        return subprocess.CompletedProcess(cmd, popen.returncode, stdout, stderr), pre, background, wall

    def _run(self, cmd: list[str], env: dict[str, str], model: str, device: str, label: str) -> RunResult:
        discarded: list[dict[str, float]] = []
        for attempt in range(self.args.max_retries + 1):
            out = self.workdir / f"run-{self.count}.json"
            self.count += 1
            proc, pre, background, wall = self._run_once(cmd, env, out)
            if proc is None:
                print(f"  {model:<12}{device:<5}{label:<44} TIMED OUT after {self.args.timeout}s", flush=True)
                return RunResult(None, f"timed out after {self.args.timeout}s", discarded=discarded)
            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                err = next((line for line in reversed(tail) if line.strip()), f"exit {proc.returncode}")
                print(f"  {model:<12}{device:<5}{label:<44} FAILED ({err[:160]})", flush=True)
                return RunResult(None, err[:500], discarded=discarded)
            report = json.loads(out.read_text(encoding="utf-8"))
            out.unlink()
            rows = _index(report).get(model)
            if rows is None or rows.training is None or rows.device != device:
                return RunResult(None, f"report has no {model} training row on {device} "
                                       f"(device={rows.device if rows else None})", discarded=discarded)
            value = rows.training.median
            noisy = background > self.args.max_background_pct
            print(f"  {model:<12}{device:<5}{label:<44}{value:10.3f} ms/step  ({wall:.0f}s, background "
                  f"{background:.0f}%{', DISCARDED' if noisy and attempt < self.args.max_retries else ''})", flush=True)
            if noisy and attempt < self.args.max_retries:
                discarded.append({"ms": round(value, 4), "backgroundCpuPct": round(background, 1)})
                time.sleep(self.args.quiet_poll)
                continue
            return RunResult(value, None, background, discarded)
        raise AssertionError("unreachable")

    def ours(self, model: str, device: str) -> RunResult:
        cmd = ["dotnet", str(self.ours_bin / "AiDotNet.PyTorchParity.dll"), "--models", model, "--device", device,
               # The harness has no training-only switch; one inference iteration per batch size keeps it negligible.
               "--inference-iterations", "1", "--warmup-iterations", "1"] + self._common()
        return self._run(cmd, self.ours_env, model, device, "AiDotNet")

    def pytorch(self, model: str, device: str, mode: PyTorchMode) -> RunResult:
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
            result = runner.pytorch(model, device, mode)
            selection[mode.label].add(result)
            if result.value is None:
                live.remove(mode)  # a mode that cannot run is not retried
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
            series.add(fn())
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
        f"`{cfg['lockName']}` on its own, ran at {cfg['priority']} priority (both sides), and was quiet-gated: a run "
        f"during which other processes used more than {cfg['maxBackgroundPct']:.0f}% of the CPUs was discarded and "
        f"redone (up to {cfg['maxRetries']}x; see *Machine noise*).",
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
    lines += ["", "## Machine noise", "",
              "CPU used by OTHER processes during each kept run (% of all logical CPUs), and runs discarded and redone "
              "because it exceeded the threshold.", "",
              "| family | device | contender | kept runs median background % | kept runs max background % | "
              f"kept runs over {cfg['maxBackgroundPct']:.0f}% | discarded runs |", "|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        contenders = [("AiDotNet", c["aidotnet"])] + [(f"PyTorch {k}", v) for k, v in c["final"].items()]
        contenders += [(f"PyTorch {k} (selection)", v) for k, v in c["selection"].items()]
        for name, s in contenders:
            bg = sorted(s.get("backgroundCpuPct") or [])
            over = sum(1 for v in bg if v > cfg["maxBackgroundPct"])
            lines.append(f"| {c['model']} | {c['device']} | {name} | {round(_quantile(bg, 0.5), 1) if bg else 'n/a'} | "
                         f"{max(bg) if bg else 'n/a'} | {over} | {len(s.get('discardedNoisyRuns') or [])} |")
    ours = report["aidotnet"]
    lines += [
        "",
        "## Provenance",
        "",
        f"- Machine: {machine['cpu']} ({machine['logicalProcessors']} logical), GPU {torch_info.get('gpu')}, {machine['os']}",
        f"- PyTorch {torch_info.get('torch')} (CUDA {torch_info.get('cuda')}, {torch_info.get('threads')} intra-op threads), "
        f"Python {machine['python']}, {report['toolchain']['triton']}, MSVC: {report['toolchain']['msvc']}",
        f"- AiDotNet harness: `{cfg['oursBin']}`{' (run from a copy taken at the start)' if cfg.get('oursBinSnapshot') else ''} "
        f"(.NET SDK {machine['dotnetSdk']}); thread pin: "
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
    parser.add_argument("--priority", choices=sorted(PRIORITY_CLASSES), default="above-normal",
                        help="Win32 priority class of every timed process (both sides)")
    parser.add_argument("--max-background-pct", type=float, default=25.0,
                        help="max CPU load (%% of all logical CPUs) from OTHER processes before/during a run; noisier runs are redone")
    parser.add_argument("--max-retries", type=int, default=2, help="re-runs of a run discarded for background load")
    parser.add_argument("--quiet-wait", type=float, default=120, help="max seconds to wait for a quiet machine before a run")
    parser.add_argument("--quiet-poll", type=float, default=10, help="seconds between quiet-machine checks")
    parser.add_argument("--snapshot", action=argparse.BooleanOptionalAction, default=True,
                        help="run AiDotNet from a private copy of --ours-bin taken at the start (default on), so a "
                             "shared directory refreshed mid-sweep cannot mix two builds into one scoreboard")
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
    # Provenance of exactly the DLLs that run (the snapshot when --snapshot is on).
    ours_provenance = {name: file_provenance(runner.ours_bin / name)
                       for name in ("AiDotNet.dll", "AiDotNet.Tensors.dll", "AiDotNet.PyTorchParity.dll")}
    for prov in ours_provenance.values():
        prov["path"] = str(args.ours_bin / Path(str(prov["path"])).name)
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
                   "maxBackgroundPct": args.max_background_pct, "maxRetries": args.max_retries, "quietWait": args.quiet_wait,
                   "priority": args.priority, "oursBinSnapshot": args.snapshot,
                   "statistic": "per run: median steady-state epoch seconds / steps (epoch 0 excluded); per cell: median of runs"},
        "machine": machine_info(args.python),
        "toolchain": {"msvc": msvc_note, "triton": triton_note},
        "aidotnet": ours_provenance,
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
