"""Run end-to-end AiDotNet/Python PyTorch training and inference comparisons from this checkout.

Every model, device and launch runs AiDotNet and each applicable PyTorch configuration in its own fresh process, in an
order that rotates between launches. AiDotNet is compared against the fastest PyTorch configuration for each row.
Every row reports the median and interquartile range over the launches (9 by default), the trainable parameter
counts of both sides, and the first and last training loss of both sides as evidence that the timed step trains.

--classical runs the classical-ML rows (ClassicalBench.cs against scikit-learn, XGBoost and LightGBM) the same way.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pytorch"))
from compare import _index

CUDA_ENGINE = "Direct GPU Engine (CUDA "
CORE_MODELS = "mlp lstm transformer cnn resnet18 resnet50 vgg11 vit gru mobilenetv2 densenet121".split()
REFERENCE = ("mlp", "cnn", "lstm", "transformer")
BATCHES = [1, 8, 16]
THREADS = 8
# The A-6 families (#2395). Each row runs its paper's input shape, so it carries its own batch sizes; the heavy rows
# take fewer timed updates and inference calls per launch (the median of 9 launches still covers 45+ updates).
# Keys: batch (training), inference (batch sizes), train_batches (updates per epoch), iterations (inference calls).
FAMILIES = {
    "ddpm": dict(family="diffusion", batch=4, inference=[1, 4], train_batches=5, iterations=10),
    "decoder-lm": dict(family="decoder-only LM"),
    "whisper-tiny": dict(family="ASR (encoder-decoder)", batch=4, inference=[1, 4], train_batches=5, iterations=10),
    "wav2vec2-base": dict(family="ASR (CTC)", batch=4, inference=[1, 4], train_batches=5, iterations=10),
    "hifigan-v1": dict(family="TTS vocoder", batch=1, inference=[1], train_batches=5, iterations=10),
    "faster-rcnn": dict(family="detection", batch=1, inference=[1], train_batches=10, iterations=10),
    "gcn": dict(family="GNN (GCN)"),
    "gat": dict(family="GNN (GAT)", batch=1, inference=[1]),
}
MODELS = CORE_MODELS + list(FAMILIES)
# --models accepts these group names as well as row names.
GROUPS = {"core": CORE_MODELS, "families": list(FAMILIES), "all": MODELS}
DEFAULT_LAUNCHES = 9
# The PyTorch configurations AiDotNet is compared against. Each row's bar is the fastest of those run on its device.
ARMS = {
    "torch-compile": dict(
        flags=["--compile"], devices=("cpu", "cuda"),
        protocol=dict(compile=True, compile_mode="default", optimizer_impl="default", dynamic_shapes=False),
        description="torch.compile default mode with PyTorch's default AdamW (foreach on CUDA)"),
    "torch-compile-fused": dict(
        flags=["--compile", "--optimizer-impl", "fused"], devices=("cpu", "cuda"),
        protocol=dict(compile=True, compile_mode="default", optimizer_impl="fused", dynamic_shapes=False),
        description="torch.compile default mode with fused AdamW"),
    "torch-cudagraphs-fused": dict(
        flags=["--compile", "--compile-mode", "reduce-overhead", "--optimizer-impl", "fused"], devices=("cuda",),
        protocol=dict(compile=True, compile_mode="reduce-overhead", optimizer_impl="fused", dynamic_shapes=False,
                      cuda_graphs=True),
        description="torch.compile reduce-overhead (CUDA graphs) with fused AdamW"),
}
# A side whose slowest launch exceeds its fastest by this factor is reported as unstable.
UNSTABLE_SPREAD = 1.5
# A CUDA launch whose loaded SM clock moved more than this fraction (max/min - 1) is flagged clock-varied, and one
# whose loaded median is this far from a requested lock is flagged clock-unlocked.
CLOCK_TOLERANCE = 0.03
# Seconds between GPU clock samples during a CUDA launch.
CLOCK_SAMPLE_SECONDS = 0.5
# System CPU use (percent) sampled just before a launch above which that launch is flagged as contended.
BUSY_CPU_PERCENT = 20.0
# GPU utilization (percent) sampled just before a CUDA launch above which that launch is flagged as contended.
BUSY_GPU_PERCENT = 10.0
# Longest wait for the machine to go idle before a launch.
SETTLE_SECONDS = 10.0


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def revision(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return dict(commit=git("rev-parse", "HEAD"), changes=git("status", "--short"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_name():
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor()


def on_external_power():
    """True on AC power, False on battery, None when the platform does not say. A desktop without a battery is AC."""
    if os.name == "nt":
        class SystemPowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        status = SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(status.ACLineStatus)
    if sys.platform == "darwin":
        try:
            return "AC Power" in subprocess.check_output(["pmset", "-g", "batt"], text=True)
        except (OSError, subprocess.CalledProcessError):
            return None
    supplies = list(Path("/sys/class/power_supply").glob("*"))
    mains = [s for s in supplies if (s / "type").is_file() and (s / "type").read_text().strip() == "Mains"]
    if mains:
        return any((s / "online").read_text().strip() == "1" for s in mains if (s / "online").is_file())
    batteries = [s for s in supplies if (s / "type").is_file() and (s / "type").read_text().strip() == "Battery"]
    return None if batteries else True


def nvidia_smi(*args):
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_state():
    """Clocks, temperature, power, utilization and the clock-limiting reasons of the first GPU."""
    fields = ["clocks.sm", "clocks.max.sm", "temperature.gpu", "power.draw", "pstate", "utilization.gpu"]
    for reasons in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):  # renamed in driver 535
        line = nvidia_smi(f"--query-gpu={','.join(fields + [reasons])}", "--format=csv,noheader,nounits")
        if line is not None:
            values = [v.strip() for v in line.splitlines()[0].split(",")]
            return dict(zip(fields + ["clocks.reasons"], values))
    return None


def conditions(device):
    """The machine state around one launch: power source, CPU use, GitHub runner jobs and the GPU's state."""
    import psutil
    state = dict(time=now(), externalPower=on_external_power(), cpuPercent=psutil.cpu_percent(interval=0.5))
    runners = sorted({p.info["name"] for p in psutil.process_iter(["name"])
                      if (p.info["name"] or "").lower().startswith("runner.worker")})
    if runners:
        state["ciJobs"] = runners
    if device == "cuda":
        state["gpu"] = gpu_state()
    return state


def condition_flags(before, after):
    """Flags for anything that could have slowed one launch. Load is judged before the launch, when only other
    processes can be using the machine; power and CI jobs on both sides of it."""
    flags = set()
    if False in (before.get("externalPower"), after.get("externalPower")):
        flags.add("battery")
    if before.get("cpuPercent", 0) > BUSY_CPU_PERCENT:
        flags.add("busy-cpu")
    if before.get("ciJobs") or after.get("ciJobs"):
        flags.add("ci-job")
    if gpu_busy(before):
        flags.add("busy-gpu")
    return sorted(flags)


def gpu_busy(state):
    try:
        return float((state.get("gpu") or {}).get("utilization.gpu", 0)) > BUSY_GPU_PERCENT
    except ValueError:
        return False  # "[N/A]" on GPUs that do not report utilization


def settled_conditions(device):
    """Wait up to SETTLE_SECONDS for the machine to go idle, then return its state.

    The previous process has just exited: the GPU utilization counter still averages over its last moments and its
    CPU teardown may still run. Waiting for idle gives every launch the same quiet start, so a flag left after the
    wait means another program was busy, not this runner's previous process.
    """
    deadline = time.monotonic() + SETTLE_SECONDS
    while True:
        state = conditions(device)
        if (state["cpuPercent"] <= BUSY_CPU_PERCENT and not gpu_busy(state)) or time.monotonic() >= deadline:
            return state


CLOCK_FIELDS = ["clocks.sm", "clocks.mem", "utilization.gpu", "temperature.gpu", "power.draw"]
# GPU utilization (percent) at or above which a clock sample counts as taken under load.
LOADED_GPU_PERCENT = 50.0
# The locked clocks requested for this run ({"sm": MHz, "mem": MHz}); set by main() and read by clock_flags().
LOCKED_CLOCKS = {}


def start_clock_sampler(stem, device):
    """Sample the GPU's clocks every CLOCK_SAMPLE_SECONDS into STEM.clocks.csv while a CUDA launch runs."""
    if device != "cuda" or not shutil.which("nvidia-smi"):
        return None
    out = stem.with_suffix(".clocks.csv").open("w", encoding="utf-8")
    process = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(CLOCK_FIELDS)}", "--format=csv,noheader,nounits",
                                "-lms", str(int(CLOCK_SAMPLE_SECONDS * 1000))], stdout=out, stderr=subprocess.DEVNULL)
    return process, out


def stop_clock_sampler(sampler, stem):
    """Stop the sampler and summarize the samples taken under load: SM and memory clock medians and range."""
    if sampler is None:
        return None
    process, out = sampler
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    out.close()
    samples = []
    for line in stem.with_suffix(".clocks.csv").read_text(encoding="utf-8").splitlines():
        try:
            samples.append(dict(zip(CLOCK_FIELDS, (float(v) for v in line.split(",")))))
        except ValueError:
            continue  # "[N/A]" or a line cut off by terminate()
    loaded = [s for s in samples if s["utilization.gpu"] >= LOADED_GPU_PERCENT]
    if not loaded:
        return dict(samples=len(samples), loadedSamples=0)
    sm = [s["clocks.sm"] for s in loaded]
    return dict(samples=len(samples), loadedSamples=len(loaded), smMedianMhz=statistics.median(sm), smMinMhz=min(sm),
                smMaxMhz=max(sm), memMedianMhz=statistics.median(s["clocks.mem"] for s in loaded),
                maxTemperatureC=max(s["temperature.gpu"] for s in loaded))


def clock_flags(clocks):
    """clock-varied when the SM clock under load moved; clock-unlocked when it missed the requested lock."""
    flags = set()
    if not clocks or not clocks.get("loadedSamples"):
        return flags
    if clocks["smMaxMhz"] / clocks["smMinMhz"] - 1 > CLOCK_TOLERANCE:
        flags.add("clock-varied")
    if LOCKED_CLOCKS.get("sm") and abs(clocks["smMedianMhz"] / LOCKED_CLOCKS["sm"] - 1) > CLOCK_TOLERANCE:
        flags.add("clock-unlocked")
    if LOCKED_CLOCKS.get("mem") and abs(clocks["memMedianMhz"] / LOCKED_CLOCKS["mem"] - 1) > CLOCK_TOLERANCE:
        flags.add("clock-unlocked")
    return flags


def lock_clocks(sm, mem):
    """Lock the GPU's SM and memory clocks with nvidia-smi -lgc / -lmc (needs an elevated prompt). Return what each
    request printed; a refused lock is recorded, not fatal, and every CUDA launch still records its clocks."""
    result = {}
    for key, flag, value in (("sm", "-lgc", sm), ("mem", "-lmc", mem)):
        if value:
            done = subprocess.run(["nvidia-smi", flag, f"{value},{value}"], text=True, capture_output=True)
            result[key] = dict(mhz=value, exitCode=done.returncode, output=(done.stdout + done.stderr).strip())
            if done.returncode == 0:
                LOCKED_CLOCKS[key] = value
            else:
                print(f"nvidia-smi {flag} {value},{value} was refused: {result[key]['output']}", flush=True)
    return result


def reset_clocks(locked):
    for key, flag in (("sm", "-rgc"), ("mem", "-rmc")):
        if key in locked:
            subprocess.run(["nvidia-smi", flag], capture_output=True)


def run(cmd, stem, env, device=None, allow_battery=False):
    """Run one process with its output in STEM.log and a receipt in STEM.receipt.json. Return its condition flags.

    A CUDA launch also samples the GPU's clocks while it runs (STEM.clocks.csv) and records them in the receipt."""
    print(stem.name, flush=True)
    before = settled_conditions(device)
    if before["externalPower"] is False and not allow_battery:
        raise SystemExit(f"Running on battery before {stem.name}. Connect external power or pass --allow-battery.")
    started = now()
    sampler = start_clock_sampler(stem, device)
    try:
        with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
            code = subprocess.run(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    finally:
        clocks = stop_clock_sampler(sampler, stem)
    finished = now()
    after = conditions(device)
    flags = sorted(set(condition_flags(before, after)) | clock_flags(clocks))
    receipt = dict(command=cmd, started=started, finished=finished, exitCode=code, before=before, after=after, flags=flags)
    if clocks is not None:
        receipt["clocks"] = clocks
    save(stem.with_suffix(".receipt.json"), receipt)
    if code:
        raise RuntimeError(f"{stem.name} exited {code}. See {stem.with_suffix('.log')}")
    return flags


class build_lock:
    """Hold an exclusive cross-process lock, released by the OS if this process dies."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    pass  # LK_LOCK gives up after 10 s; keep waiting for the other build
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        self.file.close()


def row_arguments(model, smoke):
    """The benchmark arguments for one row: the core rows' shared schedule, or a family row's own shape."""
    spec = FAMILIES.get(model, {})
    inference = spec.get("inference", BATCHES)
    epochs, batches, iters, warm = (2, 2, 3, 1) if smoke else (3, spec.get("train_batches", 20), spec.get("iterations", 100), 10)
    return ["--epochs", str(epochs), "--train-batches", str(batches), "--batch-size", str(spec.get("batch", 64)),
            "--inference-batches", ",".join(map(str, inference)), "--inference-iterations", str(iters),
            "--warmup-iterations", str(warm), "--warmup-min-ms", "1000", "--seed", "1234"]


def row_common(info, model):
    """The arguments a run gave MODEL; runs recorded before per-row arguments existed used one list for every row."""
    return info.get("rowArguments", {}).get(model, info["common"])


def schedule(common):
    """(epochs, updates per epoch, operations) from a row's arguments; operations are (operation, batch) pairs."""
    def value(flag):
        return common[common.index(flag) + 1]
    inference = [int(b) for b in value("--inference-batches").split(",")]
    return (int(value("--epochs")), int(value("--train-batches")),
            [("train", int(value("--batch-size"))), *(("inference", b) for b in inference)])


def measurements(path, side, model, device, common):
    """Validate one report against the protocol. Return (parameter count, [train ms/update, inference ms...],
    (first loss, last loss))."""
    epochs, batches, operations = schedule(common)
    inference = [b for o, b in operations if o == "inference"]
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    if len(rows) != 1 or obj["Framework" if ai else "framework"] != ("AiDotNet" if ai else "PyTorch"):
        raise ValueError("Unexpected framework or number of models")
    # The matrix runs fp32 in both arms; a report at another precision (benchmark --precision) is not comparable.
    precision = obj["Protocol"].get("Precision", "fp32") if ai else obj["protocol"].get("precision", "fp32")
    if precision != "fp32":
        raise ValueError(f"{side} ran at precision {precision!r}; the matrix compares fp32")
    if ai:
        env = obj["Environment"]
        if (env["CpuThreads"], env["EngineDevice"]) != (THREADS, {"cpu": "CpuEngine", "cuda": "DirectGpuTensorEngine"}[device]):
            raise ValueError("AiDotNet engine/thread count mismatch")
        if device == "cuda" and not env["EngineName"].startswith(CUDA_ENGINE):
            raise ValueError(f"AiDotNet measured {env['EngineName']!r}, not the CUDA backend")
    else:
        if obj["device"] != device or obj["torch_num_threads"] != THREADS or (device == "cuda" and not obj["cuda_available"]):
            raise ValueError("PyTorch device/thread count mismatch")
        expected = dict(ARMS[side]["protocol"])
        expected.setdefault("cuda_graphs", False)
        recorded = {key: obj["protocol"][key] for key in expected}
        if recorded != expected:
            raise ValueError(f"PyTorch ran {recorded}, expected {expected}")
    row = _index(obj)[model]
    if row.device != device or row.training is None or set(row.inference) != set(inference):
        raise ValueError("Incomplete or mismatched benchmark row")
    raw = rows[0]
    training = raw["Training" if ai else "training"]
    losses = (training["FirstLoss" if ai else "first_loss"], training["LastLoss" if ai else "last_loss"])
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses)
            or len(training["EpochSeconds" if ai else "epoch_seconds"]) != epochs
            or training["StepsPerEpoch" if ai else "steps_per_epoch"] != batches):
        raise ValueError("Training schedule or loss mismatch")
    times = [row.training.median, *(row.inference[b].median for b in inference)]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("Invalid timing")
    return raw["Parameters" if ai else "parameters"], times, losses


def quantile(values, q):
    """Linear interpolation between order statistics (numpy's default, as pytorch/classical.py)."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def spread(values):
    """Median, first and third quartile, minimum and maximum."""
    return dict(median=quantile(values, 0.5), q1=quantile(values, 0.25), q3=quantile(values, 0.75),
                min=min(values), max=max(values))


def summarize(rows, launches):
    """Aggregate the per-launch measurements: median, IQR and range per side, AiDotNet against the fastest PyTorch
    arm, and the median first and last training loss of each side."""
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["device"], r["operation"], r["batch"]), []).append(r)
    summary = []
    for (model, device, operation, batch), items in groups.items():
        by_side = {}
        for item in items:
            by_side.setdefault(item["configuration"], {})[item["launch"]] = item
        ai = by_side.pop("aidotnet", {})
        if not ai or not by_side:
            continue
        medians = {arm: quantile([i["ms"] for i in launches_.values()], 0.5) for arm, launches_ in by_side.items()}
        best = min(medians, key=medians.get)
        paired = sorted(set(ai) & set(by_side[best]))
        ratios = [by_side[best][l]["ms"] / ai[l]["ms"] for l in paired]
        a = spread([i["ms"] for i in ai.values()])
        t = spread([i["ms"] for i in by_side[best].values()])
        flags = set()
        for item in items:
            flags.update(f for f in item["flags"].split(";") if f)
        if a["max"] / a["min"] > UNSTABLE_SPREAD or t["max"] / t["min"] > UNSTABLE_SPREAD:
            flags.add("unstable")
        if len(ai) < launches or len(by_side[best]) < launches:
            flags.add("missing-launches")
        ai_params = next(iter(ai.values()))["parameters"]
        torch_params = next(iter(by_side[best].values()))["parameters"]
        if ai_params != torch_params:
            flags.add("parameters-differ")
        losses = {}
        if operation == "train":
            for name, side in (("ai", ai), ("torch", by_side[best])):
                first = quantile([i["first_loss"] for i in side.values()], 0.5)
                last = quantile([i["last_loss"] for i in side.values()], 0.5)
                losses.update({f"{name}_first_loss": first, f"{name}_last_loss": last})
                if not last < first:
                    flags.add("loss-not-falling")
        clocks = [i["sm_mhz"] for i in items if i.get("sm_mhz")]
        ratio = spread(ratios) if ratios else {}
        summary.append(dict(
            model=model, device=device, operation=operation, batch=batch, launches=len(paired),
            ai_parameters=ai_params, torch_parameters=torch_params,
            ai_median_ms=a["median"], ai_q1_ms=a["q1"], ai_q3_ms=a["q3"], ai_iqr_ms=a["q3"] - a["q1"],
            ai_min_ms=a["min"], ai_max_ms=a["max"],
            torch_configuration=best, torch_median_ms=t["median"], torch_q1_ms=t["q1"], torch_q3_ms=t["q3"],
            torch_iqr_ms=t["q3"] - t["q1"], torch_min_ms=t["min"], torch_max_ms=t["max"],
            torch_over_ai=t["median"] / a["median"],
            launch_ratio_q1=ratio.get("q1"), launch_ratio_q3=ratio.get("q3"),
            launch_ratio_min=ratio.get("min"), launch_ratio_max=ratio.get("max"),
            **{f"{arm}_median_ms": medians.get(arm) for arm in ARMS},
            ai_first_loss=losses.get("ai_first_loss"), ai_last_loss=losses.get("ai_last_loss"),
            torch_first_loss=losses.get("torch_first_loss"), torch_last_loss=losses.get("torch_last_loss"),
            sm_mhz_median=quantile(clocks, 0.5) if clocks else None,
            flags=";".join(sorted(flags))))
    order = {m: i for i, m in enumerate(MODELS)}
    summary.sort(key=lambda s: (s["device"] != "cuda", order.get(s["model"], 99), s["operation"] != "train", s["batch"]))
    return summary


def _number(value):
    return f"{value:.3g}" if abs(value) < 100 else f"{value:,.0f}"


def markdown(summary):
    """A README-ready table per device: every row's median and IQR over the launches on both sides, the ratio, the
    parameter counts and, for training, the median first -> last loss of both sides."""
    def ms(median, q1, q3):
        return f"{_number(median)} ({_number(q1)}–{_number(q3)})"
    lines = []
    for device in ("cuda", "cpu"):
        rows = [s for s in summary if s["device"] == device]
        if not rows:
            continue
        clock = [s["sm_mhz_median"] for s in rows if s["sm_mhz_median"]]
        lines += [f"**{device.upper()}**: milliseconds per training update or inference call, median (interquartile "
                  f"range) over {max(s['launches'] for s in rows)} launches"
                  + (f", SM clock under load {quantile(clock, 0.5):.0f} MHz" if clock else "")
                  + ". The ratio is PyTorch time ÷ AiDotNet time (IQR of the per-launch ratios); above 1 means "
                  "AiDotNet took less time. PyTorch is the fastest configuration for each row. ⚑ marks a row with a flag "
                  "in summary.csv.", "",
                  "| Model | Operation | Batch | AiDotNet ms | PyTorch ms | PyTorch ÷ AiDotNet | Parameters (AiDotNet / PyTorch) "
                  "| Loss first → last (AiDotNet; PyTorch) |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for s in rows:
            ratio = f"{s['torch_over_ai']:.2f}"
            if s["launch_ratio_q1"] is not None:
                ratio += f" ({s['launch_ratio_q1']:.2f}–{s['launch_ratio_q3']:.2f})"
            loss = ""
            if s["operation"] == "train":
                loss = (f"{s['ai_first_loss']:.4g} → {s['ai_last_loss']:.4g}; "
                        f"{s['torch_first_loss']:.4g} → {s['torch_last_loss']:.4g}")
            lines.append(f"| {s['model']} | {s['operation']} | {s['batch']} | "
                         f"{ms(s['ai_median_ms'], s['ai_q1_ms'], s['ai_q3_ms'])} | "
                         f"{ms(s['torch_median_ms'], s['torch_q1_ms'], s['torch_q3_ms'])} {s['torch_configuration']} | "
                         f"{ratio}{' ⚑' if s['flags'] else ''} | {s['ai_parameters']:,} / {s['torch_parameters']:,} | {loss} |")
        lines.append("")
    return "\n".join(lines)


COLUMNS = ("model", "device", "launch", "framework", "configuration", "operation", "batch", "parameters", "ms",
           "first_loss", "last_loss", "sm_mhz", "flags")


def collect(dest):
    """Read and validate every report a run directory should contain. Return (measurement rows, problems)."""
    info = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
    rows, problems = [], []
    for launch in range(1, info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                common = row_common(info, model)
                for side in ["aidotnet", *info["torchConfigurations"][device]]:
                    tag = dest / f"{side}-{launch}-{device}-{model}"
                    try:
                        receipt = json.loads(tag.with_suffix(".receipt.json").read_text(encoding="utf-8"))
                        if receipt["exitCode"] != 0:
                            raise RuntimeError(f"exited {receipt['exitCode']}")
                        parameters, times, losses = measurements(tag.with_suffix(".json"), side, model, device, common)
                    except (KeyError, ValueError, OSError, RuntimeError, TypeError) as error:
                        problems.append(f"{tag.name}: {error}")
                        continue
                    for (operation, batch), ms in zip(schedule(common)[2], times):
                        train = operation == "train"
                        rows.append(dict(model=model, device=device, launch=launch,
                                         framework="AiDotNet" if side == "aidotnet" else "PyTorch", configuration=side,
                                         operation=operation, batch=batch, parameters=parameters, ms=ms,
                                         first_loss=losses[0] if train else None, last_loss=losses[1] if train else None,
                                         sm_mhz=(receipt.get("clocks") or {}).get("smMedianMhz"),
                                         flags=";".join(receipt.get("flags", []))))
    return rows, problems


def write_outputs(dest, rows, launches):
    """measurements.csv (every launch), summary.csv (medians, IQRs, ranges, flags) and summary.md (the README table)."""
    with (dest / "measurements.csv").open("w", newline="", encoding="utf-8") as out:
        table = csv.writer(out, lineterminator="\n")
        table.writerow([c if c != "ms" else "median_ms" for c in COLUMNS])
        table.writerows([row[c] for c in COLUMNS] for row in rows)
    summary = summarize(rows, launches)
    with (dest / "summary.csv").open("w", newline="", encoding="utf-8") as out:
        if summary:
            writer = csv.DictWriter(out, fieldnames=list(summary[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    (dest / "summary.md").write_text(markdown(summary), encoding="utf-8", newline="\n")


def regenerate(run_directory):
    """--summarize: rebuild the CSVs and table of an existing run directory from its raw reports and receipts."""
    target = run_directory / "regenerated"
    target.mkdir(exist_ok=True)
    shutil.copyfile(run_directory / "protocol.json", target / "protocol.json")
    info = json.loads((run_directory / "protocol.json").read_text(encoding="utf-8"))
    if info.get("classical"):
        rows, problems = collect_classical(run_directory)
        write_classical_outputs(target, rows, info["launches"])
    else:
        rows, problems = collect(run_directory)
        write_outputs(target, rows, info["launches"])
    for problem in problems:
        print(problem, file=sys.stderr)
    print(target)
    return 1 if problems else 0


# The classical-ML rows (ClassicalBench.cs against pytorch/classical.py). Each launch fits every row --repeats times in
# one process and reports the median; the summary is the median and IQR of those per-launch medians.
CLASSICAL_LIBRARIES = ("scikit-learn", "xgboost", "lightgbm")


def library_key(library):
    return library.replace("-", "_").lower()


def classical_measurements(path):
    """{(row, library): (fit s, predict s or None, quality value, metric)} from one classical report."""
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {}
    for row in obj["rows"]:
        if "skipped" in row:
            continue
        name, _, library = row["model"].partition(":")
        result[(name, library or ("AiDotNet" if obj["framework"] == "AiDotNet" else "scikit-learn"))] = (
            row["fit_seconds_median"], row.get("predict_seconds_median"), row["value"], row["metric"])
    return result


def collect_classical(dest):
    """Read every classical launch's two reports. Return (measurement rows, problems)."""
    info = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
    rows, problems = [], []
    for launch in range(1, info["launches"] + 1):
        for side, report in (("aidotnet", "aidotnet_classical.json"), ("python", "python_classical.json")):
            stem = dest / f"classical-{side}-{launch}"
            try:
                receipt = json.loads(stem.with_suffix(".receipt.json").read_text(encoding="utf-8"))
                if receipt["exitCode"] != 0:
                    raise RuntimeError(f"exited {receipt['exitCode']}")
                found = classical_measurements(dest / f"classical-{launch}" / report)
            except (KeyError, ValueError, OSError, RuntimeError, TypeError) as error:
                problems.append(f"{stem.name}: {error}")
                continue
            for (name, library), (fit, predict, value, metric) in found.items():
                rows.append(dict(row=name, library=library, launch=launch, fit_s=fit, predict_s=predict,
                                 metric=metric, value=value, flags=";".join(receipt.get("flags", []))))
    return rows, problems


def summarize_classical(rows, launches):
    """Per row and library: the median and IQR of the per-launch fit and predict medians, and the median quality."""
    groups = {}
    for r in rows:
        groups.setdefault(r["row"], {}).setdefault(r["library"], []).append(r)
    summary = []
    for name, libraries in groups.items():
        ai = libraries.get("AiDotNet")
        if not ai:
            continue
        entry = dict(row=name, launches=len(ai), metric=ai[0]["metric"])
        flags = set()
        for library, items in libraries.items():
            key = library_key(library)
            fit = spread([i["fit_s"] for i in items])
            entry.update({f"{key}_fit_median_s": fit["median"], f"{key}_fit_q1_s": fit["q1"], f"{key}_fit_q3_s": fit["q3"],
                          f"{key}_value": quantile([i["value"] for i in items], 0.5)})
            predictions = [i["predict_s"] for i in items if i["predict_s"] is not None]
            if predictions:
                p = spread(predictions)
                entry.update({f"{key}_predict_median_s": p["median"], f"{key}_predict_q1_s": p["q1"],
                              f"{key}_predict_q3_s": p["q3"]})
            if len(items) < launches:
                flags.add("missing-launches")
            if fit["max"] / fit["min"] > UNSTABLE_SPREAD:
                flags.add("unstable")
            for item in items:
                flags.update(f for f in item["flags"].split(";") if f)
        rivals = {lib: entry[f"{library_key(lib)}_fit_median_s"] for lib in CLASSICAL_LIBRARIES
                  if f"{library_key(lib)}_fit_median_s" in entry}
        if rivals:
            best = min(rivals, key=rivals.get)
            entry.update(fastest_rival=best, rival_over_ai_fit=rivals[best] / entry["aidotnet_fit_median_s"])
        entry["flags"] = ";".join(sorted(flags))
        summary.append(entry)
    return summary


def classical_markdown(summary):
    def cell(entry, key, kind):
        if f"{key}_{kind}_median_s" not in entry:
            return "–"
        m, q1, q3 = (entry[f"{key}_{kind}_{s}_s"] for s in ("median", "q1", "q3"))
        return f"{_number(m * 1000)} ({_number(q1 * 1000)}–{_number(q3 * 1000)})"
    lines = [f"**CPU, classical ML**: fit and predict milliseconds, median (interquartile range) over "
             f"{max((e['launches'] for e in summary), default=0)} launches of each launch's median. The ratio is the "
             "fastest Python library's fit time ÷ AiDotNet's; above 1 means AiDotNet fit faster. Quality is held-out "
             "accuracy, R² or k-means inertia. ⚑ marks a row with a flag in classical_summary.csv.", "",
             "| Row | AiDotNet fit | scikit-learn fit | XGBoost fit | LightGBM fit | Fastest ÷ AiDotNet | AiDotNet predict "
             "| scikit-learn predict | Quality (AiDotNet / scikit-learn / XGBoost / LightGBM) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for e in summary:
        quality = " / ".join(f"{e[k + '_value']:.4g}" if k + "_value" in e else "–"
                             for k in ("aidotnet", "scikit_learn", "xgboost", "lightgbm"))
        ratio = f"{e['rival_over_ai_fit']:.2f} ({e['fastest_rival']})" if "fastest_rival" in e else "–"
        lines.append(f"| {e['row']} | {cell(e, 'aidotnet', 'fit')} | {cell(e, 'scikit_learn', 'fit')} | "
                     f"{cell(e, 'xgboost', 'fit')} | {cell(e, 'lightgbm', 'fit')} | {ratio}{' ⚑' if e['flags'] else ''} | "
                     f"{cell(e, 'aidotnet', 'predict')} | {cell(e, 'scikit_learn', 'predict')} | {e['metric']}: {quality} |")
    return "\n".join(lines) + "\n"


def write_classical_outputs(dest, rows, launches):
    """classical_measurements.csv (every launch), classical_summary.csv and classical_summary.md."""
    columns = ("row", "library", "launch", "fit_s", "predict_s", "metric", "value", "flags")
    with (dest / "classical_measurements.csv").open("w", newline="", encoding="utf-8") as out:
        table = csv.writer(out, lineterminator="\n")
        table.writerow(columns)
        table.writerows([row[c] for c in columns] for row in rows)
    summary = summarize_classical(rows, launches)
    with (dest / "classical_summary.csv").open("w", newline="", encoding="utf-8") as out:
        if summary:
            writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for e in summary for k in e)),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    (dest / "classical_summary.md").write_text(classical_markdown(summary), encoding="utf-8", newline="\n")


def package_versions(*names):
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def build_environment(env):
    """The environment for dotnet build. The Visual Studio developer prompt sets Platform=x64, which moves the build
    output to bin/x64/Release; remove it so the output lands where this runner reads it."""
    build_env = {k: v for k, v in env.items() if k.upper() != "PLATFORM"}
    return build_env


def succeeded(stem):
    """True when the process for STEM already ran and exited 0 (only possible when resuming)."""
    receipt = stem.with_suffix(".receipt.json")
    return receipt.is_file() and json.loads(receipt.read_text(encoding="utf-8")).get("exitCode") == 0


def completed(tag):
    """True when a benchmark process already finished: it exited 0 and wrote its report."""
    return succeeded(tag) and tag.with_suffix(".json").is_file()


def resume_run(run_directory, torch_version):
    """Load an interrupted run's protocol and check that it can continue unchanged: same source and same PyTorch."""
    info = json.loads((run_directory / "protocol.json").read_text(encoding="utf-8"))
    current = revision(HERE.parents[1])
    if current != info["ai"]:
        raise SystemExit(f"--resume needs the run's source revision {info['ai']}; this checkout is {current}.")
    if torch_version != info["torch"]:
        raise SystemExit(f"--resume needs PyTorch {info['torch']}; this environment has {torch_version}.")
    recorded = json.loads((run_directory / "assemblies.json").read_text(encoding="utf-8"))["sha256"]
    dll = run_directory / "bin/AiDotNet.PyTorchParity.dll"
    if {p: digest(dll.with_name(p)) for p in recorded} != recorded:
        raise SystemExit(f"The run's private binaries in {dll.parent} no longer match assemblies.json.")
    info.setdefault("resumed", []).append(now())
    save(run_directory / "protocol.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=["cuda", "cpu"], default=["cuda", "cpu"])
    parser.add_argument("--models", nargs="+", choices=MODELS + list(GROUPS), default=["core"],
                        help="Rows or groups: core (the 11 original rows, the default), families (the A-6 rows: "
                             + ", ".join(FAMILIES) + ") or all")
    parser.add_argument("--torch-configurations", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="PyTorch configurations to run (each device runs those that apply to it)")
    parser.add_argument("--launches", type=int,
                        help=f"Independent launches of every process (default {DEFAULT_LAUNCHES}, smoke 1)")
    parser.add_argument("--lock-gpu-clock", type=int, metavar="MHZ",
                        help="Lock the GPU's SM clock with nvidia-smi -lgc MHZ,MHZ for the run and reset it after (-rgc). "
                             "Needs an elevated prompt; a refused lock is recorded and the run continues.")
    parser.add_argument("--lock-memory-clock", type=int, metavar="MHZ",
                        help="Lock the GPU's memory clock with nvidia-smi -lmc MHZ,MHZ for the run and reset it after (-rmc).")
    parser.add_argument("--classical", action="store_true",
                        help="Run the classical-ML rows (ClassicalBench.cs against scikit-learn, XGBoost and LightGBM) "
                             "instead of the model rows; CPU only")
    parser.add_argument("--classical-repeats", type=int, default=3,
                        help="Fits of every classical row within one launch; each launch reports their median")
    parser.add_argument("--smoke", action="store_true", help="Two epochs of two updates, three measured inference calls.")
    parser.add_argument("--tensors", type=Path, help="Build against this Tensors checkout instead of the repository's NuGet dependency.")
    parser.add_argument("--output", type=Path, default=HERE / "results", help="Parent for unique run directories")
    parser.add_argument("--allow-battery", action="store_true", help="Run even when the machine is on battery power")
    parser.add_argument("--summarize", type=Path, metavar="RUN_DIR",
                        help="Only rebuild RUN_DIR's CSVs and table from its raw reports, into RUN_DIR/regenerated")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR",
                        help="Continue an interrupted run with its recorded settings and binaries; completed processes are kept")
    args = parser.parse_args()
    args.models = list(dict.fromkeys(m for name in args.models for m in GROUPS.get(name, [name])))
    if args.summarize:
        sys.exit(regenerate(args.summarize.resolve()))
    import torch
    import torchvision
    import torch._dynamo.config as dynamo
    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compilation-error fallback.")
    if os.name == "nt" and not shutil.which("cl") and not args.classical:
        parser.error("Use Developer PowerShell for Visual Studio with the x64 C++ build tools.")
    if on_external_power() is False and not args.allow_battery:
        parser.error("The machine is on battery power. Connect external power or pass --allow-battery.")
    env = os.environ.copy()
    # A CUDA row must not fall back to OpenCL/HIP when CUDA fails to initialise. CPU launches never read this.
    env["AIDOTNET_DIRECTGPU_BACKENDS"] = "cuda"
    for key in ("AIDOTNET_CPU_THREADS", "AIDOTNET_BLAS_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(THREADS)

    if args.resume:
        dest = args.resume.resolve()
        info = resume_run(dest, torch.__version__)
        print(f"Resuming: {dest}", flush=True)
    else:
        launches = args.launches if args.launches is not None else (1 if args.smoke else DEFAULT_LAUNCHES)
        if launches < 1 or len(set(args.devices)) != len(args.devices):
            parser.error("Select distinct devices and a positive launch count.")
        if args.classical:
            args.devices, args.models = ["cpu"], []
        arms = {device: [a for a in args.torch_configurations if device in ARMS[a]["devices"]] for device in args.devices}
        if not all(arms.values()):
            parser.error("Every device needs at least one PyTorch configuration that runs on it.")
        # "common" is the core rows' schedule; each row's own arguments are in rowArguments.
        common = row_arguments("mlp", args.smoke)
        import psutil
        args.output.mkdir(exist_ok=True, parents=True)
        dest = Path(tempfile.mkdtemp(dir=args.output.resolve(), prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")))
        print(f"Results: {dest}", flush=True)
        info = dict(started=now(), models=args.models, devices=args.devices, launches=launches, smoke=args.smoke, common=common,
                    rowArguments={m: row_arguments(m, args.smoke) for m in args.models},
                    families={m: FAMILIES[m]["family"] for m in args.models if m in FAMILIES},
                    threads=THREADS, referenceModels=[m for m in REFERENCE if m in args.models],
                    torchConfigurations={d: arms[d] for d in args.devices},
                    configurationDescriptions={a: ARMS[a]["description"] for a in args.torch_configurations},
                    ai=revision(HERE.parents[1]), torch=torch.__version__, torchvision=torchvision.__version__,
                    packages=package_versions("torch", "torchvision", "triton", "triton-windows", "numpy", "psutil",
                                             "transformers", "scikit-learn", "xgboost", "lightgbm"),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    os=platform.platform(), cpu=cpu_name(), logicalCpus=os.cpu_count(), memoryBytes=psutil.virtual_memory().total,
                    python=sys.version, sdk=subprocess.check_output(["dotnet", "--version"], text=True).strip())
        if args.classical:
            info.update(classical=True, classicalRepeats=args.classical_repeats)
        if shutil.which("nvidia-smi"):
            info["driver"] = nvidia_smi("--query-gpu=name,driver_version,memory.total", "--format=csv")
        build = ["dotnet", "build", str(HERE / "AiDotNet.PyTorchParity.csproj"), "-c", "Release", "-p:GeneratePackageOnBuild=false",
                 "-p:UseSharedCompilation=false", "-m:1"]
        if args.tensors:
            tensors = args.tensors.resolve()
            info["tensors"] = revision(tensors)
            build += ["-p:UseLocalTensors=true", f"-p:LocalTensorsRoot={tensors}"]
        else:
            build += ["-p:UseLocalTensors=false"]
        save(dest / "protocol.json", info)
        # The build output directory is shared by every invocation, so build, check and copy it under a lock,
        # then hash and launch only this invocation's private copy.
        dll = dest / "bin/AiDotNet.PyTorchParity.dll"
        output = HERE / "bin/Release/net10.0"
        with build_lock(HERE / ".run_matrix.lock"):
            run(build, dest / "build", build_environment(env), allow_battery=args.allow_battery)
            # Each assembly in the benchmark output must be the one its own project just built, never an older copy.
            library = HERE.parents[1] / "src/bin/Release/net10.0/AiDotNet.dll"
            if not (output / dll.name).is_file() or digest(output / "AiDotNet.dll") != digest(library):
                raise RuntimeError(f"{output} does not hold this build's AiDotNet.dll ({library}). See {dest / 'build.log'}")
            if args.tensors and digest(output / "AiDotNet.Tensors.dll") != digest(tensors / "src/AiDotNet.Tensors/bin/Release/net10.0/AiDotNet.Tensors.dll"):
                raise RuntimeError("Loaded Tensors assembly differs from the requested checkout")
            shutil.copytree(output, dll.parent)
        hashes = {p: digest(dll.with_name(p)) for p in (dll.name, "AiDotNet.dll", "AiDotNet.Tensors.dll")}
        deps = json.loads(dll.with_name("AiDotNet.PyTorchParity.deps.json").read_text(encoding="utf-8-sig"))
        tensors_library = {k: v.get("type") for k, v in deps["libraries"].items() if k.startswith("AiDotNet.Tensors/")}
        save(dest / "assemblies.json", dict(sha256=hashes, tensorsLibrary=tensors_library))

    if info.get("classical"):
        sys.exit(run_classical(dest, info, env, args.allow_battery))
    if "cuda" in info["devices"] and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select --devices cpu or install CUDA-enabled PyTorch.")
    locked = {}
    if "cuda" in info["devices"] and (args.lock_gpu_clock or args.lock_memory_clock):
        locked = lock_clocks(args.lock_gpu_clock, args.lock_memory_clock)
        info.setdefault("clockLocks", []).append(dict(time=now(), **locked))
        save(dest / "protocol.json", info)
    try:
        problems = run_models(dest, info, env, args.allow_battery)
    finally:
        reset_clocks(locked)
    if problems:
        sys.exit("\n".join(problems))
    print(dest / "summary.md")


def run_models(dest, info, env, allow_battery):
    """The step-1 reference check, then every launch of every row. Return the problems collect() found."""
    dll = dest / "bin/AiDotNet.PyTorchParity.dll"
    refs = info["referenceModels"]
    if refs:
        ref = dest / "reference"
        if not succeeded(dest / "reference-export"):
            run([sys.executable, str(HERE / "pytorch/export_reference.py"), "--models", ",".join(refs), "--out", str(ref)],
                dest / "reference-export", env, allow_battery=allow_battery)
        for device in info["devices"]:
            stem = dest / f"reference-{device}"
            if not succeeded(stem):
                run(["dotnet", str(dll), "--verify-step1", str(ref), "--models", ",".join(refs), "--device", device], stem, env,
                    device, allow_battery)
            lines = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace").splitlines()
            if not all(any(line.startswith(f"[step1] {m}: PASS") for line in lines) for m in refs):
                raise RuntimeError(f"Missing first-update verdict. See {stem.with_suffix('.log')}")
    for launch in range(1, info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                sides = ["aidotnet", *info["torchConfigurations"][device]]
                # Rotate the order every launch so no side always runs first (or last) on a warming machine.
                shift = (launch - 1) % len(sides)
                for side in sides[shift:] + sides[:shift]:
                    tag = dest / f"{side}-{launch}-{device}-{model}"
                    if completed(tag):
                        continue  # only when resuming: this process already finished
                    cmd = (["dotnet", str(dll)] if side == "aidotnet" else
                           [sys.executable, str(HERE / "pytorch/benchmark.py"), *ARMS[side]["flags"], "--threads", str(THREADS)])
                    try:
                        run([*cmd, "--models", model, "--device", device, *row_common(info, model),
                             "--output", str(tag.with_suffix(".json"))], tag, env, device, allow_battery)
                    except RuntimeError as error:
                        print(error, flush=True)  # its receipt records the exit code; collect() reports it below
    # Every measurement comes from re-reading the reports on disk, exactly as --summarize does for an archive.
    rows, problems = collect(dest)
    write_outputs(dest, rows, info["launches"])
    save(dest / "completion.json", dict(finished=now(), failures=problems))
    return problems


def run_classical(dest, info, env, allow_battery):
    """Every launch fits the classical rows in a fresh AiDotNet process and a fresh Python process, in an order that
    alternates between launches. Both sides use every core for the all-core rows (the thread pins the model rows set
    are removed) and one thread for the _1job rows. Return the exit code."""
    dll = dest / "bin/AiDotNet.PyTorchParity.dll"
    env = {k: v for k, v in env.items() if k not in ("AIDOTNET_CPU_THREADS", "AIDOTNET_BLAS_THREADS", "OMP_NUM_THREADS",
                                                      "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    repeats = str(info["classicalRepeats"])
    for launch in range(1, info["launches"] + 1):
        directory = dest / f"classical-{launch}"
        directory.mkdir(exist_ok=True)
        commands = {"aidotnet": ["dotnet", str(dll), "--classical", str(directory), "--repeats", repeats, "--seed", "42"],
                    "python": [sys.executable, str(HERE / "pytorch/classical.py"), str(directory), "--repeats", repeats,
                               "--seed", "42"]}
        order = ["aidotnet", "python"] if launch % 2 else ["python", "aidotnet"]
        if order[0] == "python" and not (directory / "train.csv").is_file():
            # The C# side writes the data (seeded, the same every launch); reuse launch 1's so Python can go first.
            for name in ("train.csv", "test.csv"):
                shutil.copyfile(dest / "classical-1" / name, directory / name)
        for side in order:
            stem = dest / f"classical-{side}-{launch}"
            if succeeded(stem):
                continue
            try:
                run(commands[side], stem, env, "cpu", allow_battery)
            except RuntimeError as error:
                print(error, flush=True)
    rows, problems = collect_classical(dest)
    write_classical_outputs(dest, rows, info["launches"])
    save(dest / "completion.json", dict(finished=now(), failures=problems))
    print("\n".join(problems) if problems else dest / "classical_summary.md")
    return 1 if problems else 0


if __name__ == "__main__":
    main()
