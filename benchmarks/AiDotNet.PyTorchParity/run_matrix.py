"""Run end-to-end AiDotNet/Python PyTorch training and inference comparisons from this checkout.

Every model, device and launch runs AiDotNet and each applicable PyTorch configuration in its own fresh process, in an
order that rotates between launches. AiDotNet is compared against the fastest PyTorch configuration for each row.

--scale DATA_ROOT runs the large-data comparison instead: whole epochs over the real and synthetic datasets that
scale/prepare_data.py wrote to DATA_ROOT, with the test split scored after every epoch (scale/scale_matrix.py).
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
sys.path.insert(0, str(HERE / "scale"))
import scale_matrix  # noqa: E402
sys.path.insert(0, str(HERE / "pytorch"))
from compare import _index

CUDA_ENGINE = "Direct GPU Engine (CUDA "
MODELS = "mlp lstm transformer cnn resnet18 resnet50 vgg11 vit gru mobilenetv2 densenet121".split()
REFERENCE = ("mlp", "cnn", "lstm", "transformer")
BATCHES = [1, 8, 16]
THREADS = 8
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
    # torch.compile does not trace nn.LSTM or nn.GRU unless torch._dynamo.config.allow_rnn is set: Dynamo skips the
    # whole forward and the arms above run those models eagerly (cuDNN on CUDA, oneDNN on CPU). These arms compile
    # them too, so the RNN rows are compared with whichever is faster. CPU LSTM is excluded because PyTorch 2.14 fails
    # to compile it ("expected Tensor() for op: torch.ops.aten.mkldnn_rnn_layer.default").
    "torch-compile-fused-rnn": dict(
        flags=["--compile", "--optimizer-impl", "fused", "--compile-rnn"], devices=("cpu",), models=("gru",),
        protocol=dict(compile=True, compile_mode="default", optimizer_impl="fused", dynamic_shapes=False, compile_rnn=True),
        description="torch.compile default mode with fused AdamW, tracing nn.GRU (allow_rnn)"),
    "torch-cudagraphs-fused-rnn": dict(
        flags=["--compile", "--compile-mode", "reduce-overhead", "--optimizer-impl", "fused", "--compile-rnn"],
        devices=("cuda",), models=("lstm", "gru"),
        protocol=dict(compile=True, compile_mode="reduce-overhead", optimizer_impl="fused", dynamic_shapes=False,
                      cuda_graphs=True, compile_rnn=True),
        description="torch.compile reduce-overhead (CUDA graphs) with fused AdamW, tracing nn.LSTM/nn.GRU (allow_rnn)"),
}
# Models whose forward Dynamo skips unless allow_rnn is set, so a compiled arm without it legitimately compiles nothing.
RNN_MODELS = ("lstm", "gru")


def sides_for(info, device, model):
    """AiDotNet plus every PyTorch configuration that runs this model on this device. Runs recorded before
    configurations could be limited to some models have no configurationModels entry and run every model."""
    limited = info.get("configurationModels", {})
    return ["aidotnet", *(a for a in info["torchConfigurations"][device] if model in (limited.get(a) or (model,)))]
# A side whose slowest launch exceeds its fastest by this factor is reported as unstable.
UNSTABLE_SPREAD = 1.5
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
    if gpu_clock_low(after):
        flags.add("gpu-clock-low")
    return sorted(flags)


# A CUDA launch whose GPU, busy at its end, ran below this fraction of its maximum SM clock is flagged gpu-clock-low,
# and the runner refuses to start CUDA measurements when its load probe sustains less.
LOW_CLOCK_FRACTION = 0.6
# With --gpu-clock-lock, the reference is the locked clock instead: the probe must sustain at least this fraction of
# it (so the lock is in effect and nothing caps the GPU below it), and so must every busy launch.
LOCKED_CLOCK_FRACTION = 0.9
# The SM clock the run declared locked (nvidia-smi -lgc), or None. Set from protocol.json, so a resume keeps it.
GPU_CLOCK_LOCK = None


def clock_floor(max_clock):
    """The SM clock below which a busy GPU counts as held down: 90% of the declared lock, else 60% of the maximum."""
    return LOCKED_CLOCK_FRACTION * GPU_CLOCK_LOCK if GPU_CLOCK_LOCK else LOW_CLOCK_FRACTION * max_clock


def gpu_clock_low(state):
    """True when the GPU was working (P0, over 30% utilization) below clock_floor."""
    gpu = state.get("gpu") or {}
    try:
        return (gpu.get("pstate") == "P0" and float(gpu["utilization.gpu"]) > 30
                and float(gpu["clocks.sm"]) < clock_floor(float(gpu["clocks.max.sm"])))
    except (KeyError, ValueError):
        return False


def gpu_clock_probe(env):
    """Run pytorch/gpu_clock_probe.py: the SM clock the GPU sustains under a matrix-multiplication load."""
    output = subprocess.run([sys.executable, str(HERE / "pytorch/gpu_clock_probe.py")], cwd=HERE, env=env, text=True,
                            capture_output=True, check=True).stdout
    probe = json.loads(output.strip().splitlines()[-1])
    probe["time"] = now()
    return probe


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


def run(cmd, stem, env, device=None, allow_battery=False):
    """Run one process with its output in STEM.log and a receipt in STEM.receipt.json. Return its condition flags."""
    print(stem.name, flush=True)
    before = settled_conditions(device)
    if before["externalPower"] is False and not allow_battery:
        raise SystemExit(f"Running on battery before {stem.name}. Connect external power or pass --allow-battery.")
    started = now()
    with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    finished = now()
    after = conditions(device)
    flags = condition_flags(before, after)
    save(stem.with_suffix(".receipt.json"), dict(command=cmd, started=started, finished=finished, exitCode=code,
                                                 before=before, after=after, flags=flags))
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


def check_protocol(obj, side, model, device):
    """Check that a report ran the requested framework, device, threads and PyTorch configuration."""
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
        expected.setdefault("compile_rnn", False)
        # Reports written before compile_rnn existed never set allow_rnn.
        recorded = {key: obj["protocol"].get(key, False if key == "compile_rnn" else None) for key in expected}
        if recorded != expected:
            raise ValueError(f"PyTorch ran {recorded}, expected {expected}")
        # A compiled configuration must have compiled something, except an RNN model it was not told to trace.
        graphs = (obj["protocol"].get("compiled_graphs") or {}).get(model)
        if graphs == 0 and (model not in RNN_MODELS or expected["compile_rnn"]):
            raise ValueError("torch.compile compiled no graph for this model")


def measurements(path, side, model, device, epochs, batches):
    """Validate one report against the protocol and return (parameter count, [train ms/update, inference ms...])."""
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    check_protocol(obj, side, model, device)
    row = _index(obj)[model]
    if row.device != device or row.training is None or set(row.inference) != set(BATCHES):
        raise ValueError("Incomplete or mismatched benchmark row")
    raw = rows[0]
    training = raw["Training" if ai else "training"]
    losses = [training["FirstLoss" if ai else "first_loss"], training["LastLoss" if ai else "last_loss"]]
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses)
            or len(training["EpochSeconds" if ai else "epoch_seconds"]) != epochs
            or training["StepsPerEpoch" if ai else "steps_per_epoch"] != batches):
        raise ValueError("Training schedule or loss mismatch")
    times = [row.training.median, *(row.inference[b].median for b in BATCHES)]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("Invalid timing")
    return raw["Parameters" if ai else "parameters"], times


OPERATIONS = [("train", 64), *(("inference", b) for b in BATCHES)]


def summarize(rows, launches):
    """Aggregate the per-launch measurements: medians and ranges per side, AiDotNet against the fastest PyTorch arm."""
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["device"], r["operation"], r["batch"]), []).append(r)
    # One median column per configuration this run used, so an older run's summary keeps its original columns.
    arms_run = [arm for arm in ARMS if any(r["configuration"] == arm for r in rows)]
    summary = []
    for (model, device, operation, batch), items in groups.items():
        by_side = {}
        for item in items:
            by_side.setdefault(item["configuration"], {})[item["launch"]] = item
        ai = by_side.pop("aidotnet", {})
        if not ai or not by_side:
            continue
        medians = {arm: statistics.median(i["ms"] for i in launches_.values()) for arm, launches_ in by_side.items()}
        best = min(medians, key=medians.get)
        paired = sorted(set(ai) & set(by_side[best]))
        ratios = [by_side[best][l]["ms"] / ai[l]["ms"] for l in paired]
        ai_ms = [i["ms"] for i in ai.values()]
        torch_ms = [i["ms"] for i in by_side[best].values()]
        flags = set()
        for item in items:
            flags.update(f for f in item["flags"].split(";") if f)
        if max(ai_ms) / min(ai_ms) > UNSTABLE_SPREAD or max(torch_ms) / min(torch_ms) > UNSTABLE_SPREAD:
            flags.add("unstable")
        if len(ai) < launches or len(by_side[best]) < launches:
            flags.add("missing-launches")
        ai_params = next(iter(ai.values()))["parameters"]
        torch_params = next(iter(by_side[best].values()))["parameters"]
        if ai_params != torch_params:
            flags.add("parameters-differ")
        summary.append(dict(
            model=model, device=device, operation=operation, batch=batch, launches=len(paired),
            ai_parameters=ai_params, torch_parameters=torch_params,
            ai_median_ms=statistics.median(ai_ms), ai_min_ms=min(ai_ms), ai_max_ms=max(ai_ms),
            torch_configuration=best, torch_median_ms=medians[best], torch_min_ms=min(torch_ms), torch_max_ms=max(torch_ms),
            torch_over_ai=medians[best] / statistics.median(ai_ms),
            launch_ratio_min=min(ratios) if ratios else None, launch_ratio_max=max(ratios) if ratios else None,
            **{f"{arm}_median_ms": medians.get(arm) for arm in arms_run},
            flags=";".join(sorted(flags))))
    order = {m: i for i, m in enumerate(MODELS)}
    summary.sort(key=lambda s: (s["device"] != "cuda", order.get(s["model"], 99),
                                [o for o, _ in OPERATIONS].index(s["operation"]), s["batch"]))
    return summary


def markdown(summary):
    """A README-ready table: the median ratio and the range of per-launch ratios for every row."""
    def cell(s):
        if s is None:
            return ""
        text = f"{s['torch_over_ai']:.2f}"
        if s["launch_ratio_min"] is not None:
            text += f" ({s['launch_ratio_min']:.2f}–{s['launch_ratio_max']:.2f})"
        return text + (" ⚑" if s["flags"] else "")
    lines = []
    for device in ("cuda", "cpu"):
        rows = [s for s in summary if s["device"] == device]
        if not rows:
            continue
        lines += [f"**{device.upper()}**: PyTorch time ÷ AiDotNet time, median of launches (range of launches). "
                  "Above 1 means AiDotNet took less time. ⚑ marks a row with a flag in summary.csv.", "",
                  "| Model | Training, batch 64 | Inference, batch 1 | Inference, batch 8 | Inference, batch 16 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for model in dict.fromkeys(s["model"] for s in rows):
            cells = [cell(next((s for s in rows if s["model"] == model and (s["operation"], s["batch"]) == op), None))
                     for op in OPERATIONS]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


COLUMNS = ("model", "device", "launch", "framework", "configuration", "operation", "batch", "parameters", "ms", "flags")


def scale_sides(info, device, model):
    """AiDotNet plus the PyTorch configurations a large-data row runs: the fastest one from --scale-torch-from, or all."""
    chosen = (info["scale"].get("torch") or {}).get(device, {}).get(model)
    return ["aidotnet", chosen] if chosen else sides_for(info, device, model)


def scale_jobs(info):
    """(launch, device, model, dataset, epochs, sides) for every large-data process, sides in launch order."""
    for launch in range(1, info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                for dataset, epochs in info["scale"]["plan"][device][model]:
                    sides = scale_sides(info, device, model)
                    shift = (launch - 1) % len(sides)
                    yield launch, device, model, dataset, epochs, sides[shift:] + sides[:shift]


def collect_scale(dest, info):
    rows, problems = [], []
    for launch, device, model, dataset, epochs, sides in scale_jobs(info):
        for side in sides:
            tag = dest / scale_matrix.tag_name(side, launch, device, model, dataset)
            try:
                receipt = json.loads(tag.with_suffix(".receipt.json").read_text(encoding="utf-8"))
                if receipt["exitCode"] != 0:
                    raise RuntimeError(f"exited {receipt['exitCode']}")
                report = tag.with_suffix(".json")
                check_protocol(json.loads(report.read_text(encoding="utf-8-sig")), side, model, device)
                result = scale_matrix.validate(report, side, model, device, info["scale"]["datasets"][dataset], epochs)
            except (KeyError, ValueError, OSError, RuntimeError, TypeError) as error:
                problems.append(f"{tag.name}: {error}")
                continue
            for epoch in result["epochs"]:
                rows.append(dict(model=model, device=device, dataset=dataset, launch=launch,
                                 framework="AiDotNet" if side == "aidotnet" else "PyTorch", configuration=side, **epoch,
                                 parameters=result["parameters"], rss_mb_peak=result["rss_mb_peak"],
                                 gpu_mem_mb_peak=result["gpu_mem_mb_peak"], flags=";".join(receipt.get("flags", []))))
    return rows, problems


def collect(dest):
    """Read and validate every report a run directory should contain. Return (measurement rows, problems)."""
    info = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
    if info.get("scale"):
        return collect_scale(dest, info)
    common = info["common"]
    epochs, batches = int(common[common.index("--epochs") + 1]), int(common[common.index("--train-batches") + 1])
    rows, problems = [], []
    for launch in range(1, info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                for side in sides_for(info, device, model):
                    tag = dest / f"{side}-{launch}-{device}-{model}"
                    try:
                        receipt = json.loads(tag.with_suffix(".receipt.json").read_text(encoding="utf-8"))
                        if receipt["exitCode"] != 0:
                            raise RuntimeError(f"exited {receipt['exitCode']}")
                        parameters, times = measurements(tag.with_suffix(".json"), side, model, device, epochs, batches)
                    except (KeyError, ValueError, OSError, RuntimeError, TypeError) as error:
                        problems.append(f"{tag.name}: {error}")
                        continue
                    for (operation, batch), ms in zip(OPERATIONS, times):
                        rows.append(dict(model=model, device=device, launch=launch,
                                         framework="AiDotNet" if side == "aidotnet" else "PyTorch", configuration=side,
                                         operation=operation, batch=batch, parameters=parameters, ms=ms,
                                         flags=";".join(receipt.get("flags", []))))
    return rows, problems


def write_outputs(dest, rows, launches):
    """measurements.csv (every launch), summary.csv (medians, ranges, flags) and summary.md (the README table)."""
    if rows and "dataset" in rows[0]:
        scale_matrix.write_outputs(dest, rows)
        return
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
    rows, problems = collect(run_directory)
    write_outputs(target, rows, json.loads((run_directory / "protocol.json").read_text(encoding="utf-8"))["launches"])
    for problem in problems:
        print(problem, file=sys.stderr)
    print(target)
    return 1 if problems else 0


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
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--torch-configurations", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="PyTorch configurations to run (each device runs those that apply to it)")
    parser.add_argument("--launches", type=int, help="Independent launches of every process (default 5, smoke 1)")
    parser.add_argument("--smoke", action="store_true", help="Two epochs of two updates, three measured inference calls.")
    parser.add_argument("--tensors", type=Path, help="Build against this Tensors checkout instead of the repository's NuGet dependency.")
    parser.add_argument("--output", type=Path, default=HERE / "results", help="Parent for unique run directories")
    parser.add_argument("--allow-battery", action="store_true", help="Run even when the machine is on battery power")
    parser.add_argument("--allow-gpu-clock-cap", action="store_true",
                        help="Run CUDA measurements even when the GPU sustains under 60%% of its maximum SM clock")
    parser.add_argument("--gpu-clock-lock", type=int, metavar="MHZ",
                        help="Declare that the GPU's SM clock is locked at MHZ (nvidia-smi -lgc MHZ,MHZ, run as administrator). "
                             "The run records it, refuses to start unless the GPU sustains it, and judges gpu-clock-low "
                             "against it. Use it on a GPU that computes wrong values at its top clocks.")
    parser.add_argument("--summarize", type=Path, metavar="RUN_DIR",
                        help="Only rebuild RUN_DIR's CSVs and table from its raw reports, into RUN_DIR/regenerated")
    parser.add_argument("--scale", type=Path, metavar="DATA_ROOT",
                        help="Large-data comparison over the datasets scale/prepare_data.py wrote to DATA_ROOT")
    parser.add_argument("--scale-torch-from", type=Path, metavar="SUMMARY_CSV",
                        help="With --scale, run only the fastest PyTorch configuration per training row of this run_matrix "
                             "summary.csv (default: every configuration)")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR",
                        help="Continue an interrupted run with its recorded settings and binaries; completed processes are kept")
    args = parser.parse_args()
    if args.summarize:
        sys.exit(regenerate(args.summarize.resolve()))
    import torch
    import torchvision
    import torch._dynamo.config as dynamo
    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compilation-error fallback.")
    if os.name == "nt" and not shutil.which("cl"):
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
        launches = args.launches if args.launches is not None else (1 if args.smoke else 5)
        if launches < 1 or len(set(args.devices)) != len(args.devices) or len(set(args.models)) != len(args.models):
            parser.error("Select distinct devices/models and a positive launch count.")
        arms = {device: [a for a in args.torch_configurations if device in ARMS[a]["devices"]] for device in args.devices}
        if not all(arms.values()):
            parser.error("Every device needs at least one PyTorch configuration that runs on it.")
        epochs, batches, iters, warm = (2, 2, 3, 1) if args.smoke else (3, 20, 100, 10)
        common = ["--epochs", str(epochs), "--train-batches", str(batches), "--batch-size", "64", "--inference-batches", "1,8,16",
                  "--inference-iterations", str(iters), "--warmup-iterations", str(warm), "--warmup-min-ms", "1000", "--seed", "1234"]
        scale = None
        if args.scale:
            unsupported = [m for m in args.models if m not in scale_matrix.DATASETS]
            if unsupported:
                parser.error(f"No large-data datasets for: {', '.join(unsupported)}")
            plan = scale_matrix.default_plan(args.devices, args.models)
            names = sorted({dataset for device in plan.values() for jobs in device.values() for dataset, _ in jobs})
            missing = [n for n in names if not (args.scale / n / "meta.json").is_file()]
            if missing:
                parser.error(f"Run scale/prepare_data.py first; missing under {args.scale}: {', '.join(missing)}")
            torch_choice = None
            if args.scale_torch_from:
                found = scale_matrix.fastest_configurations(args.scale_torch_from)
                torch_choice = {d: {m: found[d][m] for m in args.models} for d in args.devices}
            scale = dict(dataRoot=str(args.scale.resolve()), plan=plan, torch=torch_choice,
                         torchFrom=str(args.scale_torch_from) if args.scale_torch_from else None,
                         datasets={n: json.loads((args.scale / n / "meta.json").read_text(encoding="utf-8")) for n in names})
            common = ["--batch-size", "64", "--seed", "1234"]
        import psutil
        args.output.mkdir(exist_ok=True, parents=True)
        dest = Path(tempfile.mkdtemp(dir=args.output.resolve(), prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")))
        print(f"Results: {dest}", flush=True)
        info = dict(started=now(), models=args.models, devices=args.devices, launches=launches, smoke=args.smoke, common=common,
                    scale=scale, gpuClockLock=args.gpu_clock_lock,
                    threads=THREADS, referenceModels=[m for m in REFERENCE if m in args.models],
                    torchConfigurations={d: arms[d] for d in args.devices},
                    configurationDescriptions={a: ARMS[a]["description"] for a in args.torch_configurations},
                    configurationModels={a: list(ARMS[a]["models"]) for a in args.torch_configurations if "models" in ARMS[a]},
                    ai=revision(HERE.parents[1]), torch=torch.__version__, torchvision=torchvision.__version__,
                    packages=package_versions("torch", "torchvision", "triton", "triton-windows", "numpy", "psutil"),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    os=platform.platform(), cpu=cpu_name(), logicalCpus=os.cpu_count(), memoryBytes=psutil.virtual_memory().total,
                    python=sys.version, sdk=subprocess.check_output(["dotnet", "--version"], text=True).strip())
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

    if "cuda" in info["devices"] and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select --devices cpu or install CUDA-enabled PyTorch.")
    global GPU_CLOCK_LOCK
    if args.resume and args.gpu_clock_lock and args.gpu_clock_lock != info.get("gpuClockLock"):
        parser.error(f"This run declared --gpu-clock-lock {info.get('gpuClockLock')}; resume it with the same lock.")
    GPU_CLOCK_LOCK = info.get("gpuClockLock")
    if "cuda" in info["devices"] and shutil.which("nvidia-smi"):
        # Checked on every start, including a resume: the platform can cap a laptop GPU between sessions, and a clock
        # lock does not survive a reboot.
        probe = gpu_clock_probe(env)
        info.setdefault("gpuClockProbes", []).append(probe)
        save(dest / "protocol.json", info)
        print(f"GPU sustained {probe['medianSmClock']:.0f} of {probe['maxSmClock']:.0f} MHz under load"
              + (f" (declared lock {GPU_CLOCK_LOCK} MHz)" if GPU_CLOCK_LOCK else ""), flush=True)
        if GPU_CLOCK_LOCK and probe["medianSmClock"] > GPU_CLOCK_LOCK + 30:
            sys.exit(f"The GPU ran at {probe['medianSmClock']:.0f} MHz, above the declared {GPU_CLOCK_LOCK} MHz lock, so the "
                     f"lock is not in effect. Run nvidia-smi -lgc {GPU_CLOCK_LOCK},{GPU_CLOCK_LOCK} as administrator.")
        if probe["medianSmClock"] < clock_floor(probe["maxSmClock"]) and not args.allow_gpu_clock_cap:
            sys.exit(f"The GPU sustains only {probe['medianSmClock']:.0f} of {probe['maxSmClock']:.0f} MHz under load, so "
                     "its timings would describe the cap. Check the charger and the performance profile, or pass "
                     "--allow-gpu-clock-cap.")
    dll = dest / "bin/AiDotNet.PyTorchParity.dll"
    common = info["common"]
    refs = info["referenceModels"]
    if refs:
        ref = dest / "reference"
        if not succeeded(dest / "reference-export"):
            run([sys.executable, str(HERE / "pytorch/export_reference.py"), "--models", ",".join(refs), "--out", str(ref)],
                dest / "reference-export", env, allow_battery=args.allow_battery)
        for device in info["devices"]:
            stem = dest / f"reference-{device}"
            if not succeeded(stem):
                run(["dotnet", str(dll), "--verify-step1", str(ref), "--models", ",".join(refs), "--device", device], stem, env,
                    device, args.allow_battery)
            lines = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace").splitlines()
            if not all(any(line.startswith(f"[step1] {m}: PASS") for line in lines) for m in refs):
                raise RuntimeError(f"Missing first-update verdict. See {stem.with_suffix('.log')}")
    for launch, device, model, dataset, epochs, sides in (scale_jobs(info) if info.get("scale") else ()):
        root = Path(info["scale"]["dataRoot"])
        for side in sides:
            tag = dest / scale_matrix.tag_name(side, launch, device, model, dataset)
            if completed(tag):
                continue
            cmd = (["dotnet", str(dll)] if side == "aidotnet" else
                   [sys.executable, str(HERE / "pytorch/benchmark.py"), *ARMS[side]["flags"], "--threads", str(THREADS)])
            try:
                run([*cmd, "--models", model, "--device", device, *common, "--epochs", str(epochs), "--data", str(root / dataset),
                     "--output", str(tag.with_suffix(".json"))], tag, env, device, args.allow_battery)
            except RuntimeError as error:
                print(error, flush=True)
    for launch in range(1, 0 if info.get("scale") else info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                sides = sides_for(info, device, model)
                # Rotate the order every launch so no side always runs first (or last) on a warming machine.
                shift = (launch - 1) % len(sides)
                for side in sides[shift:] + sides[:shift]:
                    tag = dest / f"{side}-{launch}-{device}-{model}"
                    if completed(tag):
                        continue  # only when resuming: this process already finished
                    cmd = (["dotnet", str(dll)] if side == "aidotnet" else
                           [sys.executable, str(HERE / "pytorch/benchmark.py"), *ARMS[side]["flags"], "--threads", str(THREADS)])
                    try:
                        run([*cmd, "--models", model, "--device", device, *common, "--output", str(tag.with_suffix(".json"))],
                            tag, env, device, args.allow_battery)
                    except RuntimeError as error:
                        print(error, flush=True)  # its receipt records the exit code; collect() reports it below
    # Every measurement comes from re-reading the reports on disk, exactly as --summarize does for an archive.
    rows, problems = collect(dest)
    write_outputs(dest, rows, info["launches"])
    save(dest / "completion.json", dict(finished=now(), failures=problems))
    if problems:
        sys.exit("\n".join(problems))
    print(dest / ("scale-summary.md" if info.get("scale") else "summary.md"))


if __name__ == "__main__":
    main()
