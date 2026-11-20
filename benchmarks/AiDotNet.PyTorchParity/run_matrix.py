"""Run end-to-end AiDotNet/Python PyTorch training and inference comparisons from this checkout.

Every model, device and launch runs AiDotNet and each applicable PyTorch configuration in its own fresh process, in an
order that rotates between launches. AiDotNet is compared against the fastest PyTorch configuration for each row.
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
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
}
# A side whose slowest launch exceeds its fastest by this factor is reported as unstable.
UNSTABLE_SPREAD = 1.5
# System CPU use (percent) sampled just before a launch above which that launch is flagged as contended.
BUSY_CPU_PERCENT = 20.0
# GPU utilization (percent) sampled just before a CUDA launch above which that launch is flagged as contended.
BUSY_GPU_PERCENT = 10.0


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
    try:
        if float((before.get("gpu") or {}).get("utilization.gpu", 0)) > BUSY_GPU_PERCENT:
            flags.add("busy-gpu")
    except ValueError:
        pass  # "[N/A]" on GPUs that do not report utilization
    return sorted(flags)


def run(cmd, stem, env, device=None, allow_battery=False):
    """Run one process with its output in STEM.log and a receipt in STEM.receipt.json. Return its condition flags."""
    print(stem.name, flush=True)
    before = conditions(device)
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


def measurements(path, side, model, device, epochs, batches):
    """Validate one report against the protocol and return (parameter count, [train ms/update, inference ms...])."""
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    if len(rows) != 1 or obj["Framework" if ai else "framework"] != ("AiDotNet" if ai else "PyTorch"):
        raise ValueError("Unexpected framework or number of models")
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
            **{f"{arm}_median_ms": medians.get(arm) for arm in ARMS},
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


def collect(dest):
    """Read and validate every report a run directory should contain. Return (measurement rows, problems)."""
    info = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
    common = info["common"]
    epochs, batches = int(common[common.index("--epochs") + 1]), int(common[common.index("--train-batches") + 1])
    rows, problems = [], []
    for launch in range(1, info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                for side in ["aidotnet", *info["torchConfigurations"][device]]:
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
    parser.add_argument("--summarize", type=Path, metavar="RUN_DIR",
                        help="Only rebuild RUN_DIR's CSVs and table from its raw reports, into RUN_DIR/regenerated")
    args = parser.parse_args()
    if args.summarize:
        sys.exit(regenerate(args.summarize.resolve()))
    launches = args.launches if args.launches is not None else (1 if args.smoke else 5)
    if launches < 1 or len(set(args.devices)) != len(args.devices) or len(set(args.models)) != len(args.models):
        parser.error("Select distinct devices/models and a positive launch count.")
    import torch
    import torchvision
    import torch._dynamo.config as dynamo
    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compilation-error fallback.")
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select --devices cpu or install CUDA-enabled PyTorch.")
    if os.name == "nt" and not shutil.which("cl"):
        parser.error("Use Developer PowerShell for Visual Studio with the x64 C++ build tools.")
    if on_external_power() is False and not args.allow_battery:
        parser.error("The machine is on battery power. Connect external power or pass --allow-battery.")
    arms = {device: [a for a in args.torch_configurations if device in ARMS[a]["devices"]] for device in args.devices}
    if not all(arms.values()):
        parser.error("Every device needs at least one PyTorch configuration that runs on it.")
    args.output.mkdir(exist_ok=True, parents=True)
    dest = Path(tempfile.mkdtemp(dir=args.output.resolve(), prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")))
    print(f"Results: {dest}", flush=True)
    env = os.environ.copy()
    # A CUDA row must not fall back to OpenCL/HIP when CUDA fails to initialise. CPU launches never read this.
    env["AIDOTNET_DIRECTGPU_BACKENDS"] = "cuda"
    for key in ("AIDOTNET_CPU_THREADS", "AIDOTNET_BLAS_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(THREADS)
    epochs, batches, iters, warm = (2, 2, 3, 1) if args.smoke else (3, 20, 100, 10)
    common = ["--epochs", str(epochs), "--train-batches", str(batches), "--batch-size", "64", "--inference-batches", "1,8,16",
              "--inference-iterations", str(iters), "--warmup-iterations", str(warm), "--warmup-min-ms", "1000", "--seed", "1234"]
    refs = [m for m in REFERENCE if m in args.models]
    import psutil
    info = dict(started=now(), models=args.models, devices=args.devices, launches=launches, smoke=args.smoke, common=common,
                threads=THREADS, referenceModels=refs, torchConfigurations={d: arms[d] for d in args.devices},
                configurationDescriptions={a: ARMS[a]["description"] for a in args.torch_configurations},
                ai=revision(HERE.parents[1]), torch=torch.__version__, torchvision=torchvision.__version__,
                packages=package_versions("torch", "torchvision", "triton", "triton-windows", "numpy", "psutil"),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
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
        run(build, dest / "build", build_environment(env))
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
    if refs:
        ref = dest / "reference"
        run([sys.executable, str(HERE / "pytorch/export_reference.py"), "--models", ",".join(refs), "--out", str(ref)],
            dest / "reference-export", env, allow_battery=args.allow_battery)
        for device in args.devices:
            stem = dest / f"reference-{device}"
            run(["dotnet", str(dll), "--verify-step1", str(ref), "--models", ",".join(refs), "--device", device], stem, env,
                device, args.allow_battery)
            lines = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace").splitlines()
            if not all(any(line.startswith(f"[step1] {m}: PASS") for line in lines) for m in refs):
                raise RuntimeError(f"Missing first-update verdict. See {stem.with_suffix('.log')}")
    for launch in range(1, launches + 1):
        for device in args.devices:
            for model in args.models:
                sides = ["aidotnet", *arms[device]]
                # Rotate the order every launch so no side always runs first (or last) on a warming machine.
                shift = (launch - 1) % len(sides)
                for side in sides[shift:] + sides[:shift]:
                    cmd = (["dotnet", str(dll)] if side == "aidotnet" else
                           [sys.executable, str(HERE / "pytorch/benchmark.py"), *ARMS[side]["flags"], "--threads", str(THREADS)])
                    tag = dest / f"{side}-{launch}-{device}-{model}"
                    try:
                        run([*cmd, "--models", model, "--device", device, *common, "--output", str(tag.with_suffix(".json"))],
                            tag, env, device, args.allow_battery)
                    except RuntimeError as error:
                        print(error, flush=True)  # its receipt records the exit code; collect() reports it below
    # Every measurement comes from re-reading the reports on disk, exactly as --summarize does for an archive.
    rows, problems = collect(dest)
    write_outputs(dest, rows, launches)
    save(dest / "completion.json", dict(finished=now(), failures=problems))
    if problems:
        sys.exit("\n".join(problems))
    print(dest / "summary.md")


if __name__ == "__main__":
    main()
