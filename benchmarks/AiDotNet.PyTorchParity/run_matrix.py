"""Run end-to-end AiDotNet/Python PyTorch training and inference comparisons from this checkout.

Every model, device and launch runs AiDotNet and each applicable PyTorch configuration in its own fresh process, in an
order that rotates between launches. AiDotNet is compared against the fastest PyTorch configuration for each row.
Every row reports the median and interquartile range over the launches (9 by default), the trainable parameter
counts of both sides, and the first and last training loss of both sides as evidence that the timed step trains.

--classical runs the classical-ML rows (ClassicalBench.cs against scikit-learn, XGBoost and LightGBM) the same way.

--scale DATA_ROOT runs the large-data comparison instead: whole epochs over the real and synthetic datasets that
scale/prepare_data.py wrote to DATA_ROOT, with the test split scored after every epoch (scale/scale_matrix.py).
"""
import argparse
import contextlib
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


def uncovered_pairs(devices, models, configurations):
    """Each "model on device" pair that no selected PyTorch configuration runs. A configuration limited to some models
    (the RNN arms) can run on a device yet run none of the selected models there, which would leave that row with no
    PyTorch side and no comparison, so the check is per pair rather than per device."""
    return [f"{model} on {device}" for device in devices for model in models
            if not any(device in ARMS[a]["devices"] and model in ARMS[a].get("models", (model,)) for a in configurations)]


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


# The nvidia-smi selector (GPU-<uuid>) of the GPU the CUDA launches use; set by main(). None queries every GPU.
GPU_ID = None


def gpu_selector():
    """nvidia-smi arguments that restrict a query, lock or reset to the benchmark GPU."""
    return [f"--id={GPU_ID}"] if GPU_ID else []


def benchmark_gpu_id(torch):
    """The UUID selector of CUDA device 0 of this process: the GPU every launch uses. By UUID, because CUDA orders
    devices (and applies CUDA_VISIBLE_DEVICES) differently from nvidia-smi's PCI order."""
    try:
        uuid = str(torch.cuda.get_device_properties(0).uuid)
    except (AttributeError, AssertionError, RuntimeError):
        return None  # an older PyTorch without the property: queries fall back to every GPU
    return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"


def sole_gpu_id():
    """The UUID of the only GPU nvidia-smi lists, or None when it lists none, several, or cannot be queried. With a
    single GPU the benchmark GPU is known even when PyTorch does not report its UUID."""
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, capture_output=True)
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []
    return uuids[0] if len(uuids) == 1 else None


def nvidia_smi(*args):
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", *gpu_selector(), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_state():
    """Clocks, temperature, power, utilization and the clock-limiting reasons of the benchmark GPU."""
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


CLOCK_FIELDS = ["clocks.sm", "clocks.mem", "utilization.gpu", "temperature.gpu", "power.draw"]
# The fields a clock sample needs; the others are optional telemetry a GPU may report as "[N/A]".
REQUIRED_CLOCK_FIELDS = ("clocks.sm", "clocks.mem", "utilization.gpu")
# GPU utilization (percent) at or above which a clock sample counts as taken under load.
LOADED_GPU_PERCENT = 50.0
# The clock locks requested for this run ({"sm": MHz, "mem": MHz}), whether or not the driver accepted them; set by
# main() and read by clock_flags(). What was actually acquired is lock_clocks()'s result, which reset_clocks() reads.
REQUESTED_CLOCKS = {}


def start_clock_sampler(stem, device):
    """Sample the GPU's clocks every CLOCK_SAMPLE_SECONDS into STEM.clocks.csv while a CUDA launch runs.

    Only the benchmark GPU is sampled. Without its UUID (GPU_ID None) nvidia-smi would report every GPU in rows that do
    not name the device, and another GPU's clocks could pass as this launch's, so nothing is sampled and the launch
    records no clock evidence (main() records why in protocol.json)."""
    if device != "cuda" or not shutil.which("nvidia-smi") or not GPU_ID:
        return None
    out = stem.with_suffix(".clocks.csv").open("w", encoding="utf-8")
    process = subprocess.Popen(["nvidia-smi", *gpu_selector(), f"--query-gpu={','.join(CLOCK_FIELDS)}", "--format=csv,noheader,nounits",
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
        values = line.split(",")
        if len(values) != len(CLOCK_FIELDS):
            continue  # a line cut off by terminate()
        sample = {}
        for field, value in zip(CLOCK_FIELDS, values):
            try:
                sample[field] = float(value)
            except ValueError:
                sample[field] = None  # "[N/A]", or a field cut off by terminate()
        if any(sample[field] is None for field in REQUIRED_CLOCK_FIELDS):
            continue  # no usable clock reading; a missing temperature or power alone keeps the sample
        samples.append(sample)
    loaded = [s for s in samples if s["utilization.gpu"] >= LOADED_GPU_PERCENT]
    if not loaded:
        return dict(samples=len(samples), loadedSamples=0)
    sm = [s["clocks.sm"] for s in loaded]
    return dict(samples=len(samples), loadedSamples=len(loaded), smMedianMhz=statistics.median(sm), smMinMhz=min(sm),
                smMaxMhz=max(sm), memMedianMhz=statistics.median(s["clocks.mem"] for s in loaded),
                maxTemperatureC=max((s["temperature.gpu"] for s in loaded if s["temperature.gpu"] is not None),
                                    default=None))


def clock_flags(clocks):
    """clock-varied when the SM clock under load moved; clock-unlocked when it missed the requested lock, including a
    lock the driver refused."""
    flags = set()
    if not clocks or not clocks.get("loadedSamples"):
        return flags
    if clocks["smMaxMhz"] / clocks["smMinMhz"] - 1 > CLOCK_TOLERANCE:
        flags.add("clock-varied")
    if REQUESTED_CLOCKS.get("sm") and abs(clocks["smMedianMhz"] / REQUESTED_CLOCKS["sm"] - 1) > CLOCK_TOLERANCE:
        flags.add("clock-unlocked")
    if REQUESTED_CLOCKS.get("mem") and abs(clocks["memMedianMhz"] / REQUESTED_CLOCKS["mem"] - 1) > CLOCK_TOLERANCE:
        flags.add("clock-unlocked")
    return flags


def lock_clocks(sm, mem, result):
    """Lock the GPU's SM and memory clocks with nvidia-smi -lgc / -lmc (needs an elevated prompt). Record what each
    request printed in RESULT as it is made, so a failure later still leaves the taken locks for reset_clocks; a
    refused lock is recorded, not fatal, and every CUDA launch still records its clocks."""
    for key, flag, value in (("sm", "-lgc", sm), ("mem", "-lmc", mem)):
        if value:
            done = subprocess.run(["nvidia-smi", *gpu_selector(), flag, f"{value},{value}"], text=True, capture_output=True)
            result[key] = dict(mhz=value, exitCode=done.returncode, output=(done.stdout + done.stderr).strip())
            if done.returncode != 0:
                print(f"nvidia-smi {flag} {value},{value} was refused: {result[key]['output']}", flush=True)
    return result


def reset_clocks(locked):
    """Reset every clock lock_clocks() acquired (exit code 0), never one it was refused, which this run does not own.
    Every reset is attempted. Return ({key: result} for each reset, [failure messages])."""
    results, failures = {}, []
    for key, flag in (("sm", "-rgc"), ("mem", "-rmc")):
        if (locked.get(key) or {}).get("exitCode") != 0:
            continue
        done = subprocess.run(["nvidia-smi", *gpu_selector(), flag], text=True, capture_output=True)
        results[key] = dict(exitCode=done.returncode, output=(done.stdout + done.stderr).strip())
        if done.returncode != 0:
            failures.append(f"nvidia-smi {flag} exited {done.returncode}; the GPU's clocks may still be locked: "
                            f"{results[key]['output']}")
    return results, failures


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


def clock_lock_path(gpu_id):
    """The cross-process lock an invocation that locks GPU clocks holds from its lock to its reset, so another
    invocation on the same GPU cannot lock or reset those clocks while this one still measures under them."""
    name = "".join(c if c.isalnum() or c == "-" else "_" for c in (gpu_id or "all-gpus"))
    return HERE / f".run_matrix.clocks-{name}.lock"


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


def check_protocol(obj, side, model, device, train_batch=None):
    """Check that a report ran the requested framework, device, threads and PyTorch configuration, and, when
    TRAIN_BATCH is given, that it trained at that batch."""
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    if len(rows) != 1 or obj["Framework" if ai else "framework"] != ("AiDotNet" if ai else "PyTorch"):
        raise ValueError("Unexpected framework or number of models")
    # The matrix runs fp32 in both arms; a report at another precision (benchmark --precision) is not comparable.
    # AiDotNet reports gained their Protocol block with --precision; one without it predates the option and ran fp32,
    # so archived runs still regenerate.
    protocol = obj.get("Protocol" if ai else "protocol") or {}
    precision = protocol.get("Precision" if ai else "precision", "fp32")
    if precision != "fp32":
        raise ValueError(f"{side} ran at precision {precision!r}; the matrix compares fp32")
    # The training timing is filed under the schedule's training batch, so the report must have trained at it.
    recorded_batch = protocol.get("BatchSize" if ai else "batch_size")
    if train_batch is not None and recorded_batch != train_batch:
        raise ValueError(f"{side} trained at batch {recorded_batch!r}, expected {train_batch}")
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


def measurements(path, side, model, device, common):
    """Validate one report against the protocol. Return (parameter count, [train ms/update, inference ms...],
    (first loss, last loss))."""
    epochs, batches, operations = schedule(common)
    inference = [b for o, b in operations if o == "inference"]
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    check_protocol(obj, side, model, device, next(b for o, b in operations if o == "train"))
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
            **{f"{arm}_median_ms": medians.get(arm) for arm in arms_run},
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
                check_protocol(json.loads(report.read_text(encoding="utf-8-sig")), side, model, device,
                               int(info["common"][info["common"].index("--batch-size") + 1]))
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
    rows, problems = [], []
    for launch in range(1, info["launches"] + 1):
        for device in info["devices"]:
            for model in info["models"]:
                common = row_common(info, model)
                for side in sides_for(info, device, model):
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
# The report each classical side writes into its launch directory.
CLASSICAL_REPORTS = {"aidotnet": "aidotnet_classical.json", "python": "python_classical.json"}


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
        for side, report in CLASSICAL_REPORTS.items():
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


def require_cpp_tools(parser, classical):
    """The model rows compile with torch.compile, which needs the MSVC compiler (cl) on Windows; classical runs do not."""
    if os.name == "nt" and not classical and not shutil.which("cl"):
        parser.error("Use Developer PowerShell for Visual Studio with the x64 C++ build tools.")


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
    args.models = list(dict.fromkeys(m for name in args.models for m in GROUPS.get(name, [name])))
    if args.summarize:
        sys.exit(regenerate(args.summarize.resolve()))
    import torch
    import torchvision
    import torch._dynamo.config as dynamo
    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compilation-error fallback.")
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
        # The run's recorded mode decides, so a classical run resumes without repeating --classical.
        require_cpp_tools(parser, info.get("classical"))
        print(f"Resuming: {dest}", flush=True)
    else:
        require_cpp_tools(parser, args.classical)
        launches = args.launches if args.launches is not None else (1 if args.smoke else DEFAULT_LAUNCHES)
        if launches < 1 or len(set(args.devices)) != len(args.devices):
            parser.error("Select distinct devices and a positive launch count.")
        if args.classical:
            args.devices, args.models = ["cpu"], []
        arms = {device: [a for a in args.torch_configurations if device in ARMS[a]["devices"]] for device in args.devices}
        uncovered = uncovered_pairs(args.devices, args.models, args.torch_configurations)
        if uncovered:
            parser.error("No selected PyTorch configuration runs " + ", ".join(uncovered) + ".")
        # "common" is the core rows' schedule; each row's own arguments are in rowArguments.
        common = row_arguments("mlp", args.smoke)
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
                    rowArguments={m: row_arguments(m, args.smoke) for m in args.models},
                    families={m: FAMILIES[m]["family"] for m in args.models if m in FAMILIES},
                    scale=scale, gpuClockLock=args.gpu_clock_lock,
                    threads=THREADS, referenceModels=[m for m in REFERENCE if m in args.models],
                    torchConfigurations={d: arms[d] for d in args.devices},
                    configurationDescriptions={a: ARMS[a]["description"] for a in args.torch_configurations},
                    configurationModels={a: list(ARMS[a]["models"]) for a in args.torch_configurations if "models" in ARMS[a]},
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
    global GPU_ID
    if "cuda" in info["devices"]:
        GPU_ID = benchmark_gpu_id(torch) or sole_gpu_id()
        if args.resume and info.get("gpuId") not in (None, GPU_ID):
            parser.error(f"--resume needs the run's GPU {info['gpuId']}; this process sees {GPU_ID}.")
        info["gpuId"] = GPU_ID
        if GPU_ID is None:
            info["clockSampling"] = ("unavailable: PyTorch reports no device UUID and nvidia-smi lists more than one "
                                     "GPU (or none), so launches record no GPU clocks")
    try:
        lock_sm, lock_mem = resolve_clock_request(info, args.lock_gpu_clock, args.lock_memory_clock, bool(args.resume))
    except ValueError as error:
        parser.error(str(error))
    info["clockLockRequest"] = dict(sm=lock_sm, mem=lock_mem)
    REQUESTED_CLOCKS.clear()
    REQUESTED_CLOCKS.update({k: v for k, v in info["clockLockRequest"].items() if v})
    # Persisted before any launch, so a resume of this run (even an unlocked one) checks the same GPU and settings.
    save(dest / "protocol.json", info)
    locking = "cuda" in info["devices"] and bool(lock_sm or lock_mem)
    problems, reset_failures = [], []
    # Held from the lock to the reset, so another invocation on this GPU cannot reset the clocks under this one.
    with build_lock(clock_lock_path(GPU_ID)) if locking else contextlib.nullcontext():
        locked = {}
        try:
            if locking:
                lock_clocks(lock_sm, lock_mem, locked)
                info.setdefault("clockLocks", []).append(dict(time=now(), **locked))
                save(dest / "protocol.json", info)
            problems = run_models(dest, info, env, args.allow_battery)
        finally:
            resets, reset_failures = reset_clocks(locked)
            if resets:
                info.setdefault("clockResets", []).append(dict(time=now(), **resets))
                save(dest / "protocol.json", info)
            for failure in reset_failures:
                print(failure, file=sys.stderr, flush=True)
    if problems or reset_failures:
        sys.exit("\n".join(problems + reset_failures))
    print(dest / ("scale-summary.md" if info.get("scale") else "summary.md"))


def resolve_clock_request(info, sm, mem, resuming):
    """The (SM, memory) clock locks for this invocation. A new run uses the flags. A resume uses the run's recorded
    request, including an explicit no-lock request, and rejects flags that differ from it. Runs recorded before
    clockLockRequest existed fall back to their first lock attempt, or to no lock when they made none."""
    if not resuming:
        return sm, mem
    if "clockLockRequest" in info:
        recorded = (info["clockLockRequest"].get("sm"), info["clockLockRequest"].get("mem"))
    elif info.get("clockLocks"):
        first = info["clockLocks"][0]
        recorded = tuple((first.get(k) or {}).get("mhz") for k in ("sm", "mem"))
    else:
        recorded = (None, None)
    if (sm or mem) and (sm, mem) != recorded:
        raise ValueError(f"--resume continues with the run's recorded clock locks (SM {recorded[0]}, memory "
                         f"{recorded[1]} MHz); omit --lock-gpu-clock / --lock-memory-clock or pass the same values.")
    return recorded


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
    for launch, device, model, dataset, epochs, sides in (scale_jobs(info) if info.get("scale") else ()):
        root = Path(info["scale"]["dataRoot"])
        for side in sides:
            tag = dest / scale_matrix.tag_name(side, launch, device, model, dataset)
            if completed(tag):
                continue
            cmd = (["dotnet", str(dll)] if side == "aidotnet" else
                   [sys.executable, str(HERE / "pytorch/benchmark.py"), *ARMS[side]["flags"], "--threads", str(THREADS)])
            try:
                run([*cmd, "--models", model, "--device", device, *info["common"], "--epochs", str(epochs),
                     "--data", str(root / dataset), "--output", str(tag.with_suffix(".json"))], tag, env, device,
                    allow_battery)
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
        if order[0] == "python" and not all((directory / name).is_file() for name in ("train.csv", "test.csv")):
            # The C# side writes the data (seeded, the same every launch); reuse launch 1's so Python can go first.
            seed = [dest / "classical-1" / name for name in ("train.csv", "test.csv")]
            if all(path.is_file() for path in seed):
                for path in seed:
                    shutil.copyfile(path, directory / path.name)
            else:
                # Launch 1's C# side wrote no data (it failed): this launch's C# side runs first and writes its own.
                print(f"classical-{launch}: no seed data from launch 1; running AiDotNet first", flush=True)
                order.reverse()
        for side in order:
            stem = dest / f"classical-{side}-{launch}"
            if succeeded(stem) and (directory / CLASSICAL_REPORTS[side]).is_file():
                continue  # only when resuming: this side exited 0 and wrote its report
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
