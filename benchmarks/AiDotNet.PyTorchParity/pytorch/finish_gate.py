"""Alternating fresh-process parity campaign. Unknown measurements never pass.

Commands are argument arrays in a JSON config, not shell strings. Templates may use
{model}, {device}, {output}, {cache}, and the protocol keys. Each scored process has
an identical external RSS/VRAM sampler; profiled allocation processes are separate.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import threading
import time

import psutil
from measurement_contracts import CACHE_SENSITIVE_ENVIRONMENT, allocation_metrics, cache_coverage, checked_phases, fixed_windows, raw_file
from python_process import bootstrap_argv, resolve_python_process, resolve_script, stop_launch
from lstm_equivalence import lstm_correctness_gate, lstm_workload_gate
from torch_policy import compare_reference, sha as streaming_sha




def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snake(key: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


def normalize(value: object) -> object:
    if isinstance(value, dict):
        return {snake(key): normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def number(value: object) -> float | None:
    return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def paired_gate(ours: list[float | None], theirs: list[float | None], maximum: float = .9,
                resamples: int = 20000) -> dict:
    """A one-sided 95% paired-bootstrap bound of the median log ratio, with a fixed seed."""
    if len(ours) != len(theirs) or len(ours) < 5 or any(number(x) is None or x < 0 for x in ours + theirs):
        return {"status": "unverified", "reason": "At least five complete, finite, nonnegative matched pairs are required."}
    if any(x == 0 for x in theirs):
        # A zero-allocation reference cannot be beaten by 10%; equality is reported as a tie, never a win.
        return {"status": "fail", "reason": "Reference includes zero; a strict percentage win is not established."}
    ratios = [a / b for a, b in zip(ours, theirs)]
    if any(not math.isfinite(ratio) for ratio in ratios):
        return {"status": "unverified", "reason": "Derived ratios must be finite."}
    if all(ratio == 0 for ratio in ratios):
        return {"status": "pass", "median_ratio": 0.0, "upper_95_ratio": 0.0, "pairs": len(ratios)}
    # Tiny positive floor admits genuine zero allocations without log(0).
    logs = [math.log(max(ratio, 1e-300)) for ratio in ratios]
    rng = random.Random(2561)
    draws = sorted(statistics.median(rng.choices(logs, k=len(logs))) for _ in range(resamples))
    upper = math.exp(draws[math.ceil(.95 * len(draws)) - 1])
    point = midpoint_median(ratios)
    return {"status": "pass" if point <= maximum and upper <= maximum else "fail",
            "median_ratio": point, "upper_95_ratio": upper, "pairs": len(ratios)}


def warmed_cuda_calls_gate(ours: list[int | None], theirs: list[int | None], maximum: float) -> dict:
    """Score exact native CUDA calls during a warmed workload phase."""
    if (len(ours) != len(theirs) or len(ours) < 5 or
            any(type(value) is not int or value < 0 for value in ours + theirs)):
        return {"status": "unverified", "reason": "Five complete pairs of native CUDA call counts are required."}
    if any(a != 0 and b == 0 for a, b in zip(ours, theirs)):
        return {"status": "fail", "reason": "AiDotNet issued native CUDA calls against a zero-call reference."}
    positive = [(a, b) for a, b in zip(ours, theirs) if b > 0]
    if not positive:
        return {"status": "pass", "zero_floor_pairs": len(ours), "reason": "Both frameworks issued zero native calls."}
    result = paired_gate([a for a, _ in positive], [b for _, b in positive], maximum)
    result["zero_floor_pairs"] = len(ours) - len(positive)
    return result


def phase_group(phase: str) -> str:
    if phase.startswith("construction"):
        return "construction"
    if phase.startswith("training"):
        return "training"
    match = re.match(r"inference-b([0-9]+)", phase)
    return "inference_b" + match[1] if match else phase


def phase_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


def midpoint_median(values: list[float]) -> float:
    """statistics.median adds the two middle values before halving, which overflows for finite values near the
    float maximum; low + (high - low) / 2 cannot for the nonnegative ratios scored here."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2


def crosses_phase_boundary(sample: dict, boundaries: list[int]) -> bool:
    """A device query that started in one phase and returned in another has no known observation phase."""
    start = sample.get("query_start_unix_ms")
    if start is None:
        return False
    return any(start < boundary <= sample["unix_ms"] for boundary in boundaries)


def memory_metrics(samples: list[dict], phases: list[dict]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("rss_bytes", "gpu_device_bytes"):
        values = [sample[metric] for sample in samples if number(sample.get(metric)) is not None]
        result["whole_process_" + metric] = max(values) if values else None
    hwm = [sample["os_rss_hwm_bytes"] for sample in samples if number(sample.get("os_rss_hwm_bytes")) is not None]
    if hwm:
        result["whole_process_rss_bytes"] = max(hwm + [result["whole_process_rss_bytes"] or 0])
    by_phase: dict[str, list[dict]] = {}
    ordered = sorted(phases, key=lambda event: event["unix_ms"])
    index = -1
    boundaries = [event["unix_ms"] for event in ordered]
    for sample in sorted(samples, key=lambda item: item["unix_ms"]):
        while index + 1 < len(ordered) and ordered[index + 1]["unix_ms"] <= sample["unix_ms"]:
            index += 1
        if crosses_phase_boundary(sample, boundaries):
            continue
        group = phase_group(ordered[index]["phase"]) if index >= 0 else "startup"
        by_phase.setdefault(group, []).append(sample)
    for group in {phase_group(event["phase"]) for event in ordered} | {"startup"}:
        for metric in ("rss_bytes", "gpu_device_bytes"):
            values = [sample[metric] for sample in by_phase.get(group, []) if number(sample.get(metric)) is not None]
            result[group + "_" + metric] = max(values) if values else None
    return result


def gpu_allocator_peaks(phases: list[dict]) -> dict[str, float]:
    """Interval high-water marks are independent of device-wide sampled VRAM.

    Single-device fresh processes are required: adding independent device peaks would
    overstate a simultaneous process peak. Unknown scopes/missing counters stay absent.
    """
    result = {}
    for event in phases:
        interval = normalize(event.get("gpu_memory_interval", {}))
        if not isinstance(interval, dict) or interval.get("coverage") not in ("owning_backend_buffers_and_registered_driver_pools", "torch_cuda_caching_allocator"):
            continue
        devices = interval.get("devices", {})
        if not isinstance(devices, dict) or len(devices) != 1:
            continue
        device = next(iter(devices.values()))
        if not isinstance(device, dict):
            continue
        group = phase_group(interval.get("previous_phase", "startup"))
        for kind in ("allocated", "reserved"):
            value = number(device.get("peak_" + kind + "_bytes"))
            if value is not None and value >= 0:
                for label in (group, "whole_process"):
                    key = f"{label}_gpu_allocator_peak_{kind}_bytes"
                    result[key] = max(result.get(key, 0), value)
    return result


def nvidia_query(gpu_id: str, fields: str) -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", "--id=" + gpu_id, "--query-gpu=" + fields,
                                        "--format=csv,noheader,nounits"], text=True, timeout=3,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def nvidia_compute_processes(gpu_id: str) -> set[int] | None:
    """PIDs with a compute context on the GPU, or None when the driver cannot list them."""
    try:
        text = subprocess.check_output(["nvidia-smi", "--id=" + gpu_id, "--query-compute-apps=pid",
                                        "--format=csv,noheader,nounits"], text=True, timeout=3,
                                       stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    pids = set()
    for line in text.splitlines():
        with contextlib.suppress(ValueError):
            pids.add(int(line.strip()))
    return pids


def contending_processes(excluded: set[int] | None = None) -> list[dict]:
    """Record active CI workers/test hosts without stopping them or collecting their arguments."""
    excluded = (excluded or set()) | {os.getpid()}
    found = []
    for process in psutil.process_iter(["pid", "name", "create_time"]):
        if process.pid in excluded:
            continue
        try:
            name = (process.info["name"] or "").lower().removesuffix(".exe")
            active = name in ("runner.worker", "testhost", "vstest.console") or name.startswith("testhost.")
            if name == "dotnet":
                active = any(Path(arg).name.lower() in ("testhost.dll", "vstest.console.dll")
                             for arg in process.cmdline())
            if active:
                found.append({"pid": process.pid, "name": process.info["name"],
                              "create_time": process.info["create_time"]})
        except psutil.Error:
            continue
    return found


class Sampler:
    """CPU RSS every 10 ms, separate device-wide NVIDIA memory sampling every 100 ms.

    Device samples are explicitly sampled peaks, not exact allocator high-water marks.
    The same sampler is used on both processes; profiler allocator counters are separate.
    """
    def __init__(self, pid: int, gpu_id: str | None):
        self.process = psutil.Process(pid)
        self.gpu_id = gpu_id
        self.stop = threading.Event()
        self.samples: list[dict] = []
        self.contention: list[dict] = []
        self.lock = threading.Lock()
        self.threads = [threading.Thread(target=self.cpu, daemon=True),
                        threading.Thread(target=self.audit_contention, daemon=True)]
        if gpu_id is not None:
            self.threads.append(threading.Thread(target=self.gpu, daemon=True))

    def cpu(self) -> None:
        while not self.stop.is_set():
            try:
                info = self.process.memory_info()
                row = {"unix_ms": time.time_ns() // 1_000_000, "rss_bytes": info.rss,
                       "os_rss_hwm_bytes": getattr(info, "peak_wset", None)}
                with self.lock:
                    self.samples.append(row)
            except psutil.Error:
                break
            self.stop.wait(.01)

    def gpu(self) -> None:
        while not self.stop.is_set():
            # nvidia-smi can take longer than a short phase. Keep both ends of the query so a reading taken in
            # one phase is not charged to the phase in which the query happened to return.
            started = time.time_ns() // 1_000_000
            value = nvidia_query(self.gpu_id, "memory.used")
            if value is not None:
                with contextlib.suppress(ValueError):
                    with self.lock:
                        self.samples.append({"unix_ms": time.time_ns() // 1_000_000, "query_start_unix_ms": started,
                                             "gpu_device_bytes": float(value) * 1048576})
            self.audit_gpu_exclusivity()
            self.stop.wait(.1)

    def audit_gpu_exclusivity(self) -> None:
        """memory.used is device-wide: another CUDA process on this GPU would be charged to the measured side."""
        listing = nvidia_compute_processes(self.gpu_id)
        if listing is None:
            return
        own = {self.process.pid}
        with contextlib.suppress(psutil.Error):
            own.update(child.pid for child in self.process.children(recursive=True))
        foreign = sorted(pid for pid in listing if pid not in own)
        if foreign:
            self.contention.append({"unix_ms": time.time_ns() // 1_000_000, "kind": "foreign_cuda_process",
                                    "processes": [{"pid": pid} for pid in foreign]})

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def audit_contention(self) -> None:
        while not self.stop.is_set():
            excluded = {self.process.pid}
            with contextlib.suppress(psutil.Error):
                excluded.update(child.pid for child in self.process.children(recursive=True))
            blockers = contending_processes(excluded)
            if blockers:
                self.contention.append({"unix_ms": time.time_ns() // 1_000_000, "processes": blockers})
            self.stop.wait(.25)

    def finish(self) -> list[dict]:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=4)
        return self.samples


def reference_recipe_gate(run: dict) -> dict:
    """Require the actual hashed reference manifest to name the scored Torch recipe."""
    if run.get("model") == "lstm":
        return lstm_workload_gate(run)
    hashes = run.get("artifacts", {})
    try:
        manifest_path = Path(run.get("artifact_paths", {}).get("hifigan_reference_manifest", ""))
        payload = manifest_path.read_bytes()
        if not hashes.get("hifigan_reference_manifest") or hashlib.sha256(payload).hexdigest() != hashes["hifigan_reference_manifest"]:
            return {"status": "unverified", "reason": "Missing or changed reference manifest for recipe validation."}
        manifest = json.loads(payload)
        if not isinstance(manifest, dict) or not hashes.get("pytorch_benchmark") or manifest.get("benchmark_source_sha256") != hashes["pytorch_benchmark"]:
            return {"status": "unverified", "reason": "Reference manifest benchmark recipe does not match the scored PyTorch benchmark hash."}
        exporter_hash = manifest.get("exporter_source_sha256")
        if exporter_hash is not None and (not hashes.get("reference_exporter") or exporter_hash != hashes["reference_exporter"]):
            return {"status": "unverified", "reason": "Reference manifest exporter hash does not match the per-reference exporter artifact."}
    except (OSError, ValueError, TypeError):
        return {"status": "unverified", "reason": "Missing or invalid reference manifest for recipe validation."}
    if run.get("side") == "pytorch" and run.get("mode") == "score":
        return reference_policy_gate(run)
    return {"status": "pass"}


def reference_policy_gate(run: dict) -> dict:
    """Bind actual scored Torch precision to the same reference used for numerical checks."""
    try:
        if run.get("side") != "pytorch" or run.get("mode") != "score" or run.get("return_code") != 0:
            raise ValueError("Formal Torch policy evidence requires a successful scored process.")
        manifest_path = Path(run["artifact_paths"]["hifigan_reference_manifest"])
        manifest = read_json(manifest_path)
        if streaming_sha(manifest_path) != run["artifacts"]["hifigan_reference_manifest"]:
            raise ValueError("Reference manifest changed.")
        if manifest.get("model") != run["model"] or manifest.get("device") != run["device"]:
            raise ValueError("Reference precision policy belongs to another model/device.")
        policy = read_json(raw_file(run, "torch_policy"))
        if not isinstance(policy, dict):
            raise ValueError("Torch policy receipt is not an object.")
        if (policy.get("schema") != "torch-policy-native-receipt-v1" or policy.get("mode") != "strict-reference"
                or policy.get("pid") != run["pid"] or isinstance(policy.get("pid"), bool)
                or policy.get("atexit_complete") is not True or policy.get("observation_error")
                or type(policy.get("benchmark_exit_code")) is not int or policy["benchmark_exit_code"] != 0):
            raise ValueError("Torch policy receipt lacks strict mode, real PID, successful benchmark or complete shutdown observation.")
        for field, artifact in (("benchmark_sha256", "pytorch_benchmark"), ("wrapper_sha256", "torch_policy_wrapper"),
                                ("reference_manifest_sha256", "hifigan_reference_manifest")):
            if not run["artifacts"].get(artifact) or policy.get(field) != run["artifacts"][artifact]:
                raise ValueError("Torch policy source/reference binding differs: " + field)
        resolved = dict(manifest)
        attestation = None
        if policy.get("reference_environment") is not None:
            artifact = "reference_environment"
            path = Path(run["artifact_paths"][artifact])
            actual_hash = streaming_sha(path)
            if (actual_hash != run["artifacts"][artifact] or policy["reference_environment"].get("sha256") != actual_hash
                    or Path(policy["reference_environment"].get("path", "")).resolve() != path.resolve()):
                raise ValueError("Reference startup policy artifact differs.")
            attestation = read_json(path)
            if attestation.get("reference_manifest_sha256") != run["artifacts"]["hifigan_reference_manifest"]:
                raise ValueError("Reference startup policy belongs to another manifest.")
            evidence = attestation.get("evidence", [])
            if not evidence or any(streaming_sha(item["path"]) != item.get("sha256") for item in evidence):
                raise ValueError("Reference startup policy evidence changed or is missing.")
            for key, value in attestation.get("policy", {}).items():
                if key != "deterministic_warn_only" or not isinstance(value, bool) or (key in resolved and resolved[key] != value):
                    raise ValueError("Legacy attestation changes a recorded policy field or has an unsupported field.")
                resolved[key] = value
        if not isinstance(resolved.get("deterministic_warn_only"), bool):
            raise ValueError("Legacy reference warn-only choice lacks a bound source attestation.")
        if run["device"] == "cuda" and (attestation is None or "CUBLAS_WORKSPACE_CONFIG" not in attestation.get("environment", {})):
            raise ValueError("CUDA reference workspace environment is not explicitly bound.")
        observations = {}
        for name in ("applied_flags_before_benchmark", "flags_after_benchmark", "flags_at_exit"):
            snapshot = policy.get(name, {})
            if not isinstance(snapshot, dict):
                raise ValueError("Torch policy observation is missing or malformed: " + name)
            observations[name] = compare_reference(snapshot, resolved)
            if (type(snapshot.get("cuda_initialized_before_observation")) is not bool
                    or type(snapshot.get("cuda_initialized_after_observation")) is not bool
                    or snapshot["cuda_initialized_before_observation"] != snapshot["cuda_initialized_after_observation"]):
                raise ValueError("Observation changed CUDA initialization state.")
        if any(row["status"] != "match" for row in observations.values()):
            return {"status": "fail", "reason": "Actual scored Torch policy differs from the numerical reference.", "observations": observations}
        expected_env = policy.get("effective_environment")
        if not isinstance(expected_env, dict) or any(policy.get(name) != expected_env for name in ("environment_after_benchmark", "environment_at_exit")):
            raise ValueError("Precision override environment changed during the benchmark.")
        if attestation is not None and any(expected_env.get(key) != value for key, value in attestation.get("environment", {}).items()):
            raise ValueError("Effective precision environment differs from the reference startup policy.")
        native = policy.get("native")
        if not isinstance(native, dict) or native.get("status") != "observed_loaded_images":
            # A non-Windows producer states that formal scoring is Windows-only; pass that reason through.
            reason = native.get("reason") if isinstance(native, dict) else None
            raise ValueError(reason if isinstance(reason, str) and reason else "Loaded native-image observation is unavailable.")
        native_files = policy["native"].get("files")
        if not isinstance(native_files, list) or not native_files or any(
                not isinstance(item, dict) or not isinstance(item.get("path"), str) or not re.fullmatch("[a-f0-9]{64}", item.get("sha256", ""))
                or type(item.get("bytes")) is not int or item["bytes"] <= 0 for item in native_files):
            raise ValueError("Native image paths/binary hashes are missing or malformed.")
        return {"status": "pass", "observations": observations, "scope": "Explicit reference precision policy; native image presence is not vendor execution proof."}
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        return {"status": "unverified", "reason": str(error)}


def provenance(config: dict) -> dict:
    repos = {}
    for name, path in config["repositories"].items():
        sha = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain"], text=True).strip()
        if dirty:
            raise ValueError(f"Scoring source must be committed: {name} has working-tree changes.")
        repos[name] = sha
    hashes = {name: streaming_sha(path) for name, path in config["artifacts"].items()}
    receipt = read_json(Path(config["build_receipt"]))
    if receipt.get("repositories") != repos or receipt.get("artifacts") != hashes:
        raise ValueError("Build receipt does not bind these exact source commits and binary/script hashes.")
    artifact_paths = {name: str(Path(path).resolve()) for name, path in config["artifacts"].items()}
    if "hifigan-v1" in config.get("models", []):
        recipe = reference_recipe_gate({"artifacts": hashes, "artifact_paths": artifact_paths})
        if recipe["status"] != "pass":
            raise ValueError(recipe["reason"])
    return {"repositories": repos, "artifacts": hashes, "artifact_paths": artifact_paths,
            "machine": platform.uname()._asdict(),
            "logical_cpus": psutil.cpu_count(), "gpu": nvidia_query(config["gpu_id"], "uuid,name,driver_version,memory.total")
            if config.get("gpu_id") else None, "cache_policy": config["cache_policy"]}


def cache_environment(run_dir: Path, policy: dict, side: str) -> tuple[dict[str, str], dict]:
    """Own every redirected cache directory; never clear or mutate the user's shared caches.

    An independent hash-checked warm seed is copied for each fresh process. Redirecting
    known caches does not prove an inventory of every framework/vendor cache is complete.
    That coverage remains explicit and cannot be established by a `verified` boolean.
    """
    root = run_dir / "cache"
    root.mkdir()
    regime = policy.get("regime")
    if regime not in ("cold", "warm"):
        raise ValueError("cache_policy.regime must be cold or warm.")
    known = {"CUDA_CACHE_PATH": "cuda-driver", "AIDOTNET_KERNEL_CACHE_DIR": "aidotnet-kernels",
             "AIDOTNET_DIRECT_PTX_CACHE_PATH": "aidotnet-kernels/CUDA/DirectPtx"}
    if side == "pytorch":
        known.update(TORCHINDUCTOR_CACHE_DIR="torchinductor", TRITON_CACHE_DIR="triton")
    else:
        known.update(AIDOTNET_AUTOTUNE_CACHE_PATH="aidotnet-autotune",
                     AIDOTNET_CODEGEN_AUTOTUNE_CACHE="aidotnet-codegen/autotune.tsv")
    seed_receipt = None
    if regime == "warm":
        seed = Path(policy["warm_seeds"][side]["path"]).resolve()
        expected = policy["warm_seeds"][side]["sha256"]
        actual = {path.relative_to(seed).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(seed.rglob("*")) if path.is_file()}
        if not actual or actual != expected:
            raise ValueError("Warm cache seed is empty or differs from its exact file hash receipt.")
        shutil.copytree(seed, root, dirs_exist_ok=True)
        seed_receipt = actual
    env = {key: str(root / directory) for key, directory in known.items()}
    for key, directory in env.items():
        target = Path(directory)
        (target.parent if key == "AIDOTNET_CODEGEN_AUTOTUNE_CACHE" else target).mkdir(parents=True, exist_ok=True)
    initial_files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(root.rglob("*")) if path.is_file()}
    return env, {"regime": regime, "redirected_environment": env, "warm_seed": seed_receipt,
                 "initial_files": initial_files, "coverage": "known_environment_redirects_only",
                 "unverified": "Framework/vendor caches not honoring these environment variables require source and runtime verification."}


def profiled_dotnet_identity(process, phase_file: Path, model: str, timeout: float = 60) -> dict:
    """Identify the managed benchmark child when Nsight Systems owns the launcher PID."""
    try:
        launcher_birth = psutil.Process(process.pid).create_time()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if phase_file.is_file():
                with phase_file.open("rb") as stream:
                    line = stream.readline()
                if line.endswith(b"\n"):
                    marker = json.loads(line.decode("utf-8-sig"))
                    if (marker.get("phase") != "construction" or marker.get("model") != model
                            or marker.get("diagnostics") is not True or type(marker.get("pid")) is not int
                            or marker["pid"] <= 0):
                        raise ValueError("The first diagnostic marker does not identify the benchmark.")
                    actual = psutil.Process(marker["pid"])
                    birth = actual.create_time()
                    if birth + .001 < launcher_birth or process.pid not in {item.pid for item in actual.parents()}:
                        raise ValueError("The diagnostic marker belongs to a different process.")
                    return {"pid": actual.pid, "process_create_time": birth, "launcher_pid": process.pid,
                            "launcher_create_time": launcher_birth, "actual_process_image": actual.exe()}
            if process.poll() is not None:
                raise ValueError("Nsight exited before the benchmark emitted its construction marker.")
            time.sleep(.02)
        raise ValueError("Timed out waiting for the profiled benchmark construction marker.")
    except (OSError, ValueError, psutil.Error):
        stop_launch(process)
        raise


def launch(config: dict, side: str, model: str, device: str, pair: int, mode: str, output_dir: Path) -> dict:
    run_dir = output_dir / f"{model}-{device}-{mode}-{pair}-{side}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = config["commands"][side]
    protocol = config["protocol" if mode == "score" else "diagnostic_protocol"]
    bindings = dict(protocol, model=model, device=device, output=str(run_dir / "report.json"), cache=str(run_dir / "cache"), run_directory=str(run_dir))
    bindings["inference_batches"] = ",".join(map(str, protocol["inference_batch_sizes"]))
    argv = [str(arg).format_map(bindings) for arg in command["argv"]]
    if model == "lstm" and "lstm_fixture_manifest" in config["artifacts"]:
        if "--lstm-fixture" not in argv:
            argv.extend(["--lstm-fixture", str(Path(config["artifacts"]["lstm_fixture_manifest"]).resolve().parent)])
        if "--lstm-workload-receipt" not in argv:
            argv.extend(["--lstm-workload-receipt", str(run_dir / "lstm-workload.json")])
    original_argv = argv.copy()
    bootstrap_receipt = None
    if side == "pytorch":
        bootstrap_script = config["artifacts"].get("python_process_bootstrap")
        if not bootstrap_script:
            raise ValueError("PyTorch needs a hash-bound real-process bootstrap; a Windows venv launcher PID is not the framework PID.")
        bootstrap_receipt = run_dir / "python-process.json"
        argv = bootstrap_argv(argv, bootstrap_script, bootstrap_receipt, cwd=command["cwd"])
    nsys = config.get("artifacts", {}).get("nsys") if device == "cuda" and mode == "diagnostic" else None
    profiled_argv = argv.copy() if nsys else None
    if nsys:
        argv = [str(Path(nsys).resolve()), "profile", "--trace=cuda,nvtx", "--cuda-memory-usage=true",
                "--cuda-trace-all-apis=true", "--sample=none", "--cpuctxsw=none", "--export=sqlite",
                "--output", str(run_dir / "native-memory"), *profiled_argv]
    env = os.environ.copy()
    env.update({key: str(value).format_map(bindings) for key, value in command.get("env", {}).items()})
    cache_env, cache_state = cache_environment(run_dir, config["cache_policy"], side)
    env.update(cache_env)
    env.update(AIDOTNET_BENCH_PHASE_FILE=str(run_dir / "phases.jsonl"), AIDOTNET_BENCH_EXTERNAL_MONITOR="1",
               AIDOTNET_BENCH_DEVICE=device,
               AIDOTNET_BENCH_DIAGNOSTICS="1" if mode == "diagnostic" else "0",
               AIDOTNET_BUFFER_ALLOC_DIAG="1" if mode == "diagnostic" else "0",
               AIDOTNET_GPU_MEMTRACK="0", AIDOTNET_GPU_MEMTRACK_PEAKDUMP="0")
    if mode == "diagnostic" and side == "pytorch":
        env["AIDOTNET_BENCH_CPU_ALLOCATION_TRACE"] = str(run_dir / "cpu-allocations.json")
    else:
        env.pop("AIDOTNET_BENCH_CPU_ALLOCATION_TRACE", None)
    for key in ("PROFILE_RANGE", "PROFILE_INFERENCE_BATCH", "BENCH_MEMSTEP", "AIDOTNET_FUSED_DIAG"):
        env.pop(key, None)
    # Allocation callbacks belong only to diagnostic processes, never to scored timing.
    profiler_keys = ("CORECLR_ENABLE_PROFILING", "CORECLR_PROFILER", "CORECLR_PROFILER_PATH", "CORECLR_PROFILER_PATH_64",
                     "DOTNET_ENABLE_PROFILING", "DOTNET_PROFILER", "DOTNET_PROFILER_PATH", "DOTNET_PROFILER_PATH_64",
                     "AIDOTNET_ARRAY_PROFILER_OUTPUT")
    for key in profiler_keys:
        env.pop(key, None)
    if mode == "diagnostic" and side == "aidotnet" and "clr_array_profiler" in config["artifacts"]:
        env.update(CORECLR_ENABLE_PROFILING="1", CORECLR_PROFILER="{5B735E8D-5547-4DB5-B407-E9AF3E2A9D15}",
                   CORECLR_PROFILER_PATH=str(Path(config["artifacts"]["clr_array_profiler"]).resolve()),
                   CORECLR_PROFILER_PATH_64=str(Path(config["artifacts"]["clr_array_profiler"]).resolve()),
                   AIDOTNET_ARRAY_PROFILER_OUTPUT=str(run_dir / "array-allocations.json"))
    gpu_mapping = None
    if device == "cuda":
        gpu_id = str(config.get("gpu_id", "")).strip()
        identity = nvidia_query(gpu_id, "index,uuid,pci.bus_id") if gpu_id else None
        fields = [field.strip() for field in identity.split(",")] if identity and "\n" not in identity else []
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].startswith("GPU-") or not fields[2]:
            raise ValueError("CUDA scoring requires gpu_id to resolve to one physical nvidia-smi GPU.")
        gpu_mapping = dict(requested_id=gpu_id, physical_index=fields[0], uuid=fields[1], pci_bus_id=fields[2],
                           cuda_visible_devices=fields[1], logical_device=0)
        env["CUDA_VISIBLE_DEVICES"] = fields[1]
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["AIDOTNET_DIRECTGPU_BACKENDS"] = "cuda"
    metadata = provenance(config)
    receipt_path = config.get("correctness_receipts", {}).get(model, {}).get(device)
    if receipt_path:
        receipt_file = Path(receipt_path).resolve()
        metadata["correctness_receipt"] = {"path": str(receipt_file),
            "sha256": hashlib.sha256(receipt_file.read_bytes()).hexdigest() if receipt_file.is_file() else None}
    metadata.update(side=side, model=model, device=device, pair=pair, mode=mode, argv=argv, original_argv=original_argv,
                    profiled_argv=profiled_argv, protocol=protocol,
                    cache_state=cache_state, gpu_mapping=gpu_mapping,
                    cache_sensitive_environment={key: env.get(key) for key in CACHE_SENSITIVE_ENVIRONMENT},
                    # Only keys present in the final environment: an override that launch filtering removed (a
                    # disabled diagnostic such as PROFILE_RANGE) stays out of the receipt instead of raising.
                    environment_overrides={key: env[key] for key in set(command.get("env", {})) | set(cache_env) |
                        (set(profiler_keys) & set(env)) |
                        ({"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "AIDOTNET_DIRECTGPU_BACKENDS"} & set(env)) |
                        {"AIDOTNET_BENCH_DIAGNOSTICS", "AIDOTNET_BUFFER_ALLOC_DIAG", "AIDOTNET_GPU_MEMTRACK",
                         "AIDOTNET_GPU_MEMTRACK_PEAKDUMP"} if key in env})
    blockers = contending_processes()
    if mode == "score" and blockers:
        metadata.update(return_code=-2, memory={}, phase_events=[], contention=[{
            "unix_ms": time.time_ns() // 1_000_000, "processes": blockers}],
            invalid_reason="An external CI worker/testhost is active; scoring was not launched.")
        write_json(run_dir / "measurement.json", metadata)
        return {"path": str(run_dir), **metadata}
    with (run_dir / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=command["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            if bootstrap_receipt:
                identity = resolve_python_process(process, bootstrap_receipt, expected_target=resolve_script(original_argv, command["cwd"]),
                                                  expected_bootstrap=Path(bootstrap_script))
            elif nsys:
                identity = profiled_dotnet_identity(process, run_dir / "phases.jsonl", model)
            else:
                identity = {"pid": process.pid, "process_create_time": psutil.Process(process.pid).create_time(),
                            "launcher_pid": process.pid}
            metadata.update(identity)
            sampler = Sampler(identity["pid"], gpu_mapping["uuid"] if gpu_mapping else None)
        except psutil.NoSuchProcess:
            metadata.update(return_code=process.wait(), memory={}, phase_events=[], contention=[],
                            invalid_reason="Child exited before memory sampling could start.")
            write_json(run_dir / "measurement.json", metadata)
            return {"path": str(run_dir), **metadata}
        except (ValueError, OSError, psutil.Error) as error:
            metadata.update(return_code=-3, launcher_pid=process.pid, memory={}, phase_events=[],
                            invalid_reason="Benchmark process identity could not be verified: " + str(error))
            write_json(run_dir / "measurement.json", metadata)
            return {"path": str(run_dir), **metadata}
        def stop_tree() -> None:
            # Only children of this launch are terminated; compiler workers must not leak into the next pair.
            try:
                with contextlib.suppress(psutil.Error):
                    for child in psutil.Process(process.pid).children(recursive=True):
                        with contextlib.suppress(psutil.Error):
                            child.kill()
            finally:
                with contextlib.suppress(OSError):
                    process.kill()
                process.wait()

        sampler.start()
        try:
            try:
                return_code = process.wait(timeout=config.get("timeout_seconds", 3600))
            except subprocess.TimeoutExpired:
                stop_tree()
                return_code = -1
            except BaseException:
                # Ctrl+C or any other interruption: a surviving child would keep the CPU or GPU busy during the
                # next run. Stop and reap it, then let the interruption propagate.
                stop_tree()
                raise
        finally:
            samples = sampler.finish()
    phases = phase_records(run_dir / "phases.jsonl")
    raw_names = {"phases": "phases.jsonl", "cpu_allocations": "cpu-allocations.json", "clr_arrays": "array-allocations.json",
                 "torch_policy": "torch-policy.json", "cuda_nsys_sqlite": "native-memory.sqlite",
                 "lstm_workload": "lstm-workload.json"}
    metadata["diagnostic_files"] = {key: {"name": name, "sha256": streaming_sha(run_dir / name)}
                                    for key, name in raw_names.items() if (run_dir / name).is_file()}
    metadata.update(return_code=return_code, memory=memory_metrics(samples, phases), phase_events=phases,
                    gpu_allocator_peaks=gpu_allocator_peaks(phases),
                    contention=sampler.contention,
                    peak_method={"cpu": "10ms sampled RSS; Windows OS whole-process HWM when available",
                                 "gpu": "100ms device-wide nvidia-smi sampled peak; requires exclusive GPU"})
    write_json(run_dir / "measurement.json", metadata)
    write_json(run_dir / "memory-samples.json", samples)
    return {"path": str(run_dir), **metadata}


def hifigan_required_checks() -> tuple[set[str], set[str], set[str]]:
    learned, common = set(), set()
    for family, layers in (("generator", 78), ("discriminator", 54)):
        for layer in range(layers):
            prefix = f"{family}/{layer:03d}/"
            roles = {"direction", "bias"}
            spectral = family == "discriminator" and 30 <= layer <= 37
            if not spectral:
                roles.add("gain")
            learned.update(prefix + role for role in roles)
            common.update(prefix + role for role in roles | ({"u", "v"} if spectral else set()))
    required = {"isolated-gradient/" + key for key in learned}
    required.update("train/step" + str(step) + "/" + key for step in (1, 2, 3) for key in common)
    required.update("isolated-gradient-summary/" + family for family in ("generator", "discriminator"))
    required.update("inference/output/" + str(batch) for batch in (1, 8, 16))
    required.update("isolated/" + name for name in ("discriminator_loss", "adversarial", "feature_matching", "mel_l1",
                                                  "components_total", "production_generator_loss"))
    required.update("train/generator_loss/" + str(step) for step in (0, 1, 2))
    required.add("generator/output0")
    return required, learned, common


def hifigan_comparison_policy(device: str) -> dict:
    relative, cosine = {"cpu": (1e-3, .999), "cuda": (3e-3, .995)}[device]
    return {"schema": "hifigan-v1-calibrated-learned-state-v1",
            "learned_state_checks": 1164, "spectral_state_checks": 48, "unchanged_checks": 451,
            "learned_state": {"finite_required": True, "relative_l2_max": relative,
                              "cosine_min": cosine, "max_absolute_error_gate": False},
            "other_criteria": "unchanged_original"}


def hifigan_comparison_kinds() -> dict[str, str]:
    required, learned, common = hifigan_required_checks()
    kinds = dict.fromkeys(required, "standard")
    for family in ("generator", "discriminator"):
        kinds["isolated-gradient-summary/" + family] = "gradient_summary"
    for step in (1, 2, 3):
        for key in common:
            kinds[f"train/step{step}/" + key] = "learned_state" if key in learned else "spectral_state"
    return kinds


def correctness_gate(run: dict) -> dict:
    """Bind a successful full-V1 equivalence receipt to the exact scored binaries/reference."""
    if run.get("model") == "lstm":
        return lstm_correctness_gate(run)
    binding = run.get("correctness_receipt") or {}
    if run.get("model") != "hifigan-v1":
        return {"status": "unverified", "reason": "This family has no validated numerical-equivalence receipt contract."}
    path = Path(binding.get("path", ""))
    if not path.is_file() or binding.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        return {"status": "unverified", "reason": "Missing or changed numerical-equivalence receipt."}
    try:
        receipt = read_json(path)
    except ValueError:
        receipt = None
    if not isinstance(receipt, dict):
        # Hash-bound but malformed evidence is unverified; it must not abort the campaign before gate.json exists.
        return {"status": "unverified", "reason": "The numerical-equivalence receipt is not a JSON object."}
    if receipt.get("schema") != "hifigan-v1-equivalence-receipt-v1" or receipt.get("model") != run["model"] or receipt.get("device") != run["device"]:
        return {"status": "unverified", "reason": "Numerical receipt schema/model/device mismatch."}
    budgets = {"cpu": {"absolute": 2e-5, "relative": 1e-3, "gradient_relative": .01, "gradient_cosine": .999},
               "cuda": {"absolute": 2e-4, "relative": 3e-3, "gradient_relative": .03, "gradient_cosine": .995}}
    if run["device"] not in budgets or receipt.get("tolerance") != budgets.get(run["device"]):
        return {"status": "unverified", "reason": "Numerical receipt does not use the fixed reviewed error budgets."}
    if receipt.get("comparison_policy") != hifigan_comparison_policy(run["device"]):
        return {"status": "unverified", "reason": "Numerical receipt is missing or changes the authorized learned-state comparison policy."}
    hashes = run.get("artifacts", {})
    assemblies = receipt.get("assemblies_sha256") or {}
    if any(not hashes.get(name) or assemblies.get(name) != hashes[name] for name in ("AiDotNet", "Tensors")):
        return {"status": "unverified", "reason": "Numerical receipt does not bind the scored assembly hashes."}
    if not hashes.get("hifigan_reference_manifest") or receipt.get("reference_manifest_sha256") != hashes["hifigan_reference_manifest"]:
        return {"status": "unverified", "reason": "Numerical receipt reference-manifest hash mismatch."}
    recipe = reference_recipe_gate(run)
    if recipe["status"] != "pass":
        return recipe
    checks = receipt.get("checks") or []
    valid_checks = bool(checks) and all(check.get("pass") is True and check.get("finite") is True
                                       and (number(check.get("elements")) or 0) > 0 for check in checks)
    if receipt.get("pass") is not True or receipt.get("errors") != [] or not valid_checks:
        return {"status": "fail", "reason": "Numerical equivalence has a failing, empty, or nonfinite check."}
    required, learned, common = hifigan_required_checks()
    names = [check.get("name") for check in checks]
    if len(names) != len(required) or set(names) != required:
        return {"status": "unverified", "reason": "Receipt must contain exactly all 1615 required V1 checks, without duplicates or omitted tensors."}
    by_name = {check["name"]: check for check in checks}
    kinds = hifigan_comparison_kinds()
    tolerance = budgets[run["device"]]
    for name, check in by_name.items():
        kind = kinds[name]
        if check.get("comparison_kind") != kind:
            return {"status": "unverified", "reason": "Numerical check comparison kind differs from its authorized tensor scope: " + name}
        fields = ("max_absolute_error", "relative_l2", "cosine", "norm_ratio", "reference_max_absolute_value")
        measured = {field: number(check.get(field)) for field in fields}
        if any(value is None for value in measured.values()) or any(measured[field] < 0 for field in fields if field != "cosine"):
            return {"status": "unverified", "reason": "Numerical check has missing or invalid comparison metrics: " + name}
        error, relative, cosine, ratio, scale = (measured[field] for field in fields)
        if kind == "learned_state":
            accepted = relative <= tolerance["relative"] and cosine >= tolerance["gradient_cosine"]
        elif kind == "gradient_summary":
            accepted = (relative <= tolerance["gradient_relative"] and abs(ratio - 1) <= tolerance["gradient_relative"]
                        and cosine >= tolerance["gradient_cosine"])
        else:
            accepted = (error <= tolerance["absolute"] + tolerance["relative"] * scale
                        and (relative <= tolerance["relative"] or error <= tolerance["absolute"]))
        if not accepted:
            return {"status": "fail", "reason": "Numerical check violates its authorized comparison criteria: " + name}
    if sum(by_name["isolated-gradient/" + key]["elements"] for key in learned) != 84_660_721:
        return {"status": "unverified", "reason": "Receipt does not cover all 84,660,721 learned gradient scalars."}
    for step in (1, 2, 3):
        if sum(by_name[f"train/step{step}/" + key]["elements"] for key in common) != 84_679_257:
            return {"status": "unverified", "reason": "Receipt omits learned weights or active spectral state from a fused update."}
    for batch in (1, 8, 16):
        if by_name[f"inference/output/{batch}"]["elements"] != batch * 65_536:
            return {"status": "unverified", "reason": "Receipt inference is not the full 256-frame V1 configuration."}
    if by_name["generator/output0"]["elements"] != 8192:
        return {"status": "unverified", "reason": "Receipt isolated training crop is not 8192 samples."}
    if receipt.get("fused_steps") != 6 or receipt.get("eager_steps") != 0:
        return {"status": "unverified", "reason": "Receipt did not prove the three actual fused discriminator/generator updates."}
    return {"status": "pass", "receipt_sha256": binding["sha256"], "checks": len(checks)}


def cold_completion_seconds(run: dict) -> float:
    """Include process startup, construction and preparation through the first completed epoch.

    The existing epoch timer remains separate: a constructor can execute a forward
    before that timer starts. Raw phase hashes/PIDs bind this end-to-end measurement
    to the same fresh process instead of inferring startup from a sampler's first tick.
    """
    created = number(run.get("process_create_time"))
    if created is None or created <= 0:
        raise ValueError("Cold completion lacks a finite positive process creation timestamp.")
    pid = run.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("Cold completion lacks the launched process PID.")
    epochs = run.get("protocol", {}).get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Cold completion requires at least one real training epoch.")
    phases = checked_phases(run)
    if any(not isinstance(event.get("unix_ms"), int) or isinstance(event["unix_ms"], bool)
           or event["unix_ms"] <= 0 for event in phases):
        raise ValueError("Cold completion phase timestamps are missing or invalid.")
    launch_ms = created * 1000
    if launch_ms > phases[0]["unix_ms"]:
        raise ValueError("Cold completion phases predate process creation.")
    endpoint = next(event for event in phases if event["phase"] == "training-epoch-0-complete")
    elapsed = (endpoint["unix_ms"] - launch_ms) / 1000
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Cold completion interval is not positive and finite.")
    return elapsed


def metrics(run: dict) -> tuple[dict[str, float | None], list[str]]:
    path = Path(run["path"])
    if run.get("invalid_reason"):
        return {}, [run["invalid_reason"]]
    if run["return_code"] != 0 or not (path / "report.json").exists():
        return {}, ["process failed or report missing"]
    report = normalize(read_json(path / "report.json"))
    results = report.get("results", [])
    model = next((row for row in results if row.get("model") == run["model"]), None)
    if model is None:
        return {}, ["requested model missing"]
    errors = []
    if run.get("mode") == "score" and run.get("contention"):
        kinds = {record.get("kind") for record in run["contention"] if isinstance(record, dict)}
        if kinds - {"foreign_cuda_process"} or not kinds:
            errors.append("external CI worker/testhost contention observed during scoring")
        if "foreign_cuda_process" in kinds:
            errors.append("another CUDA process used the measured GPU during scoring; device-wide VRAM samples "
                          "would include its memory")
    if run["model"] == "lstm":
        workload = lstm_workload_gate(run)
        if workload["status"] != "pass":
            errors.append("LSTM workload: " + workload["reason"])
    if model.get("device") != run["device"]:
        errors.append("device mismatch")
    for name in ("epochs", "train_batches", "batch_size", "inference_iterations", "warmup_iterations", "warmup_min_ms", "seed", "inference_batch_sizes"):
        if report.get("protocol", {}).get(name) != run["protocol"].get(name):
            errors.append("protocol mismatch: " + name)
    if run["side"] == "pytorch" and report.get("protocol", {}).get("compile") is not True:
        errors.append("reference is not torch.compile")
    threads = report.get("torch_num_threads") if run["side"] == "pytorch" else report.get("environment", {}).get("cpu_threads")
    if threads != run["protocol"].get("threads"):
        errors.append("CPU thread mismatch")
    training = model.get("training") or {}
    epochs = training.get("epoch_seconds", [])
    values = {"first_epoch_seconds": number(epochs[0]) if epochs else None,
              "steady_epoch_seconds": number(training.get("steady_state_epoch_seconds_median")),
              "launch_to_first_epoch_end_seconds": None}
    try:
        values["launch_to_first_epoch_end_seconds"] = cold_completion_seconds(run)
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        errors.append("cold completion: " + str(error))
    if number(training.get("first_loss")) is None or number(training.get("last_loss")) is None:
        errors.append("missing/nonfinite training loss evidence")
    for batch in run["protocol"]["inference_batch_sizes"]:
        row = next((row for row in model.get("inference", []) if row["batch_size"] == batch), {})
        for kind in ("median", "p95"):
            values[f"inference_b{batch}_{kind}_ms"] = number(row.get("steady_state_latency_ms_" + kind))
    memory = run["memory"]
    for group in ("whole_process", "construction", "training", *("inference_b" + str(batch) for batch in run["protocol"]["inference_batch_sizes"])):
        for kind in (["rss_bytes", "gpu_device_bytes"] if run["device"] == "cuda" else ["rss_bytes"]):
            values[group + "_" + kind] = memory.get(group + "_" + kind)
        if run["device"] == "cuda":
            for kind in ("allocated", "reserved"):
                key = f"{group}_gpu_allocator_peak_{kind}_bytes"
                values[key] = run.get("gpu_allocator_peaks", {}).get(key)
    return values, errors


def evaluate(runs: list[dict], maximum: float, required: list[tuple[str, str]] | None = None) -> dict:
    if number(maximum) is None or not 0 < maximum <= .9:
        raise ValueError("Every nonzero baseline requires at least a 10% reduction.")
    groups = sorted({(run["model"], run["device"]) for run in runs} | set(required or []))
    output = {}
    for model, device in groups:
        scored = [run for run in runs if (run["model"], run["device"], run["mode"]) == (model, device, "score")]
        pairs = sorted({run["pair"] for run in scored})
        sides = {side: [next((run for run in scored if run["pair"] == pair and run["side"] == side), None)
                        for pair in pairs] for side in ("aidotnet", "pytorch")}
        rows = {side: [metrics(run) if run else ({}, ["missing pair member"]) for run in sides[side]] for side in sides}
        names = sorted({"launch_to_first_epoch_end_seconds"} | set().union(*(row[0].keys() for side in rows.values() for row in side)))
        outcomes = {name: paired_gate([row[0].get(name) for row in rows["aidotnet"]],
                                      [row[0].get(name) for row in rows["pytorch"]], maximum) for name in names}
        cache_checks = [cache_coverage(run) for run in scored]
        identities = [(run.get("pid"), run.get("process_create_time")) for run in scored]
        outcomes["first_epoch_cache_coverage"] = {
            "status": "pass" if cache_checks and all(row["status"] == "pass" for row in cache_checks)
                      and len(set(identities)) == len(identities) else "unverified",
            "receipts": cache_checks,
            "reason": "Every score needs a fresh process and source-bound persistent-cache inventory.",
        }
        policy_checks = [reference_recipe_gate(run) for run in scored if run["side"] == "pytorch"]
        outcomes["reference_precision_policy"] = {
            "status": "pass" if policy_checks and all(row["status"] == "pass" for row in policy_checks) else
                      "fail" if any(row["status"] == "fail" for row in policy_checks) else "unverified",
            "receipts": policy_checks,
        }
        diagnostics = [run for run in runs if (run["model"], run["device"], run["mode"]) == (model, device, "diagnostic")]
        allocation_rows = {side: [] for side in sides}
        for side in sides:
            for pair, score in zip(pairs, sides[side]):
                matched = [run for run in diagnostics if run["side"] == side and run["pair"] == pair]
                diagnostic = matched[0] if len(matched) == 1 else None
                keys = ("protocol", "repositories", "artifacts", "machine", "gpu", "gpu_mapping", "cache_policy")
                if score is None or diagnostic is None or any(score.get(key) != diagnostic.get(key) for key in keys):
                    allocation_rows[side].append(({}, ["Diagnostic pair/source/protocol/cache/hardware differs from scoring."]))
                elif cache_coverage(diagnostic)["status"] != "pass":
                    allocation_rows[side].append(({}, ["Allocation diagnostic cache coverage is incomplete."]))
                elif model == "lstm" and lstm_workload_gate(diagnostic)["status"] != "pass":
                    allocation_rows[side].append(({}, ["Allocation diagnostic did not run the matched LSTM workload."]))
                else:
                    allocation_rows[side].append(allocation_metrics(diagnostic))
        # Compare computed exact/upper Ai bounds against exact/lower Torch bounds.
        # Native CUDA acquisitions include async driver-pool suballocations, not physical backing events.
        for kind in ("cpu_framework_requested", "cpu_backing", *(("gpu_framework_requested", "gpu_backing") if device == "cuda" else ())):
            for unit in ("bytes", "count"):
                name = f"{kind}_{unit}"
                outcome = paired_gate([row[0].get(name) for row in allocation_rows["aidotnet"]],
                                      [row[0].get(name) for row in allocation_rows["pytorch"]], maximum)
                outcome["coverage_errors"] = {side: [error for _, errors in allocation_rows[side] for error in errors] for side in sides}
                outcome["window"] = "construction_training_epochs_inference_steady_excluding_adaptive_warmup"
                outcomes[name] = outcome
        if device == "cuda":
            windows = {window for run in scored if "protocol" in run
                       for window in fixed_windows(run) if window not in ("construction", "training_epoch_0")}
            for window in sorted(windows):
                for action in ("alloc", "free"):
                    name = f"cuda_native_{window}_{action}_calls"
                    outcomes[name] = warmed_cuda_calls_gate(
                        [row[0].get(name) for row in allocation_rows["aidotnet"]],
                        [row[0].get(name) for row in allocation_rows["pytorch"]], maximum)
        numerical = [correctness_gate(run) for run in scored]
        outcomes["numerical_equivalence"] = {
            "status": "pass" if numerical and all(row["status"] == "pass" for row in numerical) else
                      "fail" if any(row["status"] == "fail" for row in numerical) else "unverified",
            "receipts": numerical,
        }
        failures = [error for side in rows.values() for _, errors in side for error in errors]
        if not scored:
            failures.append("required model/device has no scored runs")
        if len(scored) != len(pairs) * 2:
            failures.append("scored pairs contain missing or duplicate members")
        all_identities = [(run.get("pid"), run.get("process_create_time")) for run in scored + diagnostics]
        if len(set(all_identities)) != len(all_identities):
            failures.append("score/diagnostic process identities were reused")
        signatures = {json.dumps({key: run.get(key) for key in ("repositories", "artifacts", "machine", "gpu", "gpu_mapping")}, sort_keys=True)
                      for run in scored}
        if len(signatures) != 1:
            failures.append("source, binaries, or hardware changed within the campaign")
        parameter_counts = set()
        for run in scored:
            report_path = Path(run["path"]) / "report.json"
            if report_path.exists():
                report = normalize(read_json(report_path))
                parameter_counts.update(row.get("parameters") for row in report.get("results", []) if row.get("model") == model)
        if len(parameter_counts) != 1 or None in parameter_counts:
            failures.append("model parameter counts differ or are missing")
        output[model + "/" + device] = {"status": "pass" if outcomes and not failures and all(row["status"] == "pass" for row in outcomes.values()) else "unfinished",
                                        "protocol_errors": failures, "metrics": outcomes}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    run_index = args.output / "runs.json"
    if not args.evaluate_only:
        if config.get("pairs", 0) < 5 or not 0 < config.get("maximum_ratio", .9) <= .9:
            raise ValueError("At least five pairs and a >=10% gate are required.")
        if config["cache_policy"].get("regime") not in ("cold", "warm"):
            raise ValueError("An explicit cold or warm cache regime is required.")
        runs = []
        for model in config["models"]:
            for device in config["devices"]:
                for mode in ("score", "diagnostic"):
                    for pair in range(config["pairs"]):
                        for side in (("aidotnet", "pytorch") if pair % 2 == 0 else ("pytorch", "aidotnet")):
                            print(f"{mode} pair {pair + 1}: {model}/{device}/{side}", flush=True)
                            runs.append(launch(config, side, model, device, pair, mode, args.output))
                            write_json(run_index, runs)
                            if mode == "score" and runs[-1].get("contention"):
                                required = [(name, target) for name in config["models"] for target in config["devices"]]
                                write_json(args.output / "gate.json", evaluate(runs, config.get("maximum_ratio", .9), required))
                                raise SystemExit("Scoring invalidated by an external CI worker/testhost; see measurement.json.")
    else:
        runs = read_json(run_index)
    required = [(model, device) for model in config["models"] for device in config["devices"]]
    results = evaluate(runs, config.get("maximum_ratio", .9), required)
    write_json(args.output / "gate.json", results)
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if results and all(row["status"] == "pass" for row in results.values()) else 1)


if __name__ == "__main__":
    main()
