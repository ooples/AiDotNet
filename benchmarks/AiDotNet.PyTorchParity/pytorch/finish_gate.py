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
    if all(ratio == 0 for ratio in ratios):
        return {"status": "pass", "median_ratio": 0.0, "upper_95_ratio": 0.0, "pairs": len(ratios)}
    # Tiny positive floor admits genuine zero allocations without log(0).
    logs = [math.log(max(ratio, 1e-300)) for ratio in ratios]
    rng = random.Random(2561)
    draws = sorted(statistics.median(rng.choices(logs, k=len(logs))) for _ in range(resamples))
    upper = math.exp(draws[math.ceil(.95 * len(draws)) - 1])
    point = statistics.median(ratios)
    return {"status": "pass" if point <= maximum and upper <= maximum else "fail",
            "median_ratio": point, "upper_95_ratio": upper, "pairs": len(ratios)}


def phase_group(phase: str) -> str:
    if phase.startswith("construction"):
        return "construction"
    if phase.startswith("training"):
        return "training"
    match = re.match(r"inference-b([0-9]+)", phase)
    return "inference_b" + match[1] if match else phase


def phase_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


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
    for sample in sorted(samples, key=lambda item: item["unix_ms"]):
        while index + 1 < len(ordered) and ordered[index + 1]["unix_ms"] <= sample["unix_ms"]:
            index += 1
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
            value = nvidia_query(self.gpu_id, "memory.used")
            if value is not None:
                with contextlib.suppress(ValueError):
                    with self.lock:
                        self.samples.append({"unix_ms": time.time_ns() // 1_000_000,
                                             "gpu_device_bytes": float(value) * 1048576})
            self.stop.wait(.1)

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


def provenance(config: dict) -> dict:
    repos = {}
    for name, path in config["repositories"].items():
        sha = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain"], text=True).strip()
        if dirty:
            raise ValueError(f"Scoring source must be committed: {name} has working-tree changes.")
        repos[name] = sha
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in config["artifacts"].items()}
    receipt = read_json(Path(config["build_receipt"]))
    if receipt.get("repositories") != repos or receipt.get("artifacts") != hashes:
        raise ValueError("Build receipt does not bind these exact source commits and binary/script hashes.")
    return {"repositories": repos, "artifacts": hashes, "machine": platform.uname()._asdict(),
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
    for directory in env.values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    return env, {"regime": regime, "redirected_environment": env, "warm_seed": seed_receipt,
                 "coverage": "known_environment_redirects_only",
                 "unverified": "Framework/vendor caches not honoring these environment variables require source and runtime verification."}


def launch(config: dict, side: str, model: str, device: str, pair: int, mode: str, output_dir: Path) -> dict:
    run_dir = output_dir / f"{model}-{device}-{mode}-{pair}-{side}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = config["commands"][side]
    protocol = config["protocol" if mode == "score" else "diagnostic_protocol"]
    bindings = dict(protocol, model=model, device=device, output=str(run_dir / "report.json"), cache=str(run_dir / "cache"))
    bindings["inference_batches"] = ",".join(map(str, protocol["inference_batch_sizes"]))
    argv = [str(arg).format_map(bindings) for arg in command["argv"]]
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
    metadata = provenance(config)
    receipt_path = config.get("correctness_receipts", {}).get(model, {}).get(device)
    if receipt_path:
        receipt_file = Path(receipt_path).resolve()
        metadata["correctness_receipt"] = {"path": str(receipt_file),
            "sha256": hashlib.sha256(receipt_file.read_bytes()).hexdigest() if receipt_file.is_file() else None}
    metadata.update(side=side, model=model, device=device, pair=pair, mode=mode, argv=argv, protocol=protocol,
                    cache_state=cache_state,
                    environment_overrides={key: env[key] for key in set(command.get("env", {})) | set(cache_env) |
                        {"AIDOTNET_BENCH_DIAGNOSTICS", "AIDOTNET_BUFFER_ALLOC_DIAG", "AIDOTNET_GPU_MEMTRACK",
                         "AIDOTNET_GPU_MEMTRACK_PEAKDUMP"}})
    blockers = contending_processes()
    if mode == "score" and blockers:
        metadata.update(return_code=-2, memory={}, phase_events=[], contention=[{
            "unix_ms": time.time_ns() // 1_000_000, "processes": blockers}],
            invalid_reason="An external CI worker/testhost is active; scoring was not launched.")
        write_json(run_dir / "measurement.json", metadata)
        return {"path": str(run_dir), **metadata}
    with (run_dir / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd=command["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT)
        sampler = Sampler(process.pid, config.get("gpu_id") if device == "cuda" else None)
        sampler.start()
        try:
            return_code = process.wait(timeout=config.get("timeout_seconds", 3600))
        except subprocess.TimeoutExpired:
            # Only children of this launch are terminated; compiler workers must not leak into the next pair.
            with contextlib.suppress(psutil.Error):
                for child in psutil.Process(process.pid).children(recursive=True):
                    with contextlib.suppress(psutil.Error):
                        child.kill()
            process.kill()
            process.wait()
            return_code = -1
        samples = sampler.finish()
    phases = phase_records(run_dir / "phases.jsonl")
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


def correctness_gate(run: dict) -> dict:
    """Bind a successful full-V1 equivalence receipt to the exact scored binaries/reference."""
    binding = run.get("correctness_receipt") or {}
    if run.get("model") != "hifigan-v1":
        return {"status": "unverified", "reason": "This family has no validated numerical-equivalence receipt contract."}
    path = Path(binding.get("path", ""))
    if not path.is_file() or binding.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        return {"status": "unverified", "reason": "Missing or changed numerical-equivalence receipt."}
    receipt = read_json(path)
    if receipt.get("schema") != "hifigan-v1-equivalence-receipt-v1" or receipt.get("model") != run["model"] or receipt.get("device") != run["device"]:
        return {"status": "unverified", "reason": "Numerical receipt schema/model/device mismatch."}
    budgets = {"cpu": {"absolute": 2e-5, "relative": 1e-3, "gradient_relative": .01, "gradient_cosine": .999},
               "cuda": {"absolute": 2e-4, "relative": 3e-3, "gradient_relative": .03, "gradient_cosine": .995}}
    if receipt.get("tolerance") != budgets.get(run["device"]):
        return {"status": "unverified", "reason": "Numerical receipt does not use the fixed reviewed error budgets."}
    hashes = run.get("artifacts", {})
    assemblies = receipt.get("assemblies_sha256") or {}
    if any(not hashes.get(name) or assemblies.get(name) != hashes[name] for name in ("AiDotNet", "Tensors")):
        return {"status": "unverified", "reason": "Numerical receipt does not bind the scored assembly hashes."}
    if not hashes.get("hifigan_reference_manifest") or receipt.get("reference_manifest_sha256") != hashes["hifigan_reference_manifest"]:
        return {"status": "unverified", "reason": "Numerical receipt reference-manifest hash mismatch."}
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


def metrics(run: dict) -> tuple[dict[str, float | None], list[str]]:
    path = Path(run["path"])
    if run["return_code"] != 0 or not (path / "report.json").exists():
        return {}, ["process failed or report missing"]
    report = normalize(read_json(path / "report.json"))
    results = report.get("results", [])
    model = next((row for row in results if row.get("model") == run["model"]), None)
    if model is None:
        return {}, ["requested model missing"]
    errors = []
    if run.get("mode") == "score" and run.get("contention"):
        errors.append("external CI worker/testhost contention observed during scoring")
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
              "steady_epoch_seconds": number(training.get("steady_state_epoch_seconds_median"))}
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
    groups = sorted({(run["model"], run["device"]) for run in runs} | set(required or []))
    output = {}
    for model, device in groups:
        scored = [run for run in runs if (run["model"], run["device"], run["mode"]) == (model, device, "score")]
        pairs = sorted({run["pair"] for run in scored})
        sides = {side: [next((run for run in scored if run["pair"] == pair and run["side"] == side), None)
                        for pair in pairs] for side in ("aidotnet", "pytorch")}
        rows = {side: [metrics(run) if run else ({}, ["missing pair member"]) for run in sides[side]] for side in sides}
        names = sorted(set().union(*(row[0].keys() for side in rows.values() for row in side)))
        outcomes = {name: paired_gate([row[0].get(name) for row in rows["aidotnet"]],
                                      [row[0].get(name) for row in rows["pytorch"]], maximum) for name in names}
        # Disk-cache redirection alone cannot prove a complete cold/warm regime across vendor libraries.
        outcomes["first_epoch_cache_coverage"] = {"status": "unverified", "reason":
            "Known cache directories are isolated; complete framework/vendor cache coverage is not yet proven."}
        # Requested and actual backing allocations are independent gates. Unsupported counter coverage stays red.
        for kind in ("cpu_framework_requested", "cpu_backing", *( ("gpu_framework_requested", "gpu_backing") if device == "cuda" else ())):
            for unit in ("bytes", "count"):
                outcomes[f"{kind}_{unit}"] = {"status": "unverified", "reason": "Allocator coverage contract not yet verified on both frameworks."}
        numerical = [correctness_gate(run) for run in scored]
        outcomes["numerical_equivalence"] = {
            "status": "pass" if numerical and all(row["status"] == "pass" for row in numerical) else
                      "fail" if any(row["status"] == "fail" for row in numerical) else "unverified",
            "receipts": numerical,
        }
        failures = [error for side in rows.values() for _, errors in side for error in errors]
        if not scored:
            failures.append("required model/device has no scored runs")
        signatures = {json.dumps({key: run.get(key) for key in ("repositories", "artifacts", "machine", "gpu")}, sort_keys=True)
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
