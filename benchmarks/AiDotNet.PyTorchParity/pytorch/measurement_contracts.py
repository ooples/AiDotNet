"""Source-bound allocation bounds and cache coverage for the finish gate.

Coverage is an audited inventory of active storage routes, never inferred from a
nonzero counter. Ai upper bounds may be compared with Torch lower bounds: if
U(A) <= .9 L(T), then A <= .9 T. Counter bytes/count are calculated from raw files;
the contract cannot supply numbers. Backing means successful acquisitions beyond
the framework's own reuse pools, including CUDA async driver-pool suballocations.
It does not mean physical pages acquired by the driver.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_SENSITIVE_ENVIRONMENT = (
    "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE",
    "TORCHINDUCTOR_AOT_AUTOGRAD_REMOTE_CACHE", "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCHINDUCTOR_REDIS_HOST", "TORCHINDUCTOR_REDIS_PORT", "TRITON_OVERRIDE_DIR",
    "TRITON_DUMP_DIR", "TRITON_HOME", "TRITON_CACHE_MANAGER", "TORCH_HOME",
    "CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE",
)

SCOPES = {
    "cpu_framework_requested": "cpu_numeric_storage_and_framework_workspace_requests",
    "gpu_framework_requested": "gpu_numeric_storage_and_framework_workspace_requests",
    "cpu_backing": "cpu_framework_backing_api_acquisitions",
    "gpu_backing": "gpu_framework_backing_api_acquisitions_including_async_driver_pool_suballocations",
}
ROLES = {
    "cpu_framework_requested": {"managed_fresh", "framework_pool_borrows", "framework_owned_vendor_workspaces"},
    "cpu_backing": {"managed_fresh", "framework_owned_vendor_workspaces"},
    "gpu_framework_requested": {"backend_buffer_api", "framework_pool_borrows", "framework_owned_vendor_workspaces"},
    "gpu_backing": {"backend_allocation_api", "framework_owned_vendor_workspaces"},
}
PROVIDERS = {
    "clr_arrays": ("aidotnet", {"cpu_framework_requested", "cpu_backing"}),
    "aidotnet_cpu_requests": ("aidotnet", {"cpu_framework_requested"}),
    "aidotnet_gpu_requests": ("aidotnet", {"gpu_framework_requested"}),
    "aidotnet_compiled_workspace_loans": ("aidotnet", {"gpu_framework_requested"}),
    "aidotnet_cuda_backing_api": ("aidotnet", {"gpu_backing"}),
    "torch_cpu_memory_events": ("pytorch", {"cpu_framework_requested", "cpu_backing"}),
    "torch_cuda_requests": ("pytorch", {"gpu_framework_requested"}),
    "torch_cuda_segments": ("pytorch", {"gpu_backing"}),
}
CACHE_ROOTS = {
    "cpu_autotune": "AIDOTNET_AUTOTUNE_CACHE_PATH",
    "codegen_autotune": "AIDOTNET_CODEGEN_AUTOTUNE_CACHE",
    "cuda_nvrtc": "AIDOTNET_KERNEL_CACHE_DIR",
    "cuda_direct_ptx": "AIDOTNET_DIRECT_PTX_CACHE_PATH",
    "cuda_driver": "CUDA_CACHE_PATH",
    "inductor": "TORCHINDUCTOR_CACHE_DIR",
    "triton": "TRITON_CACHE_DIR",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Allocation counter is missing, negative or not an exact integer.")
    return value


def bound_contract(run: dict) -> dict:
    path = Path(run["artifact_paths"]["measurement_contract"])
    if digest(path) != run["artifacts"]["measurement_contract"]:
        raise ValueError("Measurement contract hash differs from the launch receipt.")
    contract = load(path)
    if contract.get("schema") != "parity-measurement-contract-v1":
        raise ValueError("Unsupported measurement contract schema.")
    binding = contract["binding"]
    artifacts = {key: value for key, value in run["artifacts"].items() if key != "measurement_contract"}
    if binding.get("repositories") != run["repositories"] or binding.get("artifacts") != artifacts:
        raise ValueError("Measurement coverage does not bind the exact source, binaries and runtime artifacts.")
    if contract.get("model") != run["model"] or contract.get("device") != run["device"]:
        raise ValueError("Measurement contract belongs to a different model/device.")
    if contract.get("protocol") != run["protocol"]:
        raise ValueError("Measurement coverage belongs to a different workload protocol.")
    return contract


def raw_file(run: dict, name: str) -> Path:
    receipt = run["diagnostic_files"][name]
    if Path(receipt["name"]).name != receipt["name"]:
        raise ValueError("Diagnostic receipt must name one file in its process directory.")
    path = Path(run["path"]) / receipt["name"]
    # Evidence is local to this fresh process; absolute paths and traversal are rejected.
    if path.resolve().parent != Path(run["path"]).resolve() or digest(path) != receipt["sha256"]:
        raise ValueError("Diagnostic file path/hash differs from the process receipt: " + name)
    return path


def checked_phases(run: dict) -> list[dict]:
    path = raw_file(run, "phases")
    phases = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    expected = ["construction", "construction-complete", "training-prepare"]
    for epoch in range(run["protocol"]["epochs"]):
        expected.extend([f"training-epoch-{epoch}", f"training-epoch-{epoch}-complete"])
    expected.append("training-complete")
    for batch in run["protocol"]["inference_batch_sizes"]:
        expected.extend(f"inference-b{batch}-{part}" for part in ("prepare", "warmup", "steady", "complete"))
    expected.append("complete")
    if [event.get("phase") for event in phases] != expected:
        raise ValueError("Allocation diagnostics omit, duplicate or reorder workload phases.")
    if any(event.get("pid") != run.get("pid") or event.get("model") != run["model"] or
           event.get("diagnostics") is not (run["mode"] == "diagnostic") for event in phases):
        raise ValueError("Allocation phase PID/model/diagnostic mode differs from the launch.")
    if any(a["unix_ms"] > b["unix_ms"] for a, b in zip(phases, phases[1:])):
        raise ValueError("Allocation phase clock is nonmonotone.")
    return phases


def fixed_windows(run: dict) -> dict[str, tuple[str, str]]:
    windows = {"construction": ("construction", "construction-complete")}
    for epoch in range(run["protocol"]["epochs"]):
        windows[f"training_epoch_{epoch}"] = (f"training-epoch-{epoch}", f"training-epoch-{epoch}-complete")
    for batch in run["protocol"]["inference_batch_sizes"]:
        windows[f"inference_b{batch}"] = (f"inference-b{batch}-steady", f"inference-b{batch}-complete")
    return windows


def counter_windows(run: dict, phases: list[dict], select) -> dict[str, dict[str, int]]:
    values = [select(event) for event in phases]
    # CUDA may initialize during construction. Absence is allowed only before its first counter.
    first = next((index for index, value in enumerate(values) if value is not None), len(values))
    observed = values[first:]
    if not observed or any(value is None for value in observed):
        raise ValueError("Required allocation provider disappears or never becomes available.")
    for unit in ("bytes", "count"):
        series = [integer(value.get(unit)) for value in observed]
        if any(a > b for a, b in zip(series, series[1:])):
            raise ValueError("Cumulative allocation counters were reset or decreased.")
    by_phase = {event["phase"]: value or {"bytes": 0, "count": 0} for event, value in zip(phases, values)}
    result = {}
    for window, (start, end) in fixed_windows(run).items():
        # Including earlier startup allocations only enlarges Ai's upper bound.
        baseline = {"bytes": 0, "count": 0} if window == "construction" and run["side"] == "aidotnet" else by_phase[start]
        result[window] = {unit: by_phase[end][unit] - baseline[unit] for unit in ("bytes", "count")}
    return result


def array_totals(snapshot: object) -> dict[str, int]:
    if not isinstance(snapshot, list) or len(snapshot) != 7:
        raise ValueError("CLR array snapshot has the wrong ABI.")
    values = [integer(value) for value in snapshot]
    if values[5] or values[6] != 1 or values[1] + values[3] > values[0]:
        raise ValueError("CLR sizing failed, callback counters are invalid or profiler was not initialized.")
    return {"bytes": values[2] + values[4], "count": values[1] + values[3]}


def provider_values(run: dict, phases: list[dict], provider: str) -> dict[str, dict[str, int]]:
    if provider == "clr_arrays":
        if "clr_array_profiler" not in run["artifacts"]:
            raise ValueError("CLR profiler binary is not hash-bound to the launch.")
        receipt = load(raw_file(run, "clr_arrays"))
        if (receipt.get("schema") != "clr-array-profiler-v1" or receipt.get("pid") != run["pid"] or
            receipt.get("shutdown_complete") is not True or receipt.get("sampling") is not False or
            receipt.get("event_mask") != 0x800100 or receipt.get("unmanaged_coverage") is not False):
            raise ValueError("CLR allocation callbacks lack exact launch/shutdown/flag coverage.")
        final = array_totals(receipt["snapshot"])
        def select(event):
            value = event.get("clr_arrays") or {}
            if value.get("coverage") != "clr_array_callbacks_v1" or value.get("abi") != 1:
                raise ValueError("Exact CLR array snapshots are missing at a phase boundary.")
            totals = array_totals(value.get("snapshot"))
            if any(totals[unit] > final[unit] for unit in totals):
                raise ValueError("CLR phase counters exceed the shutdown receipt.")
            return totals
        return counter_windows(run, phases, select)
    if provider == "torch_cpu_memory_events":
        events = load(raw_file(run, "cpu_allocations"))["traceEvents"]
        prefix = "aidotnet.phase/" + run["model"] + "/"
        markers = [event for event in events if event.get("name", "").startswith(prefix)]
        def same_process(event):
            # Kineto Chrome traces use an OS process ID (JSON integer, occasionally a decimal string).
            pid = event.get("pid")
            return not isinstance(pid, bool) and str(pid).isdecimal() and int(pid) == run["pid"]
        if any(not same_process(event) for event in markers):
            raise ValueError("CPU trace phase marker PID differs from the launched process.")
        names = [event["name"][len(prefix):] for event in sorted(markers, key=lambda item: item["ts"])]
        if names != [event["phase"] for event in phases]:
            raise ValueError("CPU trace lacks the exact diagnostic phase marker sequence.")
        by_phase = {event["name"][len(prefix):]: event for event in markers}
        positive = [event for event in events if event.get("name") == "[memory]" and
                    event.get("args", {}).get("Device Type") == 0 and event.get("args", {}).get("Bytes", 0) > 0]
        if any(not same_process(event) for event in positive):
            raise ValueError("CPU allocator event PID differs from the launched process.")
        result = {}
        for window, (start, end) in fixed_windows(run).items():
            begin = by_phase[start]["ts"] + by_phase[start].get("dur", 0)
            finish = by_phase[end]["ts"]
            selected = [integer(event["args"]["Bytes"]) for event in positive if begin <= event["ts"] < finish]
            result[window] = {"bytes": sum(selected), "count": len(selected)}
        return result
    if provider.startswith("torch_cuda_"):
        key = "gpu_framework_requested" if provider == "torch_cuda_requests" else "gpu_backing"
        expected = "pytorch_native_caching_allocator_requests" if provider == "torch_cuda_requests" else "pytorch_native_caching_allocator_segments"
        def select(event):
            value = event.get("allocators", {}).get(key)
            if value is not None and value.get("source") != expected:
                raise ValueError("Torch CUDA allocator provider semantics differ.")
            return value
        return counter_windows(run, phases, select)
    key, coverage = {
        "aidotnet_cpu_requests": ("cpu_storage_requested", "partial_numeric_storage_request_boundaries"),
        "aidotnet_gpu_requests": ("gpu_buffer_requested", "primitive8_32_backend_buffer_api_only"),
        "aidotnet_compiled_workspace_loans": ("gpu_compiled_workspace_loan_requested", "successful_compiled_broker_loans_only"),
        "aidotnet_cuda_backing_api": ("gpu_native_api_requested", "cuda_tracker_hook_sites_only"),
    }[provider]
    def select(event):
        diagnostics = event.get("allocator_diagnostics") or {}
        value = diagnostics.get(key)
        if value is not None and (diagnostics.get("enabled") is not True or value.get("coverage") != coverage):
            raise ValueError("AiDotNet allocation provider is disabled or its raw scope changed.")
        return value
    return counter_windows(run, phases, select)


def source_audit(route: dict, run: dict) -> None:
    if not route.get("rationale") or not route.get("callsites"):
        raise ValueError("Source audit lacks callsites/rationale.")
    for site in route["callsites"]:
        if not isinstance(site, dict) or not site.get("location"):
            raise ValueError("Source audit callsite is not tied to an artifact/file.")
        if "artifact" in site:
            if site["artifact"] not in run["artifacts"]:
                raise ValueError("Source audit artifact is not bound to the launch.")
        elif "path" not in site or digest(Path(site["path"])) != site.get("sha256"):
            raise ValueError("Source audit callsite file hash differs.")


def audit_coverage(category: str, claim: dict, run: dict) -> list[str]:
    if claim.get("scope") != SCOPES[category] or claim.get("window") != "fixed_work_construction_training_epochs_inference_steady":
        raise ValueError("Allocation category, backing boundary or workload window differs.")
    if claim.get("exclusions") != ["vendor_internal_allocations_bypassing_framework_allocators"]:
        raise ValueError("Framework allocation scope must explicitly exclude vendor-internal bypasses on both sides.")
    allowed_bounds = {"exact", "upper"} if run["side"] == "aidotnet" else {"exact", "lower"}
    if claim.get("bound") not in allowed_bounds:
        raise ValueError("Only an Ai exact/upper bound versus a Torch exact/lower bound proves a win.")
    # Arrays include headers/other arrays and unknown-class objects: they cannot be declared exact numeric storage.
    providers = claim.get("providers", [])
    if not providers or len(set(providers)) != len(providers):
        raise ValueError("Allocation providers are empty or duplicated.")
    for provider in providers:
        if provider not in PROVIDERS or PROVIDERS[provider][0] != run["side"] or category not in PROVIDERS[provider][1]:
            raise ValueError("Provider is incompatible with allocation side/category: " + str(provider))
    if "clr_arrays" in providers and claim["bound"] != "upper":
        raise ValueError("All-array/header-inclusive coverage is a conservative upper bound, not exact numeric storage.")
    if run["side"] == "pytorch" and "torch_runtime" not in run["artifacts"]:
        raise ValueError("Torch allocator runtime identity is not hash-bound to the launch.")
    if claim.get("bound") != "lower" and claim.get("unmeasured_routes") != []:
        raise ValueError("Active allocation routes remain unmeasured.")
    routes = claim.get("routes", {})
    if set(routes) != ROLES[category] or not claim.get("reviewer"):
        raise ValueError("Allocation route review omits a required route class.")
    used = set()
    for role, route in routes.items():
        # A source-bound explanation must identify actual callsites or prove the class inactive.
        # This is a review artifact, not an automated assertion that static inventory proves execution.
        source_audit(route, run)
        if route.get("status") == "measured":
            if route.get("provider") not in providers:
                raise ValueError("Allocation route has no eligible measured provider: " + role)
            if run["side"] == "aidotnet":
                if role == "managed_fresh" and route["provider"] != "clr_arrays":
                    raise ValueError("Instrumented partial CPU requests do not bound every fresh array.")
                if role == "framework_pool_borrows" and route["provider"] == "clr_arrays":
                    raise ValueError("Array callbacks do not observe reused pool/arena borrows.")
            used.add(route["provider"])
        elif route.get("status") != "inactive" and not (claim["bound"] == "lower" and route.get("status") == "excluded_lower_bound"):
            raise ValueError("Allocation route is unknown or partially covered: " + role)
    if used != set(providers):
        raise ValueError("Allocation bound includes an unaudited provider.")
    return providers


def allocation_metrics(run: dict) -> tuple[dict[str, int], list[str]]:
    values, errors = {}, []
    try:
        if run.get("mode") != "diagnostic" or run.get("return_code") != 0:
            raise ValueError("Allocation evidence requires a successful separate diagnostic process.")
        contract = bound_contract(run)
        phases = checked_phases(run)
        claims = contract["allocations"][run["side"]]
        for category in ("cpu_framework_requested", "cpu_backing", *(("gpu_framework_requested", "gpu_backing") if run["device"] == "cuda" else ())):
            try:
                providers = audit_coverage(category, claims[category], run)
                counters = [provider_values(run, phases, provider) for provider in providers]
                for unit in ("bytes", "count"):
                    values[category + "_" + unit] = sum(window[unit] for counter in counters for window in counter.values())
            except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
                errors.append(category + ": " + str(error))
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return values, errors


def cache_coverage(run: dict) -> dict:
    try:
        if run.get("return_code") != 0:
            raise ValueError("Cache coverage requires a successful process.")
        contract = bound_contract(run)
        checked_phases(run)
        state = run["cache_state"]
        if state.get("regime") not in ("cold", "warm") or state.get("initial_files") is None:
            raise ValueError("Cache initial contents/regime were not recorded before launch.")
        if state["regime"] == "cold" and state["initial_files"]:
            raise ValueError("Cold process cache was not empty before launch.")
        if state["regime"] == "warm" and (not state.get("warm_seed") or state["initial_files"] != state["warm_seed"]):
            raise ValueError("Warm cache differs from its immutable seed receipt.")
        if not run.get("pid") or not run.get("process_create_time"):
            raise ValueError("Fresh process identity is missing.")
        audit = contract["caches"][run["side"]]
        if not audit.get("reviewer") or audit.get("unknown_roots") != []:
            raise ValueError("Persistent cache inventory remains incomplete.")
        required = {"process_local", "vendor_process_local"}
        if run["side"] == "aidotnet":
            required |= {"cpu_autotune", "codegen_autotune", "gpu_profile"}
            if run["device"] == "cuda":
                required |= {"cuda_nvrtc", "cuda_direct_ptx", "cuda_driver"}
        else:
            required |= {"inductor", "triton", "remote_fxgraph", "remote_autotune", "external_kernel_overrides"}
            if run["device"] == "cuda":
                required.add("cuda_driver")
        if set(audit.get("roots", {})) != required:
            raise ValueError("Active framework/vendor cache inventory omits required roots.")
        for name, root in audit["roots"].items():
            source_audit(root, run)
            if root.get("status") == "isolated":
                env = CACHE_ROOTS.get(name)
                target = state["redirected_environment"].get(env)
                if not env or not target or run["environment_overrides"].get(env) != target:
                    raise ValueError("Persistent cache override differs from the actual launch: " + name)
                # A codegen cache can be a file; its parent still has to live inside our private root.
                Path(target).resolve().relative_to((Path(run["path"]) / "cache").resolve())
            elif root.get("status") not in ("process_local", "inactive"):
                raise ValueError("Unisolated persistent cache root: " + name)
            elif root.get("status") == "process_local" and name not in ("process_local", "vendor_process_local"):
                raise ValueError("A known persistent cache root cannot be relabeled process-local: " + name)
        if audit.get("sensitive_environment") != run.get("cache_sensitive_environment"):
            raise ValueError("External/remote cache policy differs from the audited environment.")
        sensitive = run["cache_sensitive_environment"]
        if set(sensitive) != set(CACHE_SENSITIVE_ENVIRONMENT):
            raise ValueError("Cache-sensitive inherited environment was not completely recorded.")
        if run["side"] == "pytorch":
            remote = ("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE", "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
                      "TORCHINDUCTOR_AOT_AUTOGRAD_REMOTE_CACHE")
            if any(str(sensitive[key]).lower() in ("1", "true") for key in remote) or sensitive["TRITON_OVERRIDE_DIR"] or sensitive["TRITON_CACHE_MANAGER"]:
                raise ValueError("Active external/remote cache manager is outside the isolated local-cache contract.")
        return {"status": "pass", "regime": state["regime"], "boundary": "fresh_process_and_audited_persistent_roots"}
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        return {"status": "unverified", "reason": str(error)}
