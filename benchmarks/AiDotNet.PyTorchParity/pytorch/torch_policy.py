"""Same-process observation or explicitly disclosed reference policy around an unchanged benchmark."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


OVERRIDE_ENVIRONMENT = ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                        "CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_snapshot(torch):
    initialized = torch.cuda.is_initialized()
    getters = {"torch_version": lambda: str(torch.__version__), "torch_cuda_build": lambda: torch.version.cuda,
        "matmul_allow_tf32": lambda: torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": lambda: torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled,
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled,
        "cudnn_benchmark": lambda: torch.backends.cudnn.benchmark,
        "cudnn_deterministic": lambda: torch.backends.cudnn.deterministic}
    result = {"cuda_initialized_before_observation": initialized}
    for name, getter in getters.items():
        try:
            result[name] = getter()
        except Exception as error:
            result[name] = {"unverified": repr(error)}
    # Do not initialize CUDA or call a vendor API just to observe the runtime.
    if initialized:
        try:
            result["cuda_allocator_backend"] = torch.cuda.get_allocator_backend()
        except Exception as error:
            result["cuda_allocator_backend"] = {"unverified": repr(error)}
    result["cuda_initialized_after_observation"] = torch.cuda.is_initialized()
    if result["cuda_initialized_after_observation"] != initialized:
        raise RuntimeError("Policy observation unexpectedly initialized CUDA.")
    return result


def compare_reference(snapshot, manifest):
    expected = {"torch_version": manifest["torch_version"], "matmul_allow_tf32": manifest["tf32"],
        "cudnn_allow_tf32": manifest["tf32"], "deterministic_algorithms": manifest["deterministic_algorithms"],
        "deterministic_warn_only": manifest["deterministic_warn_only"]}
    mismatches = {name: {"expected": value, "observed": snapshot.get(name)} for name, value in expected.items()
                  if snapshot.get(name) != value}
    return {"status": "match" if not mismatches else "mismatch", "mismatches": mismatches,
            "scope": "Only precision/determinism fields actually present in the frozen reference manifest; no bitwise determinism claim."}


def loaded_native_files():
    """Read already-loaded Windows images. Never create CUDA/vendor handles or load those libraries."""
    if os.name != "nt":
        return {"status": "unverified", "reason": "This native-image observer currently supports Windows.", "files": []}
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    modules = (wintypes.HMODULE * 4096)()
    required = wintypes.DWORD()
    psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    if not psapi.EnumProcessModulesEx(process, modules, ctypes.sizeof(modules), ctypes.byref(required), 3):
        raise ctypes.WinError(ctypes.get_last_error())
    if required.value > ctypes.sizeof(modules):
        raise RuntimeError("Loaded native module list exceeded the bounded observer capacity.")
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    version = ctypes.WinDLL("version", use_last_error=True)
    version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    version.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID]
    version.GetFileVersionInfoW.restype = wintypes.BOOL
    version.VerQueryValueW.argtypes = [wintypes.LPCVOID, wintypes.LPCWSTR, ctypes.POINTER(wintypes.LPVOID), ctypes.POINTER(wintypes.UINT)]
    version.VerQueryValueW.restype = wintypes.BOOL
    files = []
    seen = set()
    for index in range(required.value // ctypes.sizeof(wintypes.HMODULE)):
        name = ctypes.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(process, modules[index], name, len(name)):
            continue
        path = Path(name.value)
        lower = path.name.lower()
        if not lower.startswith(("torch", "c10", "cudnn", "cublas", "cufft", "nvrtc", "nvcuda", "libiomp", "mkl", "dnnl", "python", "_c.")):
            continue
        if str(path).lower() in seen:
            continue
        seen.add(str(path).lower())
        row = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "file_version": None}
        unused = wintypes.DWORD()
        size = version.GetFileVersionInfoSizeW(str(path), ctypes.byref(unused))
        if size:
            buffer = ctypes.create_string_buffer(size)
            pointer = wintypes.LPVOID(); length = wintypes.UINT()
            if version.GetFileVersionInfoW(str(path), 0, size, buffer) and version.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)) and length.value >= 52:
                values = ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD))
                row["file_version"] = ".".join(map(str, (values[2] >> 16, values[2] & 65535, values[3] >> 16, values[3] & 65535)))
        files.append(row)
    return {"status": "observed_loaded_images", "scope": "Loaded images only, not proof that a vendor kernel executed. SHA256 streamed in1MiB blocks after benchmark work.", "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mode", choices=("observe", "strict-reference"), default="observe")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-environment", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("Policy receipt must be fresh; preserve previous observations.")
    benchmark = args.benchmark.resolve()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8-sig")) if args.reference_manifest else None
    manifest_hash = sha(args.reference_manifest) if args.reference_manifest else None
    if manifest is not None and manifest.get("benchmark_source_sha256") != sha(benchmark):
        raise ValueError("Reference manifest describes a different benchmark source recipe.")
    original_environment = {name: os.environ.get(name) for name in OVERRIDE_ENVIRONMENT}
    environment_binding = None
    environment = None
    reference_policy = dict(manifest) if manifest else None
    if args.reference_environment is not None:
        environment = json.loads(args.reference_environment.read_text(encoding="utf-8-sig"))
        if environment.get("reference_manifest_sha256") != manifest_hash:
            raise ValueError("Reference startup policy does not bind this exact manifest.")
        evidence = environment.get("evidence", [])
        if not evidence or any(sha(item["path"]) != item.get("sha256") for item in evidence):
            raise ValueError("Reference environment lacks actual source/runtime evidence hashes.")
        for key, value in environment.get("policy", {}).items():
            if key != "deterministic_warn_only" or not isinstance(value, bool) or reference_policy is None:
                raise ValueError("Unexpected legacy reference policy field: " + key)
            if key in reference_policy and reference_policy[key] != value:
                raise ValueError("A reference policy attestation cannot override a recorded manifest field.")
            reference_policy[key] = value
        environment_binding = {"path": str(args.reference_environment.resolve()), "sha256": sha(args.reference_environment)}
    setters = []
    if args.mode == "strict-reference":
        if manifest is None or manifest.get("tf32") is not False or manifest.get("deterministic_algorithms") is not True:
            raise ValueError("Strict mode requires the reviewed full-FP32 deterministic-choice reference manifest.")
        if not isinstance(reference_policy.get("deterministic_warn_only"), bool):
            raise ValueError("Reference warn-only policy is missing; provide a hash-bound legacy source attestation.")
        if manifest.get("device") == "cuda":
            if environment is None:
                raise ValueError("CUDA strict policy requires a real reference startup-environment receipt; historical workspace settings are not assumed.")
            if "CUBLAS_WORKSPACE_CONFIG" not in environment.get("environment", {}):
                raise ValueError("Reference environment does not bind this manifest and explicit CUBLAS workspace policy.")
        if environment is not None:
            for key, value in environment["environment"].items():
                if key not in OVERRIDE_ENVIRONMENT or (value is not None and not isinstance(value, str)):
                    raise ValueError("Unexpected reference environment override: " + key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
                setters.append({"environment_before_torch_import": key, "value": value})

    import torch
    defaults = policy_snapshot(torch)
    if args.mode == "strict-reference":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=reference_policy["deterministic_warn_only"])
        setters += [{"public_setter": "torch.backends.cuda.matmul.allow_tf32", "value": False},
                    {"public_setter": "torch.backends.cudnn.allow_tf32", "value": False},
                    {"public_setter": "torch.use_deterministic_algorithms", "value": True, "warn_only": reference_policy["deterministic_warn_only"]}]
    applied = policy_snapshot(torch)
    state = {"schema": "torch-policy-native-receipt-v1", "pid": os.getpid(), "mode": args.mode,
        "wrapper_sha256": sha(__file__), "benchmark_sha256": sha(benchmark), "target_script": str(benchmark), "reference_manifest_sha256": manifest_hash,
        "resolved_reference_policy": reference_policy,
        "reference_environment": environment_binding, "original_environment": original_environment,
        "effective_environment": {name: os.environ.get(name) for name in OVERRIDE_ENVIRONMENT},
        "default_flags_before_public_setters": defaults, "applied_flags_before_benchmark": applied,
        "setters": setters, "benchmark_exit_code": None, "atexit_complete": False,
        "policy_match_before": compare_reference(applied, reference_policy) if reference_policy and "deterministic_warn_only" in reference_policy else None}

    def finish():
        try:
            state["flags_at_exit"] = policy_snapshot(torch)
            state["native"] = loaded_native_files()
            state["atexit_complete"] = True
        except Exception as error:
            state["observation_error"] = repr(error)
        temporary = args.receipt.with_name(args.receipt.name + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(args.receipt)
    atexit.register(finish)
    if args.mode == "strict-reference" and state["policy_match_before"]["status"] != "match":
        state["benchmark_exit_code"] = -1
        raise ValueError("Applied runtime policy/version does not match the bound reference; benchmark was not executed.")
    sys.path.insert(0, str(benchmark.parent))
    sys.argv = [str(benchmark), *arguments]
    try:
        runpy.run_path(str(benchmark), run_name="__main__")
        state["benchmark_exit_code"] = 0
    except SystemExit as error:
        state["benchmark_exit_code"] = 0 if error.code is None else error.code
        raise
    except BaseException as error:
        state["benchmark_exit_code"] = -1
        state["benchmark_error"] = repr(error)
        raise
    finally:
        state["flags_after_benchmark"] = policy_snapshot(torch)
        state["policy_match_after"] = compare_reference(state["flags_after_benchmark"], reference_policy) if reference_policy and "deterministic_warn_only" in reference_policy else None


if __name__ == "__main__":
    main()
