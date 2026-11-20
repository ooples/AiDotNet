"""Untimed phase markers and explicitly scoped allocator diagnostics for the parity gate."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_previous_phase: str | None = None


def mark_phase(model: str, phase: str) -> bool:
    """Record a phase boundary. Returns True when it snapshotted and reset the CUDA allocator peaks.

    When phase telemetry is active this function owns that reset: a caller that reset the peaks first would make
    the marker record the reset values as the finished phase's peaks.
    """
    global _previous_phase
    output = os.environ.get("AIDOTNET_BENCH_PHASE_FILE")
    if not output:
        return False
    reset_peaks = False
    diagnostics = os.environ.get("AIDOTNET_BENCH_DIAGNOSTICS") == "1"
    record: dict[str, object] = {
        "schema": 1, "pid": os.getpid(), "unix_ms": time.time_ns() // 1_000_000,
        "model": model, "phase": phase, "diagnostics": diagnostics,
    }
    import torch
    if diagnostics and os.environ.get("AIDOTNET_BENCH_CPU_ALLOCATION_TRACE"):
        # A zero-duration marker maps allocator events to fixed-work windows; adaptive
        # warm-up call counts must not be mixed into the allocation efficiency score.
        with torch.profiler.record_function("aidotnet.phase/" + model + "/" + phase):
            pass
    if torch.cuda.is_initialized() and os.environ.get("AIDOTNET_BENCH_DEVICE") != "cpu":
        stats = torch.cuda.memory_stats()
        record["gpu_memory_interval"] = {
            "previous_phase": _previous_phase or "startup",
            "coverage": "torch_cuda_caching_allocator" if torch.cuda.get_allocator_backend() == "native" else "unsupported_allocator_backend",
            "scope": "CUDA allocator allocated/reserved high-water marks; excludes context/vendor allocations bypassing it.",
            "devices": {str(torch.cuda.current_device()): {
                "allocated_bytes": stats.get("allocated_bytes.all.current"),
                "peak_allocated_bytes": stats.get("allocated_bytes.all.peak"),
                "reserved_bytes": stats.get("reserved_bytes.all.current"),
                "peak_reserved_bytes": stats.get("reserved_bytes.all.peak"),
            }},
        }
        torch.cuda.reset_peak_memory_stats()
        reset_peaks = True
    else:
        record["gpu_memory_interval"] = {"coverage": "inactive_or_uninitialized"}
    if diagnostics:
        # Native CUDA caching-allocator requests are distinct from segments acquired from the driver.
        # These counters do not measure Python objects, CPU storage, or allocations inside vendor libraries.
        if torch.cuda.is_initialized() and torch.cuda.get_allocator_backend() == "native" and os.environ.get("AIDOTNET_BENCH_DEVICE") != "cpu":
            stats = torch.cuda.memory_stats()
            record["allocators"] = {
                "gpu_framework_requested": {
                    "bytes": stats.get("requested_bytes.all.allocated"),
                    "count": stats.get("allocation.all.allocated"),
                    "source": "pytorch_native_caching_allocator_requests",
                    "scope": "Framework storage and workspace requests through the CUDA caching allocator; excludes vendor allocations bypassing it.",
                },
                "gpu_backing": {
                    "bytes": stats.get("reserved_bytes.all.allocated"),
                    "count": stats.get("segment.all.allocated"),
                    "source": "pytorch_native_caching_allocator_segments",
                    "scope": "CUDA caching-allocator segments; excludes vendor allocations bypassing it.",
                },
            }
    with Path(output).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    _previous_phase = phase
    return reset_peaks


def cpu_storage_events(trace_path: Path) -> dict[str, object]:
    """Count standalone CPU allocator events once; nested operator memory totals double-count storage."""
    events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    positive = [event for event in events if event.get("name") == "[memory]"
                and event.get("args", {}).get("Device Type") == 0
                and event.get("args", {}).get("Bytes", 0) > 0]
    return {"bytes": sum(event["args"]["Bytes"] for event in positive), "count": len(positive),
            "source": "pytorch_profiler_cpu_allocator_events", "diagnostic_only": True,
            "scope": "Positive CPU allocator events including framework workspaces; excludes Python objects and vendor allocations bypassing the allocator."}
