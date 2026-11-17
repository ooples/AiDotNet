"""Untimed phase markers and explicitly scoped allocator diagnostics for the parity gate."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def mark_phase(model: str, phase: str) -> None:
    output = os.environ.get("AIDOTNET_BENCH_PHASE_FILE")
    if not output:
        return
    diagnostics = os.environ.get("AIDOTNET_BENCH_DIAGNOSTICS") == "1"
    record: dict[str, object] = {
        "schema": 1, "pid": os.getpid(), "unix_ms": time.time_ns() // 1_000_000,
        "model": model, "phase": phase, "diagnostics": diagnostics,
    }
    if diagnostics:
        import torch
        # Native CUDA caching-allocator requests are distinct from segments acquired from the driver.
        # These counters do not measure Python objects, CPU storage, or allocations inside vendor libraries.
        if torch.cuda.is_initialized() and torch.cuda.get_allocator_backend() == "native":
            stats = torch.cuda.memory_stats()
            record["allocators"] = {
                "gpu_tensor_requested": {
                    "bytes": stats.get("requested_bytes.all.allocated"),
                    "count": stats.get("allocation.all.allocated"),
                    "source": "pytorch_native_caching_allocator_requests",
                },
                "gpu_backing": {
                    "bytes": stats.get("reserved_bytes.all.allocated"),
                    "count": stats.get("segment.all.allocated"),
                    "source": "pytorch_native_caching_allocator_segments",
                },
            }
    with Path(output).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def cpu_storage_events(trace_path: Path) -> dict[str, object]:
    """Count standalone CPU allocator events once; nested operator memory totals double-count storage."""
    events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    positive = [event for event in events if event.get("name") == "[memory]"
                and event.get("args", {}).get("Device Type") == 0
                and event.get("args", {}).get("Bytes", 0) > 0]
    return {"bytes": sum(event["args"]["Bytes"] for event in positive), "count": len(positive),
            "source": "pytorch_profiler_cpu_allocator_events", "diagnostic_only": True}
