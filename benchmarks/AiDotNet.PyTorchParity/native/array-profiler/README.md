This launch-only Windows x64 profiler is for separate allocation diagnostics. Do not load it into timing-score processes. `ObjectAllocated` callbacks count every managed array, with `GetObjectSize2` full object bytes. Unknown-class objects are included conservatively; any failed size query invalidates the measurement. It does not count unmanaged memory, stack storage or borrowed existing arrays.

The numeric storage/backing upper bound includes all arrays, including harness/runtime arrays and array headers. The requested-storage upper bound additionally includes every audited framework pool/arena borrow; a new array on a cold rent may therefore be counted twice. A source-bound active-route audit must account for framework-owned native/vendor workspaces. Vendor-internal allocations bypassing framework allocators are explicitly excluded on both sides; whole-process RAM/VRAM peaks remain independent required measurements.

Build with `./build.ps1 -OutputDirectory C:/Users/FranklinMoormann/source/repos/perf2561/array-profiler-build` in an exclusive compiler slot. The script pins official CoreCLR headers to runtime commit `60629d14374c56f1cb51819049ad1fa529307f8d` and verifies their hashes, then records source/header/binary hashes and the exact MSVC command. The generated stubs implement disabled callback events; no JIT callbacks, IL rewrites or disable-optimization/inlining flags are requested. Allocation instrumentation still changes runtime overhead, which is why its results are separate from latency scores.

Before using model evidence, build `smoke/ArrayProfilerSmoke.csproj` in Release, then launch its DLL in a fresh process with these environment variables:

```text
CORECLR_ENABLE_PROFILING=1
CORECLR_PROFILER={5B735E8D-5547-4DB5-B407-E9AF3E2A9D15}
CORECLR_PROFILER_PATH=<absolute AiDotNet.ArrayProfiler.dll path>
CORECLR_PROFILER_PATH_64=<same path>
AIDOTNET_ARRAY_PROFILER_OUTPUT=<new shutdown JSON path>
```

The smoke must report three fresh arrays with at least their payload bytes, zero fresh arrays on a warmed pool borrow, and zero sizing failures. Check the matching-PID shutdown receipt as well. The benchmark runner enables these variables only in diagnostic AiDotNet launches and hashes the raw shutdown/phase files afterward. A missing or incomplete receipt cannot pass the gate.

The finish gate sums construction, fixed-step training epochs, and fixed-count inference steady intervals. Time-based adaptive warmup is excluded from allocation ratios because faster implementations execute more warmup calls. Per-phase snapshots and Torch profiler markers retain the evidence for each included interval. Ai construction includes earlier startup arrays, conservatively enlarging its upper bound. Neither CLR GC allocation ticks nor GC cumulative managed bytes are used as exact allocation counts.

The optional numeric diagnostic derivative preserves the legacy seven-slot ABI and adds separately versioned primitive-array payload/type/rank counters. Its exact layout, conservative unknown/unsupported cases, source-only smoke protocol and comparison limits are in [NUMERIC-CONTRACT.md](NUMERIC-CONTRACT.md). It does not change the finish gate or certify physical heap/backing coverage.
