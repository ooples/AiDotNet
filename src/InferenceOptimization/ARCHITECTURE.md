# AiDotNet Inference Optimization Architecture

This document describes the internal structure of the `AiDotNet.InferenceOptimization` module and its extension points.

## Design Goals

- Hardware-aware CPU acceleration (SIMD when available)
- Deterministic behavior with safe fallbacks
- Low overhead when optimizations are disabled
- Extensible operator/kernels surface for future backends
- Thread-safe initialization and registration
- Optional profiling hooks for diagnosis

## Key Components

### OptimizationInitializer

Responsibilities:
- One entrypoint to initialize platform detection and (optionally) profiling
- Ensures module initialization is safe to call multiple times

### PlatformDetector

Responsibilities:
- Detects process architecture and SIMD availability (x86/x64 and ARM)
- Exposes `PlatformCapabilities` used for selecting implementations

Notes:
- Capability checks are runtime-based; unsupported intrinsics must always fall back to scalar implementations.

### CustomOperatorRegistry

Responsibilities:
- Registers multiple implementations per operation name
- Chooses the best supported implementation at runtime
- Caches the selection to avoid repeated capability checks

### Kernels (`Kernels/*`)

Responsibilities:
- Optimized building blocks for critical inference workloads:
  - GEMM / matmul
  - attention
  - convolution

Notes:
- Kernels are implemented with safe, span-based loops and use platform intrinsics only behind runtime capability checks.

### CPU Helpers (`AiDotNet.Tensors/Engines/Optimization/*`)

Responsibilities:
- Cache-aware helpers (tiling/transposition heuristics)
- Loop tiling/unrolling utilities where beneficial
- Optional profiling (`PerformanceProfiler`) for hotspot tracking

## Integration Points

- `AiDotNet.Inference.InferenceOptimizer` selects and applies inference-time implementations (e.g., attention variants, paged KV-cache) based on `InferenceOptimizationConfig`.
- `AiDotNet.Models.Results.PredictionModelResult` exposes facade-friendly entrypoints (`Predict`, `BeginInferenceSession`) while keeping internal complexity non-user-facing by default.

