# PR676 Direct GPU Backends Plan

## Goals
- Add a NVIDIA direct GPU backend that mirrors the OpenCL direct-kernel path.
- Remove ILGPU and replace all GPU execution with direct GPU backends.
- Keep CLBlast as a fallback option when direct kernels are unavailable or fail.
- Preserve tuning + diagnostics so we can beat CLBlast on end-to-end performance.

## Progress
- CUDA backend scaffolded with cuBLAS GEMM + NVRTC kernels.
- NVRTC fallback probing added for multiple DLL/SO names.
- Elementwise/unary kernel coverage expanded for CUDA + OpenCL (HIP uses CPU fallback).
- OpenCL/CUDA sum/max reductions now use GPU partial reduction kernels.
- DirectGpu backend order is configurable via `AIDOTNET_DIRECTGPU_BACKENDS`.
- CUDA sum-axis reduction now uses a direct kernel.
- CUDA AllocateByteBuffer implemented; Cuda activation kernel source + CLBlast raw strings fixed for build.
- GpuEngine memory pool returns corrected in matmul/cached-weight paths; DirectGpuEngine buffer lookup now skips null buffers.
- RX 5500 XT benchmarks captured (see "Latest Benchmarks" below).

## Non-goals
- Multi-node/distributed GPU execution.
- Full coverage of every kernel outside GEMM unless explicitly requested.
- Automatic driver/toolkit installation.

## Decisions
- NVIDIA API: CUDA Driver API (P/Invoke) + PTX (NVRTC for runtime codegen).
- NVIDIA GEMM fallback: cuBLAS.
- NVIDIA non-GEMM fallback: cuDNN (custom CUDA kernels remain primary).
- cuDNN fallback scope: convolution, pooling, normalization (activations/custom elementwise stay custom).
- OpenCL GEMM fallback: CLBlast.
- Kernel coverage: replace all ILGPU kernels with direct backends.
- Removal: delete ILGPU packages/types after parity + tests are in place.
- Type conversion: convert to float at GPU boundary, convert back via INumericOperations.
- NVRTC: add fallback probing for multiple DLL versions.

## Decisions (TBD)
- cuDNN version support list (8/9 and platform-specific names).
- NVRTC DLL name list to probe (Windows + Linux).
- CI strategy for CUDA (env-gated tests or no CUDA in CI)?

## Latest Benchmarks (RX 5500 XT, gfx1012:xnack-)
OpenCL vs CLBlast (end-to-end, untuned):
- 256^2: CLBlast 236.3 GFLOPS, AiDotNet 44.2 GFLOPS (CLBlast 5.35x)
- 512^2: CLBlast 1110.9 GFLOPS, AiDotNet 266.9 GFLOPS (CLBlast 4.16x)
- 1024^2: CLBlast 1965.0 GFLOPS, AiDotNet 655.8 GFLOPS (CLBlast 3.00x)
- 2048^2: CLBlast 2276.8 GFLOPS, AiDotNet 946.3 GFLOPS (CLBlast 2.41x)
- 4096^2: CLBlast 435.1 GFLOPS, AiDotNet 397.3 GFLOPS (CLBlast 1.09x)
- DenseLayer (64x768x3072): CLBlast 1078.7 GFLOPS, AiDotNet 123.1 GFLOPS (CLBlast 8.76x)
- Large (128x4096x4096): CLBlast 2502.1 GFLOPS, AiDotNet 391.3 GFLOPS (CLBlast 6.39x)

DirectGpu TUNED (OpenClBackend) vs CLBlast:
- 256^2: CLBlast 242.4 GFLOPS, AiDotNet 466.0 GFLOPS (AiDotNet 1.93x)
- 512^2: CLBlast 1071.2 GFLOPS, AiDotNet 173.2 GFLOPS (CLBlast 6.18x)
- 1024^2: CLBlast 2061.0 GFLOPS, AiDotNet 349.9 GFLOPS (CLBlast 5.89x)
- 2048^2: CLBlast 2265.5 GFLOPS, AiDotNet 518.6 GFLOPS (CLBlast 4.37x)
- 4096^2: CLBlast 435.2 GFLOPS, AiDotNet 542.5 GFLOPS (AiDotNet 1.25x)
- DenseLayer (64x768x3072): CLBlast 1034.4 GFLOPS, AiDotNet 54.8 GFLOPS (CLBlast 18.89x)
- Large (128x4096x4096): CLBlast 2452.7 GFLOPS, AiDotNet 484.6 GFLOPS (CLBlast 5.06x)

## Architecture Outline
- Engine selection:
  - NVIDIA GPU -> DirectCudaBackend (custom kernels) -> cuDNN/cuBLAS fallback -> CPU
  - Non-NVIDIA GPU -> DirectOpenClBackend (primary) -> CLBlast fallback -> CPU
- Shared abstractions:
  - IGpuBackend, IGpuAllocator, IGpuKernel, IGpuStream, IGpuTuner
  - Unified diagnostics + CSV logging across backends

## Implementation Phases
1. Inventory + design
   - Map ILGPU usage and entry points to replace.
   - Define backend selection logic and configuration knobs.
2. NVIDIA backend
   - Device discovery, context, stream, memory, kernel launch.
   - GEMM baseline + tuned kernels; packing and layout parity with OpenCL.
   - NVRTC-based kernel compile for tuned kernels and elementwise ops.
   - Custom CUDA kernels are primary; cuDNN is fallback for non-GEMM primitives.
   - Add NVRTC DLL fallback probing.
   - Hook Bayesian tuning pipeline + diagnostics.
3. Remove ILGPU
   - Delete ILGPU package refs and types.
   - Replace all ILGPU-specific code paths in engines and layers.
4. Fallbacks + tests
   - Wire CLBlast fallback for OpenCL path.
   - Add cuBLAS fallback for CUDA path.
   - Add integration tests for correctness and perf gating.
5. Docs + benchmarks
   - Update GPU docs and benchmarks to include CUDA path.
   - Capture tuning runs and performance deltas.

## 100% Confidence Checklist
- Build a kernel parity matrix mapping ILGPU ops to DirectOpenCL/DirectCUDA/cuDNN/CLBlast/CPU and track per-op test coverage.
- Implement missing kernels per matrix (elementwise, reductions, softmax variants, indexing, conv/pool/norm, sparse) on OpenCL/CUDA or via fallback.
- Wire explicit fallback chain logging + failure recording (tuning DB logs failed configs; overwrite only when global best improves).
- Add integration tests that compare GPU outputs to CPU for every kernel family with deterministic seeds and tolerance targets.
- Validate performance vs CLBlast/cuBLAS across target sizes; run offline tuning on RX 5500 XT and store best configs + CSV diagnostics.
- Remove ILGPU packages/types only after parity + tests + perf baselines pass; update Engine.Default to DirectGpu-first.
- Document CI/manual GPU validation steps (AMD + NVIDIA) and required env vars for tuning/diagnostics.

## Open Questions
- Scope of kernels beyond GEMM (convs, activations, etc.)?
- Required OS/driver baseline for CUDA on target machines?
- Expected behavior when both OpenCL and CUDA are available?
