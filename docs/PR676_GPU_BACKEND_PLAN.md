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

## Open Questions
- Scope of kernels beyond GEMM (convs, activations, etc.)?
- Required OS/driver baseline for CUDA on target machines?
- Expected behavior when both OpenCL and CUDA are available?
