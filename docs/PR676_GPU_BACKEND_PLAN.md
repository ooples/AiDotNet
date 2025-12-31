# PR676 Direct GPU Backends Plan

## Goals
- Add a NVIDIA direct GPU backend that mirrors the OpenCL direct-kernel path.
- Remove ILGPU and replace all GPU execution with direct GPU backends.
- Keep CLBlast as a fallback option when direct kernels are unavailable or fail.
- Preserve tuning + diagnostics so we can beat CLBlast on end-to-end performance.

## Non-goals
- Multi-node/distributed GPU execution.
- Full coverage of every kernel outside GEMM unless explicitly requested.
- Automatic driver/toolkit installation.

## Decisions (TBD)
- NVIDIA API: CUDA Driver API (P/Invoke) with PTX or Runtime API?
- NVIDIA fallback: cuBLAS, CLBlast-over-OpenCL, or direct-only?
- CI strategy for CUDA (env-gated tests or no CUDA in CI)?

## Architecture Outline
- Engine selection:
  - NVIDIA GPU -> DirectCudaBackend (primary) -> fallback (TBD)
  - Non-NVIDIA GPU -> DirectOpenClBackend (primary) -> CLBlast fallback
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
   - Hook Bayesian tuning pipeline + diagnostics.
3. Remove ILGPU
   - Delete ILGPU package refs and types.
   - Replace all ILGPU-specific code paths in engines and layers.
4. Fallbacks + tests
   - Wire CLBlast fallback for OpenCL path.
   - Add CUDA fallback (decision above) or direct-only strategy.
   - Add integration tests for correctness and perf gating.
5. Docs + benchmarks
   - Update GPU docs and benchmarks to include CUDA path.
   - Capture tuning runs and performance deltas.

## Open Questions
- Scope of kernels beyond GEMM (convs, activations, etc.)?
- Required OS/driver baseline for CUDA on target machines?
- Expected behavior when both OpenCL and CUDA are available?
