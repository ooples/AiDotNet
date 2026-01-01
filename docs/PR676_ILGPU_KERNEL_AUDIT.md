# PR676 ILGPU Kernel Audit

## Scope
This inventory captures operations currently accelerated by ILGPU in `GpuEngine` and maps them to the direct backends that must replace them (DirectOpenClBackend + DirectCudaBackend).

## Notes
- Direct GPU path is float32-first. Double/other types are converted via INumericOperations.
- CUDA path uses custom kernels as primary, cuBLAS for GEMM fallback, and cuDNN for conv/pool/norm fallback.

## Progress
- CUDA/OpenCL elementwise + unary kernels expanded (add/sub/mul/div/min/max, abs/exp/log/log2/exp2/exp10/expm1/log1p/sqrt/sign, power-scalar).
- HIP backend mirrors these ops via CPU fallback for now.
- OpenCL/CUDA sum/max reductions now use GPU partial reduction kernels.
- CUDA sum-axis reduction now uses a direct kernel.

## Kernel Families to Replace
- Elementwise arithmetic: add/sub/mul/div, scalar ops, negate, clamp, lerp, reciprocal, rsqrt, min/max magnitude, round/floor/ceil/truncate/frac.
- Activation + math: relu/sigmoid/tanh/gelu/mish/swish/elu, sin/cos/tan, sinh/cosh/tanh, exp/log/log2/exp2/exp10/log1p/expm1, asin/acos/atan, asinh/acosh/atanh, pow/power-scalar.
- Matrix ops: matmul (tiled + transposed-B), batch matmul, matvec, transpose, add, scale, swap rows/cols, get/set column, outer product.
- Convolution: conv2d/conv3d, depthwise conv2d, locally-connected conv2d, conv-transpose2d, plus backward input/weights/bias kernels.
- Pooling: max/avg pool 2d/3d + backward + maxpool-with-indices.
- Normalization: batch norm forward/backward, layer norm forward/backward.
- Reductions: reduce sum/mean/max/min/variance, partial sums, partial dot, partial sum of squares.
- Softmax family: softmax + backward, tensor softmax, gumbel softmax + backward, taylor softmax + backward, sparsemax + backward, spherical softmax + backward.
- Indexing: gather, scatter, scatter-add, copy, tensor slice/set-slice.
- Embedding: embedding lookup + backward.
- Resampling: upsample + backward (including any-rank), pixel shuffle + backward.
- Misc: positional encoding forward/backward, volume rendering forward, trilinear interpolate + backward, crop/pad, concat.

## Backend Mapping (Initial)
- DirectOpenClBackend: implement full kernel coverage in OpenCL C sources.
- DirectCudaBackend:
  - GEMM -> custom kernel + cuBLAS fallback.
  - Conv/pool/norm/activation -> cuDNN (if accepted) or custom CUDA kernels.
  - Elementwise/reduction/indexing -> NVRTC-compiled kernels.

## Open Questions
- Should double-precision ops run via float conversion on GPU or force CPU fallback for correctness?
- Do we adopt cuDNN for NVIDIA non-GEMM primitives?
