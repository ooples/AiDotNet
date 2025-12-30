# CLBlast GEMM Kernel Analysis

This document analyzes CLBlast's GEMM (General Matrix Multiplication) kernel implementation to identify optimization techniques and potential gaps for building superior GPU kernels.

## Overview

CLBlast is a tuned OpenCL BLAS library that achieves near-vendor performance through auto-tuning. The GEMM implementation consists of multiple kernel variants:

- **Indirect GEMM** (`xgemm_part1-4.opencl`): Pre-/post-processing kernels for optimal performance
- **Direct GEMM** (`xgemm_direct_part1-3.opencl`): Single-pass kernel for all sizes
- **Batched GEMM** (`xgemm_batched.opencl`): Multiple matrix multiplications in parallel

---

## CLBlast GEMM Optimization Techniques

### 1. Tiling Strategy

CLBlast uses a hierarchical tiling approach with three levels:

**Workgroup Tiles:**
- `MWG`: Tile size in M dimension (16-128)
- `NWG`: Tile size in N dimension (16-128)
- `KWG`: Tile size in K dimension (16-32)

**Thread Tiles (Register Blocking):**
- `MDIMC`, `NDIMC`: Thread distribution (4-32)
- `MDIMA`, `NDIMB`: Alternative thread mapping (2-32)

**Example Configuration (RTX 4090 Single-Precision):**
```
MWG=128, NWG=64, KWG=32
MDIMC=8, NDIMC=8
MDIMA=8, NDIMB=8
```

### 2. Register Blocking

Each thread computes multiple output elements stored in private registers:

```opencl
// Private accumulators for NWI x MWI output elements
realM cpm[NWI * (MWI/VWM)];

// Initialize to zero
for (int _ni = 0; _ni < NWI; _ni++) {
    for (int _mi = 0; _mi < MWI/VWM; _mi++) {
        SetToZero(cpm[_ni * (MWI/VWM) + _mi]);
    }
}
```

**Key Insight:** CLBlast uses vector types (`realM` = `float4`/`float8`) to pack multiple accumulators, reducing register count while maintaining throughput.

### 3. Shared Memory Usage Patterns

**Local Memory Allocation:**
```opencl
// Conditional allocation based on SA/SB flags
#if SA == 1
    __local realM alm[KWG * MWG/VWM];
#endif
#if SB == 1
    __local realN blm[KWG * NWG/VWN];
#endif
```

**Bank Conflict Avoidance (Transpose Kernel):**
```opencl
// Padding to prevent bank conflicts
__local realT tile[TRA_WPT * TRA_DIM][TRA_DIM + TRA_PAD];
```

The `PADA` and `PADB` parameters add padding to local memory arrays to avoid bank conflicts during column-wise access.

### 4. Memory Access Coalescing

**Global-to-Local Loading with Coalescing:**
```opencl
// Vectorized coalesced load from global memory
void GlobalToLocalDirectA(__local realM* alm, const __global realM* agm,
                          const int kSizeM, const int tid, const int kwg) {
    // Multiple elements per thread with stride matching warp size
    for (int _la = 0; _la < MWAI/VWM; _la++) {
        int mg = _la * MDIMAD + lid0;     // Coalesced index
        int kg = _lb * NDIMAD + lid1;     // K dimension
        alm[kg * (MWG/VWM) + mg] = agm[kg * kSizeM + mg + kwg];
    }
}
```

**Strided vs Non-Strided Access:**
- `STRM`/`STRN` flags control whether threads access contiguous or strided elements
- Non-strided (STRM=0): Better for row-major matrices
- Strided (STRM=1): Better for transposed access patterns

### 5. Vectorized Loads

CLBlast supports vector widths from 1 to 16 elements:

```opencl
// Vector type definitions based on VWM/VWN parameters
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#elif VWM == 16
    typedef real16 realM;
#endif
```

**Vectorized Multiply-Add:**
```opencl
inline void MultiplyAddVector(realM *cvec, const realM avec, const real bval) {
    #if USE_VECTOR_MAD == 1
        *cvec += avec * bval;
    #else
        // Manual component-wise for older hardware
        (*cvec).x += avec.x * bval;
        (*cvec).y += avec.y * bval;
        (*cvec).z += avec.z * bval;
        (*cvec).w += avec.w * bval;
    #endif
}
```

### 6. Loop Unrolling Strategies

**K-Dimension Loop:**
```opencl
// Main computation loop over K tiles
for (int kwg = 0; kwg < kSizeK; kwg += KWG) {
    // Load A and B tiles into local memory
    GlobalToLocalA(alm, agm, ...);
    GlobalToLocalB(blm, bgm, ...);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Unrolled inner loop over K within tile
    #pragma unroll
    for (int _ki = 0; _ki < KWG; _ki++) {
        // Load from local to private registers
        LocalToPrivateA(apm, alm, _ki);
        LocalToPrivateB(bpm, blm, _ki);

        // Compute outer product
        MultiplyAccumulate(cpm, apm, bpm);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
```

**Register-Level Unrolling:**
```opencl
// Fully unrolled accumulation
for (int _ni = 0; _ni < NWI; _ni++) {
    for (int _mi = 0; _mi < MWI/VWM; _mi++) {
        MultiplyAddVector(&cpm[_ni*(MWI/VWM)+_mi], apm[_mi], bpm[_ni]);
    }
}
```

### 7. Local Memory Barriers

**Synchronization Points:**
```opencl
// After loading tiles to shared memory
barrier(CLK_LOCAL_MEM_FENCE);

// Before next tile load (double buffering not used)
barrier(CLK_LOCAL_MEM_FENCE);

// Optional global fence for result writing
#if GLOBAL_MEM_FENCE == 1
barrier(CLK_GLOBAL_MEM_FENCE);
#endif
```

### 8. Vendor-Specific Optimizations

**Subgroup Shuffling (GEMMK=1):**
```opencl
// NVIDIA/Intel subgroup shuffle for data exchange
#if USE_SUBGROUP_SHUFFLING == 1
    // Broadcast values within subgroup without shared memory
    real bval = sub_group_broadcast(bpm[_ki], _ni);
#endif
```

**Staggered Indices (Partition Camping Prevention):**
```opencl
#if USE_STAGGERED_INDICES == 1
    // Shuffle workgroup IDs to avoid memory bank conflicts
    int group_id = (get_group_id(0) + get_group_id(1)) % get_num_groups(0);
#endif
```

---

## Auto-Tuning Parameters

### Parameter Definitions and Ranges

| Parameter | Description | Limited Range | Extended Range |
|-----------|-------------|---------------|----------------|
| `MWG` | M-dimension workgroup tile | 16, 32, 64 | 16-128 |
| `NWG` | N-dimension workgroup tile | 16, 32, 64 | 16-128 |
| `KWG` | K-dimension tile size | 32 | 16, 32 |
| `MDIMC` | Threads in M dimension | 8, 16, 32 | 2-32 |
| `NDIMC` | Threads in N dimension | 8, 16, 32 | 2-32 |
| `MDIMA` | Alt thread mapping M | 8, 16, 32 | 2-32 |
| `NDIMB` | Alt thread mapping N | 8, 16, 32 | 2-32 |
| `VWM` | Vector width for M | 1, 2, 4 | 1, 2, 4, 8 |
| `VWN` | Vector width for N | 1, 2, 4 | 1, 2, 4, 8 |
| `SA` | Use local mem for A | 0, 1 | 0, 1 |
| `SB` | Use local mem for B | 0, 1 | 0, 1 |
| `STRM` | Strided access for M | 0, 1 | 0, 1 |
| `STRN` | Strided access for N | 0, 1 | 0, 1 |
| `KREG` | K-register blocking (GEMMK=1) | 1, 2, 4 | 1-16 |

### Optimal Parameters by GPU

**NVIDIA GPUs (Single-Precision):**
| GPU | MWG | NWG | KWG | MDIMC | NDIMC | VWM | VWN | SA | SB |
|-----|-----|-----|-----|-------|-------|-----|-----|----|----|
| RTX 4090 | 128 | 64 | 32 | 8 | 8 | 4 | 4 | 1 | 1 |
| RTX 3090 | 64 | 64 | 32 | 8 | 8 | 4 | 4 | 1 | 1 |
| Tesla V100 | 64 | 32 | 32 | 8 | 8 | 4 | 4 | 1 | 1 |

**AMD GPUs (Single-Precision):**
| GPU | MWG | NWG | KWG | MDIMC | NDIMC | VWM | VWN | SA | SB |
|-----|-----|-----|-----|-------|-------|-----|-----|----|----|
| RX 6800 XT | 128 | 64 | 16 | 8 | 32 | 4 | 1 | 1 | 0 |
| RX 5700 XT | 64 | 64 | 16 | 8 | 8 | 2 | 2 | 1 | 1 |
| RX 480 | 64 | 16 | 32 | 16 | 16 | 2 | 1 | 1 | 1 |

**Intel GPUs (Single-Precision):**
| GPU | MWG | NWG | KWG | MDIMC | NDIMC | VWM | VWN | SA | SB |
|-----|-----|-----|-----|-------|-------|-----|-----|----|----|
| Arc A770 | 64 | 32 | 16 | 4 | 4 | 1 | 8 | 0 | 0 |
| UHD 770 | 64 | 128 | 32 | 8 | 8 | 1 | 8 | 1 | 1 |

### Constraints

CLBlast enforces several parameter constraints:

```
MWG % (MDIMC * VWM) == 0
NWG % (NDIMC * VWN) == 0
KWG % ((MDIMC * NDIMC) / MDIMA) == 0
KWG % ((MDIMC * NDIMC) / NDIMB) == 0
KREG % VWN == 0  (for GEMMK=1)
```

---

## Identified Gaps and Opportunities

### 1. Missing Double-Buffering / Prefetching

**Gap:** CLBlast uses single-buffering with barriers, causing pipeline stalls.

**Current Pattern:**
```opencl
for (int kwg = 0; kwg < kSizeK; kwg += KWG) {
    // Load tile (STALL while loading)
    GlobalToLocal(...);
    barrier(CLK_LOCAL_MEM_FENCE);  // <-- STALL

    // Compute (memory subsystem idle)
    MultiplyAccumulate(...);
    barrier(CLK_LOCAL_MEM_FENCE);  // <-- STALL
}
```

**Opportunity:** Implement double-buffering to overlap compute with memory:
```opencl
// Load first tile
LoadTile(buffer[0], k=0);
barrier();

for (int kwg = 0; kwg < kSizeK-KWG; kwg += KWG) {
    // Async load NEXT tile while computing current
    LoadTileAsync(buffer[(kwg/KWG+1)%2], kwg+KWG);
    ComputeTile(buffer[kwg/KWG%2]);
    barrier();
}
// Compute last tile
ComputeTile(buffer[last]);
```

**Expected Gain:** 15-30% for memory-bound configurations.

### 2. No Tensor Core / Matrix Core Utilization

**Gap:** CLBlast uses scalar/vector FMA instructions only.

**Opportunity:** Modern GPUs have dedicated matrix units:
- NVIDIA: Tensor Cores (WMMA/MMA instructions)
- AMD: Matrix Cores (MFMA instructions)
- Intel: XMX units

**Example for NVIDIA Tensor Cores:**
```cpp
// Using CUDA's WMMA API (translatable to OpenCL via vendor extensions)
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**Expected Gain:** 3-10x for supported precisions (FP16, TF32, INT8).

### 3. Limited Fused Operations

**Gap:** CLBlast only supports `C = alpha*A*B + beta*C`.

**Opportunity:** Implement fused kernels for common patterns:

1. **GEMM + Activation:**
```opencl
// Fused: Y = activation(alpha*A*B + beta*C)
// Saves one global memory round-trip
C[idx] = activation(alpha * dot_product + beta * C[idx]);
```

2. **GEMM + Bias + Activation (Neural Network Forward):**
```opencl
// Fused: Y = activation(A*B + bias)
C[idx] = relu(dot_product + bias[col]);
```

3. **GEMM + Softmax:**
```opencl
// Fused attention pattern
// Q*K^T -> softmax -> *V in single kernel
```

4. **Batched GEMM + Reduction:**
```opencl
// Fused: sum(A_i * B_i) for ensemble methods
```

**Expected Gain:** 20-50% for memory-bound workloads.

### 4. Suboptimal Small Matrix Handling

**Gap:** Direct GEMM kernel has high overhead for small matrices (<256).

**Current Approach:**
- Same tile sizes regardless of matrix size
- Fixed workgroup dimensions
- No persistent kernel support

**Opportunity:**
1. **Warp-level GEMM for small matrices:**
```opencl
// Single warp computes entire small GEMM
if (M <= 32 && N <= 32 && K <= 32) {
    // All data in registers, no shared memory
    WarpGEMM(A, B, C, M, N, K);
}
```

2. **Persistent Kernels for Batched Small GEMM:**
```opencl
// Single kernel launch processes all batches
__kernel void PersistentBatchedGEMM(...) {
    while (batch_idx < total_batches) {
        ProcessBatch(batch_idx);
        batch_idx = atomic_inc(&global_counter);
    }
}
```

### 5. No Automatic Precision Selection

**Gap:** User must manually choose precision.

**Opportunity:** Mixed-precision computation:
```opencl
// Accumulate in FP32, load/store in FP16
half a_val = A[idx];  // FP16 load (saves bandwidth)
half b_val = B[idx];  // FP16 load
float acc = (float)a_val * (float)b_val;  // FP32 compute
// ... accumulate in FP32 ...
C[idx] = (half)acc;  // FP16 store
```

**Expected Gain:** 2x bandwidth reduction with minimal accuracy loss.

### 6. Missing Sparsity Support

**Gap:** No sparse matrix acceleration.

**Opportunity:** Structured sparsity (2:4 pattern for NVIDIA Ampere+):
```opencl
// Process 2:4 sparse matrices
// 50% sparsity = 2x compute throughput
__kernel void SparseMMA(
    __global half* A_values,  // Only non-zero values
    __global int* A_indices,  // Sparsity pattern
    __global half* B,
    __global float* C
) {
    // Use sparse tensor core instructions
}
```

### 7. Architecture-Specific Gaps

**NVIDIA:**
- No async copy (`cp.async`) for Ampere+
- No warp specialization (producer/consumer pattern)
- No tensor memory accelerator (TMA) for Hopper

**AMD:**
- No MFMA (Matrix Fused Multiply-Add) utilization
- No LDS (Local Data Share) swizzling optimization
- Limited wave32/wave64 selection

**Intel:**
- No XMX (Xe Matrix eXtensions) utilization
- No EU thread preemption hints
- Limited SLM bank conflict optimization

### 8. Memory Access Pattern Improvements

**Gap:** Fixed access patterns don't adapt to matrix layouts.

**Opportunity:** Runtime layout selection:
```opencl
// Detect optimal layout at runtime
if (is_row_major(A) && is_col_major(B)) {
    // Use NT kernel variant with specific tile shape
} else if (is_col_major(A) && is_row_major(B)) {
    // Use TN kernel variant
}
```

### 9. Kernel Fusion Framework

**Gap:** No mechanism for custom operation fusion.

**Opportunity:** JIT kernel generation:
```cpp
// Runtime kernel composition
KernelBuilder builder;
builder.addOp(GEMM, {M, N, K});
builder.addOp(BIAS_ADD, {bias_ptr});
builder.addOp(RELU, {});
builder.addOp(DROPOUT, {0.1});
auto kernel = builder.compile();
```

### 10. Improved Auto-Tuning

**Gap:** Exhaustive search or random sampling only.

**Opportunity:**
1. **Bayesian Optimization:** Model performance as function of parameters
2. **Transfer Learning:** Use tuning results from similar GPUs
3. **Analytical Models:** Predict performance without running kernels
4. **Multi-objective Optimization:** Balance performance vs. power

---

## Summary: Priority Optimization Targets

| Priority | Optimization | Expected Gain | Complexity |
|----------|-------------|---------------|------------|
| 1 | Tensor Core support | 3-10x | High |
| 2 | Double-buffering | 15-30% | Medium |
| 3 | Fused GEMM+Activation | 20-50% | Medium |
| 4 | Mixed precision | 2x bandwidth | Low |
| 5 | Small matrix optimization | 2-5x | Medium |
| 6 | Async copy (Ampere+) | 10-20% | Low |
| 7 | Structured sparsity | 2x | High |
| 8 | Kernel fusion framework | Variable | High |

---

## References

- CLBlast GitHub: https://github.com/CNugteren/CLBlast
- CLBlast Paper: Nugteren, C. (2018). CLBlast: A Tuned OpenCL BLAS Library. IWOCL.
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
- AMD rocBLAS: https://github.com/ROCmSoftwarePlatform/rocBLAS
