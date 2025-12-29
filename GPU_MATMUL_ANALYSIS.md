# GPU Matrix Multiplication Performance Analysis

## Date: 2025-12-29

## Problem Summary
- 2048x2048 matrix multiply is **650x slower** than expected
- Performance: 0.13 GFLOPS instead of expected ~130 GFLOPS
- Smaller matrices work fine: 256x256 (23 GFLOPS), 512x512 (67 GFLOPS), 1024x1024 (84 GFLOPS)

## Root Cause: Catastrophic Cache Misses on Matrix B

### The Kernel Code
```csharp
(index, a, b, result, k) =>
{
    float sum = 0;
    for (int i = 0; i < k; i++)
        sum += a[index.X, i] * b[i, index.Y];  // PROBLEM HERE
    result[index] = sum;
}
```

### Memory Access Pattern
Matrix B is stored in **row-major** order (DenseX with stride = k).

When accessing `b[i, index.Y]` for a fixed column (e.g., column 0):
- i=0: accesses b[0,0] at memory offset 0
- i=1: accesses b[1,0] at memory offset **2048**
- i=2: accesses b[2,0] at memory offset **4096**
- ... (each access is 2048 elements apart)

### Why This Is Catastrophic
1. **Cache line size** is typically 64-128 bytes (4-8 float elements)
2. **Accessing 2048 elements apart means every single access MISSES the cache**
3. **With 2048 iterations per thread, all 2048 accesses are L1 cache misses**
4. **With 4,194,304 threads (2048x2048), this becomes 8.6 billion cache misses**

### Scaling Analysis
| Size | Stride | Expected | Observed |
|------|--------|----------|----------|
| 256x256 | 256 | OK | 23 GFLOPS |
| 512x512 | 512 | OK | 67 GFLOPS |
| 1024x1024 | 1024 | Degraded | 84 GFLOPS |
| 2048x2048 | 2048 | **Catastrophic** | 0.13 GFLOPS |

## Solution: Transpose Matrix B

### Solution 1: Transpose B Before Kernel
1. Change kernel to use B^T instead of B
2. Kernel becomes: `sum += a[index.X, i] * b_transposed[index.Y, i]`
3. This makes both accesses consecutive (coalesced) in memory
4. Expected improvement: **50-100x speedup** for large matrices

### Solution 2: Tiled Kernel with Shared Memory
1. Load tiles of A and B into fast shared memory
2. Compute using shared memory (1000+ GB/s effective bandwidth)
3. Expected improvement: **10-20x speedup**

## Implementation Plan

1. Add optimized kernel that expects B^T
2. Transpose B on GPU before matmul (one-time cost)
3. Use transposed B for the actual multiply
4. For small matrices, skip transpose (overhead not worth it)

## Implementation Results (2025-12-29)

### Bug Fixes Applied

1. **Integer Overflow Bug (CRITICAL)**
   - `int totalOps = m * n * p` overflows for 2048x2048 matrices (8.6B > int.MaxValue)
   - Fix: Changed to `long totalOps = (long)m * n * p`
   - Impact: Without this fix, large matrices silently fell back to CPU!

2. **Transpose-B Optimization**
   - Added optimized kernel that expects B^T (transposed) for coalesced memory access
   - Transpose is performed on GPU before matmul
   - Threshold set to k >= 2048 (smaller matrices don't benefit from transpose overhead)

### Final Benchmark Results

| Size | Before (GFLOPS) | After (GFLOPS) | Improvement |
|------|-----------------|----------------|-------------|
| 256x256 | 23 | 22.50 | Same |
| 512x512 | 67 | 65.02 | Same |
| 1024x1024 | 84 | 86.16 | Same |
| **2048x2048** | **0.13** | **51.90** | **400x faster** |

### Why 2048x2048 Was So Slow

The combination of two bugs caused catastrophic slowdown:

1. **Integer overflow** caused the GPU path to be skipped entirely for large matrices
2. Even on GPU, the naive kernel's strided memory access pattern caused:
   - 8.6 billion cache misses for 2048x2048 matrices
   - Each thread accessed elements 2048 apart in memory
   - L1/L2 cache was completely ineffective

### Solution

For k >= 2048, we now:
1. Allocate a buffer for B^T on GPU
2. Transpose B using a separate kernel: `B[k,n] -> B^T[n,k]`
3. Use optimized kernel: `sum += a[row,i] * bT[col,i]`
4. Both memory accesses are now coalesced (consecutive memory)

For k < 2048, we use the naive kernel because the transpose overhead isn't worth it.
