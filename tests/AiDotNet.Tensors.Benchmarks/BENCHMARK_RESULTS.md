# AiDotNet Linear Algebra Benchmark Results

**Date:** 2025-12-28
**System:** Windows 11, AMD Ryzen 7 4800H @ 2.90GHz, 16 logical / 8 physical cores
**Runtime:** .NET 8.0.22 (RyuJIT x86-64-v3, AVX2)

## Libraries Compared

| Library | Version | Description |
|---------|---------|-------------|
| **AiDotNet** | Current | This library |
| **MathNet.Numerics** | 5.0.0 | Popular numerical computing library |
| **NumSharp** | 0.30.0 | NumPy-like library for .NET |
| **TensorPrimitives** | 10.0.1 | .NET built-in SIMD primitives |

---

## Vector Operations (N=100, 500, 1000)

### Dot Product

| Size | AiDotNet | MathNet | TensorPrimitives | Winner |
|------|----------|---------|------------------|--------|
| 100  | 30.8 ns  | 67.8 ns | 17.2 ns          | TensorPrimitives (1.8x faster than AiDotNet) |
| 500  | 100.5 ns | 351.3 ns | 75.7 ns         | TensorPrimitives (1.3x faster than AiDotNet) |
| 1000 | 189.3 ns | 709.1 ns | 165.6 ns        | TensorPrimitives (1.1x faster than AiDotNet) |

**AiDotNet vs MathNet:** 2.2x to 3.7x faster

### L2 Norm

| Size | AiDotNet | MathNet | TensorPrimitives | Winner |
|------|----------|---------|------------------|--------|
| 100  | 15.2 ns  | 1,010 ns | 18.3 ns         | **AiDotNet** (66x faster than MathNet) |
| 500  | 82.5 ns  | 5,049 ns | 73.9 ns         | TensorPrimitives (1.1x faster than AiDotNet) |
| 1000 | 172.0 ns | 10,029 ns | 162.8 ns       | TensorPrimitives (1.1x faster than AiDotNet) |

**AiDotNet vs MathNet:** 58x to 66x faster

### Vector Add

| Size | AiDotNet | MathNet | TensorPrimitives | NumSharp | Winner |
|------|----------|---------|------------------|----------|--------|
| 100  | 274 ns   | 106 ns  | 19.6 ns          | 4,649 ns | TensorPrimitives |
| 500  | 1,088 ns | 483 ns  | 59.0 ns          | 8,884 ns | TensorPrimitives |
| 1000 | 2,193 ns | 703 ns  | 107 ns           | 12,879 ns| TensorPrimitives |

**Note:** TensorPrimitives wins here because it does in-place operations with no allocation.

---

## Matrix Operations (N=100, 500, 1000)

### Frobenius Norm

| Size | AiDotNet | MathNet | Speedup |
|------|----------|---------|---------|
| 100x100   | 1.9 us   | 248 us  | **130x faster** |
| 500x500   | 49.2 us  | 11,128 us | **226x faster** |
| 1000x1000 | 532 us   | 104,320 us | **196x faster** |

**AiDotNet is dramatically faster for Frobenius norm**

### Matrix Multiply

| Size | AiDotNet | MathNet | NumSharp | AiDotNet vs MathNet |
|------|----------|---------|----------|---------------------|
| 100x100   | 3,120 us | 208 us  | 24,983 us | MathNet 15x faster |
| 500x500   | 581 ms   | 10.6 ms | 3,277 ms  | MathNet 55x faster |
| 1000x1000 | 4,653 ms | 88 ms   | 26,057 ms | MathNet 53x faster |

**Note:** MathNet uses highly optimized BLAS routines. AiDotNet's naive O(n^3) implementation needs optimization.

### Matrix Add

| Size | AiDotNet | MathNet | NumSharp | Winner |
|------|----------|---------|----------|--------|
| 100x100   | 5.8 us  | 8.2 us  | 31.1 us  | **AiDotNet** (1.4x faster) |
| 500x500   | 657 us  | 683 us  | 718 us   | **AiDotNet** (1.04x faster) |
| 1000x1000 | 2,530 us| 2,542 us| 1,763 us | NumSharp (in-place op) |

### Matrix Transpose

| Size | AiDotNet | MathNet | NumSharp | Winner |
|------|----------|---------|----------|--------|
| 100x100   | 19.9 us | 12.4 us | 115 us   | MathNet (1.6x faster) |
| 500x500   | 937 us  | 704 us  | 2,969 us | MathNet (1.3x faster) |
| 1000x1000 | 4,014 us| 3,120 us| 11,756 us| MathNet (1.3x faster) |

---

## Small Matrix Multiply (Fixed Sizes)

| Size | AiDotNet | MathNet | NumSharp | Notes |
|------|----------|---------|----------|-------|
| 4x4   | 199 ns  | 159 ns  | 2,104 ns | MathNet 1.25x faster |
| 16x16 | 11.4 us | 2.6 us  | 97.7 us  | MathNet 4.3x faster |
| 32x32 | 91.2 us | 20.6 us | 740 us   | MathNet 4.4x faster |

### Memory Allocation (Small Matrix Multiply)

| Size | AiDotNet | MathNet | NumSharp | Winner |
|------|----------|---------|----------|--------|
| 4x4   | 184 B   | 624 B   | 5,392 B  | **AiDotNet** (3.4x less) |
| 16x16 | 2,104 B | 4,944 B | 271 KB   | **AiDotNet** (2.4x less) |
| 32x32 | 8,248 B | 32 KB   | 2.1 MB   | **AiDotNet** (3.9x less) |

---

## Summary

### Where AiDotNet Excels

| Operation | vs MathNet | Notes |
|-----------|------------|-------|
| Frobenius Norm | **130-226x faster** | Zero allocation |
| L2 Norm | **58-66x faster** | SIMD optimized |
| Dot Product | **2-4x faster** | SIMD optimized |
| Matrix Add | **1.4x faster** | Comparable |
| Memory Allocation | **2-4x less** | Efficient memory usage |

### Where Optimization is Needed

| Operation | vs MathNet | Recommendation |
|-----------|------------|----------------|
| Matrix Multiply | 15-55x slower | Implement Strassen or BLAS bindings |
| Matrix Transpose | 1.3-1.6x slower | Cache-aware blocking |
| Vector Add | 2.5x slower | In-place operations |

### Overall Assessment

AiDotNet shows **excellent performance** for:
- Norm calculations (Frobenius, L2) - orders of magnitude faster
- Dot products - 2-4x faster with zero allocation
- Memory efficiency - consistently lower allocations

**Matrix multiplication** is the primary bottleneck and would benefit from:
1. Strassen algorithm for large matrices
2. Cache-blocking for better memory access patterns
3. Optional native BLAS bindings (like MathNet uses)

---

## Raw Results

See the detailed BenchmarkDotNet reports:
- `BenchmarkDotNet.Artifacts/results/AiDotNet.Tensors.Benchmarks.LinearAlgebraBenchmarks-report-github.md`
- `BenchmarkDotNet.Artifacts/results/AiDotNet.Tensors.Benchmarks.SmallMatrixBenchmarks-report-github.md`
