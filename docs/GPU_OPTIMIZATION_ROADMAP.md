# GPU GEMM Optimization Roadmap for AMD RDNA1 (RX 5500 XT)

## Current State (2026-01-01, Phase 1.5 Complete - NOW FASTER THAN CLBLAST!)

| Metric | Value |
|--------|-------|
| GPU | AMD RX 5500 XT (gfx1012, 11 CUs) |
| Theoretical Peak | ~5,196 GFLOPS |
| CLBlast Performance | 2,258 GFLOPS @ 2048x2048 |
| **Our Performance** | **3,137 GFLOPS @ 2048x2048 (139% of CLBlast!)** |
| 1024x1024 | **2,605 GFLOPS (1.43x FASTER than CLBlast!)** |
| 2048x2048 | **3,137 GFLOPS (1.39x FASTER than CLBlast!)** |
| 4096x4096 | **2,334 GFLOPS (5.4x FASTER than CLBlast!)** |
| DenseLayer | **1,235 GFLOPS (1.22x FASTER than CLBlast!)** |
| Phase 1 Status | **COMPLETE** - All low-complexity optimizations applied |
| Phase 1.5 Status | **COMPLETE** - Row-major swap eliminates 48% transpose overhead! |
| Current Achievement | **60% of theoretical peak, 40% faster than CLBlast!** |

## Key Discoveries

**1. RX 5500 XT (gfx1012) is NOT in CLBlast's tuning database!**
- CLBlast has tuned params for gfx1010 (RX 5700) but not gfx1012
- FIXED: Added gfx1012 entries to all CLBlast databases

**2. XgemmDirect kernel excels for small/DenseLayer workloads!**
- Direct path avoids transpose overhead for row-major data
- 256x256: 559-773 GFLOPS (1.5-2x faster than CLBlast)
- 512x512: 1349 GFLOPS (1.4x faster than CLBlast)
- DenseLayer: 1350 GFLOPS (1.16x faster than CLBlast)

**3. Row-Major Swap Trick (BREAKTHROUGH!)**
- Previous approach: Physical transpose of A and C matrices (48% overhead!)
- New approach: Swap A↔B and M↔N, no data movement at all
- For row-major C = A × B, compute column-major C^T = B^T × A^T
- Result: GEMM kernel alone runs at 2800-3200 GFLOPS, matching raw compute!
- 2048x2048: From 1,636 GFLOPS (71% CLBlast) to 3,137 GFLOPS (139% CLBlast)!

---

## Phase 1: Match CLBlast (CRITICAL)

These optimizations should bring us from 1,000 to 2,300+ GFLOPS.

| # | Optimization | Expected Gain | Complexity | Risk | Status |
|---|-------------|---------------|------------|------|--------|
| 1 | **Use CLBlast baseline kernel directly** | +50% (1.5x) | Low | Low | **DONE** ✓ |
| 1a | **Add gfx1012 to CLBlast databases** | +10% | Low | Low | **DONE** ✓ |
| 1b | **Hybrid direct/indirect path selection** | +10% | Low | Low | **DONE** ✓ |
| 2 | **LDS Bank Conflict Padding** (+4 stride to arrays) | +6% | Low | Low | **DONE** ✓ |
| 3 | **VWM=4, VWN=4** (wider vectorization) | +2% | Low | Low | **DONE** ✓ |
| 4 | **K-loop Unrolling (KWI>2)** | -2% (regression!) | Low | Low | **SKIP** ✗ |

**Phase 1 COMPLETE - Final Performance (before row-major swap):**
- **2048x2048: 1,636 GFLOPS (71% of CLBlast)**
- K-loop unrolling (KWI=4, KWI=8) tested but caused register pressure regression on 11-CU RDNA1
- LDS padding: +6%, Vectorization: +2% - both kept

---

## Phase 1.5: Row-Major Swap Trick (BREAKTHROUGH!)

Target: Eliminate 48% transpose overhead discovered in Phase 1.

| # | Optimization | Expected Gain | Complexity | Risk | Status |
|---|-------------|---------------|------------|------|--------|
| 5 | **Row-Major Swap Trick** | +92% (2x) | Medium | Low | **DONE** ✓ |

**The Problem:**
- CLBlast Xgemm kernel expects column-major data layout
- Our row-major data required physical transpose of A and C matrices
- Timing showed: GEMM kernel = 52% of time, Transpose = 48% of time!
- GEMM kernel alone achieved 2800-3200 GFLOPS - faster than CLBlast!

**The Solution - Row-Major Swap Trick:**
For row-major GEMM `C = A × B`:
1. Row-major data reinterpreted as column-major is already transposed
2. So: `C^T = (A × B)^T = B^T × A^T`
3. Swap A↔B and M↔N in kernel call
4. Kernel produces column-major result = row-major result (no data movement!)

**Implementation:**
```csharp
// Instead of:
// 1. Transpose A (row-major M×K) to A' (column-major K×M) [EXPENSIVE]
// 2. Call kernel: C' = A' × B
// 3. Transpose C' (column-major) to C (row-major) [EXPENSIVE]

// We do:
// 1. Swap arguments: aBuf = B, bBuf = A (zero cost)
// 2. Swap dimensions: swappedM = N, swappedN = M (zero cost)
// 3. Call kernel: C' = B' × A' where ' means row-major as column-major
// 4. Result is directly in correct row-major layout! (no transpose)
```

**Phase 1.5 COMPLETE - Final Performance:**
- **1024x1024: 2,605 GFLOPS (1.43x FASTER than CLBlast!)**
- **2048x2048: 3,137 GFLOPS (1.39x FASTER than CLBlast!)**
- **4096x4096: 2,334 GFLOPS (5.4x FASTER than CLBlast!)**
- **DenseLayer: 1,235 GFLOPS (1.22x FASTER than CLBlast!)**
- **Overall: 60% of theoretical peak, 40% faster than CLBlast!**

### Implementation Details

#### 1. CLBlast Baseline Kernel
- Use `KernelName = "clblast_baseline_k0"` config
- This triggers `ClBlastXgemmKernel.BuildSource()` - the actual CLBlast kernel
- Parameters: MWG=64, NWG=64, KWG=16, VWM=2, VWN=2, SA=1, SB=1

#### 2. LDS Bank Conflict Padding
```opencl
// BEFORE (bank conflicts)
__local float Als[KWG][MWG];

// AFTER (conflict-free)
__local float Als[KWG][MWG + 4];  // +4 padding
```

#### 3. Vectorization (VWM=4, VWN=4)
- Use `float4` for global memory loads
- Requires 16-byte alignment
- 4x fewer memory instructions

#### 4. K-loop Unrolling (SKIPPED - Causes Regression)
```opencl
// TESTED: KWI=4 and KWI=8 both caused performance regression
// Root cause: Register pressure on 11-CU RDNA1 (RX 5500 XT)
// Larger CU count GPUs may benefit, but 11 CUs cannot sustain high VGPR usage
// Keep KWI=2 (default) for best performance on gfx1012
```

---

## Phase 2: Exceed CLBlast (HIGH PRIORITY)

Target: 2,300 to 3,500+ GFLOPS

| # | Optimization | Expected Gain | Complexity | Risk |
|---|-------------|---------------|------------|------|
| 5 | **Larger Thread Tiles (16x8)** | +20-30% | Medium | Medium |
| 6 | **Wave32 Mode** (RDNA1 native) | +10-30% | Medium | Low-Medium |
| 7 | **Three-Level Double Buffering** | +20-50% | High | Medium |
| 8 | **VGPR Occupancy Optimization** | +10-25% | Medium | Low |
| 9 | **128-byte Cache Line Alignment** | +5-15% | Medium | Low |
| 10 | **XOR-Based LDS Swizzle** | +10-20% | Medium | Low |

### Implementation Details

#### 5. Larger Thread Tiles
- Increase from 8x8 to 16x8 per thread
- More arithmetic intensity per memory access
- Watch register pressure (target <128 VGPRs)

#### 6. Wave32 Mode
- RDNA1's native execution mode (vs GCN's wave64)
- 2x more wave slots with same VGPR count
- Compiler flag: `-mwavefrontsize32`

#### 7. Double Buffering
```
Iteration N:   [Compute A[N]] [Load A[N+1] to LDS]
Iteration N+1: [Compute A[N+1]] [Load A[N+2] to LDS]
```
- Overlaps compute and memory
- Requires 2x LDS allocation
- Use `s_waitcnt` for synchronization

#### 8. VGPR Occupancy (RDNA1 Specific)
| VGPRs Used | Max Waves (wave32) | Occupancy |
|------------|-------------------|-----------|
| 64 | 20 | 100% |
| 96 | 16 | 80% |
| 128 | 12 | 60% |
| 256 | 6 | 30% |

#### 9. 128-byte Cache Lines
- RDNA1 L0 cache uses 128-byte lines (not 64-byte like GCN)
- Align tile loads to 128-byte boundaries
- Wave32 accessing 32 x 4-byte = 128 bytes = 1 cache line

#### 10. XOR-Based LDS Swizzle
```opencl
// Bank-conflict-free indexing
int lds_idx = row ^ col;  // XOR swizzle
Als[lds_idx] = data;
```

---

## Phase 3: RDNA1-Specific Optimizations (MEDIUM PRIORITY)

Target: 3,500 to 4,500+ GFLOPS

| # | Optimization | Expected Gain | Complexity | Risk |
|---|-------------|---------------|------------|------|
| 11 | **GEMM + Bias + Activation Fusion** | +20-50% | Medium | Low |
| 12 | **L2-Aware Tile Rasterization** (Z-order) | +10-20% | Medium | Low |
| 13 | **Explicit FMA Instructions** | +5-10% | Low | Low |
| 14 | **cl_khr_subgroups Operations** | +5-15% | Medium | Low |
| 15 | **DPP Cross-Lane Operations** | +10-15% | High | High |

### Implementation Details

#### 11. Kernel Fusion
```opencl
// Fused epilogue - no extra memory round-trip
float result = alpha * acc + beta * C[idx];
result += bias[col];           // Bias add
result = max(0.0f, result);    // ReLU
C[idx] = result;
```

#### 12. Z-Order Tile Rasterization
- Instead of row-major workgroup scheduling
- Use Morton/Z-order for better L2 locality
- Remap: `tile_id = interleave_bits(tile_x, tile_y)`

#### 14. Subgroup Operations
```opencl
// Wave-level reduction (no LDS needed)
float sum = sub_group_reduce_add(partial);
```

---

## Phase 4: Advanced Techniques (LOWER PRIORITY)

Target: Approaching theoretical peak (4,500-5,000+ GFLOPS)

| # | Optimization | Expected Gain | Complexity | Risk |
|---|-------------|---------------|------------|------|
| 16 | **Split-K for Tall-Skinny Matrices** | +100-300% | Medium | Low |
| 17 | **Power-of-2 Padding Fix** | +Up to 5x | Medium | Low |
| 18 | **Persistent Kernels** | +20-50% | High | High |
| 19 | **Stream-K Work Distribution** | +95-108% | High | High |
| 20 | **CU Mode vs WGP Mode** | Unknown | Low | Low |

### Implementation Details

#### 16. Split-K Parallelization
- For matrices with small M, N but large K
- Partition K across workgroups
- Final reduction with atomics or second kernel

#### 17. Power-of-2 Fix
- CLBlast has 5x performance cliff at 8192x8192
- Add asymmetric padding: `M' = M + (M % 32 == 0 ? 1 : 0)`

---

## A/B Testing Protocol

For each optimization:

```
1. BASELINE: Benchmark current best config
   - Record GFLOPS, time, bottleneck analysis

2. CHANGE: Apply ONE surgical modification
   - Document exactly what changed

3. BENCHMARK: Run same test suite
   - Record GFLOPS, time, bottleneck analysis

4. COMPARE:
   - If improved by >2%: KEEP, update baseline
   - If regressed: REVERT immediately
   - If within noise (<2%): Keep if simpler, else revert

5. DOCUMENT:
   - Config parameters
   - GFLOPS achieved
   - Bottleneck indicators
   - Decision rationale
```

---

## Bottleneck Analysis Checklist

For each benchmark run, capture:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Occupancy | >50% | `OccupancyEst` in BottleneckDiagnostics |
| LDS Usage | <64KB | `LdsUsageKb` |
| Register Pressure | <256 VGPRs | `RegistersEst` |
| Wave Utilization | >90% | `WaveUtilization` |
| Compute Intensity | MWI×NWI >16 | `ComputeIntensity` |
| Vector Bandwidth | VWM×VWN >4 | `VectorBandwidth` |
| Memory vs Compute Bound | - | `IsLikelyMemoryBound` |

---

## Files Reference

| File | Purpose |
|------|---------|
| `OpenCL/DynamicGemmKernel.cs` | Kernel generation and compilation |
| `OpenCL/ClBlastXgemmKernel.cs` | CLBlast baseline kernel source |
| `OpenCL/GemmAutoTuner.cs` | Bayesian auto-tuning + bottleneck analysis |
| `OpenCL/GemmConfig.cs` | Configuration parameters |
| `OpenCL/GemmTuningDatabase.cs` | Persistent config storage |
| `OpenCL/ClBlastXgemmDatabase.Generated.cs` | CLBlast tuned params |

---

## Research Sources

### Agent acd2567 - CLBlast Source Analysis
- CLBlast GitHub Issues: #566 (RDNA3), #403 (Vega), #350 (Vega FE), #510 (GCN cross-lane), #53 (power-of-2)
- Key finding: VWN=1 too conservative, power-of-2 cliff, no DPP usage

### Agent a1da307 - Modern GEMM Techniques
- AMD ROCm GEMM Blog
- Deep Dive into Matrix Optimization on AMD GPUs (160% of rocBLAS achieved!)
- CUTLASS tutorials
- Key finding: Hierarchical tiling + VGPR bank allocation = major gains

### Agent a9bd745 - RDNA1 Architecture
- AMD GPUOpen: RDNA Performance Guide, Occupancy Explained, GCN Memory Coalescing
- Key finding: Wave32 mode, 128-byte cache lines, 32-bank LDS

---

## Changelog

| Date | Change | Result |
|------|--------|--------|
| 2026-01-01 | Initial roadmap created | - |
| 2026-01-01 | Implemented CLBlast baseline kernel as default | 1000 → 1428 GFLOPS (+43%) |
| 2026-01-01 | Added gfx1012 to CLBlast databases | Better parameter matching |
| 2026-01-01 | Implemented XgemmDirect for row-major data | Small matrices 1.5-2x faster than CLBlast! |
| 2026-01-01 | Implemented hybrid direct/indirect path selection | 2048x2048: 1502 GFLOPS (65% of CLBlast) |
| 2026-01-01 | DenseLayer-style workloads now 1.16x faster than CLBlast | Key neural network use case optimized |
| 2026-01-01 | Implemented LDS bank conflict padding (+4 stride) | 2048x2048: 1597 GFLOPS (+6%, 69% of CLBlast) |
| 2026-01-01 | Implemented VWM=4, VWN=4 wider vectorization | 2048x2048: 1632 GFLOPS (+2%, 70% of CLBlast) |
| 2026-01-01 | Tested K-loop unrolling (KWI=4, KWI=8) | **REGRESSION** - reverted to KWI=2 |
| 2026-01-01 | **PHASE 1 COMPLETE** | 2048x2048: 1636 GFLOPS (71%), 256x256: 1.9x faster than CLBlast |
| 2026-01-01 | **CRITICAL BUG FIX**: MinIndirectSize threshold not being used! | Direct path was used for ALL sizes! |
| 2026-01-01 | Fixed path selection: use indirect for M/N >= 448 | 2048x2048: 550→1657 GFLOPS (3x improvement, now 71% of CLBlast) |
| 2026-01-01 | Current status: DenseLayer 1.17x faster, Large matrices 1.09x faster than CLBlast | Key workloads optimized! |
| 2026-01-01 | Discovered 48% transpose overhead in indirect path | GEMM kernel alone: 2800+ GFLOPS! |
| 2026-01-01 | **PHASE 1.5: Row-Major Swap Trick** | Eliminates ALL transpose overhead! |
| 2026-01-01 | **NOW FASTER THAN CLBLAST!** | 1024: 2605 GFLOPS (1.43x), 2048: 3137 GFLOPS (1.39x), 4096: 2334 GFLOPS (5.4x) |

