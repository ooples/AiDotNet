# GPU GEMM Optimization Roadmap for AMD RDNA1 (RX 5500 XT)

## Current State (2026-01-01, Updated)

| Metric | Value |
|--------|-------|
| GPU | AMD RX 5500 XT (gfx1012, 11 CUs) |
| Theoretical Peak | ~5,196 GFLOPS |
| CLBlast Performance | 2,315 GFLOPS @ 2048x2048 |
| Our CLBlast Baseline | **1,502 GFLOPS @ 2048x2048 (65%)** |
| Small Matrices (256-512) | **AiDotNet is 1.4-2x FASTER than CLBlast!** |
| DenseLayer (64x3072x768) | **AiDotNet is 1.16x FASTER than CLBlast!** |
| Target | Match CLBlast @ 2048x2048, maintain lead on small/DenseLayer |

## Key Discoveries

**1. RX 5500 XT (gfx1012) is NOT in CLBlast's tuning database!**
- CLBlast has tuned params for gfx1010 (RX 5700) but not gfx1012
- FIXED: Added gfx1012 entries to all CLBlast databases

**2. XgemmDirect kernel excels for small/DenseLayer workloads!**
- Direct path avoids transpose overhead for row-major data
- 256x256: 559-773 GFLOPS (1.5-2x faster than CLBlast)
- 512x512: 1349 GFLOPS (1.4x faster than CLBlast)
- DenseLayer: 1350 GFLOPS (1.16x faster than CLBlast)

**3. Indirect Xgemm path still needed for large square matrices**
- 2048x2048: Direct gives 1428, Indirect gives 1502 GFLOPS
- Hybrid approach: Use direct for small/medium/non-square, indirect for 2048-class matrices

---

## Phase 1: Match CLBlast (CRITICAL)

These optimizations should bring us from 1,000 to 2,300+ GFLOPS.

| # | Optimization | Expected Gain | Complexity | Risk | Status |
|---|-------------|---------------|------------|------|--------|
| 1 | **Use CLBlast baseline kernel directly** | +50% (1.5x) | Low | Low | **DONE** ✓ |
| 1a | **Add gfx1012 to CLBlast databases** | +10% | Low | Low | **DONE** ✓ |
| 1b | **Hybrid direct/indirect path selection** | +10% | Low | Low | **DONE** ✓ |
| 2 | **LDS Bank Conflict Padding** (+4 stride to arrays) | +10-30% | Low | Low | NEXT |
| 3 | **VWM=4, VWN=4** (wider vectorization) | +15-25% | Low | Low | TODO |
| 4 | **Full K-loop Unrolling** (`#pragma unroll`) | +10% | Low | Low | TODO |

**Current Performance (after Phase 1.1):**
- 2048x2048: 1,502 GFLOPS (65% of CLBlast, up from 43%)
- Still need ~35% more to match CLBlast at 2048x2048

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

#### 4. K-loop Unrolling
```opencl
#pragma unroll 8
for (int k = 0; k < KWG; k++) {
    // FMA operations
}
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

