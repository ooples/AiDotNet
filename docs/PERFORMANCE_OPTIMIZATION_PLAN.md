# Performance Optimization Plan

## Executive Summary

This document outlines a comprehensive plan to address performance bottlenecks identified in the AiDotNet codebase. The bottlenecks affect both CI test execution time and production inference performance.

### Target Metrics
- **Test class runtime**: Under 2 minutes per test class
- **Network construction**: 50-70% reduction in initialization time
- **Forward pass**: 2-5x improvement with SIMD/vectorization
- **Memory**: 30% reduction in peak memory usage during tests

---

## Root Cause Analysis

### 1. Network Construction Bottlenecks

| Network | Layers | Block Config | Construction Issue |
|---------|--------|--------------|-------------------|
| DenseNet-121 | 121 | [6, 12, 24, 16] | Creates 58 internal layers + transitions |
| DenseNet-264 | 264 | [6, 12, 64, 48] | Creates 130 internal layers |
| EfficientNet-B7 | ~800 | 3.1x depth | Very deep with compound scaling |
| ResNet-50 | 50 | 4 stages | 16 bottleneck blocks |
| VGG-19 | 19 | 5 stages | 16 conv layers + 3 FC |

**Root Causes:**
1. **Eager weight allocation**: Every layer allocates weight tensors at construction time
2. **Random initialization**: Weight initialization uses RNG for every parameter
3. **Object graph complexity**: Each layer creates multiple sub-objects

### 2. Forward Pass Bottlenecks

**Root Causes:**
1. **Incomplete layer refactoring**: ~85 layers still use manual loops instead of IEngine operations
2. **Memory access patterns**: Non-contiguous access hurts cache performance
3. **Large input tensors**: 224x224x3 = 150,528 floats per image
4. **Dense connectivity**: DenseNet concatenates ALL previous feature maps

> **Note:** SIMD operations already exist via `TensorPrimitivesHelper<T>` (5-15x speedup for float/double).
> GPU acceleration exists via `IEngine` with cuBLAS/cuDNN bindings. See `GPU_ENGINE_OPTIMIZATION_PLAN.md`.

### 3. Test-Specific Issues

**Tests Creating Multiple Networks:**
- `DenseNet_LargerVariants_HaveMoreLayers`: Creates D121 + D169
- `EfficientNet_LargerVariants_HaveMoreLayers`: Creates B0 + B3
- `EfficientNet_Variants_HaveCorrectResolution`: Creates B0, B1, B2, B3 (4 networks)

**Tests with Large Inputs:**
- `EfficientNetB0_Forward_ProducesCorrectOutputShape`: 224x224 input
- `MobileNet_Forward_*`: 224x224 input

---

## Epic 0: Critical GPU Backend Fixes (BLOCKERS)

> **CRITICAL:** These issues prevent GPU acceleration from working entirely. Users report that
> all GPU backends fail and fall back to CPU. These must be fixed before any other GPU-related
> work can be validated.

### User Story 0.1: Fix OpenCL Attention Kernel Compilation Errors
**As a** user with an OpenCL-capable GPU,
**I want** the OpenCL backend to compile and run correctly,
**So that** I can use GPU acceleration on my hardware.

**Current Issues:**
1. `atomic_add` is ambiguous - float atomics not supported on many OpenCL drivers
2. `-INFINITY` macro usage causes undefined behavior with current floating-point options
3. Result: OpenCL backend marked as unavailable

**Acceptance Criteria:**
- OpenCL attention kernels compile without errors on common drivers (Intel, AMD, NVIDIA)
- Replace `atomic_add` with compatible atomic operations or algorithmic alternatives
- Replace `-INFINITY` with `FLT_MIN` or explicit float constant `-3.402823466e+38f`
- Add fallback paths for hardware without float atomics support

**Files to Modify:**
- `src/AiDotNet.Tensors/Engines/DirectGpu/OpenCL/AttentionKernels.cs`
- `src/AiDotNet.Tensors/Engines/DirectGpu/OpenCL/NormalizationKernels.cs`

**Priority:** P0 - Blocker

---

### User Story 0.2: Fix HIP Backend Header Dependencies
**As a** user with an AMD GPU,
**I want** the HIP backend to compile correctly,
**So that** I can use GPU acceleration on AMD hardware.

**Current Issues:**
1. `hip/hip_runtime.h` header not found (AMD HIP SDK not installed on build machine)
2. HIP backend fails to initialize

**Acceptance Criteria:**
- HIP backend gracefully handles missing HIP SDK at runtime (not compile time)
- Clear error message when HIP SDK is not installed
- Conditional compilation or runtime detection to avoid hard dependency
- Documentation on how to install AMD HIP SDK for HIP support

**Files to Modify:**
- `src/AiDotNet.Tensors/Engines/DirectGpu/HIP/HipBackend.cs`
- Build configuration to make HIP optional

**Priority:** P0 - Blocker

---

### User Story 0.3: Ensure At Least One GPU Backend Works
**As a** user with any GPU,
**I want** at least one GPU backend to work out of the box,
**So that** GPU acceleration is functional without special setup.

**Current Issues:**
- OpenCL fails (kernel compilation errors)
- HIP fails (missing headers)
- CUDA may fail if CUDA toolkit not installed
- Result: All GPU backends fail, system falls back to CPU silently

**Acceptance Criteria:**
- CUDA backend works when NVIDIA GPU and CUDA toolkit present
- OpenCL backend works on any OpenCL 1.2+ capable GPU
- Clear diagnostic output showing which backends are available/unavailable and why
- `IEngine.SupportsGPU` returns true when at least one backend works
- Add `IEngine.GetAvailableBackends()` or similar diagnostic method

**Files to Modify:**
- `src/AiDotNet.Tensors/Engines/DirectGpuTensorEngine.cs`
- `src/AiDotNet.Tensors/Engines/GpuEngine.cs`

**Priority:** P0 - Blocker

---

## Epic 1: Lazy Layer Initialization

### User Story 1.1: Implement Lazy Weight Initialization
**As a** developer running tests,
**I want** network construction to be fast,
**So that** my test feedback loop is under 2 minutes.

**Acceptance Criteria:**
- Weight tensors are not allocated until first `Forward()` call
- Network construction time reduced by 50-70%
- API remains unchanged (no breaking changes)
- Thread-safe lazy initialization

**Implementation Details:**
```csharp
// Before (current)
public DenseLayer(int inputSize, int outputSize, ...)
{
    _weights = new Tensor<T>(inputSize, outputSize); // EXPENSIVE
    InitializeWeights(); // EXPENSIVE
}

// After (lazy)
public DenseLayer(int inputSize, int outputSize, ...)
{
    _inputSize = inputSize;
    _outputSize = outputSize;
    _weights = null; // Deferred
}

public override Tensor<T> Forward(Tensor<T> input)
{
    EnsureInitialized(); // Lazy init on first use
    // ... actual forward pass
}
```

**Files to Modify:**
- `src/NeuralNetworks/Layers/DenseLayer.cs`
- `src/NeuralNetworks/Layers/ConvolutionalLayer.cs`
- `src/NeuralNetworks/Layers/BatchNormalizationLayer.cs`
- `src/NeuralNetworks/Layers/LayerBase.cs` (add `EnsureInitialized` pattern)

**Estimated Impact:** 50-70% faster network construction

---

### User Story 1.2: Add Initialization Strategy Interface
**As a** library developer,
**I want** to control when and how weights are initialized,
**So that** I can optimize for different use cases.

**Acceptance Criteria:**
- New `IInitializationStrategy` interface
- Support for: Lazy, Eager, FromFile, Zero, Custom
- Networks can be configured with strategy at construction

**Implementation:**
```csharp
public interface IInitializationStrategy<T>
{
    void Initialize(ILayer<T> layer);
    bool IsLazy { get; }
}

public class LazyInitialization<T> : IInitializationStrategy<T>
{
    public bool IsLazy => true;
    public void Initialize(ILayer<T> layer) { /* no-op until Forward */ }
}
```

---

## Epic 2: Object Pooling and Tensor Reuse

### User Story 2.1: Implement Tensor Pool
**As a** production user,
**I want** reduced GC pressure during inference,
**So that** my application has consistent latency.

**Acceptance Criteria:**
- `TensorPool<T>` class for renting/returning tensors
- Tensors are cleared before reuse (security)
- Thread-safe implementation
- Configurable pool size limits

**Implementation:**
```csharp
public class TensorPool<T>
{
    private readonly ConcurrentBag<Tensor<T>>[] _pools;

    public Tensor<T> Rent(int[] shape)
    {
        // Find pool by size class, return existing or create new
    }

    public void Return(Tensor<T> tensor)
    {
        tensor.Clear();
        _pools[GetSizeClass(tensor.Length)].Add(tensor);
    }
}
```

**Files to Create:**
- `src/Memory/TensorPool.cs`
- `src/Memory/PooledTensor.cs` (IDisposable wrapper)

**Estimated Impact:** 30% reduction in GC pause time

---

### User Story 2.2: Add Pooling to Forward Pass
**As a** production user running batch inference,
**I want** intermediate tensors to be pooled,
**So that** I avoid allocations per batch.

**Acceptance Criteria:**
- Forward pass uses pooled tensors for intermediates
- `InferenceContext` manages pooled resources
- Opt-in via API (not breaking change)

**Implementation:**
```csharp
using var context = new InferenceContext<T>(pool);
var output = network.Forward(input, context);
// All intermediate tensors returned to pool automatically
```

---

## Epic 3: Test-Specific Mini Networks

### User Story 3.1: Create Lightweight Test Variants
**As a** developer writing unit tests,
**I want** lightweight network variants,
**So that** tests run fast without sacrificing coverage.

**Acceptance Criteria:**
- `DenseNet-Tiny`: [2, 2, 2, 2] block config (8 layers vs 58)
- `EfficientNet-Test`: 1.0 depth/width, 32x32 input
- `ResNet-Micro`: 2 blocks per stage (8 vs 16)
- Factory methods: `DenseNetNetwork.ForTesting()`

**Implementation:**
```csharp
public static class TestNetworkFactory
{
    public static DenseNetNetwork<T> CreateMiniDenseNet<T>(int numClasses)
    {
        var config = new DenseNetConfiguration(
            variant: DenseNetVariant.Custom,
            blockLayers: [2, 2, 2, 2], // Minimal
            growthRate: 8, // Small
            inputHeight: 32, // CIFAR-size
            inputWidth: 32);
        return new DenseNetNetwork<T>(CreateArch(config), config);
    }
}
```

**Test Updates Required:**
- `DenseNetTests.cs`: Use mini variants for non-variant-specific tests
- `EfficientNetTests.cs`: Use test variants for forward pass tests
- `ResNetNetworkTests.cs`: Use micro variants

**Estimated Impact:** 80% faster test execution for affected tests

---

### User Story 3.2: Optimize Multi-Network Tests
**As a** developer,
**I want** tests that compare network variants to share base construction,
**So that** comparison tests don't double construction time.

**Current Problem:**
```csharp
[Fact]
public void DenseNet_LargerVariants_HaveMoreLayers()
{
    var d121 = DenseNetNetwork<float>.DenseNet121(numClasses: 10); // SLOW
    var d169 = DenseNetNetwork<float>.DenseNet169(numClasses: 10); // SLOW x2
    Assert.True(d169.Layers.Count >= d121.Layers.Count);
}
```

**Solution:**
```csharp
[Fact]
public void DenseNet_LargerVariants_HaveMoreLayers()
{
    // Use configuration to get expected layer counts without constructing
    var config121 = new DenseNetConfiguration(DenseNetVariant.DenseNet121);
    var config169 = new DenseNetConfiguration(DenseNetVariant.DenseNet169);

    int expectedLayers121 = config121.GetExpectedLayerCount();
    int expectedLayers169 = config169.GetExpectedLayerCount();

    Assert.True(expectedLayers169 >= expectedLayers121);
}
```

---

## Epic 4: Complete Layer Refactoring to Use IEngine

> **Important:** This epic continues the work defined in `GPU_ENGINE_OPTIMIZATION_PLAN.md` (Phase 4-5).
> The IEngine abstraction already provides SIMD (via TensorPrimitivesHelper), cuBLAS (~30K GFLOPS),
> cuDNN (Winograd, FFT convolution), and fused operations. The work here is to refactor remaining
> layers to use these existing optimized operations instead of manual loops.

### User Story 4.1: Refactor High-Priority Attention Layers
**As a** production user running transformers,
**I want** attention layers to use IEngine operations,
**So that** I get automatic CPU SIMD and GPU acceleration.

**Acceptance Criteria:**
- CrossAttentionLayer uses `Engine.BatchMatMul` and `Engine.Softmax` (currently 20+ manual loops)
- GraphAttentionLayer uses `Engine.ScaledDotProductAttention` (currently 85 manual loops)
- All attention layers use `Engine.FlashAttention` for long sequences (O(N) memory)
- No manual nested loops for attention computation

**Layers to Refactor:**
| Layer | Current Issue | Target IEngine Operations |
|-------|---------------|---------------------------|
| CrossAttentionLayer | Manual 4-nested matmul | `Engine.BatchMatMul`, `Engine.Softmax` |
| GraphAttentionLayer | 85 manual loops | `Engine.ScaledDotProductAttention` |
| MultiHeadAttentionLayer | Needs update | `Engine.FlashAttention` for seq > 256 |
| SelfAttentionLayer | Needs update | `Engine.ScaledDotProductAttention` |

**Estimated Impact:** 2-5x faster attention computation

---

### User Story 4.2: Refactor Graph Neural Network Layers
**As a** production user running GNNs,
**I want** graph layers to use IEngine scatter/gather operations,
**So that** message passing is GPU-accelerated.

**Acceptance Criteria:**
- MessagePassingLayer uses `Engine.ScatterAdd/Mean/Max` (currently 73+ NumOps calls)
- GraphTransformerLayer uses existing IEngine operations (currently 75+ NumOps calls)
- DiffusionConvLayer uses `Engine.Conv` operations (currently 43+ NumOps calls)
- HeterogeneousGraphLayer refactored (30+ NumOps calls)

**Layers to Refactor:**
| Layer | Manual Loop Count | Target IEngine Operations |
|-------|-------------------|---------------------------|
| MessagePassingLayer | 73+ NumOps | `Engine.ScatterAdd/Mean/Max` |
| GraphTransformerLayer | 75+ NumOps | `Engine.BatchMatMul`, `Engine.Softmax` |
| DiffusionConvLayer | 43+ NumOps | `Engine.Conv2D`, `Engine.MatMul` |
| HeterogeneousGraphLayer | 30+ NumOps | `Engine.Scatter*`, `Engine.Gather` |

**Estimated Impact:** 3-10x faster GNN inference

---

### User Story 4.3: Complete Normalization and Pooling Layer Refactoring
**As a** production user,
**I want** all normalization layers to use fused IEngine operations,
**So that** batch/layer/group norm are GPU-accelerated.

**Acceptance Criteria:**
- BatchNormalizationLayer uses `Engine.FusedBatchNorm`
- LayerNormalizationLayer uses `Engine.LayerNorm`
- GroupNormalizationLayer uses `Engine.GroupNorm`
- All layers call `RegisterTrainableParameter` for GPU tensor caching

**Layers to Refactor (Priority 4 from GPU plan):**
- BatchNormalizationLayer
- LayerNormalizationLayer
- GroupNormalizationLayer
- InstanceNormalizationLayer
- SpectralNormalizationLayer

**Estimated Impact:** 20-30% faster normalization operations

---

### User Story 4.4: Expand CpuEngine SIMD Coverage
**As a** developer,
**I want** remaining scalar operations in CpuEngine to use SIMD,
**So that** CPU inference is consistently fast.

**Current Scalar Operations to Vectorize:**
```csharp
// These operations in CpuEngine still use scalar loops:
- Sqrt (uses NumOps.Sqrt in loop)
- Power operations
- Some activation functions (LeakyReLU, GELU, Mish, Swish, ELU)
```

**Implementation:**
- Add vectorized versions to `INumericOperations<T>` interface
- Implement using `System.Numerics.Tensors.TensorPrimitives` for float/double
- Fallback to scalar for other numeric types

**Files to Modify:**
- `src/AiDotNet.Tensors/Helpers/TensorPrimitivesHelper.cs`
- `src/AiDotNet.Tensors/NumericOperations/*.cs`

**Estimated Impact:** 2-5x faster for affected operations

---

## Epic 5: Test Infrastructure Improvements

### User Story 5.1: Add Shared Test Fixtures
**As a** test author,
**I want** shared network fixtures across tests,
**So that** network construction happens once per class.

**Acceptance Criteria:**
- `IClassFixture<NetworkFixture<T>>` for xUnit
- Networks constructed once, reused across tests
- Thread-safe for parallel test execution

**Implementation:**
```csharp
public class NetworkFixture<T> : IDisposable
{
    public DenseNetNetwork<T> DenseNet121 { get; }
    public EfficientNetNetwork<T> EfficientNetB0 { get; }

    public NetworkFixture()
    {
        DenseNet121 = DenseNetNetwork<T>.DenseNet121(numClasses: 10);
        EfficientNetB0 = EfficientNetNetwork<T>.EfficientNetB0(numClasses: 10);
    }
}

public class DenseNetTests : IClassFixture<NetworkFixture<float>>
{
    private readonly NetworkFixture<float> _fixture;

    public DenseNetTests(NetworkFixture<float> fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public void Test_Something()
    {
        var network = _fixture.DenseNet121; // Already constructed
    }
}
```

---

### User Story 5.2: Add Performance Regression Tests
**As a** maintainer,
**I want** automated performance regression detection,
**So that** we don't accidentally slow down the codebase.

**Acceptance Criteria:**
- Benchmark tests using BenchmarkDotNet
- CI integration to detect regressions (>10% slowdown)
- Historical performance tracking

**Implementation:**
```csharp
[MemoryDiagnoser]
public class NetworkConstructionBenchmarks
{
    [Benchmark(Baseline = true)]
    public void DenseNet121_Construction()
    {
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);
    }

    [Benchmark]
    public void DenseNet121_WithLazyInit()
    {
        var network = DenseNetNetwork<float>.DenseNet121(
            numClasses: 10,
            initStrategy: InitializationStrategy.Lazy);
    }
}
```

---

## Implementation Roadmap

> **Gate Requirements:** Each phase MUST pass all integration tests before proceeding to the next phase.
> Run: `dotnet test --filter "Category=Phase{N}Gate"` to validate phase completion.

---

### Phase 0: Critical GPU Backend Fixes (IMMEDIATE)

**User Stories:**
- [ ] 0.1: Fix OpenCL attention kernel compilation errors
- [ ] 0.2: Fix HIP backend header dependencies
- [ ] 0.3: Ensure at least one GPU backend works

**Acceptance Criteria:**
| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| OpenCL compiles | No kernel compilation errors on Intel/AMD/NVIDIA OpenCL | CI build log |
| HIP graceful fallback | HIP backend doesn't crash when SDK missing | Runtime test |
| GPU detection | `GpuEngine.IsAvailable` returns true on GPU systems | Unit test |
| Backend diagnostics | `GetAvailableBackends()` reports status of each backend | Unit test |
| No silent fallback | Log warning when falling back to CPU | Log inspection |

**Performance Requirements:**
| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| GPU backend init | < 500ms | Stopwatch in test |
| First GPU operation | < 100ms overhead vs subsequent | Benchmark |

**Integration Tests (Phase 0 Gate):**
```csharp
[Trait("Category", "Phase0Gate")]
public class Phase0GateTests
{
    [Fact]
    public void OpenCL_Backend_Handles_Gracefully()
    {
        var backends = GpuEngine.GetAvailableBackends();
        var openCl = backends.FirstOrDefault(b => b.BackendType == GpuBackendType.OpenCl);
        // OpenCL should either be available or have clear error message (not crash)
        Assert.NotNull(openCl);
        Assert.True(openCl.IsAvailable || !string.IsNullOrEmpty(openCl.ErrorMessage));
    }

    [Fact]
    public void HIP_Backend_Graceful_When_SDK_Missing()
    {
        // Should not throw, should report appropriate error if unavailable
        var backends = GpuEngine.GetAvailableBackends();
        var hip = backends.FirstOrDefault(b => b.BackendType == GpuBackendType.Hip);
        Assert.NotNull(hip);
        // HIP should be available or have descriptive error message
        Assert.True(hip.IsAvailable || !string.IsNullOrEmpty(hip.ErrorMessage));
    }

    [Fact]
    public void AtLeastOneGpuBackend_IsAvailable_OnGpuSystem()
    {
        // On CI with GPU, at least one backend should work
        // On CPU-only systems, this test is skipped
        if (Environment.GetEnvironmentVariable("HAS_GPU") == "true")
        {
            var backends = GpuEngine.GetAvailableBackends();
            var anyAvailable = backends.Any(b => b.IsAvailable);
            Assert.True(anyAvailable, GpuEngine.GetDiagnosticReport());
        }
    }

    [Fact]
    public void GpuEngine_Reports_Fallback_Reason()
    {
        var report = GpuEngine.GetDiagnosticReport();
        Assert.False(string.IsNullOrEmpty(report));
        // Report should explain why each backend is/isn't available
    }
}
```

**Exit Criteria:** All Phase0Gate tests pass. GPU acceleration works on at least CUDA or OpenCL.

---

### Phase 1: Quick Wins (1-2 weeks)

**User Stories:**
- [ ] 3.1: Create lightweight test variants
- [ ] 3.2: Optimize multi-network tests
- [ ] 5.1: Add shared test fixtures

**Acceptance Criteria:**
| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| Mini networks exist | `DenseNet.ForTesting()`, `EfficientNet.ForTesting()` | API exists |
| Config-only tests | Variant comparison tests don't construct networks | Code review |
| Shared fixtures | `NetworkFixture<T>` implements `IClassFixture` | Compilation |
| Test isolation | Fixtures are thread-safe for parallel execution | Concurrent test run |

**Performance Requirements:**
| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| DenseNetTests class | ~5 min | < 2 min | CI timing |
| EfficientNetTests class | ~8 min | < 2 min | CI timing |
| Mini network construction | N/A | < 50ms | Benchmark |

**Integration Tests (Phase 1 Gate):**
```csharp
[Trait("Category", "Phase1Gate")]
public class Phase1GateTests
{
    [Fact]
    public void MiniDenseNet_Constructs_Under50ms()
    {
        var sw = Stopwatch.StartNew();
        var network = DenseNetNetwork<float>.ForTesting(numClasses: 10);
        sw.Stop();
        Assert.True(sw.ElapsedMilliseconds < 50, $"Took {sw.ElapsedMilliseconds}ms");
        Assert.True(network.Layers.Count < 20); // Much smaller than full D121
    }

    [Fact]
    public void MiniEfficientNet_Constructs_Under50ms()
    {
        var sw = Stopwatch.StartNew();
        var network = EfficientNetNetwork<float>.ForTesting(numClasses: 10);
        sw.Stop();
        Assert.True(sw.ElapsedMilliseconds < 50, $"Took {sw.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void DenseNetConfig_GetExpectedLayerCount_NoConstruction()
    {
        var sw = Stopwatch.StartNew();
        var config121 = new DenseNetConfiguration(DenseNetVariant.DenseNet121);
        var count = config121.GetExpectedLayerCount();
        sw.Stop();
        Assert.True(sw.ElapsedMilliseconds < 5); // Config-only, no network construction
        Assert.True(count > 100);
    }

    [Fact]
    public void NetworkFixture_IsThreadSafe()
    {
        var fixture = new NetworkFixture<float>();
        Parallel.For(0, 10, i =>
        {
            var network = fixture.MiniDenseNet;
            Assert.NotNull(network);
        });
    }
}
```

**Exit Criteria:** All Phase1Gate tests pass. CI test time for DenseNet/EfficientNet < 2 min each.

---

### Phase 2: Core Optimization (2-3 weeks)

**User Stories:**
- [ ] 1.1: Implement lazy weight initialization
- [ ] 1.2: Add initialization strategy interface
- [x] 2.1: Implement tensor pool (TensorPool<T> with Rent/Return, CAS-based thread safety, configurable size limits)

**Acceptance Criteria:**
| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| Lazy init pattern | `EnsureInitialized()` in LayerBase | Code exists |
| Strategy interface | `IInitializationStrategy<T>` with Lazy/Eager/FromFile | API exists |
| Tensor pool | `TensorPool<T>.Rent/Return` thread-safe | Unit test |
| Pool size limits | Configurable max pool size | Constructor param |
| No breaking changes | Existing API works unchanged | Regression tests |

**Performance Requirements:**
| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| DenseNet-121 construction (lazy) | ~500ms | < 150ms | Benchmark |
| DenseNet-264 construction (lazy) | ~1500ms | < 300ms | Benchmark |
| Tensor pool rent/return | N/A | < 1μs | Benchmark |
| GC allocations (pooled) | 100% | < 30% | MemoryDiagnoser |

**Integration Tests (Phase 2 Gate):**
```csharp
[Trait("Category", "Phase2Gate")]
public class Phase2GateTests
{
    [Fact]
    public void LazyInit_DenseNet121_Under150ms()
    {
        var sw = Stopwatch.StartNew();
        var network = DenseNetNetwork<float>.DenseNet121(
            numClasses: 10,
            initStrategy: InitializationStrategy.Lazy);
        sw.Stop();
        Assert.True(sw.ElapsedMilliseconds < 150, $"Took {sw.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void LazyInit_WeightsNotAllocated_UntilForward()
    {
        var network = DenseNetNetwork<float>.DenseNet121(
            numClasses: 10,
            initStrategy: InitializationStrategy.Lazy);

        // Weights should be null before first forward
        var denseLayer = network.Layers.OfType<DenseLayer<float>>().First();
        Assert.False(denseLayer.IsInitialized);

        // After forward, weights should exist
        var input = Tensor<float>.Random(1, 3, 224, 224);
        network.Forward(input);
        Assert.True(denseLayer.IsInitialized);
    }

    [Fact]
    public void TensorPool_ReducesAllocations()
    {
        var pool = new TensorPool<float>(maxPoolSize: 100);
        var allocsBefore = GC.GetTotalMemory(true);

        for (int i = 0; i < 1000; i++)
        {
            var tensor = pool.Rent(new[] { 64, 64 });
            pool.Return(tensor);
        }

        var allocsAfter = GC.GetTotalMemory(true);
        // Should reuse tensors, minimal new allocations
        Assert.True(allocsAfter - allocsBefore < 1_000_000); // < 1MB for 1000 iterations
    }

    [Fact]
    public void TensorPool_IsThreadSafe()
    {
        var pool = new TensorPool<float>();
        var exceptions = new ConcurrentBag<Exception>();

        Parallel.For(0, 100, i =>
        {
            try
            {
                var tensor = pool.Rent(new[] { 32, 32 });
                Thread.Sleep(1); // Simulate work
                pool.Return(tensor);
            }
            catch (Exception ex) { exceptions.Add(ex); }
        });

        Assert.Empty(exceptions);
    }
}
```

**Exit Criteria:** All Phase2Gate tests pass. Network construction 50-70% faster with lazy init.

---

### Phase 3: Layer Refactoring (3-4 weeks)

**User Stories:**
- [ ] 4.1: Refactor high-priority attention layers
- [ ] 4.2: Refactor graph neural network layers
- [ ] 4.3: Complete normalization layer refactoring
- [ ] 2.2: Add pooling to forward pass

**Acceptance Criteria:**
| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| No manual attention loops | CrossAttention uses `Engine.BatchMatMul` | Code review |
| FlashAttention for long seq | Sequences > 256 use `Engine.FlashAttention` | Unit test |
| GNN uses Scatter ops | MessagePassing uses `Engine.ScatterAdd` | Code review |
| Normalization fused | BatchNorm uses `Engine.FusedBatchNorm` | Code review |
| InferenceContext | Pooled forward pass via context | API exists |

**Performance Requirements:**
| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Attention (seq=1024) | Baseline | 2-5x faster | Benchmark |
| GNN message passing | Baseline | 3-10x faster | Benchmark |
| BatchNorm forward | Baseline | 20-30% faster | Benchmark |
| Forward pass allocations | 100% | < 30% with pooling | MemoryDiagnoser |

**Integration Tests (Phase 3 Gate):**
```csharp
[Trait("Category", "Phase3Gate")]
public class Phase3GateTests
{
    [Fact]
    public void CrossAttention_UsesEngineOperations()
    {
        var layer = new CrossAttentionLayer<float>(512, 8);
        var engineCallCount = 0;
        var mockEngine = new InstrumentedEngine(onCall: () => engineCallCount++);
        layer.Engine = mockEngine;

        var q = Tensor<float>.Random(1, 64, 512);
        var kv = Tensor<float>.Random(1, 64, 512);
        layer.Forward(q, kv);

        // Should use Engine.BatchMatMul, not manual loops
        Assert.True(engineCallCount >= 2, "Should call Engine for matmul operations");
    }

    [Fact]
    public void FlashAttention_UsedForLongSequences()
    {
        var layer = new SelfAttentionLayer<float>(512, 8);
        var input = Tensor<float>.Random(1, 1024, 512); // Long sequence

        // Should use FlashAttention path (O(N) memory)
        var memBefore = GC.GetTotalMemory(true);
        layer.Forward(input);
        var memAfter = GC.GetTotalMemory(true);

        // Flash attention should use O(N) not O(N^2) memory
        var memUsed = memAfter - memBefore;
        var maxExpected = 1024 * 512 * 8 * 4; // O(N * d * heads) not O(N^2)
        Assert.True(memUsed < maxExpected, $"Memory {memUsed} exceeds O(N) expectation");
    }

    [Fact]
    public void InferenceContext_ReducesAllocations()
    {
        var network = DenseNetNetwork<float>.ForTesting(numClasses: 10);
        var pool = new TensorPool<float>();
        var input = Tensor<float>.Random(1, 3, 32, 32);

        // Warmup
        network.Forward(input);

        var allocsBefore = GC.GetTotalMemory(true);
        using (var context = new InferenceContext<float>(pool))
        {
            for (int i = 0; i < 100; i++)
            {
                network.Forward(input, context);
            }
        }
        var allocsAfter = GC.GetTotalMemory(true);

        // Pooled inference should have minimal allocations
        Assert.True(allocsAfter - allocsBefore < 10_000_000); // < 10MB for 100 inferences
    }

    [Fact]
    public void Attention_Is2xFasterWithEngine()
    {
        var layer = new SelfAttentionLayer<float>(256, 8);
        var input = Tensor<float>.Random(1, 512, 256);

        // Warmup
        layer.Forward(input);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 10; i++)
            layer.Forward(input);
        sw.Stop();

        // Baseline expectation: should be significantly faster than manual loops
        // This test documents the performance, actual threshold TBD after baseline
        Assert.True(sw.ElapsedMilliseconds < 5000, $"10 forwards took {sw.ElapsedMilliseconds}ms");
    }
}
```

**Exit Criteria:** All Phase3Gate tests pass. Attention/GNN layers use IEngine operations.

---

### Phase 4: Polish and Maintenance (ongoing)

**User Stories:**
- [x] 4.4: Expand CpuEngine SIMD coverage (ReLU, LeakyReLU, GELU, Mish, Swish, ELU across all numeric types)
- [x] 5.2: Add performance regression tests (SimdActivationFunctionBenchmarks.cs, Phase4GateTests.cs)
- [ ] Continue layer refactoring per GPU plan Phase 5

**Acceptance Criteria:**
| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| Sqrt vectorized | `TensorPrimitivesHelper.Sqrt` uses SIMD | Benchmark shows speedup |
| Regression CI | BenchmarkDotNet runs in CI | GitHub Action |
| Baseline tracked | Performance history stored | CI artifacts |
| Alerts on regression | >10% slowdown fails build | CI config |

**Performance Requirements:**
| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| No regressions | < 10% slowdown vs baseline | BenchmarkDotNet comparison |
| Sqrt vectorized | 3-5x faster than scalar | Benchmark |
| CI benchmark time | < 10 min | CI timing |

**Integration Tests (Phase 4 Gate):**
```csharp
[Trait("Category", "Phase4Gate")]
public class Phase4GateTests
{
    [Fact]
    public void Sqrt_IsVectorized()
    {
        var input = Vector<float>.Random(10000);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 1000; i++)
            TensorPrimitivesHelper<float>.Sqrt(input);
        var vectorizedTime = sw.ElapsedMilliseconds;

        // Compare to scalar baseline (should be 3-5x faster)
        // Actual threshold set after measuring scalar baseline
        Assert.True(vectorizedTime < 100, $"Sqrt took {vectorizedTime}ms, expected vectorized speedup");
    }

    [Fact]
    public void PerformanceBaseline_Exists()
    {
        var baselinePath = "benchmarks/baseline.json";
        Assert.True(File.Exists(baselinePath), "Performance baseline file should exist");
    }
}
```

**Exit Criteria:** Regression tests in CI. No performance degradation from baseline.

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| DenseNet test class runtime | ~5 min | <2 min | CI test timing report |
| EfficientNet test class runtime | ~8 min | <2 min | CI test timing report |
| DenseNet-121 construction | ~500ms | <150ms | BenchmarkDotNet |
| Forward pass (224x224, DenseNet-121) | ~2s | <500ms | BenchmarkDotNet |
| Peak memory during tests | TBD | -30% | CI memory profiling |
| GC pauses during inference | TBD | -50% | BenchmarkDotNet |

---

## Dependencies and Risks

### Dependencies
- .NET 8.0 or .NET Framework 4.7.1 (multi-targeting supported with SIMD fallbacks)
- BenchmarkDotNet for performance testing
- Existing IEngine infrastructure (see `GPU_ENGINE_OPTIMIZATION_PLAN.md`)

### Known Issues (CRITICAL)
> **⚠️ GPU acceleration is currently non-functional due to these issues:**

1. **OpenCL Kernel Compilation Failures** (Epic 0, Story 0.1)
   - `atomic_add` ambiguous - float atomics not supported on many OpenCL drivers
   - `-INFINITY` macro undefined behavior
   - Status: OpenCL backend marked as unavailable

2. **HIP Backend Missing Dependencies** (Epic 0, Story 0.2)
   - `hip/hip_runtime.h` header not found
   - AMD HIP SDK required but not documented
   - Status: HIP backend fails to initialize

3. **No Working GPU Backend** (Epic 0, Story 0.3)
   - All GPU backends fail, silent fallback to CPU
   - Users unaware GPU acceleration is not working

### Risks
1. **API Breaking Changes**: Lazy initialization changes constructor semantics
   - Mitigation: Make lazy opt-in via strategy parameter
2. **Thread Safety**: Lazy initialization must be thread-safe
   - Mitigation: Use `LazyInitializer.EnsureInitialized` pattern
3. **Layer Refactoring Scope**: 85+ layers need updates to use IEngine
   - Mitigation: Prioritize by usage frequency and performance impact
4. **GPU Backend Portability**: Different GPU vendors have different capabilities
   - Mitigation: Runtime capability detection, graceful fallbacks
   - Mitigation: Follow patterns established in Phase 3 of GPU plan

### Related Documentation
- `GPU_ENGINE_OPTIMIZATION_PLAN.md` - Comprehensive GPU optimization plan (Phase 1-3 complete)
- `src/AiDotNet.Tensors/Engines/IEngine.cs` - Core compute abstraction interface
- `src/AiDotNet.Tensors/Helpers/TensorPrimitivesHelper.cs` - SIMD operations for float/double

---

## Appendix: Affected Test Classes

| Test Class | Issue | Priority |
|------------|-------|----------|
| `DenseNetTests` | Creates D121, D169, D201, D264 | High |
| `EfficientNetTests` | Creates B0-B7, forward with 224x224 | High |
| `ResNetNetworkTests` | Creates ResNet-18/34/50, forward | Medium |
| `VGGNetworkTests` | Creates VGG-11/13/16/19, forward | Medium |
| `MobileNetTests` | Forward with 224x224 | Medium |
| `BlipNeuralNetworkTests` | Constructor validation only | Low |
| `Blip2NeuralNetworkTests` | Constructor validation only | Low |
| `ClipNeuralNetworkTests` | Constructor validation only | Low |
