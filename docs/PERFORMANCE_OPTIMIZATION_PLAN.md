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

### Phase 1: Quick Wins (1-2 weeks)
- [ ] User Story 3.1: Create lightweight test variants
- [ ] User Story 3.2: Optimize multi-network tests
- [ ] User Story 5.1: Add shared test fixtures

**Expected Impact:** 50% reduction in test time

### Phase 2: Core Optimization (2-3 weeks)
- [ ] User Story 1.1: Implement lazy weight initialization
- [ ] User Story 1.2: Add initialization strategy interface
- [ ] User Story 2.1: Implement tensor pool

**Expected Impact:** Additional 30% reduction in test time, 30% memory reduction

### Phase 3: Layer Refactoring (3-4 weeks)
- [ ] User Story 4.1: Refactor high-priority attention layers (CrossAttention, GraphAttention)
- [ ] User Story 4.2: Refactor graph neural network layers (MessagePassing, GraphTransformer)
- [ ] User Story 4.3: Complete normalization layer refactoring
- [ ] User Story 2.2: Add pooling to forward pass

**Expected Impact:** 2-5x faster inference via existing IEngine optimizations

### Phase 4: Polish and Maintenance (ongoing)
- [ ] User Story 4.4: Expand CpuEngine SIMD coverage for remaining scalar ops
- [ ] User Story 5.2: Add performance regression tests
- [ ] Continue layer refactoring per `GPU_ENGINE_OPTIMIZATION_PLAN.md` Phase 5

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
- .NET 8.0 for latest SIMD intrinsics (already in use)
- BenchmarkDotNet for performance testing
- Existing IEngine infrastructure (see `GPU_ENGINE_OPTIMIZATION_PLAN.md`)

### Risks
1. **API Breaking Changes**: Lazy initialization changes constructor semantics
   - Mitigation: Make lazy opt-in via strategy parameter
2. **Thread Safety**: Lazy initialization must be thread-safe
   - Mitigation: Use `LazyInitializer.EnsureInitialized` pattern
3. **Layer Refactoring Scope**: 85+ layers need updates to use IEngine
   - Mitigation: Prioritize by usage frequency and performance impact
   - Mitigation: Follow patterns established in Phase 3 of GPU plan (DenseLayer, ConvolutionalLayer)

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
