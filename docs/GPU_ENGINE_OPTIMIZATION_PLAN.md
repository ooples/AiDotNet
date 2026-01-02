# GPU Engine Optimization Plan

## Overview

This document outlines the plan to move GPU-specific optimizations from individual layers into the `IEngine` interface, keeping layers backend-agnostic while enabling GPU acceleration.

## Goals

1. **Layers stay backend-agnostic** - No GPU/CUDA/OpenCL code in layer implementations
2. **Engine owns optimization decisions** - Engine chooses best implementation based on available hardware
3. **Reduce code duplication** - Common patterns handled in `LayerBase<T>`
4. **Type-safe API** - Use enums instead of strings, clear parameter purposes

## Proposed IEngine Additions

### 1. Persistent Tensor Registration

```csharp
/// <summary>
/// Hint to the engine that a tensor will be reused across many operations.
/// Engine may choose to cache it on GPU memory for faster access.
/// </summary>
void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role);
void UnregisterPersistentTensor<T>(Tensor<T> tensor);
```

**PersistentTensorRole Enum:**
```csharp
public enum PersistentTensorRole
{
    /// <summary>Layer weights that change only during training updates.</summary>
    Weights,

    /// <summary>Layer biases that change only during training updates.</summary>
    Biases,

    /// <summary>Normalization parameters (gamma/beta for BatchNorm, etc.).</summary>
    NormalizationParams,

    /// <summary>Embedding lookup tables.</summary>
    Embeddings,

    /// <summary>Attention key/value caches for inference.</summary>
    AttentionCache,

    /// <summary>Other persistent tensors not fitting above categories.</summary>
    Other
}
```

### 2. Fused Operations

```csharp
/// <summary>
/// Fused linear transformation: output = activation(input @ weights.T + bias)
/// Engine chooses optimal implementation (fused GPU kernel, cuBLAS, or CPU).
/// </summary>
Tensor<T> FusedLinear<T>(
    Tensor<T> input,
    Tensor<T> weights,
    Tensor<T>? bias,
    FusedActivationType activation);

/// <summary>
/// Fused batch normalization with optional activation.
/// </summary>
Tensor<T> FusedBatchNorm<T>(
    Tensor<T> input,
    Tensor<T> gamma,
    Tensor<T> beta,
    Tensor<T> runningMean,
    Tensor<T> runningVar,
    float epsilon,
    bool training,
    FusedActivationType activation);

/// <summary>
/// Fused convolution with optional bias and activation.
/// </summary>
Tensor<T> FusedConv2D<T>(
    Tensor<T> input,
    Tensor<T> kernel,
    Tensor<T>? bias,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    FusedActivationType activation);
```

**FusedActivationType Enum:**
```csharp
public enum FusedActivationType
{
    None,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    Swish,
    Softmax
}
```

### 3. LayerBase Integration

```csharp
// In LayerBase<T>
protected void RegisterTrainableParameter(Tensor<T> tensor, PersistentTensorRole role)
{
    Engine.RegisterPersistentTensor(tensor, role);
    _registeredTensors.Add(tensor);
}

protected override void Dispose(bool disposing)
{
    if (disposing)
    {
        foreach (var tensor in _registeredTensors)
        {
            Engine.UnregisterPersistentTensor(tensor);
        }
    }
    base.Dispose(disposing);
}
```

## Complete Layer Checklist (117 Layers)

### Priority 1: Dense/Linear Layers (High GPU Impact - FusedLinear)

| Layer | Has Weights | Has Bias | Status |
|-------|-------------|----------|--------|
| [ ] DenseLayer | Yes | Yes | Has GPU code, needs refactor |
| [ ] FullyConnectedLayer | Yes | Yes | Needs update |
| [ ] SparseLinearLayer | Yes | Yes | Needs update |
| [ ] LocallyConnectedLayer | Yes | Yes | Needs update |
| [ ] HyperbolicLinearLayer | Yes | Yes | Needs update |
| [ ] OctonionLinearLayer | Yes | Yes | Needs update |
| [ ] FeedForwardLayer | Yes | Yes | Needs update |
| [ ] GatedLinearUnitLayer | Yes | Yes | Needs update |
| [ ] HighwayLayer | Yes | Yes | Needs update |
| [ ] ExpertLayer | Yes | Yes | Needs update |

### Priority 2: Convolutional Layers (High GPU Impact - FusedConv2D/3D)

| Layer | Has Weights | Has Bias | Status |
|-------|-------------|----------|--------|
| [ ] ConvolutionalLayer | Yes | Yes | Needs update |
| [ ] Conv3DLayer | Yes | Yes | Needs update |
| [ ] DeconvolutionalLayer | Yes | Yes | Needs update |
| [ ] DepthwiseSeparableConvolutionalLayer | Yes | Yes | Needs update |
| [ ] DilatedConvolutionalLayer | Yes | Yes | Needs update |
| [ ] SeparableConvolutionalLayer | Yes | Yes | Needs update |
| [ ] DeformableConvolutionalLayer | Yes | Yes | Needs update |
| [ ] EdgeConditionalConvolutionalLayer | Yes | Yes | Needs update |
| [ ] SubpixelConvolutionalLayer | Yes | Yes | Needs update |
| [ ] ConvLSTMLayer | Yes | Yes | Needs update |
| [ ] PatchEmbeddingLayer | Yes | Yes | Needs update |

### Priority 3: Attention Layers (High GPU Impact - FusedAttention)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] MultiHeadAttentionLayer | Yes (Q/K/V/O) | Needs update |
| [ ] SelfAttentionLayer | Yes | Needs update |
| [ ] CrossAttentionLayer | Yes | Needs update |
| [ ] AttentionLayer | Yes | Needs update |
| [ ] GraphAttentionLayer | Yes | Needs update |
| [ ] TransformerEncoderLayer | Yes | Needs update |
| [ ] TransformerDecoderLayer | Yes | Needs update |
| [ ] DecoderLayer | Yes | Needs update |
| [ ] GraphTransformerLayer | Yes | Needs update |
| [ ] SpatialTransformerLayer | Yes | Needs update |

### Priority 4: Normalization Layers (Medium GPU Impact - FusedNorm)

| Layer | Has Params | Status |
|-------|------------|--------|
| [ ] BatchNormalizationLayer | Yes (gamma/beta) | Needs update |
| [ ] LayerNormalizationLayer | Yes | Needs update |
| [ ] GroupNormalizationLayer | Yes | Needs update |
| [ ] InstanceNormalizationLayer | Yes | Needs update |
| [ ] SpectralNormalizationLayer | Yes | Needs update |

### Priority 5: Recurrent Layers (Medium GPU Impact - FusedRNN)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] LSTMLayer | Yes | Needs update |
| [ ] GRULayer | Yes | Needs update |
| [ ] RecurrentLayer | Yes | Needs update |
| [ ] BidirectionalLayer | Yes (wraps RNN) | Needs update |
| [ ] TemporalMemoryLayer | Yes | Needs update |
| [ ] ContinuumMemorySystemLayer | Yes | Needs update |
| [ ] MemoryReadLayer | Yes | Needs update |
| [ ] MemoryWriteLayer | Yes | Needs update |
| [ ] SynapticPlasticityLayer | Yes | Needs update |

### Priority 6: Graph Neural Network Layers (Medium GPU Impact)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] GraphConvolutionalLayer | Yes | Needs update |
| [ ] GraphSAGELayer | Yes | Needs update |
| [ ] GraphIsomorphismLayer | Yes | Needs update |
| [ ] DirectionalGraphLayer | Yes | Needs update |
| [ ] HeterogeneousGraphLayer | Yes | Needs update |
| [ ] MessagePassingLayer | Yes | Needs update |
| [ ] PrincipalNeighbourhoodAggregationLayer | Yes | Needs update |
| [ ] DiffusionConvLayer | Yes | Needs update |
| [ ] SpiralConvLayer | Yes | Needs update |
| [ ] MeshEdgeConvLayer | Yes | Needs update |
| [ ] MeshPoolLayer | Yes | Needs update |

### Priority 7: Embedding Layers (Low-Medium GPU Impact)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] EmbeddingLayer | Yes | Needs update |
| [ ] PositionalEncodingLayer | Yes | Needs update |
| [ ] TimeEmbeddingLayer | Yes | Needs update |

### Priority 8: Pooling Layers (Low GPU Impact - Memory Bound)

| Layer | Has Params | Status |
|-------|------------|--------|
| [ ] MaxPoolingLayer | No | Needs update |
| [ ] AveragePoolingLayer | No | Needs update |
| [ ] GlobalPoolingLayer | No | Needs update |
| [ ] AdaptiveAveragePoolingLayer | No | Needs update |
| [ ] MaxPool3DLayer | No | Needs update |
| [ ] PoolingLayer | No | Needs update |
| [ ] SpatialPoolerLayer | No | Needs update |

### Priority 9: Capsule Network Layers (Specialized)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] CapsuleLayer | Yes | Needs update |
| [ ] PrimaryCapsuleLayer | Yes | Needs update |
| [ ] DigitCapsuleLayer | Yes | Needs update |

### Priority 10: Residual/Block Layers (Composite - Delegate to Sub-layers)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] ResidualLayer | No (wraps) | Delegate to children |
| [ ] BasicBlock | Yes | Needs update |
| [ ] BottleneckBlock | Yes | Needs update |
| [ ] DenseBlock | Yes | Needs update |
| [ ] DenseBlockLayer | Yes | Needs update |
| [ ] ResidualDenseBlock | Yes | Needs update |
| [ ] RRDBLayer | Yes | Needs update |
| [ ] InvertedResidualBlock | Yes | Needs update |
| [ ] SqueezeAndExcitationLayer | Yes | Needs update |
| [ ] TransitionLayer | Yes | Needs update |
| [ ] RepParameterizationLayer | Yes | Needs update |

### Priority 11: Specialized Compute Layers

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] RBFLayer | Yes | Needs update |
| [ ] RBMLayer | Yes | Needs update |
| [ ] ReservoirLayer | Yes | Needs update |
| [ ] SpikingLayer | Yes | Needs update |
| [ ] QuantumLayer | Yes | Needs update |
| [ ] MixtureOfExpertsLayer | Yes | Needs update |
| [ ] ReadoutLayer | Yes | Needs update |
| [ ] ConditionalRandomFieldLayer | Yes | Needs update |
| [ ] AnomalyDetectorLayer | Yes | Needs update |

### Priority 12: Utility Layers (No Trainable Params - Minimal Changes)

| Layer | Has Params | Status |
|-------|------------|--------|
| [ ] ActivationLayer | No | Uses Engine activation |
| [ ] DropoutLayer | No | GPU random |
| [ ] FlattenLayer | No | Reshape only |
| [ ] ReshapeLayer | No | Reshape only |
| [ ] InputLayer | No | Pass-through |
| [ ] AddLayer | No | Element-wise |
| [ ] MultiplyLayer | No | Element-wise |
| [ ] ConcatenateLayer | No | Memory op |
| [ ] SplitLayer | No | Memory op |
| [ ] CroppingLayer | No | Memory op |
| [ ] PaddingLayer | No | Memory op |
| [ ] MaskingLayer | No | Element-wise |
| [ ] GaussianNoiseLayer | No | GPU random |
| [ ] LambdaLayer | No | Custom op |
| [ ] MeanLayer | No | Reduction |
| [ ] LogVarianceLayer | No | Reduction |
| [ ] SequenceLastLayer | No | Indexing |
| [ ] TimeDistributedLayer | No (wraps) | Delegate |
| [ ] PixelShuffleLayer | No | Reshape |
| [ ] UpsamplingLayer | No | Interpolation |
| [ ] Upsample3DLayer | No | Interpolation |
| [ ] ReconstructionLayer | No | Composite |
| [ ] MeasurementLayer | No | Diagnostic |

### Generator/Discriminator Layers (Composite)

| Layer | Has Weights | Status |
|-------|-------------|--------|
| [ ] RRDBNetGenerator | Yes | Needs update |
| [ ] UNetDiscriminator | Yes | Needs update |
| [ ] SpyNetLayer | Yes | Needs update |

---

**Total: 117 layers**
- **With trainable parameters: ~85 layers**
- **Without trainable parameters: ~32 layers**

## Critical Findings: Missing IEngine Operations

### Missing Standard Operations (Per Research Papers)

| Operation | Reference | Current Status | Impact |
|-----------|-----------|----------------|--------|
| ScaledDotProductAttention | "Attention Is All You Need" (Vaswani 2017) | **MISSING** | All attention layers do manual matmul+softmax |
| FlashAttention | FlashAttention papers (2022-2023) | **MISSING** | No memory-efficient attention |
| GroupedQueryAttention | "GQA" (Ainslie 2023) | **MISSING** | Manual implementation |
| RMSNorm | "Root Mean Square Layer Normalization" | **MISSING** | LayerNorm without mean subtraction |
| SiLU/Swish | "Swish: A Self-Gated Activation Function" | Partial | In activations, not fused |
| GeGLU | "GLU Variants Improve Transformer" | **MISSING** | Used in modern transformers |

### Layers Using Manual Implementations Instead of IEngine

| Layer | Issue | Manual Loop Count | Should Use |
|-------|-------|-------------------|------------|
| **CrossAttentionLayer** | Manual 4-nested matmul for all ops | 20+ loops | Engine.TensorMatMul, Engine.Softmax |
| **GraphAttentionLayer** | Manual attention despite 24 Engine calls | 85 loops | Engine.ScaledDotProductAttention |
| **GraphTransformerLayer** | Manual loops | 75+ NumOps | Engine operations |
| **MessagePassingLayer** | Manual aggregation | 73+ NumOps | Engine.Scatter/Gather |
| **DiffusionConvLayer** | Manual convolution | 43+ NumOps | Engine.Conv operations |
| SpikingLayer | Mixed (uses 130 Engine calls) | 95 NumOps | More Engine ops possible |
| AnomalyDetectorLayer | Manual multiply-accumulate | High | Engine.TensorMatMul |

### Layers Correctly Using IEngine (Good Examples)

| Layer | Engine Calls | Manual Loops | Status |
|-------|--------------|--------------|--------|
| AttentionLayer | 40+ | Few | Good - uses Engine.BatchMatMul, TensorMatMul |
| LSTMLayer | 48 | 11 | Good - mostly Engine operations |
| GRULayer | 32 | 16 | Good - mostly Engine operations |
| ConvolutionalLayer | Many | Few | Good - uses Engine.Conv2D |
| BatchNormalizationLayer | Many | 5 | Good - uses Engine.BatchNorm |
| SelfAttentionLayer | Many | Few | Good - uses Engine.Softmax |

### Proposed New IEngine Operations

```csharp
/// <summary>
/// Scaled dot-product attention as per "Attention Is All You Need".
/// attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V
/// </summary>
Tensor<T> ScaledDotProductAttention<T>(
    Tensor<T> query,    // [batch, heads, seq_q, d_k]
    Tensor<T> key,      // [batch, heads, seq_k, d_k]
    Tensor<T> value,    // [batch, heads, seq_k, d_v]
    Tensor<T>? mask = null,
    float dropout = 0.0f);

Tensor<T> ScaledDotProductAttentionBackward<T>(
    Tensor<T> gradOutput,
    Tensor<T> query,
    Tensor<T> key,
    Tensor<T> value,
    Tensor<T> attentionWeights,
    out Tensor<T> gradQuery,
    out Tensor<T> gradKey,
    out Tensor<T> gradValue);

/// <summary>
/// Root Mean Square Layer Normalization (faster than LayerNorm).
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
/// </summary>
Tensor<T> RMSNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon);

Tensor<T> RMSNormBackward<T>(
    Tensor<T> gradOutput,
    Tensor<T> input,
    Tensor<T> gamma,
    double epsilon,
    out Tensor<T> gradGamma);

/// <summary>
/// Gated Linear Unit variants (GeGLU, SwiGLU, ReGLU).
/// GLU(x) = (x @ W1 + b1) * activation(x @ W2 + b2)
/// </summary>
Tensor<T> GatedLinearUnit<T>(
    Tensor<T> input,
    Tensor<T> gateWeights,
    Tensor<T> valueWeights,
    Tensor<T>? gateBias,
    Tensor<T>? valueBias,
    GatedActivationType activation);

/// <summary>
/// Graph message passing scatter operation.
/// Efficiently aggregates neighbor messages by destination node.
/// </summary>
Tensor<T> ScatterAdd<T>(Tensor<T> source, Tensor<int> indices, int dimSize);
Tensor<T> ScatterMean<T>(Tensor<T> source, Tensor<int> indices, int dimSize);
Tensor<T> ScatterMax<T>(Tensor<T> source, Tensor<int> indices, int dimSize);
```

## Layers Requiring Immediate Refactoring

These layers have significant manual implementations that should use IEngine:

### High Priority (Performance Critical)
1. **CrossAttentionLayer** - Core diffusion model component, manual matmul
2. **GraphAttentionLayer** - 85 manual loops
3. **GraphTransformerLayer** - Graph + Attention, 75+ NumOps calls
4. **MessagePassingLayer** - GNN core, 73+ NumOps calls

### Medium Priority
5. **DiffusionConvLayer** - 43+ NumOps calls
6. **SpatialTransformerLayer** - Combines attention with spatial ops
7. **TimeEmbeddingLayer** - 30+ NumOps calls
8. **HeterogeneousGraphLayer** - 30+ NumOps calls

## Implementation Order

1. **Phase 1: IEngine Interface**
   - Add `PersistentTensorRole` enum
   - Add `FusedActivationType` enum
   - Add `RegisterPersistentTensor`/`UnregisterPersistentTensor` to IEngine
   - Add `FusedLinear` to IEngine
   - Implement in CpuEngine (standard path)
   - Implement in GpuEngine (DirectGpu path)

2. **Phase 2: LayerBase Integration**
   - Add `RegisterTrainableParameter` helper
   - Add disposal cleanup for registered tensors
   - Document pattern for derived layers

3. **Phase 3: Priority 1 Layers**
   - Refactor DenseLayer to use FusedLinear
   - Refactor ConvolutionalLayer to use FusedConv2D
   - Remove GPU-specific code from layers

4. **Phase 4: Priority 2-3 Layers**
   - Add FusedBatchNorm, FusedLayerNorm
   - Add FusedAttention
   - Update normalization and attention layers

5. **Phase 5: Remaining Layers**
   - Update in priority order
   - Each layer should only use IEngine methods

## Testing Strategy

1. **Unit tests**: Each fused operation has correctness tests
2. **Performance tests**: Compare CPU vs GPU for each layer type
3. **Integration tests**: Full network forward/backward passes
4. **Regression tests**: Ensure numerical equivalence with old implementation

## Success Criteria

- [ ] No layer directly references DirectGpu, CUDA, OpenCL, or HIP
- [ ] All trainable parameters registered via LayerBase helper
- [ ] GPU acceleration works transparently when available
- [ ] CPU fallback works correctly when GPU unavailable
- [ ] Performance improvement measurable in benchmarks
