# Neural Network Layer Production-Grade Upgrade Tracker

## Production-Grade Pattern Requirements (Industry Standard)

### Core Requirements Checklist
Every layer MUST implement ALL of these patterns:

#### 1. Tensor-Only Internal Storage (NO TYPE CONVERSIONS)
- [ ] Use `Tensor<T>` for ALL weights, biases, and internal state
- [ ] **NO** `Matrix<T>` or `Vector<T>` for internal storage
- [ ] **NO** `.ToMatrix()` / `.ToVector()` / `.FromMatrix()` / `.FromVector()` in hot paths
- [ ] Exception: `GetParameters()` / `SetParameters()` may return `Vector<T>` for API compatibility

#### 2. Locality Caches (Forward Pass State)
- [ ] `_lastInput` - cache input tensor for backward pass
- [ ] `_lastOutput` - cache output tensor for backward pass
- [ ] Any intermediate values needed for gradients (e.g., `_lastAttentionWeights`)

#### 3. GPU/CPU Accelerated Operations via IEngine
**CRITICAL: Use Engine operations, NOT manual loops!**

Access via: `protected IEngine Engine => AiDotNetEngine.Current;` (from LayerBase)

**Available Tensor Operations:**
- [ ] `Engine.TensorAdd<T>(Tensor<T> a, Tensor<T> b)` - element-wise addition
- [ ] `Engine.TensorSubtract<T>(Tensor<T> a, Tensor<T> b)` - element-wise subtraction
- [ ] `Engine.TensorMultiply<T>(Tensor<T> a, Tensor<T> b)` - element-wise multiplication (Hadamard)
- [ ] `Engine.TensorMultiplyScalar<T>(Tensor<T> t, T scalar)` - scalar multiplication
- [ ] `Engine.TensorDivide<T>(Tensor<T> a, Tensor<T> b)` - element-wise division
- [ ] `Engine.TensorMatMul<T>(Tensor<T> a, Tensor<T> b)` - matrix multiplication
- [ ] `Engine.TensorTranspose<T>(Tensor<T> t)` - transpose
- [ ] `Engine.BatchMatMul<T>(Tensor<T> a, Tensor<T> b)` - batched matrix multiply

**Available Activation Operations:**
- [ ] `Engine.ReLU<T>(Tensor<T> t)` - ReLU activation
- [ ] `Engine.Sigmoid<T>(Tensor<T> t)` - Sigmoid activation
- [ ] `Engine.Tanh<T>(Tensor<T> t)` - Tanh activation
- [ ] `Engine.GELU<T>(Tensor<T> t)` - GELU activation
- [ ] `Engine.Softmax<T>(Tensor<T> t, int axis)` - Softmax

**Available Normalization:**
- [ ] `Engine.BatchNorm<T>(...)` - Batch normalization
- [ ] `Engine.LayerNorm<T>(...)` - Layer normalization

**Available Pooling/Conv:**
- [ ] `Engine.MaxPool2D<T>(...)` - Max pooling
- [ ] `Engine.AvgPool2D<T>(...)` - Average pooling
- [ ] `Engine.Conv2D<T>(...)` - 2D convolution

**Available Comparison Operations (like PyTorch torch.eq/torch.ne):**
- [ ] `Engine.TensorEquals<T>(Tensor<T> t, T value)` - element-wise equality to scalar
- [ ] `Engine.TensorEquals<T>(Tensor<T> a, Tensor<T> b)` - element-wise equality
- [ ] `Engine.TensorNotEquals<T>(Tensor<T> t, T value)` - element-wise inequality to scalar
- [ ] `Engine.TensorNotEquals<T>(Tensor<T> a, Tensor<T> b)` - element-wise inequality
- [ ] `Engine.TensorGreaterThan<T>(Tensor<T> t, T value)` - element-wise greater than scalar
- [ ] `Engine.TensorGreaterThan<T>(Tensor<T> a, Tensor<T> b)` - element-wise greater than
- [ ] `Engine.TensorLessThan<T>(Tensor<T> t, T value)` - element-wise less than scalar
- [ ] `Engine.TensorLessThan<T>(Tensor<T> a, Tensor<T> b)` - element-wise less than

**Available Multi-Tensor Operations (like PyTorch torch.sum over tensors):**
- [ ] `Engine.TensorAddMany<T>(params Tensor<T>[] tensors)` - sum multiple tensors (no loops needed)
- [ ] `Engine.TensorMultiplyMany<T>(params Tensor<T>[] tensors)` - element-wise product of multiple tensors

**Why Multi-Tensor Operations?**
```csharp
// ❌ WRONG: Loop over tensors (even with Engine.TensorAdd, still sub-optimal)
Tensor<T> result = inputs[0];
for (int i = 1; i < inputs.Length; i++)
{
    result = Engine.TensorAdd(result, inputs[i]);
}

// ✅ CORRECT: Single optimized call - GPU can batch all additions
var result = Engine.TensorAddMany(inputs);
```

**Anti-Pattern (DO NOT USE):**
```csharp
// ❌ WRONG: Manual loops are slow and don't use GPU
for (int i = 0; i < tensor.Length; i++)
{
    output.Data[i] = NumOps.Multiply(a.Data[i], b.Data[i]);
}

// ✅ CORRECT: Engine operations use vectorized/GPU acceleration
var output = Engine.TensorMultiply(a, b);
```

#### 4. Autodiff Support with Inline Topological Sort
- [ ] `BackwardViaAutodiff()` method for automatic differentiation path
- [ ] **INLINE** topological sort - NO helper method calls
- [ ] Pattern:
```csharp
var visited = new HashSet<ComputationNode<T>>();
var topoOrder = new List<ComputationNode<T>>();
var stack = new Stack<(ComputationNode<T> node, bool processed)>();
stack.Push((rootNode, false));

while (stack.Count > 0)
{
    var (node, processed) = stack.Pop();
    if (visited.Contains(node)) continue;

    if (processed)
    {
        visited.Add(node);
        topoOrder.Add(node);
    }
    else
    {
        stack.Push((node, true));
        foreach (var parent in node.Parents)
        {
            if (!visited.Contains(parent))
                stack.Push((parent, false));
        }
    }
}

for (int i = topoOrder.Count - 1; i >= 0; i--)
{
    var node = topoOrder[i];
    if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
        node.BackwardFunction(node.Gradient);
}
```

#### 5. JIT Compilation Support
- [ ] `ExportComputationGraph()` implementation
- [ ] `SupportsJitCompilation` property returning true when ready
- [ ] Use `TensorOperations<T>.Variable()`, `Constant()`, `MatrixMultiply()`, etc.

#### 6. Clean Code
- [ ] **REMOVE** all unused helper methods:
  - `GetTopologicalOrder()`
  - `MatrixToTensor()` / `TensorToMatrix()`
  - `VectorToTensor()` / `TensorToVector()`
  - `ApplyActivationAutodiff()` (if unused)
- [ ] No dead code or commented-out sections
- [ ] **NO null-forgiving operators (`!`)** - use proper null checks instead

**Anti-Pattern (DO NOT USE null-forgiving operators):**
```csharp
// ❌ WRONG: Null-forgiving operator bypasses null safety
_memoryPool!.Return(buffer);
var value = tensor!.GetFlat(i);

// ✅ CORRECT: Proper null checking
if (_memoryPool != null)
{
    _memoryPool.Return(buffer);
}

// OR throw meaningful exception
var pool = _memoryPool ?? throw new InvalidOperationException("Memory pool not initialized");
pool.Return(buffer);
```

#### 7. State Management
- [ ] `ResetState()` clears all cached values
- [ ] Proper null checks before using cached values in Backward

---

## Reference Implementation: FullyConnectedLayer.cs

See `src/NeuralNetworks/Layers/FullyConnectedLayer.cs` for the gold standard pattern:
- Uses `Tensor<T>` for `_weights`, `_biases` (not Matrix/Vector)
- Has `_lastInput`, `_lastOutput` caches
- Has inline topological sort in `BackwardViaAutodiff()`
- Has `ExportComputationGraph()` for JIT
- Clean - no helper methods for conversions

---

## Layer Status Summary

| Status | Count | Description |
|--------|-------|-------------|
| NEEDS REVIEW | 2 | Previously "completed" but need tensor-only verification |
| HAS CONVERSIONS | 34 | Still using ToMatrix/ToVector/FromMatrix/FromVector |
| HAS HELPER CALLS | 15 | Still calling GetTopologicalOrder helper |
| CLEAN | ~29 | May only need verification |

---

## All 78 Layers - Full Status

### Category 1: NEEDS REVIEW (Previously Marked Complete)
These were marked complete but used conversion pattern - need to verify tensor-only:

| # | Layer | Conversions | Helper Calls | Notes |
|---|-------|-------------|--------------|-------|
| 1 | FullyConnectedLayer.cs | NO | NO | **REFERENCE** - Uses Tensor<T> properly |
| 2 | DenseLayer.cs | YES | NO | Has TensorToMatrix/TensorToVector helpers - needs cleanup |

### Category 2: HAS GetTopologicalOrder HELPER CALLS (15 layers)
These MUST inline the topological sort:

| # | Layer | Conversions | Notes |
|---|-------|-------------|-------|
| 3 | TransformerDecoderLayer.cs | NO | Complex - multiple attention |
| 4 | TransformerEncoderLayer.cs | NO | Complex |
| 5 | LogVarianceLayer.cs | NO | VAE component |
| 6 | MeanLayer.cs | NO | Simple aggregation |
| 7 | PoolingLayer.cs | NO | Base pooling |
| 8 | GlobalPoolingLayer.cs | NO | Global aggregation |
| 9 | DeconvolutionalLayer.cs | NO | Transposed conv |
| 10 | UpsamplingLayer.cs | NO | For decoders |
| 11 | SplitLayer.cs | NO | Tensor splitting |
| 12 | SpatialTransformerLayer.cs | YES | Complex spatial attention |
| 13 | RBFLayer.cs | YES | Radial basis function |
| 14 | MemoryWriteLayer.cs | YES | Memory network |
| 15 | MaskingLayer.cs | NO | Masking ops |
| 16 | LSTMLayer.cs | YES | Complex RNN |
| 17 | GRULayer.cs | YES | RNN variant |

### Category 3: HAS TYPE CONVERSIONS (34 layers total, excluding those above)
These MUST eliminate ToMatrix/ToVector/FromMatrix/FromVector:

| # | Layer | Helper Calls | Notes |
|---|-------|--------------|-------|
| 18 | AttentionLayer.cs | NO | Has inline topo sort, but uses ToVector |
| 19 | SubpixelConvolutionalLayer.cs | NO | |
| 20 | SeparableConvolutionalLayer.cs | NO | |
| 21 | RecurrentLayer.cs | NO | |
| 22 | DropoutLayer.cs | NO | Simple layer |
| 23 | FeedForwardLayer.cs | NO | |
| 24 | GatedLinearUnitLayer.cs | NO | |
| 25 | GraphConvolutionalLayer.cs | NO | |
| 26 | HighwayLayer.cs | NO | Gated layer |
| 27 | LayerNormalizationLayer.cs | NO | Important for transformers |
| 28 | LocallyConnectedLayer.cs | NO | |
| 29 | MemoryReadLayer.cs | NO | |
| 30 | MultiHeadAttentionLayer.cs | NO | Complex, multiple heads |
| 31 | ReadoutLayer.cs | NO | |
| 32 | DilatedConvolutionalLayer.cs | NO | |
| 33 | DepthwiseSeparableConvolutionalLayer.cs | NO | |
| 34 | ContinuumMemorySystemLayer.cs | NO | |
| 35 | ConvLSTMLayer.cs | NO | |
| 36 | TemporalMemoryLayer.cs | NO | |
| 37 | SynapticPlasticityLayer.cs | NO | |
| 38 | SpikingLayer.cs | NO | |
| 39 | SpatialPoolerLayer.cs | NO | |
| 40 | ReservoirLayer.cs | NO | |
| 41 | RBMLayer.cs | NO | |
| 42 | DigitCapsuleLayer.cs | NO | |
| 43 | CapsuleLayer.cs | NO | |
| 44 | AnomalyDetectorLayer.cs | NO | |

### Category 4: LIKELY CLEAN - Need Verification (29 layers)
These may already be clean or have minimal issues:

| # | Layer | Notes |
|---|-------|-------|
| 45 | ActivationLayer.cs | Simple |
| 46 | AddLayer.cs | Simple merge |
| 47 | AvgPoolingLayer.cs | Pooling |
| 48 | BatchNormalizationLayer.cs | Complex gradients |
| 49 | BidirectionalLayer.cs | Wrapper |
| 50 | ConcatenateLayer.cs | Simple merge |
| 51 | ConditionalRandomFieldLayer.cs | |
| 52 | ConvolutionalLayer.cs | Core layer |
| 53 | CroppingLayer.cs | Simple |
| 54 | DecoderLayer.cs | |
| 55 | EmbeddingLayer.cs | |
| 56 | ExpertLayer.cs | MoE component |
| 57 | FlattenLayer.cs | Trivial |
| 58 | GaussianNoiseLayer.cs | Simple |
| 59 | InputLayer.cs | Pass-through |
| 60 | LambdaLayer.cs | Custom |
| 61 | MaxPoolingLayer.cs | |
| 62 | MeasurementLayer.cs | |
| 63 | MixtureOfExpertsLayer.cs | |
| 64 | MultiplyLayer.cs | Simple merge |
| 65 | PaddingLayer.cs | Simple |
| 66 | PatchEmbeddingLayer.cs | ViT component |
| 67 | PositionalEncodingLayer.cs | Transformer |
| 68 | PrimaryCapsuleLayer.cs | |
| 69 | QuantumLayer.cs | |
| 70 | ReconstructionLayer.cs | |
| 71 | RepParameterizationLayer.cs | |
| 72 | ReshapeLayer.cs | Simple |
| 73 | ResidualLayer.cs | |
| 74 | SelfAttentionLayer.cs | Similar to AttentionLayer |
| 75 | SqueezeAndExcitationLayer.cs | Channel attention |
| 76 | TimeDistributedLayer.cs | Wrapper |

### Not Layer Files (2 files in Layers folder)
| File | Notes |
|------|-------|
| LayerBase.cs | Base class - uses ToVector in ApplyActivation (acceptable) |
| MixtureOfExpertsBuilder.cs | Builder pattern, not a layer |

---

## Progress Tracking

### Statistics
- **Total Layer Files**: 78 (76 actual layers + LayerBase + MoEBuilder)
- **Layers needing GetTopologicalOrder inline**: 15
- **Layers needing conversion removal**: 34
- **Layers needing verification only**: ~29

### Completed
- [ ] FullyConnectedLayer.cs - REFERENCE IMPLEMENTATION

### In Progress
- [ ] (none currently)

---

## Work Order (Recommended Sequence)

### Phase 1: Simple Layers (Low Risk)
1. ReshapeLayer.cs
2. FlattenLayer.cs
3. DropoutLayer.cs
4. MaskingLayer.cs
5. ActivationLayer.cs
6. AddLayer.cs
7. MultiplyLayer.cs
8. ConcatenateLayer.cs

### Phase 2: Normalization & Pooling
9. LayerNormalizationLayer.cs
10. BatchNormalizationLayer.cs
11. PoolingLayer.cs
12. AvgPoolingLayer.cs
13. MaxPoolingLayer.cs
14. GlobalPoolingLayer.cs

### Phase 3: Attention Layers
15. AttentionLayer.cs
16. SelfAttentionLayer.cs
17. MultiHeadAttentionLayer.cs
18. SqueezeAndExcitationLayer.cs

### Phase 4: Transformer Components
19. TransformerEncoderLayer.cs
20. TransformerDecoderLayer.cs
21. PositionalEncodingLayer.cs
22. PatchEmbeddingLayer.cs
23. FeedForwardLayer.cs

### Phase 5: Recurrent Layers
24. RecurrentLayer.cs
25. LSTMLayer.cs
26. GRULayer.cs
27. ConvLSTMLayer.cs
28. BidirectionalLayer.cs

### Phase 6: Convolutional Layers
29. ConvolutionalLayer.cs
30. DeconvolutionalLayer.cs
31. DilatedConvolutionalLayer.cs
32. DepthwiseSeparableConvolutionalLayer.cs
33. SeparableConvolutionalLayer.cs
34. LocallyConnectedLayer.cs
35. SubpixelConvolutionalLayer.cs

### Phase 7: Specialized Layers
36. DenseLayer.cs (verify tensor-only)
37. HighwayLayer.cs
38. GatedLinearUnitLayer.cs
39. ResidualLayer.cs
40. UpsamplingLayer.cs
41. CroppingLayer.cs
42. PaddingLayer.cs
43. SplitLayer.cs

### Phase 8: Memory & Graph Layers
44. MemoryReadLayer.cs
45. MemoryWriteLayer.cs
46. GraphConvolutionalLayer.cs
47. ContinuumMemorySystemLayer.cs

### Phase 9: Probabilistic & Generative
48. LogVarianceLayer.cs
49. MeanLayer.cs
50. GaussianNoiseLayer.cs
51. RBFLayer.cs
52. RBMLayer.cs

### Phase 10: Capsule Networks
53. CapsuleLayer.cs
54. PrimaryCapsuleLayer.cs
55. DigitCapsuleLayer.cs

### Phase 11: Advanced/Experimental
56. SpatialTransformerLayer.cs
57. QuantumLayer.cs
58. SpikingLayer.cs
59. ReservoirLayer.cs
60. TemporalMemoryLayer.cs
61. SpatialPoolerLayer.cs
62. SynapticPlasticityLayer.cs
63. AnomalyDetectorLayer.cs

### Phase 12: Remaining
64-76. All remaining layers

---

## Notes

### What NOT to Change
- `LayerBase.cs` - The base class appropriately uses Vector<T> in some helper methods for API compatibility
- `GetParameters()` / `SetParameters()` - These are API contracts that return Vector<T>

### .NET Framework Compatibility (Critical)
**All code MUST be compatible with older .NET Framework versions without conditional compilation:**
- **NEVER** use `#if !NET462` or similar preprocessor directives in layer code
- **USE** compatibility shims or polyfills when needed
- GPU-specific code in GpuEngine.cs uses ILGPU types like `MemoryBuffer1D<T, Stride1D.Dense>`, not custom types
- Example: `GpuMemoryPool<T>.Rent()` returns `MemoryBuffer1D<T, Stride1D.Dense>`, NOT a custom `PooledMemory<T>` type

### Key Insight: Why Tensor-Only?
Converting between Tensor/Matrix/Vector creates:
1. Memory allocations (GC pressure)
2. Data copying overhead
3. Cache misses
4. Prevents GPU acceleration

By keeping everything as `Tensor<T>`:
- Operations can be batched and vectorized
- GPU can process without host transfers
- Memory layout is predictable
- IEngine can optimize operations
