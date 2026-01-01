# AI Models and Layers Vectorization Plan

This document lists all AI models and layers that need to be vectorized using the proper IEngine methods for GPU/SIMD acceleration.

## Available IEngine Tensor Operations

The following vectorized operations are available in `IEngine`:

### Math Operations
- `TensorSin<T>(Tensor<T>)` - Element-wise sine
- `TensorCos<T>(Tensor<T>)` - Element-wise cosine
- `TensorExp<T>(Tensor<T>)` - Element-wise exponential
- `TensorLog<T>(Tensor<T>)` - Element-wise natural log
- `TensorSqrt<T>(Tensor<T>)` - Element-wise square root
- `TensorAbs<T>(Tensor<T>)` - Element-wise absolute value
- `TensorPow<T>(Tensor<T>, T)` - Element-wise power
- `TensorFloor<T>(Tensor<T>)` - Element-wise floor
- `TensorCeiling<T>(Tensor<T>)` - Element-wise ceiling
- `TensorFrac<T>(Tensor<T>)` - Element-wise fractional part

### Activation Functions
- `Tanh<T>(Tensor<T>)` - Element-wise tanh
- `Sigmoid<T>(Tensor<T>)` - Element-wise sigmoid
- `ReLU<T>(Tensor<T>)` - Element-wise ReLU
- `GELU<T>(Tensor<T>)` - Element-wise GELU
- `Softmax<T>(Tensor<T>, int)` - Softmax with axis

### Tensor Arithmetic
- `TensorAdd<T>(Tensor<T>, Tensor<T>)` - Element-wise addition
- `TensorSubtract<T>(Tensor<T>, Tensor<T>)` - Element-wise subtraction
- `TensorMultiply<T>(Tensor<T>, Tensor<T>)` - Element-wise multiplication
- `TensorDivide<T>(Tensor<T>, Tensor<T>)` - Element-wise division
- `TensorMultiplyScalar<T>(Tensor<T>, T)` - Scalar multiplication
- `TensorMax<T>(Tensor<T>, Tensor<T>)` - Element-wise max
- `TensorMin<T>(Tensor<T>, Tensor<T>)` - Element-wise min
- `TensorClamp<T>(Tensor<T>, T, T)` - Element-wise clamp

### Reductions
- `TensorSum<T>(Tensor<T>)` - Sum all elements
- `TensorMean<T>(Tensor<T>)` - Mean of all elements
- `ReduceSum<T>(Tensor<T>, int[], bool)` - Reduce along axes
- `ReduceMax<T>(Tensor<T>, int[], bool, out int[])` - Reduce max along axes

---

## HIGH PRIORITY - Neural Network Layers

### 1. PositionalEncodingLayer.cs
**File:** `src/NeuralNetworks/Layers/PositionalEncodingLayer.cs`
**Location:** Lines 155-173 (InitializeEncodings method)
**Issue:** Scalar loop using `Math.Sin` and `Math.Cos`

```csharp
// CURRENT (scalar - slow)
for (int pos = 0; pos < maxSequenceLength; pos++)
{
    for (int i = 0; i < embeddingSize; i++)
    {
        double angle = pos / Math.Pow(10000, exponent);
        if (i % 2 == 0)
            encodings[pos, i] = NumOps.FromDouble(Math.Sin(angle));
        else
            encodings[pos, i] = NumOps.FromDouble(Math.Cos(angle));
    }
}
```

**Fix:** Vectorize using `Engine.TensorSin()` and `Engine.TensorCos()`:
1. Pre-compute all angles into a tensor
2. Apply `Engine.TensorSin()` to even indices
3. Apply `Engine.TensorCos()` to odd indices
4. Combine results

---

### 2. SpikingLayer.cs
**File:** `src/NeuralNetworks/Layers/SpikingLayer.cs`
**Locations:**
- Line 1021: `Math.Exp` in exponential integrate-and-fire model
- Lines 1102-1122: Multiple `Math.Exp`, `Math.Pow` in Hodgkin-Huxley model
- Line 1514: `Math.Cosh` in soft threshold

**Issue:** Hodgkin-Huxley neuron model uses many scalar Math operations

```csharp
// CURRENT (scalar - slow)
double expTerm = _deltaT * Math.Exp((v - _vT) / _deltaT);
double alphaM = 0.1 * (v + 40.0) / (1.0 - Math.Exp(-(v + 40.0) / 10.0));
double betaM = 4.0 * Math.Exp(-(v + 65.0) / 18.0);
double INa = gNa * Math.Pow(m, 3) * h * (v - ENa);
double IK = gK * Math.Pow(n, 4) * (v - EK);
```

**Fix:** Vectorize neuron state tensors and use:
- `Engine.TensorExp()` for exponential terms
- `Engine.TensorPow()` for power terms
- Batch process all neurons simultaneously

---

### 3. DiffusionConvLayer.cs
**File:** `src/NeuralNetworks/Layers/DiffusionConvLayer.cs`
**Locations:**
- Lines 268-282: `Math.Log`, `Math.Sqrt`, `Math.Exp` for time computation

```csharp
// CURRENT (scalar)
double logMin = Math.Log(minTime);
double logMax = Math.Log(maxTime);
times[0] = NumOps.FromDouble(Math.Sqrt(minTime * maxTime));
times[i] = NumOps.FromDouble(Math.Exp(logT));
```

**Fix:** Use `Engine.TensorLog()`, `Engine.TensorSqrt()`, `Engine.TensorExp()`

---

### 4. MemoryWriteLayer.cs
**File:** `src/NeuralNetworks/Layers/MemoryWriteLayer.cs`
**Locations:**
- Line 448: `Math.Sqrt` for scale factor
- Line 555: `Math.Sqrt` for scale factor
- Line 1125: `Math.Sqrt` for scale factor

```csharp
// CURRENT (scalar - called repeatedly)
T scaleFactor = NumOps.FromDouble(1.0 / Math.Sqrt(keys.Shape[1]));
```

**Fix:** These are constant computations per dimension, acceptable as-is. Low priority.

---

### 5. MeshPoolLayer.cs
**File:** `src/NeuralNetworks/Layers/MeshPoolLayer.cs`
**Location:** Line 178

```csharp
double scale = 1.0 / Math.Sqrt(InputChannels);
```

**Fix:** Single scalar computation - acceptable. Low priority.

---

### 6. AnomalyDetectorLayer.cs
**File:** `src/NeuralNetworks/Layers/AnomalyDetectorLayer.cs`
**Location:** Line 447

```csharp
double stdDev = Math.Sqrt(variance);
```

**Fix:** Single scalar - acceptable. Consider `Engine.TensorSqrt()` if computing over tensor.

---

### 7. ContinuumMemorySystemLayer.cs
**File:** `src/NeuralNetworks/Layers/ContinuumMemorySystemLayer.cs`
**Locations:**
- Line 138: `Math.Pow` for frequencies
- Line 149: `Math.Pow` for learning rates

**Issue:** Loop using `Math.Pow`

```csharp
frequencies[i] = (int)Math.Pow(10, i); // 1, 10, 100, 1000
double rate = baseLR / Math.Pow(10, i);
```

**Fix:** Pre-compute power table using `Engine.TensorPow()` if array is large.

---

### 8. ReservoirLayer.cs
**File:** `src/NeuralNetworks/Layers/ReservoirLayer.cs`
**Location:** Line 664

```csharp
if (Math.Abs(_leakingRate - 1.0) < 1e-10)
```

**Fix:** Single scalar comparison - acceptable. Low priority.

---

## HIGH PRIORITY - Point Cloud Models

### 9. DGCNN.cs
**File:** `src/PointCloud/Models/DGCNN.cs`
**Location:** Line 715

```csharp
distances.Add((Math.Sqrt(distSq), j));
```

**Issue:** Scalar sqrt in k-NN distance computation loop

**Fix:** Batch compute all distances into tensor, use `Engine.TensorSqrt()`:
```csharp
// Vectorized k-NN distances
var distancesTensor = new Tensor<T>(new[] { numPoints, numPoints });
// ... compute squared distances into tensor ...
var sqrtDistances = Engine.TensorSqrt(distancesTensor);
```

---

### 10. PointConvolutionLayer.cs
**File:** `src/PointCloud/Layers/PointConvolutionLayer.cs`
**Locations:**
- Line 90: `Math.Sqrt` for weight initialization stddev
- Line 351: `Math.Sqrt`, `Math.Log`, `Math.Sin` for random normal generation

```csharp
// Weight initialization
var stddev = Math.Sqrt(2.0 / inputDim);

// Box-Muller transform (scalar)
double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
```

**Fix:**
- Weight init stddev is single scalar - acceptable
- Box-Muller can be vectorized for batch random generation using `Engine.TensorLog()`, `Engine.TensorSqrt()`, `Engine.TensorSin()`

---

## MEDIUM PRIORITY - Neural Network Models (with scalar loops)

### 11. GenerativeAdversarialNetwork.cs
**Location:** Lines 593, 1924, 1930
**Issue:** Gradient updates in scalar loops

```csharp
for (int i = 0; i < gradients.Length; i++)
    // scalar updates
```

**Fix:** Use `Engine.TensorAdd()`, `Engine.TensorMultiplyScalar()` for bulk gradient updates

---

### 12. DeepBeliefNetwork.cs
**Locations:** Lines 433, 590, 597, 605
**Issue:** Scalar loops for gradient accumulation

**Fix:** Use tensor operations for gradient accumulation

---

### 13. DifferentiableNeuralComputer.cs
**Locations:** Lines 587, 605, 1396, 1404, 1441, 1449, 1455
**Issue:** Scalar loops in memory read/write weightings and softmax

**Fix:** Use `Engine.TensorExp()`, `Engine.Softmax()` for attention weights

---

### 14. EchoStateNetwork.cs
**Locations:** Lines 695, 709, 1600
**Issue:** Scalar loops for vector operations

**Fix:** Use `Engine.Add()`, `Engine.Multiply()` for vectors

---

### 15. BigGAN.cs
**Locations:** Lines 362, 402, 428, 555, 640, 662, 847, 888, 919
**Issue:** Many scalar loops for embedding lookups and gradient updates

**Fix:** Use tensor operations for batch operations

---

## LOW PRIORITY - Already Optimized

The following are in low-level operators and are already SIMD-optimized:
- `src/AiDotNet.Tensors/Operators/*.cs` - These are the building blocks used by IEngine

---

## Implementation Order

1. **Week 1 - High Impact Layers:**
   - PositionalEncodingLayer.cs (most common in Transformer models)
   - SpikingLayer.cs (complex neuron models)
   - DGCNN.cs k-NN distances

2. **Week 2 - Point Cloud:**
   - PointConvolutionLayer.cs random generation
   - DiffusionConvLayer.cs time steps

3. **Week 3 - GAN/Memory Networks:**
   - DifferentiableNeuralComputer.cs memory operations
   - GenerativeAdversarialNetwork.cs gradient loops
   - BigGAN.cs embedding operations

4. **Week 4 - Remaining Models:**
   - DeepBeliefNetwork.cs
   - EchoStateNetwork.cs
   - Other scalar loops

---

## Testing Strategy

For each vectorized layer:
1. Create unit test comparing scalar vs vectorized output
2. Verify numerical accuracy within tolerance (1e-6)
3. Benchmark performance improvement
4. Test on both CPU and GPU engines

---

## Notes

- All vectorizations must maintain .NET Framework 4.6.2 compatibility
- Do NOT use TensorPrimitives (not available in net462)
- Use IEngine methods which abstract CPU/GPU execution
- GPU acceleration was implemented via ILGPU kernels; DirectGpu backends are replacing them for core tensor ops

---

## NEW IENGINE OPERATIONS ✅ IMPLEMENTED

The following operations have been added to IEngine with CPU and GPU (legacy ILGPU) implementations:

### 1. PairwiseDistance / CDist
**Needed by:** DGCNN.cs (k-NN computation), PointNet++.cs (ball query)
**Purpose:** Compute pairwise Euclidean distances between two sets of points

```csharp
// Proposed signature
Tensor<T> PairwiseDistance<T>(Tensor<T> x, Tensor<T> y);
// x: [N, D] - N points with D dimensions
// y: [M, D] - M points with D dimensions  
// Returns: [N, M] distance matrix

// Alternative: squared distances (avoid sqrt)
Tensor<T> PairwiseDistanceSquared<T>(Tensor<T> x, Tensor<T> y);
```

**Current workaround:** Triple nested loop with O(N*M*D) scalar operations
**Benefit:** GPU kernel can parallelize across all N*M pairs

---

### 2. TopK
**Needed by:** DGCNN.cs (k-NN selection), attention mechanisms
**Purpose:** Find k smallest/largest values and their indices along an axis

```csharp
// Proposed signature
(Tensor<T> values, Tensor<int> indices) TopK<T>(Tensor<T> input, int k, int axis = -1, bool largest = true);
// input: [N, M] 
// Returns: values [N, k], indices [N, k]
```

**Current workaround:** Sort + Take in C# (not GPU-accelerated)
**Benefit:** Partial sort is O(N*k) vs full sort O(N*log(N))

---

### 3. TensorCosh / TensorSinh / TensorTanh (element-wise)
**Needed by:** SpikingLayer.cs (soft threshold), various activation functions
**Purpose:** Hyperbolic trigonometric functions

```csharp
// Proposed signatures
Tensor<T> TensorCosh<T>(Tensor<T> tensor);
Tensor<T> TensorSinh<T>(Tensor<T> tensor);
// Note: TensorTanh may already exist as Tanh()
```

**Current workaround:** Scalar Math.Cosh/Math.Sinh in loop
**Benefit:** SIMD/GPU acceleration for neural dynamics

---

### 4. TensorRandN (Gaussian random)
**Needed by:** PointConvolutionLayer.cs, weight initialization, dropout
**Purpose:** Generate tensor of Gaussian random numbers

```csharp
// Proposed signature
Tensor<T> TensorRandN<T>(int[] shape, T mean = 0, T stddev = 1, int? seed = null);
```

**Current workaround:** Box-Muller transform with scalar Math operations
**Benefit:** GPU-accelerated random number generation

---

### 5. TensorWhere / TensorSelect (conditional)
**Needed by:** SpikingLayer.cs (spike detection), ReLU variants
**Purpose:** Element-wise conditional selection

```csharp
// Proposed signature
Tensor<T> TensorWhere<T>(Tensor<T> condition, Tensor<T> x, Tensor<T> y);
// Returns x[i] where condition[i] > 0, else y[i]
```

**Current workaround:** Scalar if/else in loops
**Benefit:** Branchless GPU execution

---

### 6. Scatter / Gather
**Needed by:** Embedding layers, sparse operations, graph neural networks
**Purpose:** Index-based tensor operations

```csharp
// Proposed signatures
Tensor<T> Gather<T>(Tensor<T> input, Tensor<int> indices, int axis);
void Scatter<T>(Tensor<T> target, Tensor<int> indices, Tensor<T> values, int axis);
```

**Current workaround:** Scalar indexing loops
**Benefit:** Critical for embedding lookups and GNN message passing

---

### 7. TensorOuter (outer product)
**Needed by:** PositionalEncodingLayer.cs, attention mechanisms
**Purpose:** Compute outer product of two vectors

```csharp
// Proposed signature
Tensor<T> TensorOuter<T>(Tensor<T> a, Tensor<T> b);
// a: [N], b: [M]
// Returns: [N, M] where result[i,j] = a[i] * b[j]
```

**Current workaround:** Nested loop multiplication
**Benefit:** Single BLAS call or GPU kernel

---

### 8. ArgMax / ArgMin along axis
**Needed by:** Classification outputs, pooling indices, attention
**Purpose:** Find index of max/min value along specified axis

```csharp
// Proposed signature  
Tensor<int> ArgMax<T>(Tensor<T> input, int axis);
Tensor<int> ArgMin<T>(Tensor<T> input, int axis);
```

**Note:** ReduceMax with indices exists but returns flat indices

---

## Priority for New Operations

| Operation | Priority | Impact | Difficulty |
|-----------|----------|--------|------------|
| PairwiseDistance | HIGH | k-NN, ball query | Medium |
| TopK | HIGH | k-NN selection | Medium |
| TensorWhere | HIGH | Spike detection, ReLU | Low |
| Scatter/Gather | HIGH | Embeddings, GNNs | Medium |
| TensorCosh/Sinh | MEDIUM | Neuron models | Low |
| TensorOuter | MEDIUM | Attention, PE | Low |
| TensorRandN | MEDIUM | Initialization | Medium |
| ArgMax/ArgMin | LOW | Already partial support | Low |

---

## Completed Vectorizations

- [x] PositionalEncodingLayer.cs - Now uses TensorSin/TensorCos

---

## Additional Findings from Codebase Analysis

### Sort/TopK Usage Locations (50+ files)
The codebase has extensive use of `.Sort()`, `.OrderBy()`, and `.Take()` patterns:

**High Impact (Neural Network/ML Core):**
- `DistributedTraining/GradientCompressionOptimizer.cs:236` - Sparse gradient TopK
- `ContinualLearning/PackNet.cs:249` - Weight magnitude selection
- `ActiveLearning/Strategies/*.cs` - Multiple TopK for sample selection
- `AutoML/NAS/*.cs` - Architecture search ranking

**Vector Search (RAG):**
- `RetrievalAugmentedGeneration/VectorSearch/Indexes/*.cs` - HNSW, IVF, LSH indices
- `RetrievalAugmentedGeneration/Retrievers/*.cs` - Document retrieval TopK

### New Operations Summary

| Operation | Files Needing It | Category |
|-----------|-----------------|----------|
| TopK | 50+ files | Critical |
| PairwiseDistance | DGCNN, PointNet++, k-NN | Critical |
| ArgSort | 30+ files | High |
| Scatter/Gather | GNNs, Embeddings | High |
| TensorWhere | SpikingLayer, Activations | Medium |

---

## Implementation Status

### Completed
- [x] PositionalEncodingLayer.cs vectorized with TensorSin/TensorCos/TensorOuter
- [x] Legacy ILGPU kernels for Floor/Ceiling/Frac/Sin/Cos/TrilinearInterpolate
- [x] Legacy ILGPU kernels for TopK (production-ready, per-slice selection algorithm)
- [x] Legacy ILGPU kernels for ArgSort (production-ready, insertion sort per slice)
- [x] Legacy ILGPU kernels for Gather (production-ready, parallel indexed reads)
- [x] Legacy ILGPU kernels for Scatter (production-ready, copy + indexed writes)
- [x] Legacy ILGPU kernels for ScatterAdd (production-ready, uses Atomic.Add)
- [x] Legacy ILGPU kernels for TensorCosh/TensorSinh
- [x] Legacy ILGPU kernels for TensorOuter
- [x] Legacy ILGPU kernels for PairwiseDistanceSquared/PairwiseDistance
- [x] SpikingLayer.cs surrogate gradient vectorized with TensorCosh
- [x] DGCNN.cs k-NN vectorized with PairwiseDistanceSquared + TopK

### In Progress
(None - all major items completed)

### Recently Completed (Session 2)
- [x] SpikingLayer.cs Hodgkin-Huxley neuron model vectorized with TensorExp
- [x] SpikingLayer.cs AdEx model vectorized with TensorExp/TensorWhere
- [x] DiffusionConvLayer.cs spectral decay vectorized with TensorExp
- [x] DiffusionConvLayer.cs direct heat kernel vectorized with TensorExp
- [x] GenerativeAdversarialNetwork.cs Box-Muller noise generation vectorized
- [x] GenerativeAdversarialNetwork.cs gradient penalty vectorized with ReduceSum

### No Longer Blocked
- [x] TopK operation - previously available with ILGPU GPU kernel (needs DirectGpu parity)
- [x] PairwiseDistance operation - previously available with ILGPU GPU kernel (needs DirectGpu parity)
- [x] TensorWhere/Select operation - Already exists in IEngine
