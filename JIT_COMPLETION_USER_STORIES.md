# JIT Compilation Completion - User Stories

**Epic**: Complete JIT Compilation for All Activations and 76 Neural Network Layers
**Status**: In Progress
**Created**: 2025-01-23
**Working Directory**: C:\Users\cheat\source\repos\worktrees\pr-487-1763849203

---

## Executive Summary

**Baseline**: 74 existing build errors from incomplete JIT work (acceptable)
**Goal**: Complete all JIT implementations without introducing NEW errors
**Target**: 0 build errors when complete

This epic completes the JIT compilation rollout:
1. **6 pending complex activations** - Forward + backward passes
2. **76 neural network layers** - ExportComputationGraph implementations
3. **Comprehensive code review** - Ensure no regressions

---

## Agent Team Structure

| Agent | Responsibility | Dependencies | Complexity | Estimated Time |
|-------|---------------|--------------|------------|----------------|
| 16 | Sparsemax & SphericalSoftmax | Agent 9 | High | 1-2 days |
| 17 | GumbelSoftmax & TaylorSoftmax | Agent 9 | High | 1-2 days |
| 18 | HierarchicalSoftmax & Maxout | Agent 9 | High | 1-2 days |
| 19 | Core Layers (5) | Agents 9-13 | Moderate | 2-3 days |
| 20 | Recurrent Layers (3) | Agents 9-13 | High | 2-3 days |
| 21 | Attention Layers (3) | Agents 9-13 | High | 2-3 days |
| 22 | Specialized Batch 1 | Agents 9-13 | Moderate | 2-3 days |
| 23 | Specialized Batch 2 | Agents 9-13 | Moderate | 2-3 days |
| 24 | Specialized Batch 3 | Agents 9-13 | Moderate | 2-3 days |
| 25 | Code Review & Validation | Agents 16-24 | Moderate | 2-3 days |

**Timeline**: 3 phases, ~10-15 days with parallel execution

---

## PHASE 2: Complete 6 Complex Activations

---

## Story 1: Sparsemax & SphericalSoftmax (Agent 16)

**Priority**: P1 - HIGH
**Complexity**: High
**Agent**: 16
**Branch**: `feat/sparsemax-spherical-activations`
**Dependencies**: Agent 9 (architecture)
**Estimated Effort**: 1-2 days

### Problem Statement

Agent 12 identified that Sparsemax and SphericalSoftmax need full forward+backward implementation. Currently only method stubs exist with `NotImplementedException`.

### Sparsemax

**Definition**: Sparse softmax that produces sparse probability distributions.

**Forward Pass**:
```
sparsemax(z) = argmin_{p ∈ Δ^n} ||p - z||²
```
Where Δ^n is the probability simplex (elements sum to 1, all ≥ 0).

**Algorithm** (Euclidean Projection onto Simplex):
```
1. Sort z in descending order: z̃
2. Find k = max{j : 1 + j * z̃_j > Σ_{i=1}^j z̃_i}
3. τ = (Σ_{i=1}^k z̃_i - 1) / k
4. sparsemax(z) = max(z - τ, 0)
```

**Gradient**:
```
∂sparsemax(z)/∂z = diag(S) - (1/|S|) * (s * s^T)
where S = support(sparsemax(z)) = {i : sparsemax(z)_i > 0}
      s = indicator vector for S
```

### SphericalSoftmax

**Definition**: Projects onto unit sphere, then applies softmax.

**Forward Pass**:
```
spherical_softmax(x) = softmax(x / ||x||)
```

**Gradient**:
```
Let y = x / ||x|| (L2 normalization)
Let s = softmax(y)

∂spherical_softmax/∂x = (1/||x||) * J_softmax(y) * J_normalize(x)

where J_normalize(x) = (I - x*x^T/||x||²) / ||x||
```

### Acceptance Criteria

#### 1. Implement Sparsemax Forward Pass

**File**: `src/Autodiff/TensorOperations.cs`

```csharp
public static ComputationNode<T> Sparsemax<T>(ComputationNode<T> input) where T : struct
{
    if (input == null) throw new ArgumentNullException(nameof(input));
    if (input.Engine == null) throw new InvalidOperationException("Engine required");

    var result = input.Engine.Sparsemax(input.Value);
    var node = new ComputationNode<T>(result, input.Engine, "Sparsemax");

    node.Backward = (gradOutput) =>
    {
        if (input.RequiresGrad)
        {
            // Implement sparsemax Jacobian-vector product
            var inputValue = input.Value;
            var sparsemaxOutput = result;
            var gradInput = new Tensor<T>(inputValue.Shape);

            int batchSize = gradOutput.Shape[0];
            int numClasses = gradOutput.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                // Find support S = {i : sparsemax(z)_i > 0}
                var support = new List<int>();
                for (int i = 0; i < numClasses; i++)
                {
                    if (NumOps.GreaterThan(sparsemaxOutput[b, i], NumOps.Zero))
                    {
                        support.Add(i);
                    }
                }

                int supportSize = support.Count;
                if (supportSize == 0) continue;

                // Compute v_S = gradOutput restricted to support
                // Compute sum(v_S)
                T sumVS = NumOps.Zero;
                for (int j = 0; j < supportSize; j++)
                {
                    sumVS = NumOps.Add(sumVS, gradOutput[b, support[j]]);
                }

                // Compute gradient: v_S - (1/|S|) * sum(v_S)
                T avgSum = NumOps.Divide(sumVS, NumOps.FromDouble(supportSize));

                for (int i = 0; i < numClasses; i++)
                {
                    if (support.Contains(i))
                    {
                        gradInput[b, i] = NumOps.Subtract(gradOutput[b, i], avgSum);
                    }
                    else
                    {
                        gradInput[b, i] = NumOps.Zero;
                    }
                }
            }

            input.AccumulateGrad(gradInput);
        }
    };

    return node;
}
```

#### 2. Implement Sparsemax in IEngine

**File**: `src/Engines/IEngine.cs`

```csharp
/// <summary>
/// Applies Sparsemax activation function.
/// </summary>
Tensor<T> Sparsemax<T>(Tensor<T> input) where T : struct;
```

**File**: `src/Engines/CpuEngine.cs`

```csharp
public Tensor<T> Sparsemax<T>(Tensor<T> input) where T : struct
{
    if (input.Rank != 2)
        throw new ArgumentException("Sparsemax requires 2D input [batch, features]");

    int batchSize = input.Shape[0];
    int numClasses = input.Shape[1];
    var output = new Tensor<T>(input.Shape);

    for (int b = 0; b < batchSize; b++)
    {
        // Extract row for this batch
        var z = new double[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            z[i] = NumOps.ToDouble(input[b, i]);
        }

        // Sort in descending order
        var zSorted = new double[numClasses];
        var indices = new int[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            zSorted[i] = z[i];
            indices[i] = i;
        }
        Array.Sort(zSorted, indices);
        Array.Reverse(zSorted);
        Array.Reverse(indices);

        // Find k
        int k = 0;
        double cumSum = 0.0;
        for (int j = 0; j < numClasses; j++)
        {
            cumSum += zSorted[j];
            if (1.0 + (j + 1) * zSorted[j] > cumSum)
            {
                k = j + 1;
            }
        }

        // Compute threshold τ
        double tau = 0.0;
        if (k > 0)
        {
            double sumTopK = 0.0;
            for (int i = 0; i < k; i++)
            {
                sumTopK += zSorted[i];
            }
            tau = (sumTopK - 1.0) / k;
        }

        // Apply sparsemax: max(z - τ, 0)
        for (int i = 0; i < numClasses; i++)
        {
            double value = Math.Max(z[i] - tau, 0.0);
            output[b, i] = NumOps.FromDouble(value);
        }
    }

    return output;
}
```

**File**: `src/Engines/GpuEngine.cs` - Same implementation (GPU optimization later)

#### 3. Implement SphericalSoftmax Forward Pass

**File**: `src/Autodiff/TensorOperations.cs`

```csharp
public static ComputationNode<T> SphericalSoftmax<T>(ComputationNode<T> input) where T : struct
{
    if (input == null) throw new ArgumentNullException(nameof(input));
    if (input.Engine == null) throw new InvalidOperationException("Engine required");

    var result = input.Engine.SphericalSoftmax(input.Value);
    var node = new ComputationNode<T>(result, input.Engine, "SphericalSoftmax");

    node.Backward = (gradOutput) =>
    {
        if (input.RequiresGrad)
        {
            var inputValue = input.Value;
            var sphericalOutput = result;
            var gradInput = new Tensor<T>(inputValue.Shape);

            int batchSize = gradOutput.Shape[0];
            int numClasses = gradOutput.Shape[1];

            for (int b = 0; b < batchSize; b++)
            {
                // Compute ||x||
                T normSquared = NumOps.Zero;
                for (int i = 0; i < numClasses; i++)
                {
                    var xi = inputValue[b, i];
                    normSquared = NumOps.Add(normSquared, NumOps.Multiply(xi, xi));
                }
                var norm = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(normSquared)));

                // Normalized input: y = x / ||x||
                // Softmax Jacobian part (same as regular softmax)
                T dotProduct = NumOps.Zero;
                for (int i = 0; i < numClasses; i++)
                {
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(gradOutput[b, i], sphericalOutput[b, i]));
                }

                // Normalization Jacobian part
                // grad = (1/||x||) * [softmax_grad - (x/||x||^2) * (x^T * softmax_grad)]
                T xDotSoftmaxGrad = NumOps.Zero;
                for (int i = 0; i < numClasses; i++)
                {
                    var softmaxGrad = NumOps.Subtract(gradOutput[b, i],
                        NumOps.Multiply(sphericalOutput[b, i], dotProduct));
                    xDotSoftmaxGrad = NumOps.Add(xDotSoftmaxGrad,
                        NumOps.Multiply(inputValue[b, i], softmaxGrad));
                }

                var normCubed = NumOps.Multiply(norm, NumOps.Multiply(norm, norm));

                for (int i = 0; i < numClasses; i++)
                {
                    var softmaxGrad = NumOps.Subtract(gradOutput[b, i],
                        NumOps.Multiply(sphericalOutput[b, i], dotProduct));

                    var term1 = NumOps.Divide(softmaxGrad, norm);
                    var term2 = NumOps.Divide(
                        NumOps.Multiply(inputValue[b, i], xDotSoftmaxGrad),
                        normCubed);

                    gradInput[b, i] = NumOps.Subtract(term1, term2);
                }
            }

            input.AccumulateGrad(gradInput);
        }
    };

    return node;
}
```

#### 4. Implement SphericalSoftmax in IEngine

**Similar pattern to Sparsemax**: Add to IEngine interface, implement in CpuEngine and GpuEngine.

#### 5. Update Activation Classes

**File**: `src/ActivationFunctions/SparsemaxActivation.cs`

```csharp
public override bool SupportsJitCompilation => true;

public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
{
    if (input == null) throw new ArgumentNullException(nameof(input));
    return TensorOperations<T>.Sparsemax(input);
}
```

**File**: `src/ActivationFunctions/SphericalSoftmaxActivation.cs` - Same pattern

### Success Criteria

- [ ] Sparsemax forward pass mathematically correct
- [ ] Sparsemax gradient correct (use numerical gradient check)
- [ ] SphericalSoftmax forward pass correct
- [ ] SphericalSoftmax gradient correct
- [ ] Both activations set `SupportsJitCompilation => true`
- [ ] Build succeeds (≤ 74 errors)
- [ ] No new errors introduced

---

## Story 2: GumbelSoftmax & TaylorSoftmax (Agent 17)

**Priority**: P1 - HIGH
**Complexity**: High
**Agent**: 17
**Branch**: `feat/gumbel-taylor-activations`
**Dependencies**: Agent 9
**Estimated Effort**: 1-2 days

### GumbelSoftmax

**Definition**: Differentiable approximation to categorical sampling.

**Forward Pass** (training):
```
gumbel_softmax(x, τ) = softmax((log(x) + g) / τ)
where g ~ Gumbel(0, 1) = -log(-log(u)), u ~ Uniform(0, 1)
      τ = temperature parameter (typically 0.1 to 10)
```

**Forward Pass** (inference):
```
gumbel_softmax(x, τ) = softmax(x / τ)  // No noise
```

**Gradient** (straight-through estimator):
```
Forward: discrete (argmax)
Backward: continuous (softmax gradient)
```

### TaylorSoftmax

**Definition**: Taylor series approximation of softmax.

**Forward Pass** (2nd order):
```
taylor_softmax(x) = (1 + x + x²/2) / Σ(1 + x_i + x_i²/2)
```

**Gradient**:
```
Chain rule applied to rational function
```

### Implementation Pattern

Similar to Sparsemax - implement in TensorOperations, IEngine, CpuEngine, GpuEngine, and activation classes.

### Success Criteria

- [ ] GumbelSoftmax with temperature parameter
- [ ] TaylorSoftmax with configurable Taylor order (default 2)
- [ ] Both gradients mathematically correct
- [ ] Both set `SupportsJitCompilation => true`
- [ ] Build succeeds (≤ 74 errors)

---

## Story 3: HierarchicalSoftmax & Maxout (Agent 18)

**Priority**: P1 - HIGH
**Complexity**: High
**Agent**: 18
**Branch**: `feat/hierarchical-maxout-activations`
**Dependencies**: Agent 9
**Estimated Effort**: 1-2 days

### HierarchicalSoftmax

**Definition**: Softmax over hierarchical tree structure (efficient for large vocabularies).

**Forward Pass** (binary tree):
```
P(class) = Π_{node on path} σ(±x · w_node)
```

**Implementation Strategy**:
- Use balanced binary tree
- Each node has learnable weight vector
- Path probabilities multiply

### Maxout

**Definition**: Takes maximum over affine feature groups.

**Forward Pass**:
```
maxout(x) = max_{i ∈ groups} (W_i · x + b_i)
```

**Gradient**:
```
∂maxout/∂x = W_k where k = argmax_i (W_i · x + b_i)
```

### Success Criteria

- [ ] HierarchicalSoftmax with binary tree structure
- [ ] Maxout with configurable group size
- [ ] Both gradients correct
- [ ] Both set `SupportsJitCompilation => true`
- [ ] Build succeeds (≤ 74 errors)

---

## PHASE 3: Implement JIT for 76 Neural Network Layers

---

## Layer Inventory (76 Total)

### Core Layers (5)
1. ConvolutionalLayer
2. BatchNormalizationLayer
3. LayerNormalizationLayer
4. DropoutLayer
5. PoolingLayer

### Recurrent Layers (3)
6. LSTMLayer
7. GRULayer
8. RNNLayer

### Attention Layers (3)
9. AttentionLayer
10. MultiHeadAttentionLayer
11. SelfAttentionLayer

### Specialized Layers (65)
12. EmbeddingLayer
13. ResidualLayer
14. HighwayLayer
15. GroupNormalizationLayer
16. InstanceNormalizationLayer
17. AdaptivePoolingLayer
... (59 more)

---

## Story 4: Core Layers - Conv, Norm, Dropout, Pool (Agent 19)

**Priority**: P0 - CRITICAL (Most commonly used)
**Complexity**: Moderate
**Agent**: 19
**Branch**: `feat/core-layers-jit`
**Dependencies**: Agents 9-13
**Estimated Effort**: 2-3 days

### Your Task

Implement `ExportComputationGraph` for 5 core layers using the established pattern from DenseLayer.

### Pattern to Follow

**From DenseLayer.cs lines 1163-1223**:

```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // 1. Validation
    if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
    if (_weights == null) throw new InvalidOperationException("Weights not initialized");
    if (!CanActivationBeJitted()) throw new NotSupportedException("Activation not supported");

    // 2. Create placeholder inputs
    var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "input");
    var weightsNode = TensorOperations<T>.Variable(weightsPlaceholder, "weights");

    // 3. Add to inputNodes list
    inputNodes.Add(inputNode);
    inputNodes.Add(weightsNode);

    // 4. Build computation graph (layer-specific logic)
    var weightsTransposed = TensorOperations<T>.Transpose(weightsNode);
    var matmulResult = TensorOperations<T>.MatrixMultiply(inputNode, weightsTransposed);
    var outputNode = TensorOperations<T>.Add(matmulResult, biasesNode);

    // 5. Apply activation using LayerBase helper (NO if/else chains!)
    var activatedOutput = ApplyActivationToGraph(outputNode);

    return activatedOutput;
}
```

### Layer 1: ConvolutionalLayer

**File**: `src/NeuralNetworks/Layers/ConvolutionalLayer.cs`

**Inputs**: Input tensor [batch, channels, height, width], Filters, Biases
**Operation**: Conv2D → Add Bias → Activation
**Output**: [batch, out_channels, out_height, out_width]

**Implementation**:
```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // Validation
    if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
    if (_filters == null) throw new InvalidOperationException("Filters not initialized");
    if (!CanActivationBeJitted()) throw new NotSupportedException($"Activation not supported");

    // Placeholders
    var inputPlaceholder = new Tensor<T>(new int[] { 1, _inputChannels, _inputHeight, _inputWidth });
    var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "input");

    var filtersPlaceholder = new Tensor<T>(_filters.Shape, _filters);
    var filtersNode = TensorOperations<T>.Variable(filtersPlaceholder, "filters");

    inputNodes.Add(inputNode);
    inputNodes.Add(filtersNode);

    // Convolution (needs TensorOperations.Conv2D method)
    var convNode = TensorOperations<T>.Conv2D(inputNode, filtersNode, _stride, _padding);

    // Add bias if present
    ComputationNode<T> outputNode = convNode;
    if (_biases != null)
    {
        var biasesPlaceholder = new Tensor<T>(_biases.Shape, _biases);
        var biasesNode = TensorOperations<T>.Variable(biasesPlaceholder, "biases");
        inputNodes.Add(biasesNode);
        outputNode = TensorOperations<T>.Add(convNode, biasesNode);
    }

    // Apply activation using LayerBase helper
    var activatedOutput = ApplyActivationToGraph(outputNode);

    return activatedOutput;
}
```

**Prerequisites**:
- Need `TensorOperations<T>.Conv2D()` method
- Need `IEngine.Conv2D()` method

### Layer 2: BatchNormalizationLayer

**Operation**: (x - mean) / sqrt(variance + epsilon) * gamma + beta

**Implementation Pattern**:
```csharp
// Running mean and variance as constant nodes
var meanNode = TensorOperations<T>.Variable(runningMean, "mean");
var varianceNode = TensorOperations<T>.Variable(runningVariance, "variance");

// Normalize
var centered = TensorOperations<T>.Subtract(inputNode, meanNode);
var denominator = TensorOperations<T>.Sqrt(
    TensorOperations<T>.Add(varianceNode, epsilonNode));
var normalized = TensorOperations<T>.Divide(centered, denominator);

// Scale and shift
var scaled = TensorOperations<T>.Multiply(normalized, gammaNode);
var output = TensorOperations<T>.Add(scaled, betaNode);
```

### Layer 3: LayerNormalizationLayer

**Operation**: Similar to batch norm but normalizes across features

### Layer 4: DropoutLayer

**Operation** (inference): Identity
**Operation** (training with JIT): Scale by (1 - dropout_rate)

**Implementation**:
```csharp
// For JIT, dropout is typically disabled (inference mode)
// Just return input unchanged or scaled
var scaleNode = TensorOperations<T>.Variable(
    new Tensor<T>(new int[] {1}, new T[] { NumOps.FromDouble(1.0 / (1.0 - _dropoutRate)) }),
    "dropoutScale");
var outputNode = TensorOperations<T>.Multiply(inputNode, scaleNode);
```

### Layer 5: PoolingLayer (Max/Average)

**Operation**: Reduce over spatial dimensions

**Prerequisites**:
- Need `TensorOperations<T>.MaxPool2D()` method
- Need `TensorOperations<T>.AvgPool2D()` method

### Success Criteria

- [ ] All 5 layers implement ExportComputationGraph
- [ ] All use LayerBase.ApplyActivationToGraph helper
- [ ] All set SupportsJitCompilation appropriately
- [ ] Build succeeds (≤ 74 errors)
- [ ] No if/else chains for activation handling

---

## Story 5: Recurrent Layers - LSTM, GRU, RNN (Agent 20)

**Priority**: P1 - HIGH
**Complexity**: High (stateful, complex gates)
**Agent**: 20
**Branch**: `feat/recurrent-layers-jit`
**Dependencies**: Agents 9-13
**Estimated Effort**: 2-3 days

### Challenge

Recurrent layers have **sequential dependencies** and **hidden state** which makes JIT compilation more complex.

### Strategy

For JIT compilation, unroll for a **fixed sequence length** or implement as a **single time step**.

### LSTM Forward Pass (Single Time Step)

**Gates**:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  // Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  // Input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  // Output gate
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  // Cell candidate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
h_t = o_t ⊙ tanh(c_t)
```

**Implementation**:
```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // Input: x_t (current input)
    // Hidden state: h_{t-1} (passed as input)
    // Cell state: c_{t-1} (passed as input)

    // Concatenate [h_{t-1}, x_t]
    var concat = TensorOperations<T>.Concatenate(hiddenNode, inputNode, axis: 1);

    // Forget gate
    var f_t = TensorOperations<T>.Sigmoid(
        TensorOperations<T>.Add(
            TensorOperations<T>.MatrixMultiply(concat, W_f),
            b_f));

    // Input gate
    var i_t = TensorOperations<T>.Sigmoid(
        TensorOperations<T>.Add(
            TensorOperations<T>.MatrixMultiply(concat, W_i),
            b_i));

    // Output gate
    var o_t = TensorOperations<T>.Sigmoid(
        TensorOperations<T>.Add(
            TensorOperations<T>.MatrixMultiply(concat, W_o),
            b_o));

    // Cell candidate
    var c_tilde = TensorOperations<T>.Tanh(
        TensorOperations<T>.Add(
            TensorOperations<T>.MatrixMultiply(concat, W_c),
            b_c));

    // New cell state
    var c_t = TensorOperations<T>.Add(
        TensorOperations<T>.Multiply(f_t, c_prev),
        TensorOperations<T>.Multiply(i_t, c_tilde));

    // New hidden state
    var h_t = TensorOperations<T>.Multiply(
        o_t,
        TensorOperations<T>.Tanh(c_t));

    return h_t;  // Output hidden state
}
```

### Success Criteria

- [ ] LSTM single-step forward pass in JIT
- [ ] GRU single-step forward pass in JIT
- [ ] RNN (simple recurrent) forward pass in JIT
- [ ] Build succeeds (≤ 74 errors)

---

## Story 6: Attention Layers (Agent 21)

**Priority**: P1 - HIGH (Transformers)
**Complexity**: High
**Agent**: 21
**Branch**: `feat/attention-layers-jit`
**Dependencies**: Agents 9-13
**Estimated Effort**: 2-3 days

### Attention Mechanism

**Formula**:
```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

**Implementation**:
```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // Q, K, V as inputs
    var Q = inputNodes[0];
    var K = inputNodes[1];
    var V = inputNodes[2];

    // Compute scores: Q·K^T
    var K_T = TensorOperations<T>.Transpose(K);
    var scores = TensorOperations<T>.MatrixMultiply(Q, K_T);

    // Scale by sqrt(d_k)
    var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_d_k));
    var scaledScores = TensorOperations<T>.Multiply(scores, scaleNode);

    // Apply softmax
    var attention_weights = TensorOperations<T>.Softmax(scaledScores);

    // Multiply by V
    var output = TensorOperations<T>.MatrixMultiply(attention_weights, V);

    return output;
}
```

### Multi-Head Attention

**Formula**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

### Success Criteria

- [ ] AttentionLayer JIT implementation
- [ ] MultiHeadAttentionLayer JIT implementation
- [ ] SelfAttentionLayer JIT implementation
- [ ] Build succeeds (≤ 74 errors)

---

## Story 7-9: Specialized Layers Batches (Agents 22-24)

**Priority**: P2 - MEDIUM
**Complexity**: Moderate
**Agents**: 22, 23, 24
**Branches**: `feat/specialized-layers-batch-{1,2,3}`
**Dependencies**: Agents 9-13
**Estimated Effort**: 2-3 days each

### Batch 1 (Agent 22) - 22 Layers

12. EmbeddingLayer
13. ResidualLayer
14. HighwayLayer
15. GroupNormalizationLayer
16. InstanceNormalizationLayer
17. AdaptivePoolingLayer
18. FlattenLayer
19. ReshapeLayer
20. UpSamplingLayer
21. ZeroPaddingLayer
22. CroppingLayer
23. RepeatVectorLayer
24. PermuteLayer
25. MaskingLayer
26. SpatialDropoutLayer
27. AlphaDropoutLayer
28. GaussianDropoutLayer
29. GaussianNoiseLayer
30. ActivityRegularizationLayer
31. LocallyConnectedLayer
32. DepthwiseConvolutionalLayer
33. SeparableConvolutionalLayer

### Batch 2 (Agent 23) - 22 Layers

34. Deconvolution/TransposeConvLayer
35. DilatedConvolutionalLayer
36. BilinearLayer
37. TimeDistributedLayer
38. BidirectionalLayer
39. ConvLSTMLayer
40. SimpleRNNLayer
41. MinPoolingLayer
42. GlobalMaxPoolingLayer
43. GlobalAveragePoolingLayer
44. FractionalPoolingLayer
45. AdditiveAttentionLayer
46. DotProductAttentionLayer
47. LocationBasedAttentionLayer
48. ContentBasedAttentionLayer
49. ConcatenateLayer
50. AverageLayer
51. MaximumLayer
52. MinimumLayer
53. MultiplyLayer
54. DotProductLayer
55. SubtractLayer

### Batch 3 (Agent 24) - 21 Layers

56. AddLayer
57. UpsamplingBilinearLayer
58. UpsamplingNearestLayer
59. RandomRotationLayer
60. RandomZoomLayer
61. RandomFlipLayer
62. RandomCropLayer
63. RandomTranslationLayer
64. RandomContrastLayer
65. RandomBrightnessLayer
66. CenterCropLayer
67. RescalingLayer
68. NormalizationLayer (general)
69. StandardizationLayer
70. L1NormalizationLayer
71. L2NormalizationLayer
72. UnitNormalizationLayer
73. SpectralNormalizationLayer
74. WeightNormalizationLayer
75. PixelShuffleLayer
76. DepthToSpaceLayer
77. SpaceToDepthLayer

### Implementation Pattern (Same for All)

```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // 1. Validation
    if (inputNodes == null) throw new ArgumentNullException(nameof(inputNodes));
    if (!CanActivationBeJitted()) throw new NotSupportedException("Activation not supported");

    // 2. Create placeholders for layer parameters
    // 3. Build layer-specific computation graph
    // 4. Apply activation using LayerBase.ApplyActivationToGraph()

    return activatedOutput;
}

public override bool SupportsJitCompilation => CanActivationBeJitted();
```

### Success Criteria (Each Batch)

- [ ] All assigned layers implement ExportComputationGraph
- [ ] All use LayerBase helper for activations
- [ ] Build succeeds (≤ 74 errors)
- [ ] No new errors introduced

---

## PHASE 4: Code Review & Validation

---

## Story 10: Comprehensive Code Review (Agent 25)

**Priority**: P0 - CRITICAL (Final gate)
**Complexity**: Moderate
**Agent**: 25
**Branch**: N/A (reviews all PRs)
**Dependencies**: Agents 16-24
**Estimated Effort**: 2-3 days

### Your Mission

Review all work from Agents 16-24 and ensure production quality.

### Review Checklist

#### Phase 2: Activation Review

**Agent 16 (Sparsemax & SphericalSoftmax)**:
- [ ] Sparsemax forward pass correct (Euclidean projection)
- [ ] Sparsemax gradient correct (numerical check)
- [ ] SphericalSoftmax forward pass correct (normalize + softmax)
- [ ] SphericalSoftmax gradient correct
- [ ] Both set SupportsJitCompilation => true
- [ ] IEngine methods implemented in CPU and GPU engines

**Agent 17 (GumbelSoftmax & TaylorSoftmax)**:
- [ ] GumbelSoftmax with proper Gumbel noise
- [ ] GumbelSoftmax temperature parameter working
- [ ] TaylorSoftmax Taylor series correct
- [ ] Both gradients correct
- [ ] Both set SupportsJitCompilation => true

**Agent 18 (HierarchicalSoftmax & Maxout)**:
- [ ] HierarchicalSoftmax tree structure defined
- [ ] HierarchicalSoftmax path probabilities correct
- [ ] Maxout group reduction correct
- [ ] Both gradients correct
- [ ] Both set SupportsJitCompilation => true

#### Phase 3: Layer Review

**Agent 19 (Core Layers)**:
- [ ] ConvolutionalLayer JIT works with all supported activations
- [ ] BatchNormalizationLayer normalization correct
- [ ] LayerNormalizationLayer correct
- [ ] DropoutLayer inference mode correct
- [ ] PoolingLayer (max/avg) correct
- [ ] All use LayerBase.ApplyActivationToGraph (no if/else chains)
- [ ] All set SupportsJitCompilation correctly

**Agent 20 (Recurrent Layers)**:
- [ ] LSTM gates computed correctly
- [ ] GRU gates computed correctly
- [ ] RNN single-step correct
- [ ] All use LayerBase helper

**Agent 21 (Attention Layers)**:
- [ ] Attention scaling correct (sqrt(d_k))
- [ ] MultiHeadAttention concatenation correct
- [ ] SelfAttention correct
- [ ] All use LayerBase helper

**Agents 22-24 (Specialized Layers)**:
- [ ] All 65 layers implement ExportComputationGraph
- [ ] All use LayerBase helper
- [ ] All set SupportsJitCompilation correctly

#### Build Validation

**Critical Requirement**: Ensure ≤ 74 errors

```bash
# Build all target frameworks
dotnet build -c Release -f net462 2>&1 | tee build_net462.txt
dotnet build -c Release -f net471 2>&1 | tee build_net471.txt
dotnet build -c Release -f netstandard2.0 2>&1 | tee build_netstandard20.txt

# Count errors
grep "error CS" build_net462.txt | wc -l
grep "error CS" build_net471.txt | wc -l
grep "error CS" build_netstandard20.txt | wc -l

# MUST be ≤ 74 (ideally 0)
```

#### Integration Testing

Test sampling of layers:
```csharp
// Test 1: DenseLayer with all activations
for each activation in [ReLU, Sigmoid, Tanh, GELU, etc.]
{
    var layer = new DenseLayer<double>(10, 5, activation);
    var graph = layer.ExportComputationGraph(new List<ComputationNode<double>>());
    // Should succeed if activation.SupportsJitCompilation == true
}

// Test 2: ConvolutionalLayer
var conv = new ConvolutionalLayer<double>(3, 16, 3, 3, new ReLUActivation<double>());
var convGraph = conv.ExportComputationGraph(...);
// Should succeed

// Test 3: LSTM
var lstm = new LSTMLayer<double>(50, 100);
var lstmGraph = lstm.ExportComputationGraph(...);
// Should succeed

// Test 4: Attention
var attention = new AttentionLayer<double>(512);
var attentionGraph = attention.ExportComputationGraph(...);
// Should succeed
```

### Deliverables

1. **Comprehensive Validation Report**: `JIT_COMPLETION_VALIDATION_REPORT.md`

Include:
- Summary of all agent work (16-24)
- Issues found and resolution status
- Build error count (must be ≤ 74)
- Test results
- Final approval/rejection for each PR

2. **Approval Status**:
- PR from Agent 16: ✅ or ❌ with reasons
- PR from Agent 17: ✅ or ❌ with reasons
- PR from Agent 18: ✅ or ❌ with reasons
- PR from Agent 19: ✅ or ❌ with reasons
- PR from Agent 20: ✅ or ❌ with reasons
- PR from Agent 21: ✅ or ❌ with reasons
- PR from Agent 22: ✅ or ❌ with reasons
- PR from Agent 23: ✅ or ❌ with reasons
- PR from Agent 24: ✅ or ❌ with reasons

3. **Merge Order Recommendation**

4. **Final Statistics**:
- Total activations: 37/37 complete (100%)
- Total layers: 76/76 complete (100%)
- Build errors: X (must be ≤ 74, target 0)
- Code quality: ✅ or ❌

### Success Criteria

- [ ] All agents' work reviewed
- [ ] Build errors ≤ 74 (target 0)
- [ ] All PRs approved or issues documented
- [ ] Integration tests passing
- [ ] Validation report created
- [ ] Ready for production deployment

---

## Git Workflow

### Worktree Structure

```bash
# Activation agents (Phase 2)
git worktree add ../worktrees/jit-agent-16-sparsemax -b feat/sparsemax-spherical-activations master
git worktree add ../worktrees/jit-agent-17-gumbel -b feat/gumbel-taylor-activations master
git worktree add ../worktrees/jit-agent-18-hierarchical -b feat/hierarchical-maxout-activations master

# Layer agents (Phase 3)
git worktree add ../worktrees/jit-agent-19-core -b feat/core-layers-jit master
git worktree add ../worktrees/jit-agent-20-recurrent -b feat/recurrent-layers-jit master
git worktree add ../worktrees/jit-agent-21-attention -b feat/attention-layers-jit master
git worktree add ../worktrees/jit-agent-22-specialized-1 -b feat/specialized-layers-batch-1 master
git worktree add ../worktrees/jit-agent-23-specialized-2 -b feat/specialized-layers-batch-2 master
git worktree add ../worktrees/jit-agent-24-specialized-3 -b feat/specialized-layers-batch-3 master

# Review agent uses main worktree
```

### PR Strategy

- Agent 16 → PR #509
- Agent 17 → PR #510
- Agent 18 → PR #511
- Agent 19 → PR #512
- Agent 20 → PR #513
- Agent 21 → PR #514
- Agent 22 → PR #515
- Agent 23 → PR #516
- Agent 24 → PR #517

### Merge Order

**Phase 2 (Activations) can merge first**:
1. PRs #509, #510, #511 (any order)

**Phase 3 (Layers) can merge after Phase 2**:
2. PR #512 (Core Layers) - RECOMMENDED FIRST (most used)
3. PRs #513-517 (any order)

---

## Timeline

**Phase 2** (Agents 16-18): Days 1-2 (parallel)
- 3 agents working simultaneously on activations

**Phase 3** (Agents 19-24): Days 3-8 (parallel)
- 6 agents working simultaneously on layer batches

**Phase 4** (Agent 25): Days 9-11
- Code review, integration testing, validation

**Total**: 10-15 days with parallel execution

---

## Success Metrics

| Metric | Target | Critical |
|--------|--------|----------|
| Activations Complete | 37/37 (100%) | ✅ |
| Layers with JIT | 76/76 (100%) | ✅ |
| Build Errors | ≤ 74 (target 0) | ✅ |
| Code Quality Violations | 0 | ✅ |
| Open/Closed Principle | 100% compliant | ✅ |
| Test Coverage | Sampling | ⚠️ |

---

## Risk Mitigation

**Risk**: Activation gradients incorrect
**Mitigation**: Numerical gradient checking for all 6 new activations

**Risk**: Layer implementations incorrect
**Mitigation**: Agent 25 comprehensive review + integration tests

**Risk**: Build errors increase
**Mitigation**: Track error count after each agent, flag immediately if > 74

**Risk**: Performance regressions
**Mitigation**: Benchmark critical paths (defer to later if needed)

---

END OF USER STORIES
