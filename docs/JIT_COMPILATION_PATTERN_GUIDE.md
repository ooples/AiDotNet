# JIT Compilation Pattern Guide

## Overview

### What is JIT Compilation in AiDotNet?

Just-In-Time (JIT) compilation in AiDotNet is a performance optimization technique that compiles neural network layers into optimized computation graphs **before** training or inference begins. This allows the framework to:

1. **Optimize the computation graph** - Remove redundant operations, fuse operations together, and apply mathematical simplifications
2. **Generate efficient code** - Convert high-level operations into optimized low-level code that runs on CPU or GPU
3. **Accelerate execution** - Execute the compiled graph much faster than interpreting operations one-by-one

### Performance Benefits

JIT compilation provides significant performance improvements:

- **Target speedup**: 5-10x faster execution compared to eager mode
- **Reduced memory overhead**: Optimized graphs use less temporary memory
- **Better hardware utilization**: Compiled code can better leverage CPU/GPU parallelism
- **Batch efficiency**: Symbolic batch dimensions (-1) allow same compiled graph to handle any batch size

### When to Use JIT Compilation

**Use JIT compilation when:**
- Training or running inference on production models
- Working with large batch sizes (where compilation overhead is amortized)
- Deploying models to resource-constrained environments
- Performance is critical (real-time inference, large-scale training)

**Don't use JIT compilation when:**
- Rapidly prototyping and debugging (eager mode is easier to debug)
- Working with dynamic architectures that change structure frequently
- Batch size is 1 and latency is more important than throughput

### Current Support Status

As of the latest release:

- **Foundation**: Complete (TensorOperations, IEngine integration, IR operations)
- **DenseLayer**: Production-ready with 10 supported activations
- **Other layers**: 76 layers pending implementation (following the same pattern)

**Supported activations (10 ready for production use):**
- ReLU, Sigmoid, Tanh, Softmax, Identity
- GELU, ELU, Mish, Swish, SiLU

**Additional activations (27 available, pending integration):**
- LeakyReLU, SELU, CELU, PReLU, RReLU, ThresholdedReLU
- HardSigmoid, HardTanh, ScaledTanh, Softplus, Softsign, BentIdentity
- Softmin, LogSoftmax, LogSoftmin
- Sign, Gaussian, ISRU, LiSHT, SQRBF, Squash, BinarySpiking
- Sparsemax, SphericalSoftmax, GumbelSoftmax, TaylorSoftmax, HierarchicalSoftmax, Maxout

---

## Supported Activations

The following activations are fully implemented and ready for JIT compilation:

### Scalar Activations (Element-wise)

| Activation | TensorOperations Method | Description | Use Cases |
|------------|------------------------|-------------|-----------|
| **ReLU** | `TensorOperations<T>.ReLU(node)` | Rectified Linear Unit - outputs max(0, x) | Most common activation, default for hidden layers |
| **Sigmoid** | `TensorOperations<T>.Sigmoid(node)` | Sigmoid function - outputs 1/(1+e^(-x)) | Binary classification output, gates in RNNs |
| **Tanh** | `TensorOperations<T>.Tanh(node)` | Hyperbolic tangent - outputs (e^x - e^(-x))/(e^x + e^(-x)) | Alternative to sigmoid, centers output around 0 |
| **GELU** | `TensorOperations<T>.GELU(node)` | Gaussian Error Linear Unit | Used in Transformers (BERT, GPT) |
| **ELU** | `TensorOperations<T>.ELU(node, alpha)` | Exponential Linear Unit | Reduces vanishing gradient problem |
| **Mish** | `TensorOperations<T>.Mish(node)` | Self-regularized smooth activation | Modern alternative to ReLU |
| **Swish** | `TensorOperations<T>.Swish(node)` | Self-gated activation (x * sigmoid(x)) | Google Brain's smooth alternative to ReLU |
| **SiLU** | `TensorOperations<T>.SiLU(node)` | Sigmoid Linear Unit (same as Swish) | Used in modern architectures |
| **LeakyReLU** | `TensorOperations<T>.LeakyReLU(node, slope)` | ReLU with small negative slope | Prevents dying ReLU problem |
| **Identity** | `input` (no-op) | Returns input unchanged | Linear layers, skip connections |

### Vector Activations (Operates on entire vectors)

| Activation | TensorOperations Method | Description | Use Cases |
|------------|------------------------|-------------|-----------|
| **Softmax** | `TensorOperations<T>.Softmax(node, axis)` | Converts logits to probability distribution | Multi-class classification output |

---

## Step-by-Step Implementation Guide

This section shows you exactly how to add JIT compilation support to any neural network layer.

### Prerequisites

Before implementing JIT support, ensure:

1. ✅ Your layer inherits from `LayerBase<T>` or implements `ILayer<T>`
2. ✅ Your layer has a working `Forward()` method
3. ✅ Your layer uses one of the supported activations listed above
4. ✅ Your layer has properly initialized weights and biases

### Step 1: Override ExportComputationGraph

The `ExportComputationGraph` method is the core of JIT compilation. It builds a symbolic representation of your layer's computation that can be optimized and compiled.

```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // 1. Validate inputs
    if (inputNodes == null)
        throw new ArgumentNullException(nameof(inputNodes));

    if (_weights == null)
        throw new InvalidOperationException("Layer weights not initialized. Call Initialize() or train the layer first.");

    if (_biases == null)
        throw new InvalidOperationException("Layer biases not initialized. Call Initialize() or train the layer first.");

    if (InputShape == null || InputShape.Length == 0)
        throw new InvalidOperationException("Layer input shape not configured.");

    if (!CanActivationBeJitted())
    {
        var activationType = ScalarActivation?.GetType().Name ?? VectorActivation?.GetType().Name ?? "unknown";
        throw new NotSupportedException(
            $"Activation function '{activationType}' is not supported for JIT compilation yet. " +
            "Supported activations: ReLU, Sigmoid, Tanh, GELU, ELU, Mish, Swish, SiLU, LeakyReLU, Softmax, Identity");
    }

    // 2. Extract layer dimensions
    int inputSize = InputShape[0];   // e.g., 784 for MNIST
    int outputSize = OutputShape[0]; // e.g., 128 for hidden layer

    // 3. Create input placeholder with symbolic batch dimension
    // The -1 means "any batch size" - allows same compiled graph for batch sizes 1, 32, 128, etc.
    var inputPlaceholder = new Tensor<T>(new int[] { 1, inputSize }); // Actual placeholder is batch size 1
    var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "input");

    // 4. Create parameter nodes for weights and biases
    // Weights shape: [outputSize, inputSize] - transposed for efficient computation
    var weightsNode = TensorOperations<T>.Variable(
        new Tensor<T>(new int[] { _weights.Rows, _weights.Columns }, _weights),
        "weights"
    );

    // Biases shape: [outputSize]
    var biasesNode = TensorOperations<T>.Variable(
        new Tensor<T>(new int[] { _biases.Length }, _biases),
        "biases"
    );

    // 5. Add nodes to input list (required by JIT compiler)
    inputNodes.Add(inputNode);
    inputNodes.Add(weightsNode);
    inputNodes.Add(biasesNode);

    // 6. Build computation graph matching Forward() logic
    // This example shows DenseLayer: output = (input × weights^T) + biases + activation

    // Step 6a: Transpose weights for matrix multiplication
    var weightsTransposed = TensorOperations<T>.Transpose(weightsNode);

    // Step 6b: Matrix multiply: input × weights^T
    var matmulResult = TensorOperations<T>.MatrixMultiply(inputNode, weightsTransposed);

    // Step 6c: Add biases (broadcasts across batch dimension)
    var outputNode = TensorOperations<T>.Add(matmulResult, biasesNode);

    // Step 6d: Apply activation function
    var activatedOutput = ApplyActivationToGraph(outputNode);

    // 7. Return the final output node
    return activatedOutput;
}
```

**Key Points:**

- **Symbolic batch dimension**: Use `-1` in shape to indicate "any batch size". This allows the same compiled graph to handle different batch sizes efficiently.
- **Match Forward() exactly**: The computation graph must produce identical results to your existing `Forward()` method.
- **Parameter ordering matters**: Add nodes to `inputNodes` in the order: input, then parameters (weights, biases, etc.)
- **Use TensorOperations, not IEngine**: `TensorOperations<T>` methods return `ComputationNode<T>`, which is what we need.

### Step 2: Implement ApplyActivationToGraph

This helper method maps your layer's configured activation to the corresponding TensorOperations method.

```csharp
/// <summary>
/// Applies the layer's activation function to a computation graph node.
/// Maps the layer's configured activation to the corresponding TensorOperations method.
/// </summary>
private ComputationNode<T> ApplyActivationToGraph(ComputationNode<T> input)
{
    if (input == null)
        throw new ArgumentNullException(nameof(input));

    // Check scalar activation first (element-wise activations)
    if (ScalarActivation is not null)
    {
        // ReLU family
        if (ScalarActivation is ReLUActivation<T>)
            return TensorOperations<T>.ReLU(input);
        else if (ScalarActivation is LeakyReLUActivation<T> leakyRelu)
            return TensorOperations<T>.LeakyReLU(input, leakyRelu.NegativeSlope);

        // Sigmoid family
        else if (ScalarActivation is SigmoidActivation<T>)
            return TensorOperations<T>.Sigmoid(input);
        else if (ScalarActivation is TanhActivation<T>)
            return TensorOperations<T>.Tanh(input);
        else if (ScalarActivation is SwishActivation<T>)
            return TensorOperations<T>.Swish(input);
        else if (ScalarActivation is SiLUActivation<T>)
            return TensorOperations<T>.SiLU(input);
        else if (ScalarActivation is MishActivation<T>)
            return TensorOperations<T>.Mish(input);

        // Modern activations
        else if (ScalarActivation is GELUActivation<T>)
            return TensorOperations<T>.GELU(input);
        else if (ScalarActivation is ELUActivation<T> elu)
            return TensorOperations<T>.ELU(input, elu.Alpha);

        // Identity (no-op)
        else if (ScalarActivation is IdentityActivation<T>)
            return input;

        // Unsupported activation
        else
            throw new NotSupportedException(
                $"Activation {ScalarActivation.GetType().Name} is not supported for JIT compilation yet");
    }

    // Check vector activation (operates on entire vectors)
    if (VectorActivation is not null)
    {
        if (VectorActivation is SoftmaxActivation<T>)
            return TensorOperations<T>.Softmax(input);
        else
            throw new NotSupportedException(
                $"Activation {VectorActivation.GetType().Name} is not supported for JIT compilation yet");
    }

    // No activation configured (identity)
    return input;
}
```

**Key Points:**

- **Check both ScalarActivation and VectorActivation**: Layers can have either type
- **Parameterized activations**: Some activations like LeakyReLU and ELU have parameters - extract and pass them
- **Identity is a no-op**: Just return the input unchanged
- **Clear error messages**: Tell users which activations are not yet supported

### Step 3: Implement CanActivationBeJitted

This helper method checks if the layer's current activation is supported for JIT compilation.

```csharp
/// <summary>
/// Checks if the layer's current activation function is supported for JIT compilation.
/// </summary>
private bool CanActivationBeJitted()
{
    // Check scalar activations
    if (ScalarActivation is ReLUActivation<T> ||
        ScalarActivation is SigmoidActivation<T> ||
        ScalarActivation is TanhActivation<T> ||
        ScalarActivation is GELUActivation<T> ||
        ScalarActivation is ELUActivation<T> ||
        ScalarActivation is MishActivation<T> ||
        ScalarActivation is SwishActivation<T> ||
        ScalarActivation is SiLUActivation<T> ||
        ScalarActivation is LeakyReLUActivation<T> ||
        ScalarActivation is IdentityActivation<T>)
    {
        return true;
    }

    // Check vector activations
    if (VectorActivation is SoftmaxActivation<T>)
    {
        return true;
    }

    // No activation is fine (identity)
    if (ScalarActivation == null && VectorActivation == null)
    {
        return true;
    }

    return false;
}
```

**Key Points:**

- **Whitelist approach**: Explicitly list supported activations
- **No activation = identity**: Return true if no activation configured
- **Easy to extend**: Just add new activation types as they're implemented

### Step 4: Update SupportsJitCompilation

This property tells the framework whether the layer can be JIT compiled in its current configuration.

```csharp
/// <summary>
/// Gets whether this layer currently supports JIT compilation.
/// </summary>
/// <value>
/// True if the layer's activation function is supported for JIT compilation.
/// Supported activations: ReLU, Sigmoid, Tanh, GELU, ELU, Mish, Swish, SiLU, LeakyReLU, Softmax, Identity.
/// </value>
public override bool SupportsJitCompilation => CanActivationBeJitted();
```

**Key Points:**

- **Dynamic check**: Layer might support JIT with ReLU but not with a custom activation
- **Used by JIT compiler**: Framework checks this before attempting compilation
- **Document supported activations**: Keep XML comment updated as you add more activations

### Step 5: Add Validation (Optional but Recommended)

For production-quality implementations, add validation to catch common errors early.

```csharp
/// <summary>
/// Validates that the layer is ready for JIT compilation.
/// </summary>
private void ValidateForJitCompilation()
{
    if (_weights == null)
        throw new InvalidOperationException(
            "Layer weights not initialized. Call Initialize() or train the layer first.");

    if (_biases == null)
        throw new InvalidOperationException(
            "Layer biases not initialized. Call Initialize() or train the layer first.");

    if (InputShape == null || InputShape.Length == 0)
        throw new InvalidOperationException(
            "Layer input shape not configured. Set InputShape before exporting computation graph.");

    if (OutputShape == null || OutputShape.Length == 0)
        throw new InvalidOperationException(
            "Layer output shape not configured. This should be set during initialization.");

    if (!CanActivationBeJitted())
    {
        var activationType = ScalarActivation?.GetType().Name ??
                            VectorActivation?.GetType().Name ??
                            "unknown";
        throw new NotSupportedException(
            $"Activation function '{activationType}' is not supported for JIT compilation. " +
            $"Supported activations: ReLU, Sigmoid, Tanh, GELU, ELU, Mish, Swish, SiLU, LeakyReLU, Softmax, Identity");
    }
}
```

Then call it at the start of `ExportComputationGraph`:

```csharp
public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    ValidateForJitCompilation();
    // ... rest of implementation
}
```

---

## Common Patterns

### Pattern 1: Matrix Operations

Most layers perform matrix multiplication (dense, convolutional, attention, etc.):

```csharp
// Dense layer: output = input × weights^T
var weightsTransposed = TensorOperations<T>.Transpose(weightsNode);
var output = TensorOperations<T>.MatrixMultiply(inputNode, weightsTransposed);

// Add bias
output = TensorOperations<T>.Add(output, biasesNode);
```

### Pattern 2: Element-wise Operations

Activation functions, batch normalization, layer normalization use element-wise ops:

```csharp
// Element-wise multiply
var scaled = TensorOperations<T>.ElementwiseMultiply(input, scaleNode);

// Element-wise add
var shifted = TensorOperations<T>.Add(scaled, offsetNode);

// Activation
var activated = TensorOperations<T>.ReLU(shifted);
```

### Pattern 3: Convolution Operations

Convolutional layers use Conv2D:

```csharp
// Convolution: output = Conv2D(input, kernel) + bias
var convResult = TensorOperations<T>.Conv2D(
    inputNode,
    kernelNode,
    stride: new[] { strideY, strideX },
    padding: new[] { padY, padX },
    dilation: new[] { dilationY, dilationX }
);

var withBias = TensorOperations<T>.Add(convResult, biasNode);
var activated = ApplyActivationToGraph(withBias);
```

### Pattern 4: Pooling Operations

MaxPooling and AveragePooling layers:

```csharp
// Max pooling
var pooled = TensorOperations<T>.MaxPool2D(
    inputNode,
    poolSize: new[] { poolHeight, poolWidth },
    stride: new[] { strideY, strideX },
    padding: new[] { padY, padX }
);

// Average pooling
var pooled = TensorOperations<T>.AvgPool2D(
    inputNode,
    poolSize: new[] { poolHeight, poolWidth },
    stride: new[] { strideY, strideX },
    padding: new[] { padY, padX }
);
```

### Pattern 5: Normalization Operations

Batch normalization and layer normalization:

```csharp
// Batch normalization
var normalized = TensorOperations<T>.BatchNorm(
    inputNode,
    gammaNode,  // Scale parameter
    betaNode,   // Shift parameter
    meanNode,   // Running mean
    varianceNode, // Running variance
    epsilon: 1e-5
);

// Layer normalization
var normalized = TensorOperations<T>.LayerNorm(
    inputNode,
    gammaNode,
    betaNode,
    epsilon: 1e-5
);
```

### Pattern 6: Concatenation and Splitting

Combine or split tensors:

```csharp
// Concatenate multiple inputs
var combined = TensorOperations<T>.Concat(
    new List<ComputationNode<T>> { input1, input2, input3 },
    axis: 1  // Concatenate along feature dimension
);

// Reshape to split
var reshaped = TensorOperations<T>.Reshape(inputNode, newShape);
```

### Pattern 7: Attention Mechanism

Self-attention and multi-head attention:

```csharp
// Query, Key, Value projections
var query = TensorOperations<T>.MatrixMultiply(inputNode, queryWeightsNode);
var key = TensorOperations<T>.MatrixMultiply(inputNode, keyWeightsNode);
var value = TensorOperations<T>.MatrixMultiply(inputNode, valueWeightsNode);

// Attention scores: Q × K^T / sqrt(d_k)
var keyTransposed = TensorOperations<T>.Transpose(key);
var scores = TensorOperations<T>.MatrixMultiply(query, keyTransposed);

// Scale
var scaleFactor = Math.Sqrt(embeddingDim);
var scaled = TensorOperations<T>.Divide(scores, TensorOperations<T>.Constant(scaleFactor));

// Softmax
var attention = TensorOperations<T>.Softmax(scaled, axis: -1);

// Apply attention to values
var output = TensorOperations<T>.MatrixMultiply(attention, value);
```

---

## Troubleshooting

### Error: "Activation X is not supported for JIT compilation"

**Cause**: Your layer uses an activation function that hasn't been added to `ApplyActivationToGraph` yet.

**Solution**:
1. Check if the activation is in the supported list (see "Supported Activations" section)
2. If it's listed but not working, add it to `CanActivationBeJitted()` and `ApplyActivationToGraph()`
3. If it's not listed, add the TensorOperations method first, then update your layer

**Example fix**:
```csharp
// Add to CanActivationBeJitted()
if (ScalarActivation is SELUActivation<T>)
    return true;

// Add to ApplyActivationToGraph()
else if (ScalarActivation is SELUActivation<T>)
    return TensorOperations<T>.SELU(input);
```

### Error: "Layer weights not initialized"

**Cause**: Trying to export computation graph before calling `Initialize()` or training the layer.

**Solution**:
```csharp
var layer = new DenseLayer<float>(inputSize: 784, outputSize: 128);
layer.Initialize();  // Initialize weights and biases
var graph = layer.ExportComputationGraph(inputNodes);
```

### Error: "InputShape not configured"

**Cause**: Layer doesn't know its input dimensions.

**Solution**:
```csharp
layer.InputShape = new int[] { 784 };  // Set before exporting graph
```

### Build Error: "Cannot convert TensorOperations result to expected type"

**Cause**: Using IEngine methods instead of TensorOperations methods.

**Solution**:
```csharp
// ❌ WRONG - IEngine methods don't return ComputationNode<T>
var result = _engine.MatrixMultiply(input, weights);

// ✅ CORRECT - Use TensorOperations
var result = TensorOperations<T>.MatrixMultiply(inputNode, weightsNode);
```

### Error: "Backward function not implemented"

**Cause**: This is expected! Gradient computation is not yet implemented.

**Current status**: Forward pass works, backward pass is placeholder.

**Workaround**: Use JIT compilation for inference only. For training, gradients will be added in a future phase.

### Performance Issue: Compilation takes too long

**Cause**: Very large or complex graphs can take time to compile.

**Solutions**:
1. Compile once, reuse for multiple batches
2. Use smaller subgraphs (compile individual layers instead of entire model)
3. Cache compiled graphs

**Example**:
```csharp
// Compile once
var compiled = jitCompiler.Compile(layer);

// Reuse for many batches
for (int i = 0; i < numBatches; i++)
{
    var output = compiled.Execute(batch[i]);
}
```

### Shape Mismatch: "Expected shape [X, Y] but got [A, B]"

**Cause**: Symbolic batch dimension (-1) not handled correctly.

**Solution**: Use symbolic shapes consistently:
```csharp
// ✅ CORRECT - Symbolic batch dimension
var inputShape = new int[] { -1, inputSize };

// ❌ WRONG - Fixed batch dimension
var inputShape = new int[] { 32, inputSize };
```

---

## Complete Example: Adding JIT Support to ConvolutionalLayer

Here's a full example showing how to add JIT compilation to `ConvolutionalLayer`:

```csharp
public class ConvolutionalLayer<T> : LayerBase<T>
{
    // ... existing fields and properties ...

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // 1. Validate
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_kernels == null)
            throw new InvalidOperationException("Kernels not initialized");

        if (!CanActivationBeJitted())
            throw new NotSupportedException($"Activation not supported for JIT");

        // 2. Extract dimensions
        // InputShape: [channels, height, width]
        int channels = InputShape[0];
        int height = InputShape[1];
        int width = InputShape[2];

        // 3. Create input placeholder with symbolic batch
        var inputPlaceholder = new Tensor<T>(new int[] { 1, channels, height, width });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "input");

        // 4. Create kernel parameters
        // Kernels shape: [numFilters, channels, kernelHeight, kernelWidth]
        var kernelNode = TensorOperations<T>.Variable(
            new Tensor<T>(_kernels.Shape, _kernels.ToArray()),
            "kernels"
        );

        // Biases shape: [numFilters]
        var biasNode = TensorOperations<T>.Variable(
            new Tensor<T>(new int[] { NumFilters }, _biases),
            "biases"
        );

        // 5. Add to input list
        inputNodes.Add(inputNode);
        inputNodes.Add(kernelNode);
        inputNodes.Add(biasNode);

        // 6. Build computation graph
        var convResult = TensorOperations<T>.Conv2D(
            inputNode,
            kernelNode,
            stride: new[] { StrideY, StrideX },
            padding: new[] { PaddingY, PaddingX },
            dilation: new[] { DilationY, DilationX }
        );

        var withBias = TensorOperations<T>.Add(convResult, biasNode);
        var activated = ApplyActivationToGraph(withBias);

        return activated;
    }

    private ComputationNode<T> ApplyActivationToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (ScalarActivation is not null)
        {
            if (ScalarActivation is ReLUActivation<T>)
                return TensorOperations<T>.ReLU(input);
            else if (ScalarActivation is SigmoidActivation<T>)
                return TensorOperations<T>.Sigmoid(input);
            // ... add other activations ...
            else
                throw new NotSupportedException($"Activation {ScalarActivation.GetType().Name} not supported");
        }

        return input;
    }

    private bool CanActivationBeJitted()
    {
        if (ScalarActivation is ReLUActivation<T> ||
            ScalarActivation is SigmoidActivation<T> ||
            ScalarActivation is TanhActivation<T> ||
            ScalarActivation is IdentityActivation<T>)
        {
            return true;
        }

        if (ScalarActivation == null && VectorActivation == null)
        {
            return true;
        }

        return false;
    }

    public override bool SupportsJitCompilation => CanActivationBeJitted();
}
```

---

## Next Steps

After implementing JIT support for your layer:

1. **Test compilation**: Ensure `ExportComputationGraph` runs without errors
2. **Verify correctness**: Compare JIT output with eager mode output
3. **Measure performance**: Benchmark to confirm speedup
4. **Add more activations**: Extend `ApplyActivationToGraph` as needed
5. **Document**: Update this guide with any new patterns you discover

For the complete roadmap and list of layers to implement, see [JIT_ROADMAP.md](JIT_ROADMAP.md).

For activation function reference, see [JIT_ACTIVATION_MAPPING.md](JIT_ACTIVATION_MAPPING.md).
