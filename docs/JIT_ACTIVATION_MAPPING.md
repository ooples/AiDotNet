# JIT Activation Mapping Reference

This document provides a complete reference for all activation functions available in AiDotNet, their JIT compilation support status, and how to use them in your layers.

## Quick Reference

**Total Activations**: 37
**Production-Ready**: 10
**Available (Pending Integration)**: 27

---

## Production-Ready Activations (10)

These activations are fully integrated into DenseLayer and ready for use in JIT compilation.

### ReLU Family (1)

| Activation Class | TensorOperations Method | IEngine Method | Parameters | Status |
|------------------|-------------------------|----------------|------------|--------|
| `ReLUActivation<T>` | `TensorOperations<T>.ReLU(node)` | `IEngine<T>.ReLU(tensor)` | None | ✅ Ready |

**Usage Example:**
```csharp
// In CanActivationBeJitted()
if (ScalarActivation is ReLUActivation<T>)
    return true;

// In ApplyActivationToGraph()
if (ScalarActivation is ReLUActivation<T>)
    return TensorOperations<T>.ReLU(input);
```

**Forward Function**: `f(x) = max(0, x)`

**Use Cases**: Default activation for hidden layers in most neural networks.

---

### Sigmoid Family (5)

| Activation Class | TensorOperations Method | IEngine Method | Parameters | Status |
|------------------|-------------------------|----------------|------------|--------|
| `SigmoidActivation<T>` | `TensorOperations<T>.Sigmoid(node)` | `IEngine<T>.Sigmoid(tensor)` | None | ✅ Ready |
| `TanhActivation<T>` | `TensorOperations<T>.Tanh(node)` | `IEngine<T>.Tanh(tensor)` | None | ✅ Ready |
| `SwishActivation<T>` | `TensorOperations<T>.Swish(node)` | `IEngine<T>.Swish(tensor)` | None | ✅ Ready |
| `SiLUActivation<T>` | `TensorOperations<T>.SiLU(node)` | `IEngine<T>.SiLU(tensor)` | None | ✅ Ready |
| `MishActivation<T>` | `TensorOperations<T>.Mish(node)` | `IEngine<T>.Mish(tensor)` | None | ✅ Ready |

**Usage Example (Sigmoid):**
```csharp
// In CanActivationBeJitted()
if (ScalarActivation is SigmoidActivation<T>)
    return true;

// In ApplyActivationToGraph()
if (ScalarActivation is SigmoidActivation<T>)
    return TensorOperations<T>.Sigmoid(input);
```

**Forward Functions**:
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Tanh**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- **Swish**: `f(x) = x * sigmoid(x)` (also known as SiLU)
- **SiLU**: Same as Swish
- **Mish**: `f(x) = x * tanh(softplus(x))`

**Use Cases**:
- **Sigmoid**: Binary classification output layers, LSTM gates
- **Tanh**: RNN hidden states, centered outputs (-1 to 1)
- **Swish/SiLU**: Modern alternative to ReLU with smooth gradients
- **Mish**: Self-regularized activation, good for deep networks

---

### Modern Activations (2)

| Activation Class | TensorOperations Method | IEngine Method | Parameters | Status |
|------------------|-------------------------|----------------|------------|--------|
| `GELUActivation<T>` | `TensorOperations<T>.GELU(node)` | `IEngine<T>.GELU(tensor)` | None | ✅ Ready |
| `ELUActivation<T>` | `TensorOperations<T>.ELU(node, alpha)` | `IEngine<T>.ELU(tensor, alpha)` | `alpha` (default: 1.0) | ✅ Ready |

**Usage Example (GELU):**
```csharp
// In CanActivationBeJitted()
if (ScalarActivation is GELUActivation<T>)
    return true;

// In ApplyActivationToGraph()
if (ScalarActivation is GELUActivation<T>)
    return TensorOperations<T>.GELU(input);
```

**Usage Example (ELU with parameter):**
```csharp
// In CanActivationBeJitted()
if (ScalarActivation is ELUActivation<T>)
    return true;

// In ApplyActivationToGraph()
if (ScalarActivation is ELUActivation<T> elu)
    return TensorOperations<T>.ELU(input, elu.Alpha);
```

**Forward Functions**:
- **GELU**: `f(x) = x * Φ(x)` where Φ is the cumulative distribution function of the standard normal distribution
- **ELU**: `f(x) = x if x > 0, else alpha * (e^x - 1)`

**Use Cases**:
- **GELU**: Used in Transformers (BERT, GPT), superior to ReLU for NLP tasks
- **ELU**: Reduces vanishing gradient problem, smooth negative values

---

### Vector Activations (1)

| Activation Class | TensorOperations Method | IEngine Method | Parameters | Status |
|------------------|-------------------------|----------------|------------|--------|
| `SoftmaxActivation<T>` | `TensorOperations<T>.Softmax(node, axis)` | `IEngine<T>.Softmax(tensor, axis)` | `axis` (default: -1) | ✅ Ready |

**Usage Example:**
```csharp
// In CanActivationBeJitted()
if (VectorActivation is SoftmaxActivation<T>)
    return true;

// In ApplyActivationToGraph()
if (VectorActivation is SoftmaxActivation<T>)
    return TensorOperations<T>.Softmax(input);
```

**Forward Function**: `f(x_i) = e^(x_i) / Σ(e^(x_j))`

**Use Cases**: Multi-class classification output layers, attention mechanisms.

---

### Identity (1)

| Activation Class | TensorOperations Method | IEngine Method | Parameters | Status |
|------------------|-------------------------|----------------|------------|--------|
| `IdentityActivation<T>` | `input` (no-op) | N/A | None | ✅ Ready |

**Usage Example:**
```csharp
// In CanActivationBeJitted()
if (ScalarActivation is IdentityActivation<T>)
    return true;

// In ApplyActivationToGraph()
if (ScalarActivation is IdentityActivation<T>)
    return input;  // No transformation
```

**Forward Function**: `f(x) = x`

**Use Cases**: Linear layers, skip connections, output layers for regression.

---

## Available Activations - Pending Integration (27)

These activations have TensorOperations methods implemented but are not yet integrated into layer implementations. To use them, follow the pattern shown in the "Production-Ready" section above.

### ReLU Family (7)

| Activation Class | TensorOperations Method | Parameters | Forward Function | IEngine Status |
|------------------|-------------------------|------------|------------------|----------------|
| `LeakyReLUActivation<T>` | `TensorOperations<T>.LeakyReLU(node, negativeSlope)` | `negativeSlope` (default: 0.01) | `f(x) = max(negativeSlope*x, x)` | ✅ Integrated |
| `SELUActivation<T>` | `TensorOperations<T>.SELU(node)` | None | `f(x) = scale * (max(0,x) + min(0, alpha*(e^x-1)))` | ✅ Integrated |
| `CELUActivation<T>` | `TensorOperations<T>.CELU(node, alpha)` | `alpha` (default: 1.0) | `f(x) = max(0,x) + min(0, alpha*(e^(x/alpha)-1))` | ✅ Integrated |
| `PReLUActivation<T>` | `TensorOperations<T>.PReLU(node, alpha)` | `alpha` (default: 0.25) | `f(x) = max(alpha*x, x)` | ✅ Integrated |
| `RReLUActivation<T>` | `TensorOperations<T>.RReLU(node, lower, upper)` | `lower` (0.125), `upper` (0.333) | `f(x) = max(a*x, x)` where a ~ U(lower, upper) | ✅ Integrated |
| `ThresholdedReLUActivation<T>` | `TensorOperations<T>.ThresholdedReLU(node, threshold)` | `threshold` (default: 1.0) | `f(x) = x if x > threshold, else 0` | ✅ Integrated |

**Integration Example (LeakyReLU):**
```csharp
// Add to CanActivationBeJitted()
if (ScalarActivation is LeakyReLUActivation<T>)
    return true;

// Add to ApplyActivationToGraph()
if (ScalarActivation is LeakyReLUActivation<T> leakyRelu)
    return TensorOperations<T>.LeakyReLU(input, leakyRelu.NegativeSlope);
```

---

### Sigmoid Family (9)

| Activation Class | TensorOperations Method | Parameters | Forward Function | IEngine Status |
|------------------|-------------------------|------------|------------------|----------------|
| `HardSigmoidActivation<T>` | `TensorOperations<T>.HardSigmoid(node)` | None | `f(x) = clip((x+1)/2, 0, 1)` | ✅ Integrated |
| `HardTanhActivation<T>` | `TensorOperations<T>.HardTanh(node)` | None | `f(x) = clip(x, -1, 1)` | ✅ Integrated |
| `ScaledTanhActivation<T>` | `TensorOperations<T>.ScaledTanh(node, alpha, beta)` | `alpha` (1.0), `beta` (1.0) | `f(x) = alpha * tanh(beta * x)` | ✅ Integrated |
| `SoftplusActivation<T>` | `TensorOperations<T>.Softplus(node)` | None | `f(x) = log(1 + e^x)` | ✅ Integrated |
| `SoftsignActivation<T>` | `TensorOperations<T>.Softsign(node)` | None | `f(x) = x / (1 + abs(x))` | ✅ Integrated |
| `BentIdentityActivation<T>` | `TensorOperations<T>.BentIdentity(node)` | None | `f(x) = (sqrt(x^2 + 1) - 1)/2 + x` | ✅ Integrated |

**Integration Example (Softplus):**
```csharp
// Add to CanActivationBeJitted()
if (ScalarActivation is SoftplusActivation<T>)
    return true;

// Add to ApplyActivationToGraph()
if (ScalarActivation is SoftplusActivation<T>)
    return TensorOperations<T>.Softplus(input);
```

---

### Softmax Family (3)

| Activation Class | TensorOperations Method | Parameters | Forward Function | IEngine Status |
|------------------|-------------------------|------------|------------------|----------------|
| `SoftminActivation<T>` | `TensorOperations<T>.Softmin(node, axis)` | `axis` (default: -1) | `f(x_i) = e^(-x_i) / Σ(e^(-x_j))` | ✅ Integrated |
| `LogSoftmaxActivation<T>` | `TensorOperations<T>.LogSoftmax(node, axis)` | `axis` (default: -1) | `f(x_i) = log(e^(x_i) / Σ(e^(x_j)))` | ✅ Integrated |
| `LogSoftminActivation<T>` | `TensorOperations<T>.LogSoftmin(node, axis)` | `axis` (default: -1) | `f(x_i) = log(e^(-x_i) / Σ(e^(-x_j)))` | ✅ Integrated |

**Integration Example (LogSoftmax):**
```csharp
// Add to CanActivationBeJitted() - check VectorActivation
if (VectorActivation is LogSoftmaxActivation<T>)
    return true;

// Add to ApplyActivationToGraph() - check VectorActivation
if (VectorActivation is LogSoftmaxActivation<T>)
    return TensorOperations<T>.LogSoftmax(input);
```

---

### Special Activations (8)

| Activation Class | TensorOperations Method | Parameters | Forward Function | IEngine Status |
|------------------|-------------------------|------------|------------------|----------------|
| `SignActivation<T>` | `TensorOperations<T>.Sign(node)` | None | `f(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0` | ✅ Integrated |
| `GaussianActivation<T>` | `TensorOperations<T>.Gaussian(node)` | None | `f(x) = e^(-x^2)` | ✅ Integrated |
| `ISRUActivation<T>` | `TensorOperations<T>.ISRU(node, alpha)` | `alpha` (default: 1.0) | `f(x) = x / sqrt(1 + alpha*x^2)` | ✅ Integrated |
| `LiSHTActivation<T>` | `TensorOperations<T>.LiSHT(node)` | None | `f(x) = x * tanh(x)` | ✅ Integrated |
| `SQRBFActivation<T>` | `TensorOperations<T>.SQRBF(node, center, width)` | `center` (0.0), `width` (1.0) | `f(x) = e^(-((x-center)/width)^2)` | ✅ Integrated |
| `SquashActivation<T>` | `TensorOperations<T>.Squash(node)` | None | `f(x) = (norm^2 / (1 + norm^2)) * (x / norm)` | ✅ Integrated |
| `BinarySpikingActivation<T>` | `TensorOperations<T>.BinarySpiking(node, threshold)` | `threshold` (default: 0.0) | `f(x) = 1 if x > threshold, else 0` | ✅ Integrated |

**Integration Example (Gaussian):**
```csharp
// Add to CanActivationBeJitted()
if (ScalarActivation is GaussianActivation<T>)
    return true;

// Add to ApplyActivationToGraph()
if (ScalarActivation is GaussianActivation<T>)
    return TensorOperations<T>.Gaussian(input);
```

---

### Complex Activations - Placeholder Status (6)

These activations have placeholder implementations in TensorOperations. Full implementation requires complex algorithms and will be completed in the gradient computation phase.

| Activation Class | TensorOperations Method | Parameters | Description | Status |
|------------------|-------------------------|------------|-------------|--------|
| `SparsemaxActivation<T>` | `TensorOperations<T>.Sparsemax(node, axis)` | `axis` (default: -1) | Projects onto simplex, produces sparse outputs | ⚠️ Placeholder |
| `SphericalSoftmaxActivation<T>` | `TensorOperations<T>.SphericalSoftmax(node, axis)` | `axis` (default: -1) | Normalizes to unit sphere | ⚠️ Placeholder |
| `GumbelSoftmaxActivation<T>` | `TensorOperations<T>.GumbelSoftmax(node, temp, axis)` | `temp` (1.0), `axis` (-1) | Differentiable sampling | ⚠️ Placeholder |
| `TaylorSoftmaxActivation<T>` | `TensorOperations<T>.TaylorSoftmax(node, order, axis)` | `order` (2), `axis` (-1) | Taylor approximation of softmax | ⚠️ Placeholder |
| `HierarchicalSoftmaxActivation<T>` | `TensorOperations<T>.HierarchicalSoftmax(node)` | None | Tree-structured softmax | ⚠️ Placeholder |
| `MaxoutActivation<T>` | `TensorOperations<T>.Maxout(node, numPieces)` | `numPieces` (default: 2) | Learnable piecewise linear | ⚠️ Placeholder |

**Note**: These activations currently throw `NotImplementedException` for backward pass. Do not use in production until fully implemented.

---

## Backward Pass Status

**Current Status**: Placeholder implementations only

All TensorOperations activation methods currently have placeholder backward functions:

```csharp
backward: (gradOutput) =>
{
    throw new NotImplementedException("Backward pass for [Activation] not yet implemented");
}
```

**Future Work**: Gradient computation will be implemented in a future phase. This includes:
- Analytical gradient formulas for all 37 activations
- Efficient backward pass implementations
- Support for training with JIT-compiled graphs

**Current Limitation**: JIT compilation is only suitable for **inference** (forward pass only). For **training**, use eager mode until backward pass is implemented.

---

## Activation Selection Guide

### For Image Classification (CNNs)

**Recommended**:
- Hidden layers: `ReLUActivation<T>` (fast, effective)
- Modern alternative: `GELUActivation<T>` (smoother gradients)
- Output layer: `SoftmaxActivation<T>` (multi-class)

**Example**:
```csharp
var conv1 = new ConvolutionalLayer<float>(filters: 32, kernelSize: 3, activation: new ReLUActivation<float>());
var conv2 = new ConvolutionalLayer<float>(filters: 64, kernelSize: 3, activation: new ReLUActivation<float>());
var dense = new DenseLayer<float>(inputSize: 1024, outputSize: 10, activation: new SoftmaxActivation<float>());
```

### For Natural Language Processing (Transformers)

**Recommended**:
- Hidden layers: `GELUActivation<T>` (used in BERT, GPT)
- Alternative: `SwishActivation<T>` or `MishActivation<T>`
- Output layer: `SoftmaxActivation<T>` (classification) or `IdentityActivation<T>` (regression)

**Example**:
```csharp
var feedForward = new DenseLayer<float>(inputSize: 768, outputSize: 3072, activation: new GELUActivation<float>());
var output = new DenseLayer<float>(inputSize: 3072, outputSize: 768, activation: new IdentityActivation<float>());
```

### For Recurrent Networks (RNNs, LSTMs, GRUs)

**Recommended**:
- Gates: `SigmoidActivation<T>` (LSTM/GRU gates)
- Hidden state: `TanhActivation<T>` (LSTM/GRU hidden state)
- Output layer: `SoftmaxActivation<T>` (classification)

**Example**:
```csharp
// LSTM uses both Sigmoid (for gates) and Tanh (for cell state)
var lstm = new LSTMLayer<float>(inputSize: 100, hiddenSize: 128);
// Gates internally use Sigmoid, cell state uses Tanh
```

### For Generative Models (GANs, VAEs)

**Recommended**:
- Generator hidden: `LeakyReLUActivation<T>` or `ELUActivation<T>` (avoid dying ReLU)
- Generator output: `TanhActivation<T>` (normalize to [-1, 1])
- Discriminator: `LeakyReLUActivation<T>` (stable gradients)

**Example**:
```csharp
var genHidden = new DenseLayer<float>(inputSize: 100, outputSize: 256, activation: new LeakyReLUActivation<float>());
var genOutput = new DenseLayer<float>(inputSize: 256, outputSize: 784, activation: new TanhActivation<float>());
```

---

## Integration Checklist

When adding JIT support for an activation to your layer:

- [ ] Check if activation is in "Production-Ready" list
- [ ] If not, check "Available Activations - Pending Integration" list
- [ ] Add activation type check to `CanActivationBeJitted()`
- [ ] Add activation mapping to `ApplyActivationToGraph()`
- [ ] Handle parameterized activations correctly (extract parameters)
- [ ] Update `SupportsJitCompilation` property
- [ ] Update XML documentation with supported activations
- [ ] Test with sample data
- [ ] Verify JIT compilation succeeds
- [ ] Benchmark performance

---

## See Also

- [JIT_COMPILATION_PATTERN_GUIDE.md](JIT_COMPILATION_PATTERN_GUIDE.md) - Complete implementation guide
- [JIT_ROADMAP.md](JIT_ROADMAP.md) - Current status and future work
