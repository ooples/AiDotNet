---
title: "Adam8BitOptimizer<T, TInput, TOutput>"
description: "Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format.

## For Beginners

When training a neural network, the optimizer needs to remember information about
past gradients. Standard Adam stores two numbers per parameter (momentum and variance), which can use a lot of
memory for large models. 8-bit Adam compresses these numbers, similar to how images are compressed, reducing
memory usage while maintaining training quality.

## How It Works

8-bit Adam provides the same optimization algorithm as standard Adam but uses quantized 8-bit representations
for storing the first moment (m) and second moment (v) estimates. This reduces memory usage by approximately
4x for optimizer states, which is particularly beneficial when training large models.

**How It Works:**

- Optimizer states are divided into blocks (default 2048 elements each)
- Each block has its own scaling factor for accurate quantization
- States are dequantized before computing updates, then requantized after
- The actual parameter updates use full precision for accuracy

**When to Use:**

- Training large models where optimizer memory is a bottleneck
- GPU training with limited VRAM
- Distributed training where memory per GPU is constrained

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(0);
var trainX = new Tensor<double>(new[] { 32, 8 });
var trainY = new Tensor<double>(new[] { 32, 2 });
for (int i = 0; i < 32; i++)
{
    for (int j = 0; j < 8; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 2 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 8, numClasses: 2, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new Adam8BitOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with Adam8BitOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Adam8BitOptimizer(IFullModel<,,>,Adam8BitOptimizerOptions<,,>)` | Initializes a new instance of the Adam8BitOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllocateTapeState(Int32)` | Allocates a freshly-zeroed `QuantizedTapeState` sized for a parameter tensor of the given length. |
| `Bf16ToTensor(Vector<UInt16>,Int32[])` | Expands a BF16 (2 bytes/element) moment buffer into a freshly-allocated full-precision tensor of the given shape. |
| `Dequantize(Vector<Byte>,Vector<Double>,Boolean)` | Dequantizes an 8-bit representation back to full precision. |
| `DequantizeTensor(Vector<Byte>,Vector<Double>,Int32[],Int32,Boolean)` | Block-dequantizes an 8-bit byte buffer into a freshly-allocated tensor of the supplied shape. |
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetMemoryUsage` | Gets the memory usage statistics for this optimizer. |
| `GetOptions` | Gets the current optimizer options. |
| `GetTapeStateSnapshotForTests` | Test hook: returns a structural snapshot of every tape-state entry. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the Adam optimizer. |
| `InitializeQuantizedState(Int32)` | Initializes the quantized optimizer state buffers. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the 8-bit Adam algorithm. |
| `Quantize(Vector<>,Vector<Byte>,Vector<Double>,Boolean)` | Quantizes a full-precision vector to 8-bit representation. |
| `Reset` | Resets the optimizer's internal state. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `TensorToBf16(Tensor<>,Vector<UInt16>)` | Packs a full-precision tensor's values back into a pre-allocated BF16 (2 bytes/element) buffer with round-to-nearest-even. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters using the 8-bit Adam optimization algorithm. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the 8-bit Adam optimization algorithm. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the 8-bit Adam update rule. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Adam8BitV2Magic` | Magic header for the v2 checkpoint format ("A8B1" in ASCII LE). |
| `ChunkBytes` | Block-quantizes a tensor's values into a pre-allocated byte buffer. |
| `StateFormatVersion` | Current checkpoint format version. |
| `_currentBeta1` | The current value of beta1 (exponential decay rate for first moment estimates). |
| `_currentBeta2` | The current value of beta2 (exponential decay rate for second moment estimates). |
| `_mFullPrecision` | Full-precision first moment vector (used when CompressBothMoments is false). |
| `_mQuantized` | Quantized first moment vector (moving average of gradients). |
| `_mScales` | Scaling factors for first moment quantization blocks. |
| `_numBlocks` | Number of quantization blocks. |
| `_options` | The options specific to the 8-bit Adam optimizer. |
| `_parameterLength` | Length of the parameter vector. |
| `_t` | The current time step (iteration count). |
| `_vQuantized` | Quantized second moment vector (moving average of squared gradients). |
| `_vScales` | Scaling factors for second moment quantization blocks. |

