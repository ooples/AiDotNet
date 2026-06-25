---
title: "LARSOptimizer<T, TInput, TOutput>"
description: "Implements the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm.

## For Beginners

When training with very large batches (common in self-supervised
learning like SimCLR), regular optimizers can become unstable because gradients get averaged
over more samples, making them smaller. LARS solves this by looking at each layer and asking
"how big are the weights compared to the gradients?" and scaling the learning rate accordingly.
This allows stable training with batch sizes of 4096 or even larger.

## How It Works

LARS is specifically designed for training with very large batch sizes (4096-32768).
It automatically adapts the learning rate for each layer based on the ratio of
parameter norm to gradient norm, which helps maintain stable training at scale.

**Key Formula:**

Based on the paper "Large Batch Training of Convolutional Networks" by You et al. (2017).

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
    .ConfigureOptimizer(new LARSOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LARSOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LARSOptimizer(IFullModel<,,>,LARSOptimizerOptions<,,>)` | Initializes a new instance of the LARSOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Momentum` | Gets the current momentum coefficient. |
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |
| `TrustCoefficient` | Gets the LARS trust coefficient. |
| `WeightDecay` | Gets the current weight decay coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `ExtractLayerVector(Vector<>,Int32,Int32)` | Extracts a layer's parameters from the full parameter vector. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetDataSize(OptimizationInputData<,,>)` | Gets the size of the training data. |
| `GetOptions` | Gets the current optimizer options. |
| `GetWarmupLearningRate` | Gets the learning rate with warmup applied. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the LARS optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes LARS optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the LARS algorithm. |
| `Reset` | Resets the optimizer's internal state. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a LARS gradient update to recover original parameters. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateLayerVector(Vector<>,Vector<>,Int32,Int32)` | Updates a layer's values in the full parameter vector. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options. |
| `UpdateParameters(Matrix<>,Matrix<>)` | Updates a matrix of parameters using the LARS optimization algorithm. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the LARS optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the LARS kernel. |
| `UpdateParametersLARS(Vector<>,Vector<>,Double)` | Internal LARS parameter update. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the LARS update rule. |
| `UpdateSolutionWithLARS(IFullModel<,,>,Vector<>,Double)` | Updates the solution using the LARS algorithm with layer-wise adaptive rate scaling. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuVelocity` | GPU buffer for velocity state. |
| `_options` | The options specific to the LARS optimizer. |
| `_previousT` | Previous step count for reverse updates. |
| `_previousVelocity` | Previous velocity for reverse updates. |
| `_t` | The current time step (iteration count). |
| `_velocity` | The velocity/momentum buffer for each parameter. |
| `_warmupSteps` | The total number of warmup steps. |

