---
title: "AdagradOptimizer<T, TInput, TOutput>"
description: "Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization.

## For Beginners

Adagrad is like a smart learning assistant that adjusts how much it learns
for each piece of information based on how often it has seen similar information before.

- It learns more from new or rare information
- It learns less from common or frequently seen information
- This helps it focus on the most important parts of what it's learning

This can be especially useful when some parts of your data are more important or occur less frequently.

## How It Works

The Adagrad optimizer adapts the learning rate for each parameter based on the historical gradients.
It performs larger updates for infrequent parameters and smaller updates for frequent ones.

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
    .ConfigureOptimizer(new AdagradOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdagradOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdagradOptimizer(IFullModel<,,>,AdagradOptimizerOptions<,,>)` | Initializes a new instance of the AdagradOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this Adagrad instance for the fused kernel (Tensors `OptimizerType.Adagrad` = `AdagradUpdateSimd(lr, eps)`): Epsilon → eps; the running squared-gradient accumulator lives in the plan. |
| `Deserialize(Byte[])` | Deserializes the Adagrad optimizer from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients based on the model, input data, and Adagrad-specific parameters. |
| `GetOptions` | Retrieves the current options of the Adagrad optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the Adagrad optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes Adagrad optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Adagrad algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses an Adagrad gradient update to recover original parameters. |
| `Serialize` | Serializes the Adagrad optimizer to a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAccumulatedSquaredGradients(Vector<>)` | Updates the accumulated squared gradients used in the Adagrad algorithm. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the Adagrad optimizer. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Adagrad optimizer. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the Adagrad optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the Adagrad kernel. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the Adagrad update rule. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_accumulatedSquaredGradients` | Stores the sum of squared gradients for each parameter during optimization. |
| `_gpuAccumulatedGrad` | GPU buffer for accumulated squared gradients. |
| `_options` | The configuration options specific to the Adagrad optimizer. |

