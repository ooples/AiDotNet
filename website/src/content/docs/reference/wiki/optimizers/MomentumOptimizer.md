---
title: "MomentumOptimizer<T, TInput, TOutput>"
description: "Implements the Momentum optimization algorithm for gradient-based optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Momentum optimization algorithm for gradient-based optimization.

## For Beginners

Imagine you're rolling a ball down a hill to find the lowest point. The Momentum optimizer is like giving
that ball some "memory" of its previous movements. This helps it move faster in consistent directions and
resist getting stuck in small bumps or divots along the way.

## How It Works

The Momentum optimizer is an extension of gradient descent that helps accelerate the optimization process
in relevant directions and dampens oscillations. It does this by adding a fraction of the update vector
of the past time step to the current update vector.

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
    .ConfigureOptimizer(new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with MomentumOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MomentumOptimizer(IFullModel<,,>,MomentumOptimizerOptions<,,>)` | Initializes a new instance of the MomentumOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Deserializes a byte array to restore the optimizer's state. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients based on the model and input data. |
| `GetOptions` | Gets the current optimization algorithm options. |
| `InitializeAdaptiveParameters` | Initializes adaptive parameters for the optimization process. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes Momentum optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Momentum algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a momentum-based gradient update to recover original parameters. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the Momentum optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the SGD with momentum kernel. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution based on the calculated velocity. |
| `UpdateVelocity(Vector<>)` | Updates the velocity vector based on the current gradient and momentum. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuVelocity` | GPU buffer for velocity state. |
| `_options` | The configuration options specific to the Momentum optimizer. |
| `_velocity` | Stores the current velocity vector for each parameter in the model. |

