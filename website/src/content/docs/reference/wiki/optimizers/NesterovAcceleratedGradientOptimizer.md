---
title: "NesterovAcceleratedGradientOptimizer<T, TInput, TOutput>"
description: "Implements the Nesterov Accelerated Gradient optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Nesterov Accelerated Gradient optimization algorithm.

## For Beginners

Imagine you're skiing down a hill. Regular gradient descent is like looking at your current position to decide where to go next.
NAG is like looking ahead to where you'll be after your next move, and then deciding how to adjust your path.
This "look-ahead" helps you navigate the slope more efficiently, especially around tricky turns.

## How It Works

The Nesterov Accelerated Gradient (NAG) is an optimization algorithm that improves upon standard gradient descent.
It introduces a smart prediction of the next position of the parameters, which helps to dampen oscillations and
improve convergence, especially in scenarios with high curvature or small but consistent gradients.

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
    .ConfigureOptimizer(new NesterovAcceleratedGradientOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NesterovAcceleratedGradientOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NesterovAcceleratedGradientOptimizer(IFullModel<,,>,NesterovAcceleratedGradientOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the NesterovAcceleratedGradientOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuUpdate` | Gets whether this optimizer supports GPU-accelerated parameter updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Deserializes the Nesterov Accelerated Gradient optimizer from a byte array. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients in the Nesterov Accelerated Gradient optimizer. |
| `GetLookaheadSolution(IFullModel<,,>)` | Calculates the lookahead solution based on the current solution and velocity. |
| `GetOptions` | Gets the current options of the Nesterov Accelerated Gradient optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the NAG optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes NAG optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Nesterov Accelerated Gradient algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a Nesterov Accelerated Gradient update to recover original parameters. |
| `Serialize` | Serializes the Nesterov Accelerated Gradient optimizer to a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the NAG optimizer. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the Nesterov Accelerated Gradient algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using the NAG kernel. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the velocity vector. |
| `UpdateVelocity(Vector<>)` | Updates the velocity vector based on the current gradient. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuVelocity` | GPU buffer for velocity state. |
| `_options` | The options specific to the Nesterov Accelerated Gradient optimizer. |
| `_velocity` | The velocity vector used in the NAG algorithm. |

