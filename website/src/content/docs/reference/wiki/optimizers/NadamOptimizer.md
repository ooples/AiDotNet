---
title: "NadamOptimizer<T, TInput, TOutput>"
description: "Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm.

## For Beginners

Imagine you're rolling a smart ball down a hill. This ball can adjust its speed for different parts of the hill (adaptive learning rates),
and it can look ahead to anticipate slopes (Nesterov's method). This combination helps it find the lowest point more efficiently.

## How It Works

Nadam combines the ideas of Adam (adaptive learning rates) and Nesterov accelerated gradient (NAG).
It adapts the learning rates of each parameter and incorporates momentum using Nesterov's method.

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
    .ConfigureOptimizer(new NadamOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NadamOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NadamOptimizer(IFullModel<,,>,NadamOptimizerOptions<,,>)` | Initializes a new instance of the NadamOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this Nadam instance for the fused-compiled training kernel (Tensors `OptimizerType.Nadam` — Nesterov-accelerated Adam). |
| `Deserialize(Byte[])` | Deserializes a byte array to restore the optimizer's state. |
| `DisposeGpuState` | Disposes GPU-allocated optimizer state. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetOptions` | Gets the current optimization algorithm options. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the Nadam optimizer. |
| `InitializeGpuState(Int32,IDirectGpuBackend)` | Initializes Nadam optimizer state on the GPU. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Nadam algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a Nadam gradient update to recover original parameters. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the Nadam optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on GPU using Nadam optimization. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution based on the calculated gradient using the Nadam algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuM` | GPU buffer for first moment estimates (m). |
| `_gpuV` | GPU buffer for second moment estimates (v). |
| `_m` | The first moment vector (momentum). |
| `_options` | The options specific to the Nadam optimizer. |
| `_previousM` | Stores the pre-update snapshot of first moment vector for accurate reverse updates. |
| `_previousT` | Stores the pre-update snapshot of the time step for accurate reverse updates. |
| `_previousV` | Stores the pre-update snapshot of second moment vector for accurate reverse updates. |
| `_t` | The current time step. |
| `_v` | The second moment vector (adaptive learning rates). |

