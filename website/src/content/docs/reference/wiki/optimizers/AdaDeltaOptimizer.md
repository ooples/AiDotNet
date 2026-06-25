---
title: "AdaDeltaOptimizer<T, TInput, TOutput>"
description: "Implements the AdaDelta optimization algorithm for training neural networks and other machine learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the AdaDelta optimization algorithm for training neural networks and other machine learning models.

## For Beginners

AdaDelta is like a smart assistant that helps your model learn more efficiently.

Imagine you're learning a new skill:

- Sometimes you need to practice more on difficult parts (bigger learning steps)
- Other times you need to be more careful with easier parts (smaller learning steps)

AdaDelta does this automatically for each part of your model, helping it learn better and faster.
It remembers recent changes and uses this information to decide how big the next learning step should be.

## How It Works

AdaDelta is an adaptive learning rate method that dynamically adjusts the learning rate for each parameter
based on a moving window of gradient updates. This optimizer addresses some of the drawbacks of AdaGrad,
particularly its aggressive, monotonically decreasing learning rate.

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
    .ConfigureOptimizer(new AdaDeltaOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AdaDeltaOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaDeltaOptimizer(IFullModel<,,>,AdaDeltaOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the `AdaDeltaOptimizer<T>` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Optimizers#Fused#IFusedOptimizerSpec#TryGetFusedOptimizerConfig(FusedOptimizerConfig)` | Describes this AdaDelta instance for the fused kernel (Tensors `OptimizerType.AdaDelta` = `AdaDeltaUpdateSimd(lr, rho, eps)`): Rho → Beta2 slot, Epsilon → eps; the two accumulators live in the plan. |
| `Deserialize(Byte[])` | Deserializes the AdaDelta optimizer from a byte array. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients. |
| `GetOptions` | Gets the current optimizer options. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the AdaDelta optimizer. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the AdaDelta algorithm. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses an AdaDelta gradient update to recover original parameters. |
| `Serialize` | Serializes the AdaDelta optimizer to a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the AdaDelta optimizer. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer options. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates a vector of parameters using the AdaDelta optimization algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on GPU using the AdaDelta optimization algorithm. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution using the AdaDelta update rule. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_accumulatedSquaredGradients` | Stores the exponential moving average of squared gradients for each parameter. |
| `_accumulatedSquaredUpdates` | Stores the exponential moving average of squared parameter updates for each parameter. |
| `_options` | The configuration options specific to the AdaDelta optimizer. |
| `_previousAccumulatedSquaredGradients` | Stores the pre-update snapshot of accumulated squared gradients for accurate reverse updates. |
| `_previousAccumulatedSquaredUpdates` | Stores the pre-update snapshot of accumulated squared updates for accurate reverse updates. |

