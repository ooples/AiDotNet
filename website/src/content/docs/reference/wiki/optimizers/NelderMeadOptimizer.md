---
title: "NelderMeadOptimizer<T, TInput, TOutput>"
description: "Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method.

## For Beginners

Imagine you're trying to find the lowest point in a hilly landscape. The Nelder-Mead method is like
having a group of explorers who work together, moving and reshaping their search pattern to find the lowest point.
They don't need to know which way is downhill; they just compare their positions and adjust accordingly.

## How It Works

The Nelder-Mead method is a heuristic search method that can optimize a problem with N variables.
It attempts to minimize a scalar-valued nonlinear function of n real variables using only function values,
without any derivative information.

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
    .ConfigureOptimizer(new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NelderMeadOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NelderMeadOptimizer(IFullModel<,,>,NelderMeadOptimizerOptions<,,>)` | Initializes a new instance of the NelderMeadOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCentroid(List<IFullModel<,,>>,Int32)` | Calculates the centroid of the simplex, excluding the worst point. |
| `Contract(IFullModel<,,>,IFullModel<,,>)` | Performs the contraction operation in the Nelder-Mead algorithm. |
| `Deserialize(Byte[])` | Deserializes the Nelder-Mead optimizer from a byte array. |
| `Expand(IFullModel<,,>,IFullModel<,,>)` | Performs the expansion operation in the Nelder-Mead algorithm. |
| `GetOptions` | Gets the current options of the Nelder-Mead optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the Nelder-Mead optimizer. |
| `InitializeSimplex(,Int32)` | Initializes the simplex for the Nelder-Mead algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Nelder-Mead algorithm. |
| `PerformVectorOperation(IFullModel<,,>,IFullModel<,,>,,Func<,,,>)` | Performs a vector operation on two symbolic models. |
| `Reflect(IFullModel<,,>,IFullModel<,,>)` | Performs the reflection operation in the Nelder-Mead algorithm. |
| `Serialize` | Serializes the Nelder-Mead optimizer to a byte array. |
| `Shrink(List<IFullModel<,,>>)` | Performs the shrink operation in the Nelder-Mead algorithm. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the Nelder-Mead algorithm. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The reflection coefficient. |
| `_beta` | The contraction coefficient. |
| `_delta` | The shrinkage coefficient. |
| `_gamma` | The expansion coefficient. |
| `_iteration` | The current iteration count. |
| `_options` | The options specific to the Nelder-Mead optimizer. |

