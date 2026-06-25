---
title: "ProximalGradientDescentOptimizer<T, TInput, TOutput>"
description: "Implements a Proximal Gradient Descent optimization algorithm which combines gradient descent with regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements a Proximal Gradient Descent optimization algorithm which combines gradient descent with regularization.

## For Beginners

Proximal Gradient Descent is like walking downhill while staying within certain boundaries.

Imagine you're hiking down a mountain to find the lowest point:

- Standard gradient descent is like always walking directly downhill
- Proximal gradient descent adds boundaries or "guardrails" to your path
- These guardrails keep you from wandering into areas that might look good but are actually not helpful
- For example, the guardrails might prevent solutions that are too complex and would overfit the data

This approach helps find solutions that not only fit the data well but also have desirable properties
like simplicity or stability.

## How It Works

Proximal Gradient Descent (PGD) is an extension of standard gradient descent that handles regularization more
efficiently. The algorithm alternates between performing a gradient descent step to minimize the loss function
and applying a proximal operator to enforce regularization. This approach is particularly effective for
problems where regularization is important to prevent overfitting or to enforce specific properties in the solution.

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
    .ConfigureOptimizer(new ProximalGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ProximalGradientDescentOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProximalGradientDescentOptimizer(IFullModel<,,>,ProximalGradientDescentOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the `ProximalGradientDescentOptimizer<T>` class with the specified options and components. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Reconstructs the proximal gradient descent optimizer from a serialized byte array. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients based on the model, input data, and optimizer state. |
| `GetOptions` | Gets the current options for this optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the Proximal Gradient Descent algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the proximal gradient descent optimization to find the best solution for the given input data. |
| `ReverseUpdate(Vector<>,Vector<>)` | Reverses a Proximal Gradient Descent update to recover original parameters. |
| `Serialize` | Serializes the proximal gradient descent optimizer to a byte array for storage or transmission. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates adaptive parameters based on optimization progress. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with the provided options. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated proximal gradient descent. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the solution by applying a gradient step followed by regularization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iteration` | The current iteration count of the optimization process. |
| `_options` | Configuration options specific to Proximal Gradient Descent optimization. |
| `_previousParameters` | Stores the pre-update parameters for approximate reverse updates. |
| `_regularization` | The regularization strategy applied to the optimization process. |

