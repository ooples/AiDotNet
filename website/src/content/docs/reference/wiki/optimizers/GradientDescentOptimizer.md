---
title: "GradientDescentOptimizer<T, TInput, TOutput>"
description: "Represents a Gradient Descent optimizer for machine learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Represents a Gradient Descent optimizer for machine learning models.

## For Beginners

Imagine you're trying to find the lowest point in a valley:

- You start at a random point (initial model parameters)
- You look around to see which way is steepest downhill (calculate the gradient)
- You take a step in that direction (update the parameters)
- You repeat this process until you reach the bottom of the valley (optimize the model)

This optimizer helps the model learn by gradually adjusting its parameters to minimize errors.

## How It Works

Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.
It takes steps proportional to the negative of the gradient of the function at the current point.

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
    .ConfigureOptimizer(new GradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with GradientDescentOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientDescentOptimizer(IFullModel<,,>,GradientDescentOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the GradientDescentOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateLoss(IFullModel<,,>,,)` | Calculates the loss for a given solution and input data. |
| `Deserialize(Byte[])` | Restores the state of the Gradient Descent optimizer from a byte array. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients specific to the Gradient Descent optimizer. |
| `GetOptions` | Retrieves the current options for the Gradient Descent optimizer. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the Gradient Descent algorithm. |
| `Serialize` | Converts the current state of the Gradient Descent optimizer into a byte array for storage or transmission. |
| `Step(TapeStepContext<>)` | Reverses a Gradient Descent update to recover original parameters. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Gradient Descent optimizer. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters on the GPU using vanilla SGD. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution based on the calculated gradient. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gdOptions` | Options specific to the Gradient Descent optimizer. |
| `_regularization` | The regularization technique used to prevent overfitting. |

