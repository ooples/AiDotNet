---
title: "ConjugateGradientOptimizer<T, TInput, TOutput>"
description: "Implements the Conjugate Gradient optimization algorithm for numerical optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Conjugate Gradient optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer is like a smart hiker trying to find the lowest point in a hilly landscape. 
It uses information about the slope (gradient) and its previous steps to decide on the best direction to move next, 
allowing it to find the lowest point (optimal solution) more efficiently than simpler methods.

## How It Works

The Conjugate Gradient method is an algorithm for the numerical solution of particular systems of linear equations, 
namely those whose matrix is symmetric and positive-definite. It is often used to solve unconstrained optimization problems 
such as energy minimization.

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
    .ConfigureOptimizer(new ConjugateGradientOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ConjugateGradientOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConjugateGradientOptimizer(IFullModel<,,>,ConjugateGradientOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the ConjugateGradientOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBeta(Vector<>)` | Calculates the beta factor used in the Conjugate Gradient method. |
| `CalculateDirection(Vector<>)` | Calculates the search direction for the current iteration. |
| `Deserialize(Byte[])` | Deserializes a byte array to restore the state of the Conjugate Gradient optimizer. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients in the Conjugate Gradient optimizer. |
| `GetOptions` | Gets the current options of the Conjugate Gradient optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the Conjugate Gradient algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the Conjugate Gradient algorithm. |
| `Serialize` | Serializes the current state of the Conjugate Gradient optimizer into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the Conjugate Gradient optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Conjugate Gradient optimizer. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated conjugate gradient. |
| `UpdateSolution(IFullModel<,,>,Vector<>,Vector<>,OptimizationInputData<,,>)` | Updates the current solution based on the calculated direction and gradient. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iteration` | The current iteration count. |
| `_options` | The options specific to the Conjugate Gradient optimization algorithm. |
| `_previousDirection` | The direction vector from the previous iteration. |
| `_previousGradient` | The gradient vector from the previous iteration. |

