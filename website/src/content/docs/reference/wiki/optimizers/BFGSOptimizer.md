---
title: "BFGSOptimizer<T, TInput, TOutput>"
description: "Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm.

## For Beginners

BFGS is an advanced optimization algorithm that tries to find the best solution
by making smart steps based on the function's behavior. It's particularly good at handling complex problems
where the function being optimized is smooth but potentially has many variables.

## How It Works

BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
It approximates the Hessian matrix of second derivatives of the function to be minimized.

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
    .ConfigureOptimizer(new BFGSOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with BFGSOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BFGSOptimizer(IFullModel<,,>,BFGSOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the BFGSOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Restores the state of the BFGS optimizer from a byte array. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients in the BFGS optimization process. |
| `GetOptions` | Gets the current options for the BFGS optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the BFGS algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the BFGS algorithm. |
| `Serialize` | Converts the current state of the BFGS optimizer into a byte array for storage or transmission. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer. |
| `UpdateInverseHessian(Vector<>,Vector<>)` | Updates the approximation of the inverse Hessian matrix. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the BFGS optimizer. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates parameters using the BFGS algorithm with inverse Hessian approximation. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated BFGS. |
| `UpdateSolution(IFullModel<,,>,Vector<>,OptimizationInputData<,,>)` | Updates the current solution using the BFGS update formula. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inverseHessian` | The approximation of the inverse Hessian matrix. |
| `_iteration` | The current iteration count. |
| `_options` | The options specific to the BFGS optimization algorithm. |
| `_previousGradient` | The gradient from the previous iteration. |
| `_previousParameters` | The parameters from the previous iteration. |

