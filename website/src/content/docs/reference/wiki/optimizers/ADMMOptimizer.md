---
title: "ADMMOptimizer<T, TInput, TOutput>"
description: "Implements the Alternating Direction Method of Multipliers (ADMM) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Alternating Direction Method of Multipliers (ADMM) optimization algorithm.

## For Beginners

ADMM is like solving a complex puzzle by breaking it into smaller, manageable pieces.
It's particularly good at handling problems with constraints or when you want to distribute the computation across multiple processors.

## How It Works

ADMM is an algorithm for solving convex optimization problems, particularly useful for large-scale and distributed optimization.
It combines the benefits of dual decomposition and augmented Lagrangian methods.

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
    .ConfigureOptimizer(new ADMMOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ADMMOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ADMMOptimizer(IFullModel<,,>,ADMMOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the ADMMOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckConvergence(Vector<>)` | Checks if the optimization has converged based on primal and dual residuals. |
| `Deserialize(Byte[])` | Restores the optimizer's state from a byte array previously created by the Serialize method. |
| `GenerateGradientCacheKey(IFullModel<,,>,,)` | Generates a unique key for caching gradients based on the current state of the optimizer and input data. |
| `GetOptions` | Retrieves the current options of the optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the ADMM optimizer. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the ADMM algorithm. |
| `Serialize` | Converts the current state of the optimizer into a byte array for storage or transmission. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated ADMM. |
| `UpdateU(Vector<>)` | Updates the dual variable u in the ADMM algorithm. |
| `UpdateX(IFullModel<,,>,,)` | Updates the primal variable x in the ADMM algorithm. |
| `UpdateZ(Vector<>)` | Updates the auxiliary variable z in the ADMM algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iteration` | The current iteration count. |
| `_options` | The options specific to the ADMM optimizer. |
| `_regularization` | The regularization method used in the optimization. |
| `_u` | The dual variable in ADMM algorithm. |
| `_z` | The auxiliary variable in ADMM algorithm. |

