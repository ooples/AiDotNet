---
title: "CoordinateDescentOptimizer<T, TInput, TOutput>"
description: "Implements the Coordinate Descent optimization algorithm for numerical optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Coordinate Descent optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer is like adjusting the knobs on a complex machine one at a time. 
It focuses on improving one aspect of the solution at a time, which can be more manageable and sometimes 
more effective than trying to adjust everything at once.

## How It Works

Coordinate Descent is an optimization algorithm that minimizes a multivariable function by solving a series of 
single-variable optimization problems. It cycles through each variable (coordinate) and optimizes it while holding 
the others constant.

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
    .ConfigureOptimizer(new CoordinateDescentOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with CoordinateDescentOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CoordinateDescentOptimizer(IFullModel<,,>,CoordinateDescentOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the CoordinateDescentOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePartialDerivative(IFullModel<,,>,OptimizationInputData<,,>,Int32)` | Calculates the partial derivative (gradient) for a specific coordinate (variable). |
| `CalculateUpdate(,Int32)` | Calculates the update for a specific coordinate based on its gradient and momentum. |
| `Deserialize(Byte[])` | Deserializes the Coordinate Descent optimizer from a byte array. |
| `GetOptions` | Retrieves the current options of the Coordinate Descent optimizer. |
| `InitializeAdaptiveParameters(IFullModel<,,>)` | Initializes the adaptive parameters used in the Coordinate Descent algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the Coordinate Descent algorithm. |
| `Serialize` | Serializes the Coordinate Descent optimizer to a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters (learning rates and momentums) based on the optimization progress. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Coordinate Descent optimizer. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated coordinate descent. |
| `UpdateSolution(IFullModel<,,>,OptimizationInputData<,,>)` | Updates the current solution by optimizing each coordinate (variable) individually. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_learningRates` | Vector of learning rates for each coordinate (variable) in the optimization problem. |
| `_momentums` | Vector of momentum values for each coordinate (variable) in the optimization problem. |
| `_options` | The options specific to the Coordinate Descent optimization algorithm. |
| `_previousUpdate` | Vector of previous update values for each coordinate (variable) in the optimization problem. |

