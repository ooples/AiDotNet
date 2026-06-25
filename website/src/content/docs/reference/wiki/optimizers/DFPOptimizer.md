---
title: "DFPOptimizer<T, TInput, TOutput>"
description: "Implements the Davidon-Fletcher-Powell (DFP) optimization algorithm for numerical optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Davidon-Fletcher-Powell (DFP) optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer is like a smart navigator that learns from its past steps
to make better decisions about which direction to move in the future. It's particularly good at
handling complex optimization problems where the landscape of possible solutions is intricate.

## How It Works

The DFP algorithm is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
It approximates the inverse Hessian matrix to determine the search direction, combining the efficiency
of Newton's method with the stability of gradient descent.

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
    .ConfigureOptimizer(new DFPOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with DFPOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DFPOptimizer(IFullModel<,,>,DFPOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the DFPOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDirection(Vector<>)` | Calculates the search direction using the inverse Hessian approximation and the current gradient. |
| `Deserialize(Byte[])` | Deserializes the DFP optimizer from a byte array. |
| `GetOptions` | Retrieves the current options of the DFP optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the DFP algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the DFP algorithm. |
| `Serialize` | Serializes the DFP optimizer to a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters based on the optimization progress. |
| `UpdateInverseHessian(IFullModel<,,>,IFullModel<,,>,Vector<>)` | Updates the inverse Hessian approximation using the DFP update formula. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the DFP optimizer. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates parameters using the DFP algorithm with inverse Hessian approximation. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated DFP. |
| `UpdateSolution(IFullModel<,,>,Vector<>,Vector<>,OptimizationInputData<,,>)` | Updates the current solution by moving in the calculated direction. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adaptiveLearningRate` | The current adaptive learning rate. |
| `_inverseHessian` | The inverse Hessian matrix approximation used in the DFP algorithm. |
| `_options` | The options specific to the DFP optimization algorithm. |
| `_previousGradient` | The gradient from the previous iteration. |
| `_previousParameters` | The parameters from the previous iteration for UpdateParameters method. |

