---
title: "LevenbergMarquardtOptimizer<T, TInput, TOutput>"
description: "Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems.

## For Beginners

This optimizer is like a smart problem-solver that's really good at fitting curves to data points. It's especially 
useful when the relationship between your inputs and outputs isn't a straight line. It works by making small 
adjustments to its guess, getting closer to the best solution with each step.

## How It Works

The Levenberg-Marquardt algorithm is a popular method for solving non-linear least squares problems. It combines 
the Gauss-Newton algorithm and the method of gradient descent, providing a robust solution that works well in 
many situations.

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
    .ConfigureOptimizer(new LevenbergMarquardtOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LevenbergMarquardtOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LevenbergMarquardtOptimizer(IFullModel<,,>,LevenbergMarquardtOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the LevenbergMarquardtOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateJacobian(IFullModel<,,>,)` | Calculates the Jacobian matrix for the current model and input data. |
| `CalculatePartialDerivative(IFullModel<,,>,,Int32)` | Calculates the partial derivative of the model output with respect to a specific parameter. |
| `CalculateResiduals(IFullModel<,,>,,)` | Calculates the residuals (differences between predicted and actual values) for the current model. |
| `Deserialize(Byte[])` | Deserializes a byte array to restore the optimizer's state. |
| `GetOptions` | Retrieves the current options of the optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the optimizer. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Levenberg-Marquardt algorithm. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `SolveLinearSystem(Matrix<>,Vector<>)` | Solves the linear system of equations in the Levenberg-Marquardt algorithm. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated Levenberg-Marquardt. |
| `UpdateSolution(IFullModel<,,>,Matrix<>,Vector<>)` | Updates the current solution based on the Levenberg-Marquardt algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dampingFactor` | The damping factor used in the Levenberg-Marquardt algorithm to balance between gradient descent and Gauss-Newton steps. |
| `_iteration` | The current iteration count of the optimization process. |
| `_options` | The options specific to the Levenberg-Marquardt algorithm. |

