---
title: "LBFGSOptimizer<T, TInput, TOutput>"
description: "Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm.

## For Beginners

L-BFGS is an advanced optimization algorithm that efficiently finds the minimum of a function, especially useful 
for problems with many variables. It uses information from previous iterations to make intelligent decisions 
about where to search next, while keeping memory usage low.

## How It Works

L-BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems. It approximates the
Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using a limited amount of computer memory, making it suitable 
for optimization problems with many variables.

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
    .ConfigureOptimizer(new LBFGSOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with LBFGSOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LBFGSOptimizer(IFullModel<,,>,LBFGSOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the LBFGSOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDirection(Vector<>)` | Calculates the search direction using the L-BFGS algorithm. |
| `Deserialize(Byte[])` | Deserializes a byte array to restore the optimizer's state. |
| `GetOptions` | Retrieves the current options of the optimizer. |
| `InitializeAdaptiveParameters` | Initializes or resets the adaptive parameters used in the optimization process. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the L-BFGS algorithm. |
| `Serialize` | Serializes the optimizer's state into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateLBFGSMemory(Vector<>,Vector<>,Vector<>,Vector<>)` | Updates the L-BFGS memory with the latest step information. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates parameters using the L-BFGS algorithm. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated L-BFGS. |
| `UpdateSolution(IFullModel<,,>,Vector<>,Vector<>,OptimizationInputData<,,>)` | Updates the current solution based on the calculated direction. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iteration` | The current iteration count of the optimization process. |
| `_lbfgsPreviousGradient` | Stores the previous gradient for computing gradient differences in UpdateParameters. |
| `_lbfgsPreviousParameters` | Stores the previous parameters for computing position differences in UpdateParameters. |
| `_options` | Options specific to the L-BFGS optimizer. |
| `_s` | List of position (solution) differences used in the L-BFGS update. |
| `_y` | List of gradient differences used in the L-BFGS update. |

