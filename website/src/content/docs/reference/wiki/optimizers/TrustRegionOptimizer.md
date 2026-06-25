---
title: "TrustRegionOptimizer<T, TInput, TOutput>"
description: "Implements the Trust Region optimization algorithm for machine learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Trust Region optimization algorithm for machine learning models.

## For Beginners

Think of this optimizer as an explorer with a map:

- The "trust region" is like the area on the map the explorer trusts to be accurate.
- In each step, the explorer looks at this trusted area to decide where to go next.
- If the predictions (map) match reality well, the explorer might expand the trusted area.
- If the predictions are off, the explorer shrinks the trusted area and becomes more cautious.

This approach helps the optimizer make good decisions even in complex landscapes, balancing between
making progress and staying reliable.

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
    .ConfigureOptimizer(new TrustRegionOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with TrustRegionOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrustRegionOptimizer(IFullModel<,,>,TrustRegionOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the TrustRegionOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateHessian(IFullModel<,,>,OptimizationInputData<,,>)` | Calculates the Hessian matrix for the current solution. |
| `CalculatePredictedReduction(Vector<>,Matrix<>,Vector<>)` | Calculates the predicted reduction in the objective function for a given step. |
| `ComputeBoundaryStep(Vector<>,Vector<>)` | Computes a step that lies on the boundary of the trust region. |
| `Deserialize(Byte[])` | Deserializes the optimizer's state from a byte array. |
| `GetOptions` | Retrieves the current options of the optimizer. |
| `InitializeAdaptiveParameters` | Initializes adaptive parameters for the Trust Region optimizer. |
| `MoveInDirection(IFullModel<,,>,Vector<>,)` | Moves the current solution in the specified direction with the given step size. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using the Trust Region algorithm. |
| `Serialize` | Serializes the current state of the optimizer into a byte array. |
| `ShrinkTrustRegionRadius` | Reduces the size of the trust region. |
| `SolveQuadratic(,,)` | Solves a quadratic equation of the form ax^2 + bx + c = 0. |
| `SolveSubproblem(Vector<>,Matrix<>)` | Solves the trust region subproblem using the Conjugate Gradient method. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates adaptive parameters based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer options. |
| `UpdateParameters(Vector<>,Vector<>)` | Updates parameters using a simplified Cauchy-point Trust Region step. |
| `UpdateTrustRegionRadius()` | Updates the trust region radius based on the success of the last step. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iteration` | The current iteration count of the optimization process. |
| `_options` | The options for configuring the Trust Region optimizer. |
| `_trustRegionPreviousGradient` | The previous gradient, used for trust region update computation. |
| `_trustRegionPreviousParameters` | The previous parameters, used for trust region update computation. |
| `_trustRegionRadius` | The current radius of the trust region. |

