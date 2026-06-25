---
title: "NewtonMethodOptimizer<T, TInput, TOutput>"
description: "Implements the Newton's Method optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Newton's Method optimization algorithm.

## For Beginners

Imagine you're trying to find the lowest point in a valley. Gradient descent is like rolling a ball and letting it follow the slope.
Newton's Method is like using a telescope to look at the whole valley, predicting where the lowest point is, and jumping directly there.
It's often faster but requires more complex calculations at each step.

## How It Works

Newton's Method is a powerful optimization algorithm that uses both first and second derivatives of the objective function.
It often converges faster than first-order methods, especially near the optimum, but can be computationally expensive due to the need to compute and invert the Hessian matrix.

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
    .ConfigureOptimizer(new NewtonMethodOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NewtonMethodOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NewtonMethodOptimizer(IFullModel<,,>,NewtonMethodOptimizerOptions<,,>)` | Initializes a new instance of the NewtonMethodOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDirection(Vector<>,Matrix<>)` | Calculates the direction for the next step in Newton's Method. |
| `CalculateHessian(IFullModel<,,>,OptimizationInputData<,,>)` | Calculates the Hessian matrix for the current model and input data. |
| `CalculateSecondPartialDerivative(IFullModel<,,>,OptimizationInputData<,,>,Int32,Int32)` | Calculates the second partial derivative of the loss function with respect to two parameters. |
| `ComputeDotProduct(Dictionary<Tensor<>,Tensor<>>,Dictionary<Tensor<>,Tensor<>>)` | Computes the global dot product across all parameter tensors. |
| `Deserialize(Byte[])` | Deserializes a byte array to restore the optimizer's state. |
| `GetOptions` | Gets the current options of the optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters for the Newton's Method optimizer. |
| `NewtonCGStep(TapeStepContext<>)` | Newton-CG step using exact Hessian-vector products from the gradient tape. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process using Newton's Method algorithm. |
| `Serialize` | Serializes the current state of the optimizer into a byte array. |
| `Step(TapeStepContext<>)` |  |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters of the optimizer based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with new settings. |
| `UpdateParametersGpu(IGpuBuffer,IGpuBuffer,Int32,IDirectGpuBackend)` | Updates parameters using GPU-accelerated Newton's Method. |
| `UpdateSolution(IFullModel<,,>,Vector<>)` | Updates the current solution based on the calculated direction. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_iteration` | The current iteration count of the optimization process. |
| `_options` | The options specific to the Newton's Method optimizer. |

