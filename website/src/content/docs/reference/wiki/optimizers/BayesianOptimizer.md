---
title: "BayesianOptimizer<T, TInput, TOutput>"
description: "Represents a Bayesian Optimizer for optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Represents a Bayesian Optimizer for optimization problems.

## For Beginners

Think of this optimizer as a smart guessing game. It tries to find the best solution
by making educated guesses based on what it has learned from previous attempts. It's particularly useful when
each guess is time-consuming or expensive to evaluate.

## How It Works

Bayesian Optimization is a powerful technique for optimizing black-box functions that are expensive to evaluate.
It uses a probabilistic model to make predictions about the function's behavior and decides where to sample next.

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
    .ConfigureOptimizer(new BayesianOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with BayesianOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianOptimizer(IFullModel<,,>,BayesianOptimizerOptions<,,>,IGaussianProcess<>,IEngine)` | Initializes a new instance of the BayesianOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAcquisitionFunction(Vector<>)` | Calculates the value of the acquisition function for a given point. |
| `Deserialize(Byte[])` | Restores the state of the optimizer from a byte array. |
| `GenerateRandomPoint(Int32)` | Generates a random point within the specified bounds of the optimization space. |
| `GetOptions` | Gets the current options for the Bayesian Optimization algorithm. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the Bayesian Optimization algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the Bayesian Optimization algorithm. |
| `OptimizeAcquisitionFunction(Int32)` | Optimizes the acquisition function to determine the next point to sample. |
| `Serialize` | Converts the current state of the optimizer into a byte array for storage or transmission. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Bayesian Optimization algorithm. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gaussianProcess` | The Gaussian Process model used to approximate the objective function. |
| `_options` | The options for configuring the Bayesian Optimization algorithm. |
| `_sampledPoints` | A matrix storing the points that have been sampled during the optimization process. |
| `_sampledValues` | A vector storing the corresponding function values for the sampled points. |

