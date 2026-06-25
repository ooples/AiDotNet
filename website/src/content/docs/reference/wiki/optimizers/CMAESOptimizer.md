---
title: "CMAESOptimizer<T, TInput, TOutput>"
description: "Implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm.

## For Beginners

CMA-ES is like an advanced search algorithm that tries to find the best solution
by learning from previous attempts. It's especially good at solving complex problems where the relationship
between inputs and outputs isn't straightforward.

## How It Works

CMA-ES is a powerful optimization algorithm for non-linear, non-convex optimization problems.
It is particularly effective for problems with up to about 100 dimensions and is known for its
robustness and ability to handle complex fitness landscapes.

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
    .ConfigureOptimizer(new CMAESOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with CMAESOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CMAESOptimizer(IFullModel<,,>,CMAESOptimizerOptions<,,>,IEngine)` | Initializes a new instance of the CMAESOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Deserializes a byte array to restore the state of the CMA-ES optimizer. |
| `EvaluatePopulationWithModels(Matrix<>,OptimizationInputData<,,>,IFullModel<,,>)` | Evaluates the fitness of each individual in the population and returns the best model. |
| `GenerateMultivariateNormalSample(Int32)` | Generates a sample from a multivariate normal distribution. |
| `GeneratePopulation` | Generates a new population of candidate solutions. |
| `GenerateStandardNormal` | Generates a standard normal random number. |
| `GetOptions` | Gets the current options of the CMA-ES optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the CMA-ES algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the CMA-ES algorithm. |
| `Serialize` | Serializes the current state of the CMA-ES optimizer into a byte array. |
| `UpdateDistribution(Matrix<>,Vector<>)` | Updates the distribution parameters of the CMA-ES algorithm based on the current population and their fitness values. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the CMA-ES optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_C` | The covariance matrix of the distribution. |
| `_mean` | The mean of the current distribution. |
| `_options` | The options specific to the CMA-ES optimization algorithm. |
| `_pc` | Evolution path for covariance matrix adaptation. |
| `_population` | The current population of candidate solutions. |
| `_ps` | Evolution path for step-size adaptation. |
| `_sigma` | The current step size. |

