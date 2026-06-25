---
title: "AntColonyOptimizer<T, TInput, TOutput>"
description: "Implements the Ant Colony Optimization algorithm for solving optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Ant Colony Optimization algorithm for solving optimization problems.

## For Beginners

Think of this algorithm as a group of ants searching for the best path to food.
Each ant leaves a trail (pheromone) that other ants can follow. Over time, the best paths get stronger trails,
leading to better solutions.

## How It Works

Ant Colony Optimization is inspired by the behavior of ants in finding paths between their colony and food sources.
It uses virtual "ants" to explore the solution space and find optimal solutions.

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
    .ConfigureOptimizer(new AntColonyOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with AntColonyOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AntColonyOptimizer(IFullModel<,,>,AntColonyOptimizationOptions<,,>,IEngine)` | Initializes a new instance of the AntColonyOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConstructSolution(Matrix<>,)` | Constructs a solution (model) based on the current pheromone levels and input data. |
| `Deserialize(Byte[])` | Restores the state of the optimizer from a byte array. |
| `GetOptions` | Gets the current options for the Ant Colony Optimization algorithm. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the Ant Colony Optimization algorithm. |
| `InitializePheromones(Int32)` | Initializes the pheromone matrix for the Ant Colony Optimization algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the Ant Colony Optimization algorithm. |
| `SelectNextFeature(Int32,Matrix<>,,Boolean[])` | Selects the next feature to be included in the solution based on pheromone levels and heuristics. |
| `Serialize` | Converts the current state of the optimizer into a byte array for storage or transmission. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters based on the current and previous optimization steps. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Ant Colony Optimization algorithm. |
| `UpdatePheromoneEvaporationRate(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the pheromone evaporation rate based on the current and previous optimization steps. |
| `UpdatePheromoneIntensity(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the pheromone intensity based on the current and previous optimization steps. |
| `UpdatePheromones(Matrix<>,List<IFullModel<,,>>)` | Updates the pheromone levels based on the quality of solutions found in the current iteration. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_antColonyOptions` | Options specific to the Ant Colony Optimization algorithm. |
| `_currentPheromoneEvaporationRate` | The current rate at which pheromone evaporates from the trails. |
| `_currentPheromoneIntensity` | The current intensity of pheromone deposited by ants. |

