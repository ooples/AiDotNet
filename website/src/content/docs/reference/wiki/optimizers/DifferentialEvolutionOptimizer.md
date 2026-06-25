---
title: "DifferentialEvolutionOptimizer<T, TInput, TOutput>"
description: "Implements the Differential Evolution optimization algorithm for numerical optimization problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements the Differential Evolution optimization algorithm for numerical optimization problems.

## For Beginners

This optimizer works by evolving a population of candidate solutions over time.
It's inspired by biological evolution and is good at finding global optima in complex problem spaces.

## How It Works

Differential Evolution is a population-based optimization algorithm that is particularly well-suited
for solving non-linear, non-differentiable continuous space functions. It's known for its simplicity,
robustness, and effectiveness in various optimization scenarios.

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
    .ConfigureOptimizer(new DifferentialEvolutionOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with DifferentialEvolutionOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DifferentialEvolutionOptimizer(IFullModel<,,>,DifferentialEvolutionOptions<,,>,IEngine)` | Initializes a new instance of the DifferentialEvolutionOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Deserializes the Differential Evolution optimizer from a byte array. |
| `GenerateTrialModel(List<IFullModel<,,>>,Int32,Int32)` | Generates a trial model using the Differential Evolution algorithm's mutation and crossover operations. |
| `GetOptions` | Retrieves the current options of the Differential Evolution optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the Differential Evolution algorithm. |
| `InitializePopulation(,Int32)` | Initializes the population for the Differential Evolution algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the main optimization process using the Differential Evolution algorithm. |
| `Serialize` | Serializes the Differential Evolution optimizer to a byte array. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates the adaptive parameters based on the optimization progress. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Differential Evolution optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentCrossoverRate` | The current crossover rate used in the optimization process. |
| `_currentMutationRate` | The current mutation rate used in the optimization process. |
| `_deOptions` | Configuration options specific to the Differential Evolution algorithm. |

