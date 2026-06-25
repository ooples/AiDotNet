---
title: "TabuSearchOptimizer<T, TInput, TOutput>"
description: "Represents a Tabu Search optimizer for machine learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Represents a Tabu Search optimizer for machine learning models.

## For Beginners

Think of Tabu Search as a smart explorer:

- The explorer (optimizer) looks for the best solution in a complex landscape
- It remembers recently visited places (tabu list) to avoid going in circles
- It adapts its search strategy over time to balance between exploring new areas and refining good solutions

This method is particularly effective for problems with many local optima.

## How It Works

The TabuSearchOptimizer implements the Tabu Search algorithm, a metaheuristic search method used in optimization.
It explores the solution space by iteratively moving from one solution to the best solution in its neighborhood,
while keeping a list of recently visited solutions (the tabu list) to avoid cycling and encourage exploration of new areas.

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
    .ConfigureOptimizer(new TabuSearchOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with TabuSearchOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabuSearchOptimizer(IFullModel<,,>,TabuSearchOptions<,,>,GeneticBase<,,>,IFitnessCalculator<,,>,IEngine)` | Initializes a new instance of the TabuSearchOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Deserializes the TabuSearchOptimizer from a byte array. |
| `GenerateNeighbors(IFullModel<,,>,OptimizationInputData<,,>)` | Generates a list of neighboring solutions from the current solution. |
| `GetOptions` | Gets the current options for the Tabu Search algorithm. |
| `GetSolutionHash(IFullModel<,,>)` | Creates a hash representation of a solution for the tabu list. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used in the Tabu Search algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the optimization process to find the best solution for the given input data. |
| `Serialize` | Serializes the TabuSearchOptimizer to a byte array. |
| `UpdateAdaptiveParameters(Int32)` | Updates the adaptive parameters used in the Tabu Search algorithm. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the options for the Tabu Search algorithm. |
| `UpdateTabuList(HashSet<String>,String)` | Updates the tabu list with a new solution hash. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentMutationRate` | The current mutation rate used in generating neighboring solutions. |
| `_currentNeighborhoodSize` | The current size of the neighborhood to explore in each iteration. |
| `_currentTabuListSize` | The current size of the tabu list. |
| `_geneticAlgorithm` | The genetic algorithm used to handle mutations and generate neighboring solutions. |
| `_tabuOptions` | The options specific to the Tabu Search algorithm. |

