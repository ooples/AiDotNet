---
title: "ParticleSwarmOptimizer<T, TInput, TOutput>"
description: "Implements a Particle Swarm Optimization algorithm for finding optimal solutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Implements a Particle Swarm Optimization algorithm for finding optimal solutions.

## For Beginners

Particle Swarm Optimization is like a group of birds searching for food.

Imagine a flock of birds looking for the best food source in a field:

- Each bird is a "particle" in the swarm
- Each bird remembers where it personally found the most food
- The flock shares information about where the most food has been found overall
- Birds adjust their flight based on their own experience and what they learn from others
- Over time, the whole flock converges on the best food source

This approach is very effective for finding good solutions to complex problems where
traditional methods might get stuck in suboptimal areas.

## How It Works

Particle Swarm Optimization (PSO) is a population-based stochastic optimization technique inspired by the social
behavior of birds flocking or fish schooling. The algorithm maintains a population (swarm) of candidate solutions
(particles) that move around in the search space according to simple mathematical formulas that consider the
particle's position and velocity.

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
    .ConfigureOptimizer(new ParticleSwarmOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with ParticleSwarmOptimizer.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParticleSwarmOptimizer(IFullModel<,,>,ParticleSwarmOptimizationOptions<,,>)` | Initializes a new instance of the ParticleSwarmOptimizer class with the specified options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` | Reconstructs the particle swarm optimizer from a serialized byte array. |
| `GetOptions` | Gets the current options for this optimizer. |
| `InitializeAdaptiveParameters` | Initializes the adaptive parameters used by the PSO algorithm. |
| `Optimize(OptimizationInputData<,,>)` | Performs the particle swarm optimization to find the best solution for the given input data. |
| `Serialize` | Serializes the particle swarm optimizer to a byte array for storage or transmission. |
| `UpdateAdaptiveParameters(OptimizationStepData<,,>,OptimizationStepData<,,>)` | Updates adaptive parameters based on optimization progress. |
| `UpdateOptions(OptimizationAlgorithmOptions<,,>)` | Updates the optimizer's options with the provided options. |
| `UpdateVelocity(Vector<>,Vector<>,Vector<>,Vector<>)` | Updates the velocity of a particle based on inertia, cognitive, and social components. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentCognitiveWeight` | The current cognitive weight that controls a particle's attraction to its personal best position. |
| `_currentInertia` | The current inertia weight that controls a particle's tendency to continue its current trajectory. |
| `_currentSocialWeight` | The current social weight that controls a particle's attraction to the global best position. |
| `_psoOptions` | Configuration options specific to Particle Swarm Optimization. |
| `_random` | Random number generator for stochastic components of the algorithm. |

