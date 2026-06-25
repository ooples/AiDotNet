---
title: "CMAESOptimizer"
description: "Implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm."
section: "Reference"
---

_Optimizers_

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

