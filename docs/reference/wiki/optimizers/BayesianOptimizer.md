---
title: "BayesianOptimizer"
description: "Represents a Bayesian Optimizer for optimization problems."
section: "Reference"
---

_Optimizers_

Represents a Bayesian Optimizer for optimization problems.

## For Beginners

Think of this optimizer as a smart guessing game. It tries to find the best solution by making educated guesses based on what it has learned from previous attempts. It's particularly useful when each guess is time-consuming or expensive to evaluate.

## How It Works

Bayesian Optimization is a powerful technique for optimizing black-box functions that are expensive to evaluate. It uses a probabilistic model to make predictions about the function's behavior and decides where to sample next.

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

