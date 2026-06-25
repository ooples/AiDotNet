---
title: "NadamOptimizer"
description: "Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm."
section: "Reference"
---

_Optimizers_

Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm.

## For Beginners

Imagine you're rolling a smart ball down a hill. This ball can adjust its speed for different parts of the hill (adaptive learning rates), and it can look ahead to anticipate slopes (Nesterov's method). This combination helps it find the lowest point more efficiently.

## How It Works

Nadam combines the ideas of Adam (adaptive learning rates) and Nesterov accelerated gradient (NAG). It adapts the learning rates of each parameter and incorporates momentum using Nesterov's method.

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
    .ConfigureOptimizer(new NadamOptimizer<double, Tensor<double>, Tensor<double>>(model))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained with NadamOptimizer.");
```

